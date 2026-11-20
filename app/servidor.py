# app/servidor.py
"""Servidor da Pokédex: stream da câmera com as detecções + JSON do Pokémon identificado."""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import base64

import numpy as np
from fastapi import File, HTTPException, UploadFile


from pokedex import Pokedex, desenhar
from app.pokeapi import buscar

STATIC = Path(__file__).resolve().parent / "static"
CAMERA = 0


class Camera:
    """Lê a webcam numa thread separada e guarda o último frame já processado."""

    def __init__(self, pokedex, indice=CAMERA):
        self.pokedex = pokedex
        self.cap = cv2.VideoCapture(indice)
        self.lock = threading.Lock()
        self.jpeg = None
        self.info = {"detectado": False}
        self.rodando = True
        self.lock = threading.Lock()
        self.lock_modelo = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()


    def identificar(self, frame):
        with self.lock_modelo:
            return self.pokedex.identificar(frame)

    def rastrear(self, frame):
        with self.lock_modelo:
            return self.pokedex.rastrear(frame)


    def _loop(self):
        while self.rodando:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            deteccoes = self.rastrear(frame)
            #desenhar(frame, deteccoes)
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            with self.lock:
                self.jpeg = buffer.tobytes()
                self.info = self._melhor(deteccoes)

    @staticmethod
    def _melhor(deteccoes):
        if not deteccoes:
            return {"detectado": False}
        _, nome, conf = max(deteccoes, key=lambda d: d[2])
        return {"detectado": True, "nome": nome, "confianca": round(conf, 3)}

    def ultimo(self):
        with self.lock:
            return self.jpeg, self.info

    def parar(self):
        self.rodando = False
        self.cap.release()


camera = None


@asynccontextmanager
async def lifespan(app):
    global camera
    camera = Camera(Pokedex())   # carrega os modelos e abre a câmera uma vez só
    yield
    camera.parar()               #ao encerrar o servidor (Ctrl+C)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def gerar_mjpeg():
    while True:
        jpeg, _ = camera.ultimo()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 30)


@app.get("/video")
def video():
    return StreamingResponse(gerar_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/info")
def info():
    _, dados = camera.ultimo()
    return dados

@app.post("/identificar")
def identificar(arquivo: UploadFile = File(...)):
    dados = np.frombuffer(arquivo.file.read(), dtype=np.uint8)
    frame = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "O arquivo não é uma imagem válida")

    deteccoes = camera.identificar(frame)
    desenhar(frame, deteccoes)
    _, buffer = cv2.imencode(".jpg", frame)

    return {
        **Camera._melhor(deteccoes),
        "todos": [{"nome": n, "confianca": round(c, 3)} for _, n, c in deteccoes],
        "imagem": "data:image/jpeg;base64," + base64.b64encode(buffer).decode(),
    }

@app.get("/pokemon/{nome}")
def pokemon(nome: str):
    dados = buscar(nome)
    if dados is None:
        raise HTTPException(404, f"Não encontrei {nome} na PokéAPI")
    return dados