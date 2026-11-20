import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from classifier.treinar_resnet import criar_modelo, eval_tf

from collections import deque

ROOT = Path(__file__).resolve().parent
DETECTOR = ROOT / "runs/detect/gen1-1classe/weights/best.pt"
CLASSIFICADOR = ROOT / "models/classifier/resnet18_best.pth"
CLASSES = ROOT / "models/classifier/classes.json"


CONF_DETECCAO = 0.5
CONF_CLASSE = 0.6
MARGEM = 0.10

JANELA_FRAMES = 10
ESQUECER_APOS = 30


class Pokedex:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(DETECTOR)
        self.rastreador = YOLO(DETECTOR)
        self.classes = json.loads(CLASSES.read_text(encoding="utf-8"))
        self.classificador = criar_modelo(len(self.classes), self.device)
        self.classificador.load_state_dict(torch.load(CLASSIFICADOR, map_location=self.device))
        self.classificador.eval()

        self.historico = {}
        self.visto_em = {}
        self.frame_n = 0


    def recortar(self, frame, caixa):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = caixa
        mx, my = int((x2 - x1) * MARGEM), int((y2 - y1) * MARGEM)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)

        recorte = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        return Image.fromarray(recorte)

    def _classificar(self, frame, caixas):
        lote = torch.stack([eval_tf(self.recortar(frame, c)) for c in caixas]).to(self.device)
        
        return torch.softmax(self.classificador(lote), dim=1).cpu()

    def _esquecer(self):

        velhos = [t for t, n in self.visto_em.items() if self.frame_n - n > ESQUECER_APOS]
        for t in velhos:
            del self.historico[t]
            del self.visto_em[t]

    @torch.inference_mode()
    def rastrear(self, frame):
        """Frame de vídeo: segue cada Pokémon e decide pela média dos últimos frames."""
        self.frame_n += 1
        resultado = self.rastreador.track(frame, persist=True, conf=CONF_DETECCAO,
                                          tracker="bytetrack.yaml", verbose=False)[0]
        boxes = resultado.boxes
        if boxes.id is None:              #nada sendo seguido neste frame
            self._esquecer()
            return []

        caixas = boxes.xyxy.cpu().int().tolist()
        ids = boxes.id.int().tolist()
        probs = self._classificar(frame, caixas)

        saida = []
        for caixa, tid, p in zip(caixas, ids, probs):
            hist = self.historico.setdefault(tid, deque(maxlen=JANELA_FRAMES))
            hist.append(p)
            self.visto_em[tid] = self.frame_n

            media = torch.stack(list(hist)).mean(dim=0)
            conf, idx = media.max(dim=0)
            saida.append((caixa, self.classes[idx.item()], conf.item()))

        self._esquecer()
        return saida


    @torch.inference_mode()
    def identificar(self, frame):
        """
        Retorna uma lista de (caixa, nome, confianca) para cada Pokemon no frame.
        """

        resultado = self.detector(frame, conf=CONF_DETECCAO, verbose=False)[0]
        caixas = resultado.boxes.xyxy.cpu().int().tolist()
        if not caixas:
            return []

        confs, idxs = self._classificar(frame, caixas).max(dim=1)
        return [(c, self.classes[i], p) for c, i, p in zip(caixas, idxs.tolist(), confs.tolist())]


def desenhar(frame, deteccoes):
    for (x1, y1, x2, y2), nome, conf in deteccoes:
        if conf >= CONF_CLASSE:
            texto, cor = f"{nome} {conf:.0%}", (0, 200, 0)
        else:
            texto, cor = f"", (0,165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(frame, texto, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2, cv2.LINE_AA)


    return frame


def rodar_webcam(pokedex, camera):
    cap = cv2.VideoCapture(camera)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Erro webcam")
            break

        cv2.imshow("Pokedex", desenhar(frame, pokedex.rastrear(frame)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def rodar_imagem(pokedex, caminho):
    frame = cv2.imread(str(caminho))

    if frame is None:
        raise SystemExit(f"Nao abriu {caminho}")
    deteccoes = pokedex.identificar(frame)

    for _, nome, conf in deteccoes:
        print(f"{nome}: {conf:.1%}")
    cv2.imshow("Pokedex", desenhar(frame, deteccoes))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokedex com visao computacional")
    parser.add_argument("fonte", nargs="?", default="0")

    args = parser.parse_args()

    pokedex = Pokedex()
    if args.fonte.isdigit():
        rodar_webcam(pokedex, int(args.fonte))
    else:
        rodar_imagem(pokedex, args.fonte)