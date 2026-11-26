"""Fundos aleatórios para as imagens com transparência.

Sem isso, o convert("RGB") deixa o fundo transparente preto, e o modelo aprende
"fundo preto = sprite". No treino, cada imagem transparente ganha um fundo novo a
cada época; na avaliação, fundo branco (determinístico).

Se existir a pasta data/fundos/ com fotos (de preferência do ambiente onde a câmera
vai ser usada), metade dos fundos sai de recortes dessas fotos.
"""
import random
from pathlib import Path

from PIL import Image

PASTA_FOTOS = Path(__file__).resolve().parents[1] / "data/fundos"
_fotos = None


def _listar_fotos():
    global _fotos
    if _fotos is None:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        _fotos = [p for p in PASTA_FOTOS.rglob("*") if p.suffix.lower() in exts] \
            if PASTA_FOTOS.exists() else []
    return _fotos


def _cor():
    return tuple(random.randint(0, 255) for _ in range(3))


def _gradiente(tamanho):
    vertical = random.random() < 0.5
    base = Image.new("RGB", (1, 2) if vertical else (2, 1))
    base.putpixel((0, 0), _cor())
    base.putpixel((0, 1) if vertical else (1, 0), _cor())
    return base.resize(tamanho, Image.BILINEAR)


def _manchas(tamanho):
    """Grade pequena de cores aleatórias ampliada: manchas suaves, tipo cenário desfocado."""
    k = random.randint(2, 6)
    base = Image.new("RGB", (k, k))
    base.putdata([_cor() for _ in range(k * k)])
    return base.resize(tamanho, Image.BICUBIC)


def _recorte_foto(caminho, tamanho):
    foto = Image.open(caminho).convert("RGB")
    w, h = foto.size
    escala = random.uniform(0.3, 1.0)
    cw, ch = max(1, int(w * escala)), max(1, int(h * escala))
    x, y = random.randint(0, w - cw), random.randint(0, h - ch)
    return foto.crop((x, y, x + cw, y + ch)).resize(tamanho, Image.BILINEAR)


def fundo_aleatorio(tamanho):
    fotos = _listar_fotos()
    if fotos and random.random() < 0.5:
        return _recorte_foto(random.choice(fotos), tamanho)
    r = random.random()
    if r < 0.15:
        return Image.new("RGB", tamanho, "white")
    if r < 0.40:
        return Image.new("RGB", tamanho, _cor())
    if r < 0.70:
        return _gradiente(tamanho)
    return _manchas(tamanho)


def compor(img, aleatorio):
    """Cola uma imagem RGBA sobre um fundo. Imagens sem alfa passam direto."""
    if img.mode != "RGBA":
        return img
    fundo = fundo_aleatorio(img.size) if aleatorio else Image.new("RGB", img.size, "white")
    fundo.paste(img, mask=img.getchannel("A"))
    return fundo
