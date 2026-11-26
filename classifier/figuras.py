"""Acha imagens com mais de um Pokémon: separa as figuras do fundo e classifica cada uma.

O CLIP não sabe contar (marca Exeggcute e Dugtrio, que são um Pokémon só, como
"vários"), e o detector YOLO junta grupos numa caixa só. Aqui a ideia é mais simples:

1. Em imagens de fundo liso (transparente ou de cor única na borda), separa o
   primeiro plano do fundo e pega as figuras grandes que não se tocam.
2. Cada figura é classificada pela ResNet treinada.

Duas espécies diferentes com confiança viram "2+ espécies: A + B" na triagem; várias
figuras da mesma espécie (shiny lado a lado, folha de sprites) viram "várias figuras".
Imagens com cenário no fundo não dão sinal (cerca de 30% do dataset), e Pokémon
encostados um no outro contam como uma figura só.

As caixas não dependem do modelo e ficam em cache por MD5 em
data/classificador/figuras.json. A classificação das figuras é refeita a cada rodada
(é rápida e muda quando a ResNet é retreinada).
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from classifier.preparar_dados import abrir

CACHE = Path(__file__).resolve().parents[1] / "data/classificador/figuras.json"
LADO = 256             # a separação é feita numa miniatura com este lado
FRACAO_SEGUNDA = 0.25  # cada figura precisa ter ao menos 25% da área da maior
FRACAO_MINIMA = 0.02   # e 2% da imagem (ignora mãos soltas do Haunter, faíscas, texto)
MAX_FIGURAS = 4


def caixas_figuras(img):
    """Caixas (x1, y1, x2, y2), em pixels da imagem, das figuras grandes separadas.

    None quando o fundo não é liso (não dá para separar sem um modelo de segmentação);
    lista vazia quando há uma figura só.
    """
    w, h = img.size
    esc = LADO / max(w, h)
    p = img.resize((max(1, round(w * esc)), max(1, round(h * esc))), Image.BILINEAR)
    if p.mode == "RGBA":
        mask = np.asarray(p.getchannel("A")) > 128
    else:
        a = np.asarray(p.convert("RGB")).astype(np.int16)
        borda = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        fundo = np.median(borda, 0)
        if (np.abs(borda - fundo).max(1) < 30).mean() < 0.85:
            return None
        mask = np.abs(a - fundo).max(2) > 40
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    rotulos, n = ndimage.label(mask)
    if n < 2:
        return []
    areas = ndimage.sum(mask, rotulos, range(1, n + 1))
    ordem = np.argsort(-areas)
    fatias = ndimage.find_objects(rotulos)
    caixas = []
    for k in ordem[:MAX_FIGURAS]:
        if areas[k] < FRACAO_SEGUNDA * areas[ordem[0]] or areas[k] < FRACAO_MINIMA * mask.size:
            break
        ys, xs = fatias[k]
        mx, my = (xs.stop - xs.start) * 0.1, (ys.stop - ys.start) * 0.1
        caixas.append((max(0, int((xs.start - mx) / esc)), max(0, int((ys.start - my) / esc)),
                       min(w, int((xs.stop + mx) / esc)), min(h, int((ys.stop + my) / esc))))
    return caixas if len(caixas) >= 2 else []


def _caixas_de(caminho):
    try:
        return caixas_figuras(abrir(caminho))
    except Exception:
        return None


def caixas(linhas):
    """md5 -> caixas das figuras (None = sem sinal, [] = uma figura). Usa e atualiza o cache."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    faltam = [l for l in linhas if l["md5"] not in cache]
    if faltam:
        print(f"Separando figuras do fundo em {len(faltam)} imagens...")
        with Pool() as pool:
            for l, c in zip(faltam, pool.imap(_caixas_de, [l["caminho"] for l in faltam], chunksize=64)):
                cache[l["md5"]] = c
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return {l["md5"]: cache[l["md5"]] for l in linhas}


def classificar_figuras(linhas, por_md5, modelo, classes, transform, device):
    """md5 -> [(espécie, prob), ...] de cada figura, só para imagens com 2+ figuras."""
    import torch
    from classifier.fundos import compor

    out = {}
    modelo.eval()
    with torch.inference_mode():
        for l in linhas:
            cx = por_md5.get(l["md5"])
            if not cx:
                continue
            img = compor(abrir(l["caminho"]), aleatorio=False)
            x = torch.stack([transform(img.crop(tuple(c))) for c in cx]).to(device)
            p = torch.softmax(modelo(x), 1).max(1)
            out[l["md5"]] = [(classes[i], v) for i, v in zip(p.indices.tolist(), p.values.tolist())]
    return out
