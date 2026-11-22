import argparse
import json
from pathlib import Path
import numpy as np

import cv2
import torch
from PIL import Image

from treinar_resnet import criar_modelo, eval_tf

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICADOR = ROOT / "models/classifier/resnet18_best.pth"
CLASSES = ROOT / "models/classifier/classes.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def carregar_modelo(device):
    """Cria a ResNet, carrega os pesos e as classes. Retorna (modelo, classes)."""

    classes = json.loads(CLASSES.read_text(encoding="utf-8"))

    classificador = criar_modelo(len(classes), device)
    classificador.load_state_dict(torch.load(CLASSIFICADOR, map_location=device))
    classificador.eval()

    return classificador, classes
    


def preparar_imagem(caminho, device):
    """Abre a imagem com PIL e aplica o eval_tf. Retorna um tensor 1×3×224×224."""
    img = Image.open(caminho) 
    batch = eval_tf(img).unsqueeze(0).to(device)
    return batch


def prever(modelo, x, classes, k=3):
    """Roda o modelo e imprime as k classes mais prováveis com a probabilidade."""


    with torch.no_grad():
        logits = modelo(x)


        probs = torch.softmax(logits, dim=1)


        valores, indices = probs.topk(k, dim=1)

    for valor, indice in zip(valores[0], indices[0]):
        nome = classes[indice.item()]
        print(F"{nome}: {valor.item():.1%}")


    return indices[0][0].item()

def gradcam(modelo, x, classe):
    """Retorna o mapa de calor (NumPy, valores 0 a 1) da classe, na grade da layer4."""

    guardado = {}

    def guardar_gradiente(grad):
        guardado["gradientes"] = grad

    def capturar(modulo, entrada, saida):
        guardado["ativacoes"] = saida
        saida.register_hook(guardar_gradiente)



    handle = modelo.layer4.register_forward_hook(capturar)

    try:    
        logits = modelo(x)
        modelo.zero_grad()
        logits[0, classe].backward()

    finally:
        handle.remove()


    A = guardado["ativacoes"]
    G = guardado["gradientes"]

    alfa = G.mean(dim=(2,3), keepdim=True)

    mapa = (alfa * A).sum(dim=1)
    mapa = torch.relu(mapa)
    mapa = mapa/ (mapa.max() + 1e-8)

    return mapa[0].detach().cpu().numpy()


def visualizar(caminho, mapa, titulo, saida=None, altura=300):
    """Mostra original | mapa de calor | sobreposição, lado a lado."""
    original = np.array(Image.open(caminho).convert("RGB"))     
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    h, w = original.shape[:2]
    largura = int(w * altura / h)
    original = cv2.resize(original, (largura, altura))         

    calor = cv2.resize(mapa, (largura, altura), interpolation=cv2.INTER_LINEAR)
    calor = cv2.applyColorMap(np.uint8(255 * calor), cv2.COLORMAP_JET)

    sobreposto = cv2.addWeighted(original, 0.55, calor, 0.45, 0)

    painel = np.hstack([original, calor, sobreposto])
    for cor, espessura in (((0, 0, 0), 4), ((255, 255, 255), 2)):  
        cv2.putText(painel, titulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.8, cor, espessura, cv2.LINE_AA)

    if saida:
        cv2.imwrite(str(saida), painel)
    cv2.imshow("Grad-CAM", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        


if __name__ == "__main__":
    #python classifier/gradcam.py images/Charmander.png              
    parser = argparse.ArgumentParser(description="Grad-CAM da ResNet em uma imagem")
    parser.add_argument("imagem", help="caminho de uma imagem com um Pokémon só")
    parser.add_argument("--classe", help="classe a explicar (padrão: a mais provável)")
    parser.add_argument("--saida", help="salva o painel neste arquivo")
    args = parser.parse_args()

    modelo, classes = carregar_modelo(DEVICE)
    x = preparar_imagem(args.imagem, DEVICE)

    classe = prever(modelo, x, classes)
    if args.classe:
        classe = classes.index(args.classe)

    mapa = gradcam(modelo, x, classe)
    visualizar(args.imagem, mapa, f"Grad-CAM: {classes[classe]}", args.saida)
