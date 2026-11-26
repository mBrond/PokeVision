"""Dá uma nota de suspeita a cada imagem do manifesto, para a revisão manual começar
pelas imagens com mais chance de estarem erradas (fusões, fan art humanizada, rótulo
trocado, vários personagens...).

Usa o CLIP ViT-L/14 (OpenAI) de quatro jeitos, e a nota final é a média dos percentis:

    p_rotulo     o CLIP, sem treino nenhum, acha que a imagem é da espécie do rótulo?
    knn          os 10 vizinhos mais parecidos (no espaço do CLIP) têm o mesmo rótulo?
    z_centroide  quão longe a imagem está do "jeito típico" da própria classe
    estranho     parece mais "fusão / humanizado / quadrinho..." do que "um Pokémon"?

O CLIP não sabe contar, então imagens com mais de um Pokémon são procuradas à parte
(classifier/figuras.py): em fundo liso, as figuras separadas são classificadas pela
ResNet. Duas espécies diferentes vão para o topo da lista ("2+ espécies: A + B").

    python classifier/triagem_clip.py

Os embeddings ficam em cache (data/classificador/clip_vitl14.npz, por MD5), então rodar
de novo depois de excluir imagens é rápido. Saída: data/classificador/triagem.csv.
"""
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from classifier.fundos import compor
from classifier.preparar_dados import GEN1, abrir

ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "data/classificador"
CACHE = DADOS / "clip_vitl14.npz"
SAIDA = DADOS / "triagem.csv"
MODELO, PESOS = "ViT-L-14", "openai"
VIZINHOS = 10

# nomes como o CLIP os conhece
NOME_TEXTO = {"Nidoran-F": "Nidoran♀ (female Nidoran)", "Nidoran-M": "Nidoran♂ (male Nidoran)",
              "Farfetchd": "Farfetch'd"}
MODELOS_ESPECIE = [
    "an illustration of {}, a pokemon.",
    "a picture of the pokemon {}.",
    "{} from pokemon.",
    "a pixel art sprite of {}, a pokemon.",
]

VARIOS = "vários Pokémon"
# espécies que já parecem "várias" (ovos do Exeggcute, filhote da Kangaskhan, Shellder do
# Slowbro...): o CLIP as marcava como grupo, e a separação de figuras acha pedaços soltos
VARIAS_POR_NATUREZA = {"Exeggcute", "Exeggutor", "Dugtrio", "Magneton", "Doduo", "Dodrio",
                       "Weezing", "Kangaskhan", "Slowbro"}
# confiança mínima da ResNet para contar a espécie de uma figura. Com 0,6, 7 de 33 marcadas
# eram pedaços de um Pokémon só (asas do Zubat) ou balões de fala; com 0,8, nenhum acerto saiu
CONF_FIGURA = 0.8

# descrição em inglês -> motivo mostrado na revisão
ESTRANHO = {
    "a fusion of two different pokemon combined into one creature": "fusão",
    "a pokemon drawn as a human person": "humanizado",
    "a person wearing a pokemon cosplay costume": "cosplay",
    "a group of several different pokemon together": VARIOS,
    "a person or pokemon trainer next to a pokemon": "pessoa junto",
    "a comic page with speech bubbles and panels": "quadrinho",
    "a screenshot with a lot of text": "muito texto",
    "an evolution chart showing a pokemon and its evolutions": "linha evolutiva",
    "a very dark, almost black image": "escura",
    # "a pokemon seen from behind" foi testado e removido: em 32 imagens marcadas,
    # quase todas estavam de frente. Sprites de costas aparecem pelos outros sinais.
}
# o que é normal (cartas, pelúcias e frames do anime são bem-vindos: a câmera vai ver isso)
NORMAL = [
    "official artwork of a single pokemon",
    "a fan drawing of a single pokemon",
    "a pixel art sprite of a single pokemon",
    "a pokemon trading card",
    "a screenshot of a single pokemon from the anime",
    "a plush toy or figure of a pokemon",
    "a 3d render of a single pokemon",
]


class Imagens(Dataset):
    def __init__(self, caminhos, preprocess):
        self.caminhos, self.preprocess = caminhos, preprocess

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        return self.preprocess(compor(abrir(self.caminhos[i]), aleatorio=False))


def embeddings(linhas, modelo, preprocess, device):
    """Embeddings normalizados de todas as imagens, reaproveitando o cache por MD5."""
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE)
        cache = dict(zip(z["md5"].tolist(), z["emb"]))

    def salvar():
        tmp = CACHE.with_suffix(".tmp.npz")
        np.savez(tmp, md5=np.array(list(cache)), emb=np.stack(list(cache.values())))
        tmp.replace(CACHE)

    faltam = [l for l in linhas if l["md5"] not in cache]
    if faltam:
        print(f"Calculando {len(faltam)} embeddings ({len(linhas) - len(faltam)} em cache)...")
        lote = 32
        loader = DataLoader(Imagens([l["caminho"] for l in faltam], preprocess),
                            batch_size=lote, num_workers=4)
        dtype = modelo.visual.conv1.weight.dtype
        with torch.inference_mode():
            for k, x in enumerate(loader):
                e = modelo.encode_image(x.to(device, dtype))
                e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
                for l, v in zip(faltam[k * lote:(k + 1) * lote], e):
                    cache[l["md5"]] = v.astype(np.float16)
                if k % 50 == 49:                     # salva a cada ~1.600 imagens
                    salvar()
                    print(f"  {(k + 1) * lote}/{len(faltam)}", flush=True)
        salvar()

    return np.stack([cache[l["md5"]] for l in linhas]).astype(np.float32)


def textos(modelo, tokenizer, frases, device):
    with torch.inference_mode():
        t = modelo.encode_text(tokenizer(frases).to(device)).float()
    return torch.nn.functional.normalize(t, dim=-1).cpu().numpy()


def percentil(x):
    """Posição de cada valor entre 0 (menor) e 1 (maior)."""
    ordem = np.argsort(np.argsort(x))
    return ordem / max(1, len(x) - 1)


def main():
    import open_clip

    with open(DADOS / "manifest.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    if "md5" not in linhas[0]:
        raise SystemExit("Manifesto sem coluna md5. Rode de novo: python classifier/preparar_dados.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp32 de propósito: na GTX 1650 (sem tensor cores) o fp16 ficou 5x mais lento
    # (2 vs 10 imagens/s). Em fp32 o ViT-L/14 usa ~2,1 GB de VRAM.
    modelo, _, preprocess = open_clip.create_model_and_transforms(
        MODELO, pretrained=PESOS, device=device, precision="fp32")
    modelo.eval()
    tokenizer = open_clip.get_tokenizer(MODELO)

    emb = embeddings(linhas, modelo, preprocess, device)
    rotulo = np.array([GEN1.index(l["classe"]) for l in linhas])
    grupo = np.array([int(l["grupo"]) for l in linhas])

    # 1. classificação zero-shot: média de alguns modelos de frase por espécie
    t_esp = np.stack([
        textos(modelo, tokenizer, [m.format(NOME_TEXTO.get(c, c)) for m in MODELOS_ESPECIE], device).mean(0)
        for c in GEN1])
    t_esp /= np.linalg.norm(t_esp, axis=1, keepdims=True)
    logits = 100 * emb @ t_esp.T
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    p_rotulo = prob[np.arange(len(linhas)), rotulo]
    pred = prob.argmax(1)

    # 2. vizinhos: fração dos 10 mais parecidos com o mesmo rótulo
    #    (ignora o próprio grupo de quase-duplicatas, que concordaria sempre)
    emb_t = torch.from_numpy(emb).to(device).half()
    knn = np.zeros(len(linhas))
    for i in range(0, len(linhas), 1024):
        sim = (emb_t[i:i + 1024] @ emb_t.T).float().cpu().numpy()
        for j in range(sim.shape[0]):
            sim[j, grupo == grupo[i + j]] = -1
        viz = np.argpartition(-sim, VIZINHOS, axis=1)[:, :VIZINHOS]
        knn[i:i + 1024] = (rotulo[viz] == rotulo[i:i + 1024, None]).mean(1)

    # 3. distância ao centro da classe, padronizada dentro da classe
    z = np.zeros(len(linhas))
    for c in range(len(GEN1)):
        m = rotulo == c
        centro = emb[m].mean(0)
        centro /= np.linalg.norm(centro)
        s = emb[m] @ centro
        z[m] = (s - s.mean()) / (s.std() + 1e-6)

    # 4. "estranho": softmax entre as descrições estranhas e as normais
    frases_estranhas = list(ESTRANHO)
    t_desc = textos(modelo, tokenizer, frases_estranhas + NORMAL, device)
    ld = 100 * emb @ t_desc.T
    pd = np.exp(ld - ld.max(1, keepdims=True))
    pd /= pd.sum(1, keepdims=True)
    pe = pd[:, :len(frases_estranhas)]
    naturais = np.array([l["classe"] in VARIAS_POR_NATUREZA for l in linhas])
    pe[naturais, frases_estranhas.index(next(k for k, v in ESTRANHO.items() if v == VARIOS))] = 0
    estranho = pe.sum(1)
    motivo = [ESTRANHO[frases_estranhas[k]] for k in pe.argmax(1)]

    suspeita = np.mean([percentil(-p_rotulo), percentil(-knn), percentil(-z), percentil(estranho)], axis=0)

    # 5. mais de um Pokémon: figuras separadas no fundo liso, cada uma classificada pela ResNet
    del modelo
    torch.cuda.empty_cache()
    n_fig, esp_fig, motivo_fig = figuras(linhas, device)
    for k, m in enumerate(motivo_fig):
        if m.startswith("2+"):
            suspeita[k] = 0.9 + 0.1 * suspeita[k]     # vão para o topo da lista geral
        elif m:
            suspeita[k] = max(suspeita[k], 0.75)

    with open(SAIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["md5", "caminho", "classe", "split", "fonte", "suspeita", "clip_pred",
                    "clip_pred_prob", "p_rotulo", "knn", "z_centroide", "estranho", "n_figuras",
                    "especies_figuras", "motivo"])
        for k, l in enumerate(linhas):
            motivos = [motivo_fig[k], motivo[k] if estranho[k] >= 0.5 else ""]
            w.writerow([l["md5"], l["caminho"], l["classe"], l["split"], l["fonte"],
                        f"{suspeita[k]:.4f}", GEN1[pred[k]], f"{prob[k, pred[k]]:.3f}",
                        f"{p_rotulo[k]:.3f}", f"{knn[k]:.2f}", f"{z[k]:.2f}",
                        f"{estranho[k]:.3f}", n_fig[k], esp_fig[k],
                        " · ".join(dict.fromkeys(m for m in motivos if m))])

    acerto = (pred == rotulo).mean()
    print(f"CLIP zero-shot concorda com o rótulo em {acerto:.1%} das imagens")
    print(f"Discordâncias com o CLIP confiante (>80%): {((pred != rotulo) & (prob.max(1) > 0.8)).sum()}")
    print(f"Imagens com 'estranho' >= 0,5: {(estranho >= 0.5).sum()}")
    print(f"Separação de figuras: {sum(n != '' for n in n_fig)} imagens com fundo liso, "
          f"{sum(m.startswith('2+') for m in motivo_fig)} com 2+ espécies, "
          f"{sum(m == 'várias figuras' for m in motivo_fig)} com várias figuras da mesma espécie")
    print(f"Triagem salva em {SAIDA}")


def figuras(linhas, device):
    """Por imagem: nº de figuras ("" = fundo não liso), espécies das figuras e motivo."""
    from classifier import figuras as fg
    from classifier.treinar_resnet import CLASSES_JSON, PESOS, criar_modelo, eval_tf

    classes = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
    resnet = criar_modelo(len(classes), device)
    resnet.load_state_dict(torch.load(PESOS, map_location=device))
    caixas = fg.caixas(linhas)
    por_figura = fg.classificar_figuras(linhas, caixas, resnet, classes, eval_tf, device)

    n_fig, esp_fig, motivo = [], [], []
    for l in linhas:
        cx = caixas[l["md5"]]
        n_fig.append("" if cx is None else max(1, len(cx)))
        figs = por_figura.get(l["md5"], [])
        especies = sorted({e for e, p in figs if p >= CONF_FIGURA}, key=GEN1.index)
        esp_fig.append("+".join(e for e, _ in figs))
        if len(especies) >= 2:
            motivo.append("2+ espécies: " + " + ".join(especies))
        elif figs and l["classe"] not in VARIAS_POR_NATUREZA:
            motivo.append("várias figuras")
        else:
            motivo.append("")
    return n_fig, esp_fig, motivo


if __name__ == "__main__":
    main()
