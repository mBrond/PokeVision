"""Monta o dataset do classificador com as 151 espécies da 1ª geração.

Junta três fontes, normaliza os nomes, remove duplicatas e separa treino/val/teste
sem deixar cópias da mesma imagem em splits diferentes. Não copia imagens: gera um
manifesto CSV que o treino lê. Imagens listadas em classifier/excluir.txt (revisão
manual) ficam de fora, e cada imagem que está em classifier/splits.csv continua no
mesmo split.

    python classifier/preparar_dados.py [--refazer-split]

Saída em data/classificador/:
    manifest.csv    caminho, classe, split, grupo, fonte, transparente, md5
    classes.json    as 151 classes em ordem da Pokédex
    relatorio.txt   contagens por classe e o que foi descartado

e em classifier/splits.csv (versionado no git): o split de cada imagem, por MD5. Com
ele, um clone novo do repositório monta exatamente o mesmo treino/val/teste.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
SAIDA = ROOT / "data/classificador"
EXCLUIR = Path(__file__).resolve().parent / "excluir.txt"   # versionado no git: é trabalho manual
SPLITS = Path(__file__).resolve().parent / "splits.csv"     # versionado no git: fixa o teste
SEED = 42
PROPORCOES = {"train": 0.70, "val": 0.15, "test": 0.15}

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
LADO_MINIMO = 24          # imagens menores que isso são descartadas
DIST_QUASE_IGUAL = 4      # distância de Hamming entre dHashes para considerar "quase igual"
FRACAO_TRANSPARENTE = 0.05

GEN1 = [
    "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
    "Squirtle", "Wartortle", "Blastoise", "Caterpie", "Metapod", "Butterfree",
    "Weedle", "Kakuna", "Beedrill", "Pidgey", "Pidgeotto", "Pidgeot", "Rattata",
    "Raticate", "Spearow", "Fearow", "Ekans", "Arbok", "Pikachu", "Raichu",
    "Sandshrew", "Sandslash", "Nidoran-F", "Nidorina", "Nidoqueen", "Nidoran-M",
    "Nidorino", "Nidoking", "Clefairy", "Clefable", "Vulpix", "Ninetales",
    "Jigglypuff", "Wigglytuff", "Zubat", "Golbat", "Oddish", "Gloom", "Vileplume",
    "Paras", "Parasect", "Venonat", "Venomoth", "Diglett", "Dugtrio", "Meowth",
    "Persian", "Psyduck", "Golduck", "Mankey", "Primeape", "Growlithe", "Arcanine",
    "Poliwag", "Poliwhirl", "Poliwrath", "Abra", "Kadabra", "Alakazam", "Machop",
    "Machoke", "Machamp", "Bellsprout", "Weepinbell", "Victreebel", "Tentacool",
    "Tentacruel", "Geodude", "Graveler", "Golem", "Ponyta", "Rapidash", "Slowpoke",
    "Slowbro", "Magnemite", "Magneton", "Farfetchd", "Doduo", "Dodrio", "Seel",
    "Dewgong", "Grimer", "Muk", "Shellder", "Cloyster", "Gastly", "Haunter",
    "Gengar", "Onix", "Drowzee", "Hypno", "Krabby", "Kingler", "Voltorb",
    "Electrode", "Exeggcute", "Exeggutor", "Cubone", "Marowak", "Hitmonlee",
    "Hitmonchan", "Lickitung", "Koffing", "Weezing", "Rhyhorn", "Rhydon", "Chansey",
    "Tangela", "Kangaskhan", "Horsea", "Seadra", "Goldeen", "Seaking", "Staryu",
    "Starmie", "Mr. Mime", "Scyther", "Jynx", "Electabuzz", "Magmar", "Pinsir",
    "Tauros", "Magikarp", "Gyarados", "Lapras", "Ditto", "Eevee", "Vaporeon",
    "Jolteon", "Flareon", "Porygon", "Omanyte", "Omastar", "Kabuto", "Kabutops",
    "Aerodactyl", "Snorlax", "Articuno", "Zapdos", "Moltres", "Dratini",
    "Dragonair", "Dragonite", "Mewtwo", "Mew",
]


def chave(nome):
    """'Mr. Mime', 'MrMime', 'mr-mime' -> 'mrmime'; 'Nidoran♀', 'nidoran-f' -> 'nidoranf'.

    As classes usam nomes ASCII ('Nidoran-F') porque ♀/♂ quebram o console do
    Windows e o cv2.putText; o slug da PokéAPI ('nidoran-f') continua certo.
    """
    nome = nome.lower().replace("♀", "f").replace("♂", "m")
    return "".join(c for c in nome if c.isalnum())


CANONICO = {chave(n): n for n in GEN1}


def fontes():
    """(nome da fonte, pasta com uma subpasta por espécie). Ordem = prioridade nas duplicatas."""
    import kagglehub
    kaggle = Path(kagglehub.dataset_download(
        "mikoajkolman/pokemon-images-first-generation17000-files")) / "pokemon"
    return [
        ("kaggle-17k", kaggle),                                    # arte variada, ~17 mil
        ("pokemondata", ROOT / "data/PokemonData"),                # quase tudo repetido do anterior
        ("sprites-1000", ROOT / "data/pokemon-dataset-1000/dataset"),  # 40 sprites de jogos por espécie
    ]


def listar():
    arquivos, ignorados = [], collections.Counter()
    for fonte, pasta in fontes():
        if not pasta.exists():
            print(f"Aviso: {fonte} não encontrada em {pasta}, pulando")
            continue
        for sub in sorted(pasta.iterdir()):
            if not sub.is_dir():
                continue
            classe = CANONICO.get(chave(sub.name))
            if classe is None:
                ignorados[f"{fonte}/{sub.name}"] += 1    # outras gerações ou formas regionais
                continue
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in EXTENSOES:
                    arquivos.append((str(f), classe, fonte))
    return arquivos, ignorados


def abrir(caminho):
    """Abre qualquer formato (GIF: 1º frame). Devolve RGBA se a imagem tiver transparência."""
    im = Image.open(caminho)
    im.seek(0)
    tem_alfa = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    return im.convert("RGBA") if tem_alfa else im.convert("RGB")


def analisar(item):
    caminho, classe, fonte = item
    try:
        dados = open(caminho, "rb").read()
        im = abrir(caminho)
    except Exception:
        return None
    if min(im.size) < LADO_MINIMO:
        return None

    transparente = False
    if im.mode == "RGBA":
        alfa = im.getchannel("A")
        transparente = bool((np.asarray(alfa) < 128).mean() >= FRACAO_TRANSPARENTE)
        caixa = alfa.point(lambda a: 255 if a >= 128 else 0).getbbox()
        branco = Image.new("RGB", im.size, "white")
        branco.paste(im, mask=alfa)
        im = branco
    else:
        # fundo = cor do canto superior esquerdo
        fundo = Image.new("RGB", im.size, im.getpixel((0, 0)))
        dif = ImageChops.difference(im, fundo).convert("L")
        caixa = dif.point(lambda v: 255 if v > 30 else 0).getbbox()

    # o hash é feito só sobre o Pokémon: em sprites pequenos com muita margem, a margem
    # domina a miniatura e espécies diferentes acabam com hashes parecidos
    if caixa and (caixa[2] - caixa[0]) >= 8 and (caixa[3] - caixa[1]) >= 8:
        im = im.crop(caixa)

    # dHash 8x8: compara o brilho de pixels vizinhos numa miniatura 9x8
    cinza = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (cinza[:, 1:] > cinza[:, :-1]).flatten()
    dhash = int(np.packbits(bits).view(">u8")[0])

    return caminho, classe, fonte, hashlib.md5(dados).hexdigest(), dhash, transparente


class UniaoBusca:
    def __init__(self, n):
        self.pai = list(range(n))

    def raiz(self, i):
        while self.pai[i] != i:
            self.pai[i] = self.pai[self.pai[i]]
            i = self.pai[i]
        return i

    def unir(self, a, b):
        self.pai[self.raiz(a)] = self.raiz(b)


def pares_quase_iguais(hashes, classes, bloco=1024):
    """Pares (i, j) da mesma classe com dHash a no máximo DIST_QUASE_IGUAL bits de distância.

    Só compara dentro da classe: uma quase-cópia com outro rótulo não vaza a resposta
    para o teste, e comparar entre classes junta sprites diferentes num grupo gigante.
    """
    h = np.array(hashes, dtype=np.uint64)
    c = np.array(classes)
    bits1 = np.bitwise_count(h)
    util = (bits1 > 3) & (bits1 < 61)      # hash quase todo 0 ou 1 = imagem lisa, colide à toa
    for i in range(0, len(h), bloco):
        d = np.bitwise_count(h[i:i + bloco, None] ^ h[None, :])
        a, b = np.nonzero(d <= DIST_QUASE_IGUAL)
        a += i
        ok = (b > a) & util[a] & util[b] & (c[a] == c[b])
        yield from zip(a[ok].tolist(), b[ok].tolist())


def ler_excluidos():
    """MD5s marcados para sair do dataset (classifier/excluir.txt, gerado pela revisão)."""
    if not EXCLUIR.exists():
        return set()
    linhas = EXCLUIR.read_text(encoding="utf-8").splitlines()
    return {l.split()[0] for l in linhas if l.strip() and not l.startswith("#")}


def ler_splits_anteriores():
    """Split já definido de cada imagem (por MD5 e por caminho), para mantê-lo.

    classifier/splits.csv vale mais; o manifesto local só cobre o caso de ele não existir.
    """
    anterior = {}
    manifesto = SAIDA / "manifest.csv"
    if manifesto.exists():
        with open(manifesto, encoding="utf-8") as f:
            for l in csv.DictReader(f):
                anterior[l["caminho"]] = l["split"]
                if l.get("md5"):
                    anterior[l["md5"]] = l["split"]
    if SPLITS.exists():
        with open(SPLITS, encoding="utf-8") as f:
            anterior.update((l["md5"], l["split"]) for l in csv.DictReader(f))
    return anterior


def separar(itens, grupos, rng, anterior):
    """Cada grupo de quase-duplicatas vai inteiro para um split, equilibrando por classe.

    Imagens que já tinham split no manifesto anterior continuam nele: excluir imagens
    ou acrescentar fontes não embaralha o teste, e modelos treinados antes continuam
    comparáveis.
    """
    membros = collections.defaultdict(list)
    for i, g in enumerate(grupos):
        membros[g].append(i)

    split, por_classe = {}, collections.defaultdict(list)
    for g, ids in membros.items():
        antigos = [anterior.get(itens[i]["md5"]) or anterior.get(Path(itens[i]["caminho"]).as_posix())
                   for i in ids]
        antigos = [s for s in antigos if s]
        if antigos:
            split[g] = collections.Counter(antigos).most_common(1)[0][0]
            continue
        classe = collections.Counter(itens[i]["classe"] for i in ids).most_common(1)[0][0]
        por_classe[classe].append(g)

    # limites acumulados: [0, 0.70) treino, [0.70, 0.85) val, [0.85, 1] teste
    limites, acc = [], 0.0
    for s, p in PROPORCOES.items():
        acc += p
        limites.append((acc, s))

    for classe in GEN1:
        gs = por_classe.get(classe, [])
        rng.shuffle(gs)
        n = sum(len(membros[g]) for g in gs)
        feitos = 0
        for g in gs:
            # o grupo cai no split onde fica o seu ponto médio na fila embaralhada;
            # assim grupos grandes não tendem a ir para um split específico
            meio = (feitos + len(membros[g]) / 2) / n
            split[g] = next(s for lim, s in limites if meio < lim + 1e-9)
            feitos += len(membros[g])
    return [split[g] for g in grupos]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refazer-split", action="store_true",
                        help="sorteia os splits de novo em vez de manter os de classifier/splits.csv")
    args = parser.parse_args()
    anterior = {} if args.refazer_split else ler_splits_anteriores()
    excluidos = ler_excluidos()

    arquivos, ignorados = listar()
    print(f"{len(arquivos)} arquivos em {len({a[1] for a in arquivos})} classes. Analisando...")

    with Pool() as pool:
        resultados = pool.map(analisar, arquivos, chunksize=64)
    ilegiveis = sum(r is None for r in resultados)
    itens = [dict(zip(("caminho", "classe", "fonte", "md5", "dhash", "transparente"), r))
             for r in resultados if r is not None]

    # 0. imagens descartadas na revisão manual (todas as cópias saem)
    antes = len(itens)
    itens = [it for it in itens if it["md5"] not in excluidos]
    n_excluidas = antes - len(itens)

    # 1. duplicatas exatas: mesma imagem com rótulos diferentes é descartada (não dá para saber qual vale)
    por_md5 = collections.defaultdict(list)
    for it in itens:
        por_md5[it["md5"]].append(it)
    conflitos, repetidos, unicos = [], 0, []
    for copias in por_md5.values():
        if len({c["classe"] for c in copias}) > 1:
            conflitos.append(copias)
            continue
        repetidos += len(copias) - 1
        unicos.append(copias[0])          # fica a da fonte com maior prioridade
    itens = unicos

    # 2. quase-duplicatas (redimensionadas, recomprimidas) ficam no mesmo split
    uf = UniaoBusca(len(itens))
    n_pares = 0
    for a, b in pares_quase_iguais([it["dhash"] for it in itens], [it["classe"] for it in itens]):
        uf.unir(a, b)
        n_pares += 1
    grupos = [uf.raiz(i) for i in range(len(itens))]
    splits = separar(itens, grupos, random.Random(SEED), anterior)

    SAIDA.mkdir(parents=True, exist_ok=True)
    with open(SAIDA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caminho", "classe", "split", "grupo", "fonte", "transparente", "md5"])
        for it, g, s in zip(itens, grupos, splits):
            w.writerow([Path(it["caminho"]).as_posix(), it["classe"], s, g, it["fonte"],
                        int(it["transparente"]), it["md5"]])
    (SAIDA / "classes.json").write_text(json.dumps(GEN1, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    with open(SPLITS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["md5", "split"])
        w.writerows(sorted((it["md5"], s) for it, s in zip(itens, splits)))

    # relatório
    tabela = collections.Counter((it["classe"], s) for it, s in zip(itens, splits))
    total_split = collections.Counter(splits)
    por_fonte = collections.Counter(it["fonte"] for it in itens)
    linhas = [
        f"Imagens no manifesto: {len(itens)}  ({dict(total_split)})",
        f"Por fonte: {dict(por_fonte)}",
        f"Com transparência: {sum(it['transparente'] for it in itens)}",
        f"Descartadas: {n_excluidas} na revisão manual, {ilegiveis} ilegíveis ou menores que "
        f"{LADO_MINIMO}px, {repetidos} cópias exatas, {sum(len(c) for c in conflitos)} em "
        f"{len(conflitos)} grupos de cópias com rótulos diferentes",
        f"Pares quase iguais agrupados: {n_pares} (maior grupo: "
        f"{max(collections.Counter(grupos).values())} imagens)",
        f"Pastas ignoradas (fora da 1ª geração): {len(ignorados)}",
        "",
        f"{'classe':12} {'train':>6} {'val':>5} {'test':>5}",
        *[f"{c:12} {tabela[c, 'train']:6} {tabela[c, 'val']:5} {tabela[c, 'test']:5}" for c in GEN1],
        "",
        "Conflitos de rótulo (todas as cópias descartadas):",
        *[" | ".join(f"{c['classe']}: {Path(c['caminho']).name}" for c in cs) for cs in conflitos],
        "",
        f"Pastas ignoradas: {sorted(ignorados)}",
    ]
    (SAIDA / "relatorio.txt").write_text("\n".join(linhas), encoding="utf-8")
    print("\n".join(linhas[:6]))

    poucas = [c for c in GEN1 if sum(tabela[c, s] for s in PROPORCOES) < 100]
    if poucas:
        print(f"Classes com menos de 100 imagens: {poucas}")
    print(f"Manifesto salvo em {SAIDA / 'manifest.csv'}")


if __name__ == "__main__":
    main()
