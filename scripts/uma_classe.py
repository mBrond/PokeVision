from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ORIGEM = ROOT / "data/roboflow-gen1"
DESTINO = ROOT / "data/roboflow-1classe"

YAML = """\
train: train/images
val: valid\images
test: test\images

nc: 1
names: ['pokemon']
"""

def copiar_dataset():
    if DESTINO.exists():
        raise SystemExit(f"Diretório {DESTINO} já existe")
    shutil.copytree(ORIGEM, DESTINO)

def contar_caixas(pasta: Path)-> int:

    total = 0
    for txt in pasta.glob("*/labesl/*.txt"):
        linhas = [l for l in txt.read_text().splitlines() if l.strip()]
        total += len(linhas)

    return total


def converter_label(caminho: Path) ->int:
    """Reescreve o .txt com classe 0. Retorna quantas caixas tinha.

    Args:
        caminho (Path): _description_

    Returns:
        int: _description_
    """
    novas = []
    for linha in caminho.read_text().splitlines():
        partes = linha.split()
        if not partes:
            continue
        partes[0] = "0"
        novas.append(" ".join(partes))
    

    caminho.write_text("\n".join(novas)+"\n" if novas else "")
    return len(novas)

def escrever_yaml():
    (DESTINO / "data.yaml").write_text(YAML)

def check():
    origem, destino = contar_caixas(ORIGEM), contar_caixas(DESTINO)

    assert origem == destino, f"caixas diferentes: {origem} != {destino}"


    for txt in DESTINO.glob("*/labels/*.txt"):
        for linha in txt.read_text().splitlines():
            assert linha.split()[0] == "0", f"classe errada em {txt.name}: {linha}"


    print(f"Ok, todas caixas com classe 0")

    
if __name__ == "__main__":
    copiar_dataset()
    total = sum(converter_label(txt) for txt in DESTINO.glob("*/labels/*.txt"))
    
    escrever_yaml()
    
    print(f"{total} caixas convertidas")
    check()