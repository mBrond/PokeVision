import random
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório


PASTA_ORIGEM = ROOT / "data/PokemonData"
PASTA_DESTINO = ROOT / "data/yolo_classification_pokemon"
PROPORCAO_TREINO = 0.8
EXTENSOES = tuple(['.jpg'])


def organizador():
	classes = [d for d in os.listdir(PASTA_ORIGEM) if os.path.isdir(os.path.join(PASTA_ORIGEM, d))]
	#classes = ["Charmander", "Charizard", "Charmeleon", "Pikachu", "Raichu",
	# "Bulbasaur", "Ivysaur", "Venusaur"]
	classes = classes[:int(len(classes)/2)]
	print(f"classes: {classes}")

	for split in ['train', 'val']:
		for classe in classes:
			os.makedirs(os.path.join(PASTA_DESTINO, split, classe), exist_ok=True)


	for classe in classes:
		caminho_classe_origem = os.path.join(PASTA_ORIGEM, classe)

		imagens = [f for f in os.listdir(caminho_classe_origem) if f.lower().endswith(EXTENSOES)]
		random.shuffle(imagens)

		limite_treino = int(len(imagens) * PROPORCAO_TREINO)

		imagens_treino = imagens[:limite_treino]
		imagens_val = imagens[limite_treino:]

		for img in imagens_treino:
			origem = os.path.join(caminho_classe_origem, img)
			destino = os.path.join(PASTA_DESTINO, 'train', classe, img)
			shutil.copy2(origem, destino)

		for img in imagens_val:
			origem = os.path.join(caminho_classe_origem, img)
			destino = os.path.join(PASTA_DESTINO, 'val', classe, img)
			shutil.copy2(origem, destino)


if __name__ == "__main__":
	organizador()