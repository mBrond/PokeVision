# Pokédex com Visão Computacional

Este projeto tem como objetivo criar uma experiência semelhante a uma Pokédex, utilizando técnicas de visão computacional para identificar e detectar Pokémon em tempo real pela câmera.

## Objetivo

O projeto combina dois fluxos principais:

- **Classificação de imagens:** identifica qual Pokémon está presente em uma imagem usando uma rede neural convolucional (ResNet18).
- **Detecção de objetos:** localiza Pokémon em imagens ou vídeo com o modelo YOLO.

## Tecnologias utilizadas

Python, PyTorch, torchvision, OpenCV, Ultralytics YOLO e Pillow.

## Estrutura do projeto

```
pokedex/
├── classifier/
│   ├── pokemon_classifier.ipynb   # treino e teste do classificador ResNet18 (72 classes)
│   ├── resnet_camera.py           # classificação em tempo real pela webcam
│   └── treinar_yolo_cls.py        # alternativa: classificador YOLOv8m-cls
├── detector/
│   ├── treinar_yolo_4classes.py   # YOLOv8n com 4 classes (Bulbasaur, Charmander, Charmeleon, Charizard)
│   ├── treinar_yolo_gen1.py       # YOLOv8m com as 149 classes do dataset Roboflow Gen 1
│   └── camera.py                  # detecção em tempo real pela webcam
├── scripts/
│   ├── separador.py               # gera o split train/val de data/PokemonData para o YOLO-cls
│   └── utils.py                   # ajuste de IDs de classe e organização images/labels
├── docs/                          # anotações antigas (ideias de datasets, fontes)
├── data/        (ignorado no git) # datasets
├── models/      (ignorado no git) # pesos treinados (classifier/) e pré-treinados (base/)
├── runs/        (ignorado no git) # resultados de treino do Ultralytics
└── images/      (ignorado no git) # imagens de exemplo para teste
```

Os scripts usam caminhos relativos à raiz do repositório, então podem ser executados de qualquer pasta.

## Dados

| Pasta em `data/` | Conteúdo | Fonte |
|---|---|---|
| `pokemon-dataset-1000/` | imagens por Pokémon | [Kaggle – pokemon-1000-yolo](https://www.kaggle.com/datasets/vedantkasargod/pokemon-1000-yolo) |
| `pokemon-reduzido/` | subconjunto com bounding boxes, 4 classes | derivado do anterior |
| `roboflow-gen1/` | 917 imagens com bounding boxes, 149 classes (CC BY 4.0) | [Roboflow – Pokemon Gen 1](https://universe.roboflow.com/anthonyk/pokemon-gen-1/dataset/2) |
| `PokemonData/` | ~6.800 imagens por classe (150 classes) | Kaggle |

O classificador ResNet baixa seu dataset automaticamente via `kagglehub` ([pokemon-images-first-generation17000-files](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files)).

## Requisitos

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Como executar

**Classificador (ResNet18)**

1. Abra `classifier/pokemon_classifier.ipynb` e execute as células para treinar; os pesos são salvos em `models/classifier/`.
2. Webcam: `python classifier/resnet_camera.py`

**Detector (YOLO)**

- Treinar: `python detector/treinar_yolo_gen1.py` (ou `treinar_yolo_4classes.py`)
- Webcam: `python detector/camera.py`

Os resultados de treino vão para `runs/detect/<nome>/`.

## Resultados atuais

| Modelo | Classes | Métrica |
|---|---|---|
| ResNet18 (classificação) | 72 | ~86–90% de acurácia de validação |
| YOLOv8n (detecção) | 4 | mAP50 ≈ 0,93 |
| YOLOv8m (detecção) | 149 | mAP50 ≈ 0,77 / mAP50-95 ≈ 0,70 |

## Próximos passos

- Pipeline em duas etapas: detector de 1 classe ("pokemon") + classificador no recorte.
- Expandir o classificador para as 151 classes e avaliar no conjunto de teste.
- Montar um conjunto de teste próprio com fotos da webcam (cartas, pelúcias, telas).
- API (FastAPI) integrada à PokéAPI.
