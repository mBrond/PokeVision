# Pokédex com Visão Computacional

Uma Pokédex que identifica Pokémon da 1ª geração em tempo real pela webcam ou a partir de imagens enviadas, e mostra a ficha de cada um (número, tipos, altura, peso, descrição e sprite) numa interface web.

O reconhecimento usa **dois modelos em sequência**: um detector **YOLOv8** que só localiza "um Pokémon" na imagem, e um classificador **ResNet18** que identifica a espécie no recorte. Os dados da ficha vêm da [PokéAPI](https://pokeapi.co/).

## Funcionalidades

- Detecção e identificação ao vivo pela webcam, com stream de vídeo no navegador
- Análise de imagens arrastadas para o visor ou coladas
- Vários Pokémon por imagem (uma caixa e uma classificação para cada)(Não funcionando corretamente)
- Ficha com dados da PokéAPI
- Reconhecimento offfline, internet usada apenas para a ficha do pokémon

## Resultados

Medidos em conjuntos de teste separados, nunca usados no treino nem na escolha do melhor modelo.

| Modelo | Tarefa | Dados de teste | Resultado |
|---|---|---|---|
| YOLOv8m, 1 classe | localizar Pokémon | 95 imagens | mAP50 0,987 · mAP50-95 0,850 · recall 1,0 |
| ResNet18 | identificar a espécie (142 classes) | ~2.500 imagens | 95,9% top-1 · 99,2% top-5 |

Os erros que sobram no classificador são quase todos entre evoluções da mesma linha (Poliwhirl/Poliwrath, Rhyhorn/Rhydon, Pidgeotto/Pidgeot).

## Arquitetura
**Pipeline de reconhecimento** (`pokedex.py`, classe `Pokedex`):

1. O **YOLOv8m** recebe o frame inteiro e devolve uma caixa para cada Pokémon encontrado (confiança mínima de 0,5).
2. Cada caixa ganha **10% de margem** e é recortada, porque as imagens de treino do classificador têm um pouco de fundo em volta do Pokémon.
3. Os recortes passam juntos, num único lote, pela **ResNet18**, que devolve a espécie e a probabilidade de cada um.

**Servidor** (`app/servidor.py`):

- Uma **thread separada** lê a webcam e roda o pipeline sem parar, guardando só o último resultado. As rotas apenas leem esse resultado, então o navegador nunca espera pela inferência.
- `GET /video` entrega o vídeo como **MJPEG**, um fluxo de JPEGs que o navegador exibe num simples `<img>`.
- `GET /info` devolve o Pokémon de maior confiança no frame atual.
- `POST /identificar` recebe uma imagem enviada e devolve as detecções e a imagem anotada.
- `GET /pokemon/{nome}` busca e resume os dados na PokéAPI, com cache em memória.
- Câmera e upload compartilham o mesmo modelo, protegido por um lock, porque o YOLO não é seguro para uso simultâneo por várias threads.

**Interface** (`app/static/`): HTML, CSS e JavaScript puros. O navegador consulta `/info` a cada 500 ms e decide o que mostrar:

- a ficha só aparece quando a confiança chega a **60%**;
- ela fica na tela por **30 segundos** depois da última vez que o Pokémon foi visto com essa confiança;
- outro Pokémon com 60% ou mais substitui a ficha na hora.

## Por que dois modelos em vez de um

A primeira versão usava um único YOLOv8m treinado para detectar e classificar as 149 classes ao mesmo tempo, com o dataset do Roboflow (917 imagens com caixas). Ele chegou a mAP50-95 de 0,70 no teste.

Para entender de onde vinham os erros, o mesmo modelo foi avaliado ignorando a classe (`single_cls=True`), medindo só se as caixas estavam no lugar certo. O resultado foi **mAP50-95 de 0,861**: o modelo localizava muito bem, e quase toda a perda vinha de errar a espécie.

O motivo está nos dados. Com 917 imagens para 149 classes, o detector via cerca de **6 exemplos por Pokémon**, pouco para separar espécies parecidas. Já existia um dataset de classificação com **~17 mil imagens**, mas sem caixas, então não servia para treinar um detector.

Separar as tarefas deixa cada modelo aprender com os dados que tem:

| | Detector (YOLOv8m) | Classificador (ResNet18) |
|---|---|---|
| Pergunta que responde | Onde há um Pokémon? | Qual Pokémon é este? |
| Dados | 917 imagens com caixas | ~17 mil imagens por classe, sem caixas |
| Classes | 1 ("pokemon") | 142 espécies |
| Entrada | o frame inteiro | o recorte da caixa, 224×224 |

**Ganhos:** o detector de 1 classe encontrou todos os Pokémon do teste (recall 1,0), e a espécie passou a ser decidida por um modelo com cerca de 100 vezes mais exemplos por classe. Cada modelo também pode ser retreinado ou trocado sem mexer no outro.

**Custos:** são duas redes por frame, e a classificação depende de a caixa ser boa. Se o detector errar a região, o classificador recebe um recorte ruim.

Os dois modelos partem de pesos pré-treinados (COCO para o YOLO, ImageNet para a ResNet) e passam por *fine-tuning*. Por isso o classificador usa a normalização do ImageNet: a rede espera entradas na mesma escala com que foi pré-treinada.

## Estrutura do projeto

```
pokedex/
├── pokedex.py                     # pipeline: classe Pokedex (YOLO + ResNet); também roda sozinho
├── app/
│   ├── servidor.py                # FastAPI: câmera, /video, /info, /identificar, /pokemon
│   ├── pokeapi.py                 # busca e cache dos dados da PokéAPI
│   └── static/                    # interface: index.html, style.css, app.js
├── detector/
│   ├── treinar_yolo_1classe.py    # treino do detector atual
│   ├── treinar_yolo_gen1.py       # experimento: YOLO com 149 classes
│   ├── treinar_yolo_4classes.py   # experimento: YOLO com 4 classes
│   └── camera.py                  # experimento: YOLO puro na webcam
├── classifier/
│   ├── treinar_resnet.py          # treino e avaliação do classificador atual
│   ├── pokemon_classifier.ipynb   # experimento: primeira versão do classificador
│   ├── resnet_camera.py           # experimento: classificação na webcam (incompatível com o modelo atual)
│   └── treinar_yolo_cls.py        # experimento: classificador YOLOv8-cls
├── scripts/
│   ├── uma_classe.py              # converte o dataset do Roboflow para 1 classe
│   ├── separador.py               # split train/val para o YOLOv8-cls
│   └── utils.py                   # utilitários de labels e pastas
├── docs/                          # anotações antigas
├── data/      (fora do git)       # datasets
├── models/    (fora do git)       # pesos do classificador, classes.json, pesos base do YOLO
└── runs/      (fora do git)       # resultados de treino do Ultralytics
```

## Dados

| Dataset | Uso | Fonte |
|---|---|---|
| Pokemon Gen 1 v2 (917 imagens, caixas em 149 classes, CC BY 4.0) | detector, convertido para 1 classe | [Roboflow](https://universe.roboflow.com/anthonyk/pokemon-gen-1/dataset/2) |
| Pokemon Images First Generation (~17 mil imagens) | classificador | [Kaggle](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files) |

No dataset do Kaggle, as pastas `MrMime` e `Mr. Mime` são o mesmo Pokémon e precisam ser juntadas antes do treino, o que resulta em 142 classes.

## Como usar

### Instalação

```bash
python -m venv pokeenv
pokeenv\Scripts\activate
pip install -r requirements.txt
```

Para treinar em GPU NVIDIA, instale o PyTorch com CUDA pelo seletor de [pytorch.org](https://pytorch.org/get-started/locally/).

### Treino

1. Baixe o dataset do Roboflow em `data/roboflow-gen1/` e converta para 1 classe:
   ```bash
   python scripts/uma_classe.py
   ```
2. Treine o detector (resultado em `runs/detect/gen1-1classe/`):
   ```bash
   python detector/treinar_yolo_1classe.py
   ```
3. Treine o classificador. O dataset do Kaggle é baixado pelo `kagglehub`, e os pesos e a lista de classes são salvos em `models/classifier/`:
   ```bash
   python classifier/treinar_resnet.py
   ```

### Interface web

Da raiz do projeto:

```bash
python -m uvicorn app.servidor:app
```

Abra http://127.0.0.1:8000. Os botões do aparelho alternam entre a câmera e o modo imagem, em que dá para arrastar uma foto até o visor ou colar com Ctrl+V.


### Linha de comando

```bash
python pokedex.py                          # webcam, numa janela do OpenCV
python pokedex.py images/Charmander.png    # uma imagem
```

## Limitações

- **Fotos de telas:** apontando a câmera para uma imagem no celular, o detector costuma marcar a tela inteira em vez do Pokémon. Reflexo, moiré e cores alteradas também pesam. Os dois datasets são de artes digitais, sem imagens capturadas por câmera.
- **Pelúcias e objetos reais:** o treino tem só desenhos, então bonecos e pelúcias estão fora do domínio.
- **Vários Pokémon juntos:** só 3 imagens do dataset de detecção têm mais de um Pokémon, e Pokémon sobrepostos podem ser unidos numa só caixa.
- **Evoluções parecidas:** são a principal fonte de erro do classificador.

## Próximos passos

- Coletar e rotular frames reais (telas e cartas fotografadas) para *fine-tuning* do detector
- Augmentations que imitam a câmera no classificador (`ColorJitter`, desfoque, perspectiva)
- Usar a câmera do celular pelo navegador, servindo a página por HTTPS

## Créditos

- Dados de Pokémon: [PokéAPI](https://pokeapi.co/)
- Dataset de detecção: [Pokemon Gen 1, Roboflow Universe](https://universe.roboflow.com/anthonyk/pokemon-gen-1/dataset/2) (CC BY 4.0)
- Dataset de classificação: [Kaggle, mikoajkolman](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files)

Este é apenas um projeto pessoal.
