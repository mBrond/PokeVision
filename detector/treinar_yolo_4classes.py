from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório


def treinar_modelo():
    modelo = YOLO(ROOT / "models/base/yolov8n.pt") 

    resultados = modelo.train(
        data=str(ROOT / "data/pokemon-reduzido/pokemon.yaml"),   # Caminho para o seu arquivo de configuração
        epochs=100,         # Quantidade de épocas (iterações completas no dataset)
        imgsz=640,          # Tamanho da imagem de entrada (padrão é 640)
        batch=16,           # Tamanho do lote (reduza se der erro de memória/Out of Memory)
        device='cpu',           # Use 0 para rodar na placa de vídeo (GPU) ou 'cpu' se não tiver GPU
        workers=8,          # Número de threads para carregar os dados
        project=str(ROOT / "runs/detect"),
        name="pokemon-4classes",
    )
    
    print("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    treinar_modelo()