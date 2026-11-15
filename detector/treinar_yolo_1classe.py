from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório


def treinar_deteccao():
    model = YOLO(ROOT / "models/base/yolov8m.pt") 

    # Detector de 1 classe ("pokemon"): só localiza; a espécie fica com o classificador
    results = model.train(
        data=str(ROOT / "data/roboflow-1classe/data.yaml"),    # Aponta para o arquivo de configuração do seu dataset
        epochs=100,           # Número de épocas (com early stopping via patience)
        imgsz=416,             # Resolução recomendada (mude para 640 se quiser máxima precisão)
        patience=20,           # Para o treino se ficar 20 épocas sem melhorar
        device=0,              # GPU (requer PyTorch com CUDA)
        batch=-1,              # AutoBatch: maior batch que cabe na GPU
        project=str(ROOT / "runs/detect"),
        name="gen1-1classe",
    )
    
    print("Treinamento de detecção (1 classe) concluído!")

if __name__ == "__main__":
    treinar_deteccao()