from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório


def treinar_deteccao():
    model = YOLO(ROOT / "models/base/yolov8m.pt") 

    # 2. Inicia o treinamento de detecção de caixas (bounding boxes)
    results = model.train(
        data=str(ROOT / "data/roboflow-gen1/data.yaml"),    # Aponta para o arquivo de configuração do seu dataset
        epochs=25,            # Número de épocas (com early stopping padrão)
        imgsz=416,             # Resolução recomendada (mude para 640 se quiser máxima precisão)
        patience=15,           # Para o treino se ficar 15 épocas sem melhorar
        project=str(ROOT / "runs/detect"),
        name="gen1-149",
    )
    
    print("Treinamento de detecção concluído!")

if __name__ == "__main__":
    treinar_deteccao()