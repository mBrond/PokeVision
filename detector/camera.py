from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório


model = YOLO(ROOT / "runs/detect/gen1-149/weights/best.pt")  # 149 classes (Roboflow Gen 1)
# Alternativa, só 4 classes: ROOT / "runs/detect/pokemon-4classes/weights/best.pt"

# source=0 indica a webcam integrada do computador
model.predict(source=0, show=True)