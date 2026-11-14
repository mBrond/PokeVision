from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório


def treinar():
	model = YOLO(ROOT / "models/base/yolov8m-cls.pt")

	results = model.train(
		data=str(ROOT / "data/yolo_classification_pokemon"),
		project=str(ROOT / "runs/classify"),
		epochs=50,
		imgsz=224,
		workers=4
		)

	print("fim treino")

if __name__ == '__main__':
	treinar()