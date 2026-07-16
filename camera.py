from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

# source=0 indica a webcam integrada do computador
model.predict(source=0, show=True)