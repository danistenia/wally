from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # modelo pequeño y rápido

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=4
)
