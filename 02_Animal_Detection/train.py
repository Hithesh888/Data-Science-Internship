from ultralytics import YOLO

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data=r"D:\intern\project2\dataset\data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    project=r"D:\intern\project2\animal_detection",
    name="animal_model"
)

print("Training completed!")