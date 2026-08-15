from ultralytics import YOLO
import cv2

model = YOLO(
    r"D:\intern\project2\animal_detection\animal_model-4\weights\best.pt"
)
# Change this to your test image
image_path = r"D:\intern\project2\dataset\test\images"

import os

files = os.listdir(image_path)

image_file = None

for file in files:
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_file = os.path.join(image_path, file)
        break

if image_file is None:
    print("No image found!")
    exit()

print("Testing image:", image_file)

results = model(image_file, conf=0.40)

carnivores = {
    "Bear",
    "Leopard",
    "Lion",
    "Tiger"
}

carnivore_count = 0

for result in results:

    image = result.orig_img.copy()

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        animal = model.names[class_id]

        if animal in carnivores:
            color = (0, 0, 255)
            carnivore_count += 1

        else:
            color = (0, 255, 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = f"{animal} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.putText(
        image,
        f"Carnivores: {carnivore_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    output_path = r"D:\intern\project2\detection_result.jpg"
    cv2.imwrite(output_path, image)
    cv2.imshow("Animal Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("--------------------------------")
    print("Detection completed!")
    print("Carnivorous animals:", carnivore_count)
    print("Result saved to:", output_path)