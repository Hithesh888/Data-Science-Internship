import cv2
import os
from ultralytics import YOLO
from predict import predict_color

model = YOLO("models/yolov8n.pt")


def detect_objects(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not load image.")

    results = model(image)

    car_count = 0
    people_count = 0

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.4:
                continue

            class_name = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name == "person":

                people_count += 1

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    image,
                    "Person",
                    (x1, max(20, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

            elif class_name in ["car","bus","truck"]:

                car_count += 1

                pad = 8

                xx1 = max(0, x1 + pad)
                yy1 = max(0, y1 + pad)

                xx2 = min(image.shape[1], x2 - pad)
                yy2 = min(image.shape[0], y2 - pad)

                car_crop = image[yy1:yy2, xx1:xx2]

                color = predict_color(car_crop)

                if color.lower() == "blue":

                    box_color = (0,0,255)

                else:

                    box_color = (255,0,0)

                cv2.rectangle(
                    image,
                    (x1,y1),
                    (x2,y2),
                    box_color,
                    3
                )

                cv2.putText(
                    image,
                    f"{class_name} : {color}",
                    (x1, max(20,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

    cv2.rectangle(image,(10,10),(300,100),(40,40,40),-1)

    cv2.putText(
        image,
        f"Cars : {car_count}",
        (20,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        image,
        f"People : {people_count}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    os.makedirs("output", exist_ok=True)

    output_path = "output/result.jpg"

    cv2.imwrite(output_path, image)

    return output_path, car_count, people_count