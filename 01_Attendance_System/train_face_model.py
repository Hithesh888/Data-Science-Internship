import cv2
import os
import numpy as np

BASE_DIR = r"D:\intern\project1"
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
STUDENTS_PATH = os.path.join(BASE_DIR, "students.txt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
student_names = {}
label_id = 0

for student in os.listdir(DATASET_PATH):
    student_path = os.path.join(DATASET_PATH, student)

    if not os.path.isdir(student_path):
        continue

    student_names[label_id] = student
    print("Reading:", student)

    for image_name in os.listdir(student_path):
        image_path = os.path.join(student_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            faces.append(face)
            labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("No faces found in dataset.")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_PATH)

with open(STUDENTS_PATH, "w") as f:
    for key, value in student_names.items():
        f.write(f"{key},{value}\n")

print("Face model trained successfully!")