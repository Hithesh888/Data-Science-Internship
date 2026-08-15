import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, time

BASE_DIR = r"D:\intern\project1"

FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
STUDENTS_PATH = os.path.join(BASE_DIR, "students.txt")

EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
EMOTION_CLASSES_PATH = os.path.join(BASE_DIR, "emotion_classes.txt")

CSV_PATH = os.path.join(BASE_DIR, "attendance.csv")
EXCEL_PATH = os.path.join(BASE_DIR, "attendance.xlsx")

START_TIME = time(9, 30)
END_TIME = time(10, 0)

if not os.path.exists(FACE_MODEL_PATH):
    print("face_model.yml not found. Run train_face_model.py first.")
    exit()

if not os.path.exists(STUDENTS_PATH):
    print("students.txt not found. Run train_face_model.py first.")
    exit()

if not os.path.exists(EMOTION_MODEL_PATH):
    print("emotion_model.h5 not found. Run train_emotion_model.py first.")
    exit()

if not os.path.exists(EMOTION_CLASSES_PATH):
    print("emotion_classes.txt not found. Run train_emotion_model.py first.")
    exit()

current_time = datetime.now().time()

if not (START_TIME <= current_time <= END_TIME):
    print("Attendance system works only between 9:30 AM and 10:00 AM")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(FACE_MODEL_PATH)

emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)

students = {}

with open(STUDENTS_PATH, "r") as f:
    for line in f:
        label, name = line.strip().split(",")
        students[int(label)] = name

with open(EMOTION_CLASSES_PATH, "r") as f:
    emotion_classes = [line.strip() for line in f.readlines()]

attendance = {
    name: {
        "Status": "Absent",
        "Time": "",
        "Emotion": ""
    }
    for name in students.values()
}

def detect_emotion(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    prediction = emotion_model.predict(face, verbose=0)
    emotion_index = np.argmax(prediction)

    return emotion_classes[emotion_index]

cap = cv2.VideoCapture(0)

print("Attendance system started... Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected.")
        break

    now = datetime.now()

    if now.time() > END_TIME:
        print("Attendance time ended.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_original = gray[y:y+h, x:x+w]
        face_for_recognition = cv2.resize(face_original, (200, 200))

        label, confidence = face_recognizer.predict(face_for_recognition)

        if confidence < 70:
            name = students[label]
            emotion = detect_emotion(face_original)

            if attendance[name]["Status"] == "Absent":
                attendance[name]["Status"] = "Present"
                attendance[name]["Time"] = now.strftime("%H:%M:%S")
                attendance[name]["Emotion"] = emotion

            text = f"{name} - {emotion}"
        else:
            text = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame([
    {
        "Student Name": name,
        "Status": data["Status"],
        "Time": data["Time"],
        "Emotion": data["Emotion"]
    }
    for name, data in attendance.items()
])

df.to_csv(CSV_PATH, index=False)

try:
    df.to_excel(EXCEL_PATH, index=False)
except:
    print("Excel file not saved. Install openpyxl.")

print("Attendance saved successfully!")
print(df)