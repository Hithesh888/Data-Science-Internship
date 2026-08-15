import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from tensorflow.keras.models import load_model

MODEL_PATH = r"D:\intern\project3\drowsiness_model.h5"
CLASS_PATH = r"D:\intern\project3\classes.txt"
YOLO_PATH = r"D:\intern\project3\yolov8n.pt"

model = load_model(MODEL_PATH)
person_model = YOLO(YOLO_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = [line.strip().lower() for line in f.readlines()]

print("Loaded classes:", class_names)

cap = None
running = False


def predict_drowsiness(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        crop = crop[y:y+h, x:x+w]

    crop = cv2.resize(crop, (224, 224))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype("float32")
    crop = np.expand_dims(crop, axis=0)

    pred = model.predict(crop, verbose=0)

    awake_score = float(pred[0][class_names.index("awake")])
    sleeping_score = float(pred[0][class_names.index("sleeping")])

    print("Awake Score:", awake_score)
    print("Sleeping Score:", sleeping_score)

    if awake_score >= sleeping_score:
        return "awake", awake_score
    else:
        return "sleeping", sleeping_score

def process_frame(frame, popup=False):
    results = person_model(frame, verbose=False)

    total_people = 0
    sleeping_people = 0
    awake_people = 0
    sleeping_details = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])

            if cls_id != 0:
                continue

            total_people += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            label, conf = predict_drowsiness(person_crop)

            age = "20-30"

            if label == "sleeping" and conf >= 0.80:
                sleeping_people += 1
                color = (0, 0, 255)
                text = f"Sleeping {conf:.2f} | Age: {age}"
                sleeping_details.append(f"Person {sleeping_people}: Age {age}")
            else:
                awake_people += 1
                color = (0, 255, 0)
                text = f"Awake {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.putText(
        frame,
        f"Total People: {total_people}  Sleeping: {sleeping_people}  Awake: {awake_people}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    if popup:
        details = "\n".join(sleeping_details) if sleeping_details else "No sleeping person detected"

        messagebox.showinfo(
            "Detection Result",
            f"Total People: {total_people}\n"
            f"Sleeping People: {sleeping_people}\n"
            f"Awake People: {awake_people}\n\n"
            f"{details}"
        )

    return frame


def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((850, 500))

    imgtk = ImageTk.PhotoImage(image=img)

    preview.imgtk = imgtk
    preview.config(image=imgtk)


def upload_image():
    global running, cap

    running = False

    if cap:
        cap.release()
        cap = None

    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not path:
        return

    frame = cv2.imread(path)

    if frame is None:
        messagebox.showerror("Error", "Image not loaded.")
        return

    result = process_frame(frame, popup=True)
    show_frame(result)


def upload_video():
    global cap, running

    if cap:
        cap.release()
        cap = None

    path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )

    if not path:
        return

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Video not loaded.")
        return

    running = True
    play_video()


def play_video():
    global cap, running

    if not running or cap is None:
        return

    ret, frame = cap.read()

    if not ret:
        running = False
        cap.release()
        cap = None
        messagebox.showinfo("Completed", "Video detection completed.")
        return

    result = process_frame(frame, popup=False)
    show_frame(result)

    root.after(30, play_video)


def stop_video():
    global cap, running

    running = False

    if cap:
        cap.release()
        cap = None

    messagebox.showinfo("Stopped", "Video stopped.")


def exit_app():
    global cap, running

    running = False

    if cap:
        cap.release()

    root.destroy()


root = tk.Tk()
root.title("Drowsiness Detection Model")
root.geometry("1000x700")
root.configure(bg="#222222")

title = tk.Label(
    root,
    text="Drowsiness Detection Model",
    font=("Arial", 24, "bold"),
    bg="#222222",
    fg="white"
)
title.pack(pady=15)

preview = tk.Label(root, bg="black")
preview.pack(pady=10)

btn_frame = tk.Frame(root, bg="#222222")
btn_frame.pack(pady=20)

tk.Button(
    btn_frame,
    text="Upload Image",
    font=("Arial", 14),
    width=15,
    command=upload_image
).grid(row=0, column=0, padx=10)

tk.Button(
    btn_frame,
    text="Upload Video",
    font=("Arial", 14),
    width=15,
    command=upload_video
).grid(row=0, column=1, padx=10)

tk.Button(
    btn_frame,
    text="Stop Video",
    font=("Arial", 14),
    width=15,
    command=stop_video
).grid(row=0, column=2, padx=10)

tk.Button(
    btn_frame,
    text="Exit",
    font=("Arial", 14),
    width=15,
    bg="red",
    fg="white",
    command=exit_app
).grid(row=0, column=3, padx=10)

root.mainloop()