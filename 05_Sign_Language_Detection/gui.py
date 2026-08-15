import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime, time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = r"D:\intern\project5"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sign_model.h5"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "labels.txt"
)

START_TIME = time(18, 0)
END_TIME = time(22, 0)

IMG_SIZE = 224

def check_time():
    current_time = datetime.now().time()
    return START_TIME <= current_time <= END_TIME

if not os.path.exists(MODEL_PATH):
    messagebox.showerror(
        "Model Error",
        f"Model not found:\n{MODEL_PATH}"
    )
    exit()

if not os.path.exists(LABEL_PATH):
    messagebox.showerror(
        "Label Error",
        f"Labels file not found:\n{LABEL_PATH}"
    )
    exit()

print("Loading model...")

model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    class_names = [
        line.strip()
        for line in f.readlines()
    ]

print("Classes:", class_names)
print("Model loaded successfully.")

cap = None
camera_running = False
current_image = None
photo = None

def predict_frame(frame):
    image = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    )

    image = preprocess_input(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    index = np.argmax(
        prediction
    )

    confidence = float(
        prediction[0][index]
    )

    label = class_names[index]

    return label, confidence

def show_image(frame):
    global photo

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        rgb
    )

    image.thumbnail(
        (650, 500),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(
        image
    )

    preview_label.config(
        image=photo,
        text=""
    )

    preview_label.image = photo

def update_result(label, confidence):
    result_label.config(
        text=f"Predicted Sign: {label}"
    )

    confidence_label.config(
        text=f"Confidence: {confidence * 100:.2f}%"
    )

def upload_image():
    global current_image

    stop_camera_silent()

    if not check_time():
        messagebox.showwarning(
            "System Unavailable",
            "Sign Language Detection is available only between 6:00 PM and 10:00 PM."
        )
        return

    path = filedialog.askopenfilename(
        title="Select Sign Language Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp"
            )
        ]
    )

    if not path:
        return

    frame = cv2.imread(path)

    if frame is None:
        messagebox.showerror(
            "Image Error",
            "Unable to load the selected image."
        )
        return

    current_image = frame.copy()

    label, confidence = predict_frame(
        frame
    )

    show_image(
        frame
    )

    update_result(
        label,
        confidence
    )

    status_label.config(
        text="Image prediction completed.",
        fg="green"
    )

def start_camera():
    global cap
    global camera_running

    if not check_time():
        messagebox.showwarning(
            "System Unavailable",
            "Sign Language Detection is available only between 6:00 PM and 10:00 PM."
        )
        return

    if camera_running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror(
            "Camera Error",
            "Cannot open webcam."
        )

        cap = None
        return

    camera_running = True

    status_label.config(
        text="Real-time detection running...",
        fg="green"
    )

    update_camera()

def update_camera():
    global cap
    global camera_running

    if not camera_running:
        return

    if not check_time():
        stop_camera()

        messagebox.showinfo(
            "Time Limit",
            "The allowed detection time has ended."
        )

        return

    ret, frame = cap.read()

    if not ret:
        stop_camera()

        messagebox.showerror(
            "Camera Error",
            "Unable to read camera frame."
        )

        return

    frame = cv2.flip(
        frame,
        1
    )

    label, confidence = predict_frame(
        frame
    )

    text = (
        f"{label} "
        f"({confidence * 100:.2f}%)"
    )

    cv2.rectangle(
        frame,
        (20, 20),
        (650, 80),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        frame,
        text,
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 128, 0),
        2
    )

    show_image(
        frame
    )

    update_result(
        label,
        confidence
    )

    root.after(
        30,
        update_camera
    )

def stop_camera_silent():
    global cap
    global camera_running

    camera_running = False

    if cap is not None:
        cap.release()
        cap = None

def stop_camera():
    stop_camera_silent()

    status_label.config(
        text="Camera stopped.",
        fg="red"
    )

def clear_result():
    global current_image
    global photo

    stop_camera_silent()

    current_image = None
    photo = None

    preview_label.config(
        image="",
        text="No image selected"
    )

    result_label.config(
        text="Predicted Sign: ---"
    )

    confidence_label.config(
        text="Confidence: ---"
    )

    status_label.config(
        text="Ready.",
        fg="#555555"
    )

def exit_application():
    stop_camera_silent()
    root.destroy()

root = tk.Tk()

root.title(
    "Sign Language Detection"
)

root.geometry(
    "1100x750"
)

root.minsize(
    950,
    650
)

root.configure(
    bg="#f5f5f5"
)

title = tk.Label(
    root,
    text="SIGN LANGUAGE DETECTION",
    font=("Arial", 26, "bold"),
    bg="#f5f5f5",
    fg="#222222"
)

title.pack(
    pady=(20, 5)
)

subtitle = tk.Label(
    root,
    text="ASL Sign Recognition using MobileNetV2",
    font=("Arial", 13),
    bg="#f5f5f5",
    fg="#555555"
)

subtitle.pack(
    pady=(0, 15)
)

main_frame = tk.Frame(
    root,
    bg="#f5f5f5"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=10
)

preview_frame = tk.LabelFrame(
    main_frame,
    text="Image / Camera Preview",
    font=("Arial", 15, "bold"),
    bg="white",
    padx=10,
    pady=10
)

preview_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)

preview_label = tk.Label(
    preview_frame,
    text="No image selected",
    font=("Arial", 16),
    bg="#eeeeee",
    fg="#777777"
)

preview_label.pack(
    fill="both",
    expand=True
)

result_frame = tk.LabelFrame(
    main_frame,
    text="Prediction Result",
    font=("Arial", 15, "bold"),
    bg="white",
    padx=20,
    pady=20
)

result_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(10, 0)
)

result_label = tk.Label(
    result_frame,
    text="Predicted Sign: ---",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="#222222"
)

result_label.pack(
    pady=30
)

confidence_label = tk.Label(
    result_frame,
    text="Confidence: ---",
    font=("Arial", 16),
    bg="white",
    fg="#444444"
)

confidence_label.pack(
    pady=10
)

time_label = tk.Label(
    result_frame,
    text="Operating Time: 6:00 PM - 10:00 PM",
    font=("Arial", 13),
    bg="white",
    fg="#555555"
)

time_label.pack(
    pady=30
)

button_frame = tk.Frame(
    root,
    bg="#f5f5f5"
)

button_frame.pack(
    pady=15
)

tk.Button(
    button_frame,
    text="Upload Image",
    font=("Arial", 13, "bold"),
    width=15,
    height=2,
    command=upload_image
).grid(
    row=0,
    column=0,
    padx=5
)

tk.Button(
    button_frame,
    text="Start Camera",
    font=("Arial", 13, "bold"),
    width=15,
    height=2,
    command=start_camera
).grid(
    row=0,
    column=1,
    padx=5
)

tk.Button(
    button_frame,
    text="Stop Camera",
    font=("Arial", 13, "bold"),
    width=15,
    height=2,
    command=stop_camera
).grid(
    row=0,
    column=2,
    padx=5
)

tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 13, "bold"),
    width=12,
    height=2,
    command=clear_result
).grid(
    row=0,
    column=3,
    padx=5
)

tk.Button(
    button_frame,
    text="Exit",
    font=("Arial", 13, "bold"),
    width=12,
    height=2,
    bg="red",
    fg="white",
    command=exit_application
).grid(
    row=0,
    column=4,
    padx=5
)

status_label = tk.Label(
    root,
    text="Ready.",
    font=("Arial", 12),
    bg="#f5f5f5",
    fg="#555555"
)

status_label.pack(
    pady=(0, 15)
)

root.mainloop()