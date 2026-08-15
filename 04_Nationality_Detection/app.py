import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf


BASE_DIR = r"D:\intern\project4\models"

# Nationality model was saved inside models\models
NATIONALITY_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "nationality_model.h5"
)

AGE_MODEL_PATH = os.path.join(
    BASE_DIR, "age_model.keras"
)

EMOTION_MODEL_PATH = os.path.join(
    BASE_DIR, "emotion_model.keras"
)

DRESS_COLOR_MODEL_PATH = os.path.join(
    BASE_DIR, "dress_color_model.keras"
)

NATIONALITY_CLASSES = [
    "African",
    "American",
    "Indian",
    "Other"
]

EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

DRESS_COLOR_CLASSES = [
    "black",
    "blue",
    "brown",
    "green",
    "pink",
    "red",
    "white",
    "yellow"
]

print("Loading models...")

try:
    nationality_model = tf.keras.models.load_model(
        NATIONALITY_MODEL_PATH
    )

    age_model = tf.keras.models.load_model(
        AGE_MODEL_PATH
    )

    emotion_model = tf.keras.models.load_model(
        EMOTION_MODEL_PATH
    )

    dress_color_model = tf.keras.models.load_model(
        DRESS_COLOR_MODEL_PATH
    )

    print("All models loaded successfully!")

except Exception as e:
    print("MODEL LOADING ERROR:")
    print(e)
    raise

def preprocess_rgb(image):

    image = image.convert("RGB")
    image = image.resize((128, 128))

    arr = np.array(image, dtype=np.float32) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr


def preprocess_emotion(image):

    image = image.convert("L")
    image = image.resize((128, 128))

    arr = np.array(image, dtype=np.float32) / 255.0

    # (128,128) -> (128,128,1)
    arr = np.expand_dims(arr, axis=-1)

    # (128,128,1) -> (1,128,128,1)
    arr = np.expand_dims(arr, axis=0)

    return arr


def predict_nationality(image):

    img = preprocess_rgb(image)

    prediction = nationality_model.predict(
        img,
        verbose=0
    )

    probabilities = prediction[0]

    index = np.argmax(probabilities)

    nationality = NATIONALITY_CLASSES[index]

    confidence = float(probabilities[index]) * 100

    return nationality, confidence


def predict_emotion(image):

    img = preprocess_emotion(image)

    prediction = emotion_model.predict(
        img,
        verbose=0
    )

    probabilities = prediction[0]

    index = np.argmax(probabilities)

    emotion = EMOTION_CLASSES[index]

    confidence = float(probabilities[index]) * 100

    return emotion, confidence


def predict_age(image):

    img = preprocess_rgb(image)

    prediction = age_model.predict(
        img,
        verbose=0
    )

    age = float(np.squeeze(prediction))

    # Prevent impossible values
    age = max(0, min(100, age))

    return age


def predict_dress_color(image):

    img = preprocess_rgb(image)

    prediction = dress_color_model.predict(
        img,
        verbose=0
    )

    probabilities = prediction[0]

    index = np.argmax(probabilities)

    color = DRESS_COLOR_CLASSES[index]

    confidence = float(probabilities[index]) * 100

    return color, confidence


def analyze_image():

    global current_image

    if current_image is None:

        messagebox.showwarning(
            "No Image",
            "Please upload an image first."
        )

        return

    try:

        nationality, nationality_conf = predict_nationality(
            current_image
        )

        emotion, emotion_conf = predict_emotion(
            current_image
        )


        for widget in result_frame.winfo_children():
            widget.destroy()


        title = tk.Label(
            result_frame,
            text="PREDICTION RESULTS",
            font=("Arial", 20, "bold"),
            bg="white"
        )

        title.pack(
            pady=(10, 20)
        )

        create_result_row(
            "Nationality",
            f"{nationality}",
            f"{nationality_conf:.2f}% confidence"
        )

        create_result_row(
            "Emotion",
            emotion.capitalize(),
            f"{emotion_conf:.2f}% confidence"
        )

        additional_title = tk.Label(
            result_frame,
            text="Additional Predictions",
            font=("Arial", 15, "bold"),
            bg="white"
        )

        additional_title.pack(
            pady=(25, 10)
        )

        if nationality == "Indian":

            age = predict_age(current_image)

            color, color_conf = predict_dress_color(
                current_image
            )

            create_result_row(
                "Age",
                f"{age:.0f} years",
                ""
            )

            create_result_row(
                "Dress Color",
                color.capitalize(),
                f"{color_conf:.2f}% confidence"
            )


        elif nationality == "American":

            age = predict_age(current_image)

            create_result_row(
                "Age",
                f"{age:.0f} years",
                ""
            )

        elif nationality == "African":

            color, color_conf = predict_dress_color(
                current_image
            )

            create_result_row(
                "Dress Color",
                color.capitalize(),
                f"{color_conf:.2f}% confidence"
            )

        elif nationality == "Other":

            label = tk.Label(
                result_frame,
                text="No additional predictions",
                font=("Arial", 13),
                bg="white"
            )

            label.pack(
                pady=10
            )


        status_label.config(
            text="Analysis completed successfully.",
            fg="green"
        )

    except Exception as e:

        print("Prediction Error:")
        print(e)

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )

        status_label.config(
            text="Prediction failed.",
            fg="red"
        )


def create_result_row(label_text, value_text, confidence_text):

    frame = tk.Frame(
        result_frame,
        bg="#f3f3f3"
    )

    frame.pack(
        fill="x",
        padx=20,
        pady=5
    )

    label = tk.Label(
        frame,
        text=f"{label_text}:",
        font=("Arial", 13, "bold"),
        bg="#f3f3f3",
        anchor="w",
        width=18
    )

    label.pack(
        side="left",
        padx=10,
        pady=10
    )

    value = tk.Label(
        frame,
        text=value_text,
        font=("Arial", 13),
        bg="#f3f3f3",
        anchor="w"
    )

    value.pack(
        side="left",
        padx=10
    )

    if confidence_text:

        confidence = tk.Label(
            frame,
            text=confidence_text,
            font=("Arial", 11),
            bg="#f3f3f3",
            fg="#555555"
        )

        confidence.pack(
            side="left",
            padx=10
        )



def upload_image():

    global current_image
    global image_path
    global photo

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("JPG Files", "*.jpg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    try:

        image = Image.open(file_path)
        image = image.convert("RGB")
        current_image = image
        image_path = file_path


        preview = image.copy()

        preview.thumbnail(
            (520, 500),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            preview
        )

        image_label.config(
            image=photo,
            text=""
        )

        image_label.image = photo

        filename = os.path.basename(
            file_path
        )

        filename_label.config(
            text=filename
        )


        for widget in result_frame.winfo_children():
            widget.destroy()

        result_title = tk.Label(
            result_frame,
            text="Upload an image and click Analyze Image",
            font=("Arial", 14),
            bg="white",
            fg="#555555"
        )

        result_title.pack(
            pady=30
        )

        status_label.config(
            text="Image uploaded successfully.",
            fg="green"
        )

    except Exception as e:

        messagebox.showerror(
            "Image Error",
            f"Could not open image:\n\n{e}"
        )

def clear_image():

    global current_image
    global image_path
    global photo

    current_image = None
    image_path = None
    photo = None

    image_label.config(
        image="",
        text="No image selected",
        font=("Arial", 16),
        fg="#777777"
    )

    filename_label.config(
        text="No image selected"
    )

    for widget in result_frame.winfo_children():
        widget.destroy()

    message = tk.Label(
        result_frame,
        text="Upload an image to begin analysis",
        font=("Arial", 14),
        bg="white",
        fg="#777777"
    )

    message.pack(
        pady=30
    )

    status_label.config(
        text="Ready.",
        fg="#555555"
    )

root = tk.Tk()

root.title(
    "AI Nationality & Person Analysis"
)

root.geometry(
    "1350x900"
)

root.minsize(
    1100,
    750
)

root.configure(
    bg="#f5f7fa"
)


header = tk.Frame(
    root,
    bg="#f5f7fa"
)

header.pack(
    fill="x",
    pady=(20, 5)
)

title_label = tk.Label(
    header,
    text="AI NATIONALITY & PERSON ANALYSIS",
    font=("Arial", 27, "bold"),
    bg="#f5f7fa",
    fg="#111111"
)

title_label.pack()

subtitle_label = tk.Label(
    header,
    text="Nationality  •  Emotion  •  Age  •  Dress Color",
    font=("Arial", 14),
    bg="#f5f7fa",
    fg="#444444"
)

subtitle_label.pack(
    pady=5
)

main_frame = tk.Frame(
    root,
    bg="#f5f7fa"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=15
)

left_panel = tk.LabelFrame(
    main_frame,
    text="Image Preview",
    font=("Arial", 16, "bold"),
    bg="white",
    padx=15,
    pady=15
)

left_panel.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)

image_container = tk.Frame(
    left_panel,
    bg="#eeeeee",
    width=550,
    height=560
)

image_container.pack(
    fill="both",
    expand=True,
    pady=5
)

image_container.pack_propagate(False)


image_label = tk.Label(
    image_container,
    text="No image selected",
    font=("Arial", 16),
    fg="#777777",
    bg="#eeeeee"
)

image_label.pack(
    expand=True
)

filename_label = tk.Label(
    left_panel,
    text="No image selected",
    font=("Arial", 11),
    bg="white",
    fg="#555555"
)

filename_label.pack(
    pady=8
)

button_frame = tk.Frame(
    left_panel,
    bg="white"
)

button_frame.pack(
    pady=10
)


upload_button = tk.Button(
    button_frame,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 13, "bold"),
    width=16,
    height=2,
    cursor="hand2"
)

upload_button.pack(
    side="left",
    padx=5
)


analyze_button = tk.Button(
    button_frame,
    text="Analyze Image",
    command=analyze_image,
    font=("Arial", 13, "bold"),
    width=16,
    height=2,
    cursor="hand2"
)

analyze_button.pack(
    side="left",
    padx=5
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_image,
    font=("Arial", 13, "bold"),
    width=10,
    height=2,
    cursor="hand2"
)

clear_button.pack(
    side="left",
    padx=5
)

right_panel = tk.LabelFrame(
    main_frame,
    text="Prediction Results",
    font=("Arial", 16, "bold"),
    bg="white",
    padx=15,
    pady=15
)

right_panel.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(10, 0)
)

result_frame = tk.Frame(
    right_panel,
    bg="white"
)

result_frame.pack(
    fill="both",
    expand=True
)


initial_message = tk.Label(
    result_frame,
    text="Upload an image to begin analysis",
    font=("Arial", 14),
    bg="white",
    fg="#777777"
)

initial_message.pack(
    pady=30
)

rules_frame = tk.LabelFrame(
    right_panel,
    text="Prediction Rules",
    font=("Arial", 13, "bold"),
    bg="white",
    padx=15,
    pady=10
)

rules_frame.pack(
    fill="x",
    pady=15
)


rules = (
    "Indian    → Age + Dress Color + Emotion\n"
    "American  → Age + Emotion\n"
    "African   → Dress Color + Emotion\n"
    "Other     → Emotion"
)

rules_label = tk.Label(
    rules_frame,
    text=rules,
    font=("Arial", 12),
    bg="white",
    justify="left",
    anchor="w"
)

rules_label.pack(
    fill="x"
)


status_label = tk.Label(
    root,
    text="Ready.",
    font=("Arial", 12),
    bg="#f5f7fa",
    fg="#555555"
)

status_label.pack(
    pady=(0, 15)
)


current_image = None
image_path = None
photo = None
root.mainloop()