import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

BASE_DIR = r"D:\intern\project4\models"

NATIONALITY_MODEL = os.path.join(
    BASE_DIR, "models", "nationality_model.h5"
)

AGE_MODEL = os.path.join(
    BASE_DIR, "age_model.keras"
)

EMOTION_MODEL = os.path.join(
    BASE_DIR, "emotion_model.keras"
)

DRESS_MODEL = os.path.join(
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

DRESS_CLASSES = [
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
        NATIONALITY_MODEL,
        compile=False
    )

    age_model = tf.keras.models.load_model(
        AGE_MODEL,
        compile=False
    )

    emotion_model = tf.keras.models.load_model(
        EMOTION_MODEL,
        compile=False
    )

    dress_model = tf.keras.models.load_model(
        DRESS_MODEL,
        compile=False
    )

    print("All models loaded successfully!")

except Exception as e:

    print("MODEL LOADING ERROR:")
    print(e)

    raise

def prepare_rgb_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((128, 128))

    img = np.array(image).astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def prepare_gray_image(image_path):

    image = Image.open(image_path).convert("L")

    image = image.resize((128, 128))

    img = np.array(image).astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)

    img = np.expand_dims(img, axis=0)

    return img

def predict_nationality(image_path):

    img = prepare_rgb_image(image_path)

    prediction = nationality_model.predict(
        img,
        verbose=0
    )

    prediction = prediction[0]

    index = np.argmax(prediction)

    nationality = NATIONALITY_CLASSES[index]

    confidence = float(prediction[index]) * 100

    return nationality, confidence


def predict_emotion(image_path):

    img = prepare_gray_image(image_path)

    prediction = emotion_model.predict(
        img,
        verbose=0
    )

    prediction = prediction[0]

    index = np.argmax(prediction)

    emotion = EMOTION_CLASSES[index]

    confidence = float(prediction[index]) * 100

    return emotion, confidence


def predict_age(image_path):

    img = prepare_rgb_image(image_path)

    prediction = age_model.predict(
        img,
        verbose=0
    )

    age = float(np.squeeze(prediction))

    # Prevent unrealistic values
    age = max(0, min(age, 100))

    return age


def predict_dress_color(image_path):

    img = prepare_rgb_image(image_path)

    prediction = dress_model.predict(
        img,
        verbose=0
    )

    prediction = prediction[0]

    index = np.argmax(prediction)

    color = DRESS_CLASSES[index]

    confidence = float(prediction[index]) * 100

    return color, confidence

root = tk.Tk()

root.title("AI Nationality & Person Analysis")

root.geometry("1250x850")

root.configure(bg="#f4f6f8")

selected_image = None
preview_image = None

title = tk.Label(
    root,
    text="AI NATIONALITY & PERSON ANALYSIS",
    font=("Arial", 28, "bold"),
    bg="#f4f6f8",
    fg="#111111"
)

title.pack(pady=(25, 5))


subtitle = tk.Label(
    root,
    text="Nationality • Emotion • Age • Dress Color",
    font=("Arial", 15),
    bg="#f4f6f8",
    fg="#333333"
)

subtitle.pack(pady=(0, 20))

main_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=10
)

left_frame = tk.LabelFrame(
    main_frame,
    text=" Image Preview ",
    font=("Arial", 16, "bold"),
    bg="#ffffff",
    padx=15,
    pady=15
)

left_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 12)
)


preview_label = tk.Label(
    left_frame,
    text="No image selected",
    font=("Arial", 15),
    bg="#eeeeee",
    width=45,
    height=20
)

preview_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


filename_label = tk.Label(
    left_frame,
    text="",
    font=("Arial", 13),
    bg="#ffffff",
    fg="#333333"
)

filename_label.pack(
    pady=8
)

button_frame = tk.Frame(
    left_frame,
    bg="#ffffff"
)

button_frame.pack(
    pady=10
)

right_frame = tk.LabelFrame(
    main_frame,
    text=" Prediction Results ",
    font=("Arial", 16, "bold"),
    bg="#ffffff",
    padx=20,
    pady=20
)

right_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(12, 0)
)

nationality_result = tk.StringVar(
    value="Nationality: —"
)

emotion_result = tk.StringVar(
    value="Emotion: —"
)

age_result = tk.StringVar(
    value="Age: —"
)

dress_result = tk.StringVar(
    value="Dress Color: —"
)

def create_result_label(variable):

    label = tk.Label(
        right_frame,
        textvariable=variable,
        font=("Arial", 17, "bold"),
        bg="#f1f1f1",
        fg="#111111",
        anchor="w",
        padx=15,
        pady=12
    )

    label.pack(
        fill="x",
        pady=7
    )

    return label


nationality_label = create_result_label(
    nationality_result
)

emotion_label = create_result_label(
    emotion_result
)

age_label = create_result_label(
    age_result
)

dress_label = create_result_label(
    dress_result
)

rules_frame = tk.LabelFrame(
    right_frame,
    text=" Prediction Rules ",
    font=("Arial", 14, "bold"),
    bg="#ffffff",
    padx=15,
    pady=15
)

rules_frame.pack(
    fill="x",
    pady=25
)


rules_text = (
    "Indian   → Age + Dress Color + Emotion\n"
    "American → Age + Emotion\n"
    "African  → Dress Color + Emotion\n"
    "Other    → Emotion"
)


rules_label = tk.Label(
    rules_frame,
    text=rules_text,
    font=("Arial", 13),
    bg="#ffffff",
    justify="left",
    anchor="w"
)

rules_label.pack(
    fill="x"
)

status_label = tk.Label(
    root,
    text="Select an image to begin.",
    font=("Arial", 12),
    bg="#f4f6f8",
    fg="#444444"
)

status_label.pack(
    pady=10
)

def upload_image():

    global selected_image
    global preview_image

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    selected_image = file_path

    try:

        image = Image.open(file_path).convert("RGB")

        # Preview size
        max_width = 500
        max_height = 500

        image.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        preview_image = ImageTk.PhotoImage(image)

        preview_label.config(
            image=preview_image,
            text=""
        )

        preview_label.image = preview_image

        filename_label.config(
            text=os.path.basename(file_path)
        )

        nationality_result.set("Nationality: —")
        emotion_result.set("Emotion: —")
        age_result.set("Age: —")
        dress_result.set("Dress Color: —")

        status_label.config(
            text="Image uploaded successfully. Click Analyze Image."
        )

    except Exception as e:

        messagebox.showerror(
            "Image Error",
            str(e)
        )


def analyze_image():

    if selected_image is None:

        messagebox.showwarning(
            "No Image",
            "Please upload an image first."
        )

        return

    try:

        status_label.config(
            text="Analyzing image..."
        )

        root.update()

        nationality, nationality_conf = predict_nationality(
            selected_image
        )

        nationality_result.set(
            f"Nationality: {nationality} "
            f"({nationality_conf:.2f}% confidence)"
        )

        emotion, emotion_conf = predict_emotion(
            selected_image
        )

        emotion_result.set(
            f"Emotion: {emotion.capitalize()} "
            f"({emotion_conf:.2f}% confidence)"
        )


        age_result.set("Age: Not required")

        dress_result.set("Dress Color: Not required")

        if nationality == "Indian":

            age = predict_age(
                selected_image
            )

            color, color_conf = predict_dress_color(
                selected_image
            )

            age_result.set(
                f"Age: {age:.0f} years"
            )

            dress_result.set(
                f"Dress Color: {color.capitalize()} "
                f"({color_conf:.2f}% confidence)"
            )

        elif nationality == "American":

            age = predict_age(
                selected_image
            )

            age_result.set(
                f"Age: {age:.0f} years"
            )


        elif nationality == "African":

            color, color_conf = predict_dress_color(
                selected_image
            )

            dress_result.set(
                f"Dress Color: {color.capitalize()} "
                f"({color_conf:.2f}% confidence)"
            )

        elif nationality == "Other":

            age_result.set(
                "Age: Not required"
            )

            dress_result.set(
                "Dress Color: Not required"
            )


        status_label.config(
            text="Analysis completed successfully."
        )


    except Exception as e:

        status_label.config(
            text="Prediction failed."
        )

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )

upload_button = tk.Button(
    button_frame,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 15, "bold"),
    width=16,
    height=2,
    bg="#e8e8e8",
    relief="raised",
    cursor="hand2"
)

upload_button.pack(
    side="left",
    padx=8
)


analyze_button = tk.Button(
    button_frame,
    text="Analyze Image",
    command=analyze_image,
    font=("Arial", 15, "bold"),
    width=16,
    height=2,
    bg="#e8e8e8",
    relief="raised",
    cursor="hand2"
)

analyze_button.pack(
    side="left",
    padx=8
)


root.mainloop()