import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

MODEL_PATH = r"D:\intern\project5\model\sign_model.h5"
LABEL_PATH = r"D:\intern\project5\model\labels.txt"

model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

def predict_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found!")
        return

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32)
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    index = np.argmax(prediction)
    confidence = np.max(prediction)

    print("Prediction :", class_names[index])
    print("Confidence :", round(confidence * 100, 2), "%")

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    predict_image(image_path)