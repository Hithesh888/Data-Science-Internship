import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/car_colour_model.h5")

with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


def predict_color(car_img):

    if car_img is None or car_img.size == 0:
        return "Unknown"

    img = cv2.resize(car_img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < 0.50:
        return "Unknown"

    return class_names[index]