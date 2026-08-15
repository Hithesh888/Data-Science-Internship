import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

BASE_DIR = r"D:\intern\project1"
TRAIN_DIR = os.path.join(BASE_DIR, "emotion_dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "emotion_dataset", "test")

MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "emotion_classes.txt")

IMG_SIZE = 48
BATCH_SIZE = 32

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names
print("Emotion classes:", class_names)

train_data = train_data.map(lambda x, y: (x / 255.0, y))
test_data = test_data.map(lambda x, y: (x / 255.0, y))

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=test_data, epochs=5)

model.save(MODEL_PATH)

with open(CLASSES_PATH, "w") as f:
    for emotion in class_names:
        f.write(emotion + "\n")

print("Emotion model trained successfully!")