import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

DATASET_PATH = r"D:\intern\project5\dataset\asl_alphabet_train"
MODEL_PATH = r"D:\intern\project5\model"

os.makedirs(MODEL_PATH, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClasses:")
print(class_names)
print("\nTotal Classes:", num_classes)

with open(os.path.join(MODEL_PATH, "labels.txt"), "w") as f:
    for label in class_names:
        f.write(label + "\n")

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset
    .map(lambda x, y: (preprocess_input(x), y))
    .prefetch(AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(lambda x, y: (preprocess_input(x), y))
    .prefetch(AUTOTUNE)
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_PATH, "sign_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

model.save(os.path.join(MODEL_PATH, "sign_model.h5"))

print("\n===================================")
print("Training Completed Successfully!")
print("===================================")
print("Model saved to:", os.path.join(MODEL_PATH, "sign_model.h5"))
print("Labels saved to:", os.path.join(MODEL_PATH, "labels.txt"))