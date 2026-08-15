import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

DATASET_DIR = r"D:\intern\project4\dataset\UTKFace\UTKFace"
MODEL_DIR = r"D:\intern\project4\models"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

print("Reading UTKFace dataset...")

file_paths = []
ages = []

for filename in os.listdir(DATASET_DIR):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    parts = filename.split("_")

    try:
        age = int(parts[0])
    except (ValueError, IndexError):
        continue

    if age < 0 or age > 116:
        continue

    file_paths.append(
        os.path.join(DATASET_DIR, filename)
    )

    ages.append(age)

print("Total images:", len(file_paths))

train_paths, val_paths, train_ages, val_ages = train_test_split(
    file_paths,
    ages,
    test_size=0.2,
    random_state=42
)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))

def load_image(path, age):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    age = tf.cast(
        age,
        tf.float32
    )

    return image, age

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_ages)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_ages)
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    augmentation,

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="linear"
    )
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="mae",
    metrics=["mae"]
)
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),

    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            MODEL_DIR,
            "age_model.keras"
        ),
        monitor="val_mae",
        save_best_only=True
    )
]

print("\nStarting Age Model Training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


model.save(
    os.path.join(
        MODEL_DIR,
        "age_model.keras"
    )
)

print("\n====================================")
print("Age Model Training Completed!")
print("====================================")