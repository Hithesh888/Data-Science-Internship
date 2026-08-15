import os
import shutil
import random

SOURCE_DIR = r"D:\intern\project4\dataset\dress_color"

OUTPUT_DIR = r"D:\intern\project4\dataset\dress_color_processed"

TRAIN_RATIO = 0.8

COLORS = [
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow"
]

for split in ["train", "val"]:
    for color in COLORS:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, color),
            exist_ok=True
        )


color_images = {color: [] for color in COLORS}

for folder in os.listdir(SOURCE_DIR):

    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    folder_lower = folder.lower()

    detected_color = None

    for color in COLORS:

        if folder_lower.startswith(color + "_"):
            detected_color = color
            break

    if detected_color is None:
        print("Skipped folder:", folder)
        continue

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):

            image_path = os.path.join(
                folder_path,
                filename
            )

            color_images[detected_color].append(
                image_path
            )

print("\nImages found by color:")

for color in COLORS:

    print(
        f"{color:10s}: "
        f"{len(color_images[color])}"
    )

random.seed(42)

for color in COLORS:

    images = color_images[color]

    random.shuffle(images)

    split_index = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:split_index]
    val_images = images[split_index:]

    for i, source in enumerate(train_images):

        extension = os.path.splitext(source)[1]

        destination = os.path.join(
            OUTPUT_DIR,
            "train",
            color,
            f"{color}_train_{i}{extension}"
        )

        shutil.copy2(source, destination)


    for i, source in enumerate(val_images):

        extension = os.path.splitext(source)[1]

        destination = os.path.join(
            OUTPUT_DIR,
            "val",
            color,
            f"{color}_val_{i}{extension}"
        )

        shutil.copy2(source, destination)

    print(
        f"{color}: "
        f"{len(train_images)} train | "
        f"{len(val_images)} validation"
    )

print("\n======================================")
print("Dress Color Dataset Prepared!")
print("======================================")

print("\nOutput:")
print(OUTPUT_DIR)
