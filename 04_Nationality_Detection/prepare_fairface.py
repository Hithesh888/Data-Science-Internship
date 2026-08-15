import os
import shutil
import pandas as pd

BASE_DIR = r"D:\intern\project4\dataset\FairFace"

TRAIN_CSV = os.path.join(BASE_DIR, "fairface_label_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "fairface_label_val.csv")

OUTPUT_DIR = r"D:\intern\project4\dataset\nationality"


def map_race(race):
    race = race.strip().lower()

    if race == "indian":
        return "Indian"
    elif race == "white":
        return "American"
    elif race == "black":
        return "African"
    else:
        return "Other"


def prepare(csv_path, split):

    df = pd.read_csv(csv_path)

    copied = 0
    skipped = 0

    for _, row in df.iterrows():

        filename = row["file"]          
        race = row["race"]

        category = map_race(race)

        source = os.path.join(BASE_DIR, filename)

        destination_folder = os.path.join(
            OUTPUT_DIR,
            split,
            category
        )

        os.makedirs(destination_folder, exist_ok=True)

        destination = os.path.join(
            destination_folder,
            os.path.basename(filename)
        )

        if os.path.exists(source):
            shutil.copy2(source, destination)
            copied += 1
        else:
            skipped += 1

    print(f"\n{split.upper()} SET")
    print("Copied :", copied)
    print("Skipped:", skipped)

prepare(TRAIN_CSV, "train")
prepare(VAL_CSV, "val")

print("\nDataset Prepared Successfully!")