import pandas as pd
import os

csv_path = r"D:\intern\project4\dataset\FairFace\fairface_label_train.csv"

df = pd.read_csv(csv_path)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nFirst filename:", df.iloc[0]["file"])

image_path = os.path.join(
    r"D:\intern\project4\dataset\FairFace\train",
    df.iloc[0]["file"]
)

print("\nImage path:")
print(image_path)

print("\nImage exists:", os.path.exists(image_path))