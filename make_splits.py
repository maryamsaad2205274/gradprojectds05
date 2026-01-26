import os
import random

BASE = r"C:\Users\marya\Desktop\GradLandmarksDataset"
IMG_DIR = os.path.join(BASE, "processed", "images")
LBL_DIR = os.path.join(BASE, "processed", "labels")
OUT_DIR = os.path.join(BASE, "splits")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
VAL_RATIO = 0.2  # with 30 images => ~6 val, 24 train

# list images that have matching labels
images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

paired = []
for img in images:
    stem = os.path.splitext(img)[0]
    label = stem + ".json"
    if os.path.exists(os.path.join(LBL_DIR, label)):
        paired.append(img)

if len(paired) == 0:
    raise RuntimeError("No paired images found in processed/images and processed/labels.")

random.seed(SEED)
random.shuffle(paired)

val_count = max(1, int(len(paired) * VAL_RATIO))
val = paired[:val_count]
train = paired[val_count:]

train_path = os.path.join(OUT_DIR, "train.txt")
val_path = os.path.join(OUT_DIR, "val.txt")

with open(train_path, "w", encoding="utf-8") as f:
    for x in train:
        f.write(x + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for x in val:
        f.write(x + "\n")

print("✅ Total paired:", len(paired))
print("✅ Train:", len(train), "->", train_path)
print("✅ Val  :", len(val), "->", val_path)
