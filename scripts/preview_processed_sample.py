import os, json
import cv2
import csv

BASE = r"C:\Users\marya\Desktop\GradLandmarksDataset"

IMG_DIR = os.path.join(BASE, "processed", "images")
LBL_DIR = os.path.join(BASE, "processed", "labels")

# Change this to preview any processed image
IMG_NAME = "Image 4.jpg"
JSON_NAME = "Image 4.json"

img_path = os.path.join(IMG_DIR, IMG_NAME)
lbl_path = os.path.join(LBL_DIR, JSON_NAME)

# --- Load ---
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read processed image: {img_path}")

data = json.load(open(lbl_path, "r", encoding="utf-8"))
landmarks = data["landmarks"]

# --- Draw DOTS ONLY (no numbers on image) ---
for lm in landmarks:
    x = int(round(lm["x"]))
    y = int(round(lm["y"]))
    cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

# --- Save image proof ---
out_dir = os.path.join(BASE, "outputs")
os.makedirs(out_dir, exist_ok=True)

img_out_path = os.path.join(out_dir, "processed_sample_DOTS_ONLY.jpg")
cv2.imwrite(img_out_path, img)

# --- Save numbers separately (CSV table) ---
csv_out_path = os.path.join(out_dir, "processed_sample_landmarks_table.csv")
with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "x", "y"])
    for lm in sorted(landmarks, key=lambda d: d["id"]):
        writer.writerow([lm["id"], float(lm["x"]), float(lm["y"])])

print("✅ Saved image:", img_out_path)
print("✅ Saved landmark table:", csv_out_path)
print("Processed image size:", img.shape)
print("Landmarks:", len(landmarks))
