import os, json
import cv2
import numpy as np

BASE = r"C:\Users\marya\Desktop\GradLandmarksDataset"

SRC_IMG_DIR = os.path.join(BASE, "labeled", "images")
SRC_LBL_DIR = os.path.join(BASE, "labeled", "labels")

OUT_IMG_DIR = os.path.join(BASE, "processed", "images")
OUT_LBL_DIR = os.path.join(BASE, "processed", "labels")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

TARGET = 256  # you can change later to 384 for better accuracy

def letterbox(image, target=256):
    h, w = image.shape[:2]
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    pad_x = (target - nw) // 2
    pad_y = (target - nh) // 2

    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas, scale, pad_x, pad_y

label_files = sorted([f for f in os.listdir(SRC_LBL_DIR) if f.lower().endswith(".json")])

for jf in label_files:
    src_lbl_path = os.path.join(SRC_LBL_DIR, jf)
    data = json.load(open(src_lbl_path, "r", encoding="utf-8"))

    img_name = data["image"]
    src_img_path = os.path.join(SRC_IMG_DIR, img_name)

    img = cv2.imread(src_img_path)
    if img is None:
        print("SKIP (cannot read):", src_img_path)
        continue

    orig_h, orig_w = img.shape[:2]
    out_img, scale, pad_x, pad_y = letterbox(img, TARGET)

    # transform landmarks to new coords (pixel space in 256x256)
    new_landmarks = []
    for lm in data["landmarks"]:
        x, y = lm["x"], lm["y"]
        xs = x * scale + pad_x
        ys = y * scale + pad_y
        new_landmarks.append({"id": lm["id"], "x": float(xs), "y": float(ys)})

    # save processed image
    out_img_name = img_name  # keep same name
    out_img_path = os.path.join(OUT_IMG_DIR, out_img_name)
    cv2.imwrite(out_img_path, out_img)

    # save processed label
    out_data = {
        "image": out_img_name,
        "width": TARGET,
        "height": TARGET,
        "landmarks": new_landmarks
    }
    out_lbl_path = os.path.join(OUT_LBL_DIR, jf)
    json.dump(out_data, open(out_lbl_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("✅ Done. Processed:", len(label_files), "files")
print("Saved to:", os.path.join(BASE, "processed"))
