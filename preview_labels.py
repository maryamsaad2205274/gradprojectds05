import os
import json
import glob
import cv2

IMG_DIR = os.path.join("labeled", "images")
LBL_DIR = os.path.join("labeled", "labels")
OUT_DIR = os.path.join("outputs", "previews_labeled")

os.makedirs(OUT_DIR, exist_ok=True)

for i in range(1, 31):
    # 🔹 Find image regardless of extension
    matches = glob.glob(os.path.join(IMG_DIR, f"Image {i}.*"))
    if len(matches) == 0:
        raise FileNotFoundError(f"Cannot find image for Image {i}")

    img_path = matches[0]
    lbl_path = os.path.join(LBL_DIR, f"Image {i}.json")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    with open(lbl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # draw landmarks
    for lm in data["landmarks"]:
        x, y, lid = lm["x"], lm["y"], lm["id"]

        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            img,
            str(lid),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    out_path = os.path.join(OUT_DIR, f"Image {i}_preview.jpg")
    cv2.imwrite(out_path, img)

print("✅ DONE: previews saved in outputs/previews_labeled/")
