import os
import json
import glob
from PIL import Image
from coords import COORDS

# Base folder = GradLandmarksDataset
BASE = "."
IMG_DIR = os.path.join(BASE, "labeled", "images")
LBL_DIR = os.path.join(BASE, "labeled", "labels")

# Create labels folder if it does not exist
os.makedirs(LBL_DIR, exist_ok=True)

for i in COORDS.keys():
    # 🔹 Find image regardless of extension (.jpg / .JPG / .jpeg)
    matches = glob.glob(os.path.join(IMG_DIR, f"Image {i}.*"))
    if len(matches) == 0:
        raise FileNotFoundError(f"Missing image for Image {i}")

    img_path = matches[0]

    # 🔹 Read image size
    with Image.open(img_path) as im:
        width, height = im.size

    points = COORDS[i]

    # 🔹 Validate number of landmarks
    if len(points) != 17:
        raise ValueError(f"Image {i} has {len(points)} landmarks (expected 17)")

    # 🔹 Validate coordinates are inside image
    for idx, (x, y) in enumerate(points, start=1):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Image {i} landmark {idx} out of bounds: ({x}, {y}) for image size ({width}, {height})"
            )

    # 🔹 Create label data
    label_data = {
        "image": os.path.basename(img_path),
        "width": width,
        "height": height,
        "landmarks": [
            {"id": j, "x": int(points[j - 1][0]), "y": int(points[j - 1][1])}
            for j in range(1, 18)
        ]
    }

    # 🔹 Save JSON label file
    label_path = os.path.join(LBL_DIR, f"Image {i}.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label_data, f, indent=2)

print("✅ SUCCESS: 30 label files created in labeled/labels/")
