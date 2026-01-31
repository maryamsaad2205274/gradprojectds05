import os, json, glob
import numpy as np
import cv2

def load_landmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    w = int(data["width"])
    h = int(data["height"])

    # Sort landmarks by id to keep consistent order
    lms = sorted(data["landmarks"], key=lambda x: int(x["id"]))
    pts = np.array([[float(p["x"]), float(p["y"])] for p in lms], dtype=np.float32)

    img_name = data.get("image", None)
    return img_name, (w, h), pts

# pick any json
json_files = sorted(glob.glob(os.path.join(LANDMARK_DIR, "*.json")))
print("Landmark JSON files:", len(json_files))
sample_json = json_files[0]
print("Sample:", sample_json)

img_name, (w, h), pts = load_landmarks(sample_json)
print("Image name inside JSON:", img_name)
print("Original size from JSON:", (w, h))
print("Num landmarks:", len(pts))
print("First 5 points:\n", pts[:5])
