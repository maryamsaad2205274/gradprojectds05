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


def find_image_path(img_name):
    # search labeled folder for exact filename
    p = os.path.join(LABELED_DIR, img_name)
    if os.path.exists(p):
        return p

    # fallback: search by basename (in case of different extension)
    base = os.path.splitext(img_name)[0]
    for ext in [".jpg", ".jpeg", ".png"]:
        cand = os.path.join(LABELED_DIR, base + ext)
        if os.path.exists(cand):
            return cand

    # last fallback: brute search
    for p in glob.glob(os.path.join(LABELED_DIR, "*")):
        if os.path.splitext(os.path.basename(p))[0] == base:
            return p
    return None

img_path = find_image_path(img_name)
print("Matched image path:", img_path)
assert img_path is not None, "❌ Could not find the image file mentioned in the JSON."


import matplotlib.pyplot as plt

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print("Image loaded shape:", img_rgb.shape)  # should match (height,width,3) near JSON size

plt.figure(figsize=(5,6))
plt.imshow(img_rgb)
plt.scatter(pts[:,0], pts[:,1], s=20)
plt.title("Landmarks overlay (GT)")
plt.axis("off")
plt.show()

