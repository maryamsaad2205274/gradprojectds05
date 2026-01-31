import os, glob, json
import numpy as np

LABELED_DIR   = "/content/drive/MyDrive/projectdataset/images"
LANDMARK_DIR  = "/content/drive/MyDrive/projectdataset/landmarks"

def load_landmarks_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w = int(data["width"])
    h = int(data["height"])
    lms = sorted(data["landmarks"], key=lambda x: int(x["id"]))
    pts = np.array([[float(p["x"]), float(p["y"])] for p in lms], dtype=np.float32)
    img_name = data.get("image", None)
    return img_name, (w, h), pts

def find_image_path(img_name):
    p = os.path.join(LABELED_DIR, img_name)
    if os.path.exists(p):
        return p
    base = os.path.splitext(img_name)[0]
    for ext in [".jpg", ".jpeg", ".png"]:
        cand = os.path.join(LABELED_DIR, base + ext)
        if os.path.exists(cand):
            return cand
    # brute fallback
    for p in glob.glob(os.path.join(LABELED_DIR, "*")):
        if os.path.splitext(os.path.basename(p))[0] == base:
            return p
    return None

pairs = []
json_files = sorted(glob.glob(os.path.join(LANDMARK_DIR, "*.json")))

missing_imgs = 0
bad_pts = 0
for jf in json_files:
    img_name, (w, h), pts = load_landmarks_json(jf)
    if pts.shape[0] != 17:
        bad_pts += 1
        continue
    img_path = find_image_path(img_name)
    if img_path is None:
        missing_imgs += 1
        continue
    pairs.append((img_path, jf))

print("✅ JSON files:", len(json_files))
print("✅ Pairs found:", len(pairs))
print("❌ Missing images:", missing_imgs)
print("❌ Wrong landmark count:", bad_pts)

assert len(pairs) > 0, "No image/json pairs found. Check folder paths."
