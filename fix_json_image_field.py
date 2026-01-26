import os, json, re

BASE = r"C:\Users\marya\Desktop\GradLandmarksDataset"
LBL_DIR = os.path.join(BASE, "labeled", "labels")

pattern = re.compile(r"^(Image\s+\d+)\..+$", re.IGNORECASE)  # keeps "Image 12"

fixed = 0

for fname in os.listdir(LBL_DIR):
    if not fname.lower().endswith(".json"):
        continue

    path = os.path.join(LBL_DIR, fname)
    data = json.load(open(path, "r", encoding="utf-8"))

    img_name = data.get("image", "")
    m = pattern.match(img_name)

    if m:
        correct = m.group(1) + ".jpg"  # force Image N.jpg
        if img_name != correct:
            data["image"] = correct
            json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            fixed += 1

print("✅ Updated JSON image fields:", fixed)
