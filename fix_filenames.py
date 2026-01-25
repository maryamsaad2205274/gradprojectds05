import os
import re

BASE = r"C:\Users\marya\Desktop\GradLandmarksDataset"
IMG_DIR = os.path.join(BASE, "labeled", "images")

# Match: Image 1.jpg.jpeg, Image 1.JPG, Image 22.jpg.jpg, Image 3.jpeg, etc.
pattern = re.compile(r"^Image\s+(\d+)\.(jpg|jpeg|png|bmp)(\.(jpg|jpeg|png|bmp))?$", re.IGNORECASE)

files = os.listdir(IMG_DIR)
renamed = 0

for f in files:
    m = pattern.match(f)
    if not m:
        continue

    idx = m.group(1)               # image number
    new_name = f"Image {idx}.jpg"  # force .jpg

    old_path = os.path.join(IMG_DIR, f)
    new_path = os.path.join(IMG_DIR, new_name)

    if f.lower() == new_name.lower():
        continue

    if os.path.exists(new_path):
        print("SKIP (target exists):", new_name, "<-", f)
        continue

    os.rename(old_path, new_path)
    print("RENAMED:", f, "->", new_name)
    renamed += 1

print("✅ Done. Renamed:", renamed)
