import os, json
import cv2

BASE = r"C:\Users\marya\Desktop\GradLandmarksDataset"
IMG_DIR = os.path.join(BASE, "labeled", "images")
LBL_DIR = os.path.join(BASE, "labeled", "labels")

# choose which image to test
IMG_NAME = "Image 1.jpg"
JSON_NAME = "Image 1.json"

img_path = os.path.join(IMG_DIR, IMG_NAME)
lbl_path = os.path.join(LBL_DIR, JSON_NAME)

# load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

# load labels
data = json.load(open(lbl_path, "r", encoding="utf-8"))
landmarks = data["landmarks"]  # list of dicts {id,x,y}

for lm in landmarks:
    x, y = int(lm["x"]), int(lm["y"])
    i = lm["id"]

    # small clean dot
    cv2.circle(img, (x, y), 3, (0, 255, 0), 1)

    # tiny label
    cv2.putText(
        img,
        str(i),
        (x + 4, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 0),
        1
    )

out_path = os.path.join(BASE, "outputs")
os.makedirs(out_path, exist_ok=True)

save_path = os.path.join(out_path, "sample_with_landmarks.jpg")
cv2.imwrite(save_path, img)

print("✅ Saved:", save_path)
print("Image size:", img.shape)
print("Landmarks:", len(landmarks))
