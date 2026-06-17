import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.inference import load_model, predict_landmarks

IMAGE_PATH = r"static/uploads/6_side_8558142492f5458bb8ccf69d08347214.jpg"

print("Using image:", IMAGE_PATH)
print("Exists:", os.path.exists(IMAGE_PATH))

model = load_model()
print("Model loaded successfully.")

result = predict_landmarks(IMAGE_PATH)
points = result["landmarks"]
overlay = result["overlay_image"]

print("First 10 points:", points[:10])

overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 10))
plt.imshow(overlay_rgb)
plt.title("Local Debug Inference")
plt.axis("off")
plt.show()