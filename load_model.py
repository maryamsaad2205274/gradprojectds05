import os, glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Re-create model exactly the same as training
model = HRNetKeypoint(num_keypoints=17, heatmap_size=HEATMAP_SIZE).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

print("✅ Loaded model from:", CKPT_PATH)
