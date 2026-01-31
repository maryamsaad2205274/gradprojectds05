import torch, cv2, numpy as np, matplotlib.pyplot as plt

# ---------- SETTINGS ----------
IMG_SIZE = 256
HEATMAP_SIZE = 64

# ---------- LOAD IMAGE ----------
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h0, w0 = img_rgb.shape[:2]

img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

# scale GT landmarks
sx = IMG_SIZE / w0
sy = IMG_SIZE / h0
gt_resized = gt_pts.copy()
gt_resized[:,0] *= sx
gt_resized[:,1] *= sy

# ---------- PREPROCESS ----------
img_norm = img_resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_norm = ((img_norm - mean) / std).astype(np.float32)

x = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(device).float()

# ---------- MODEL INFERENCE ----------
model = model.to(device).float()
model.eval()

with torch.no_grad():
    pred_hm = model(x).cpu()

# ---------- HEATMAP → POINTS ----------
def heatmaps_to_points(hm):
    B, K, H, W = hm.shape
    idx = hm.view(B, K, -1).argmax(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    return torch.stack([xs, ys], dim=-1)

scale = IMG_SIZE / HEATMAP_SIZE
pred_pts = heatmaps_to_points(pred_hm)[0].numpy() * scale

# ---------- DISPLAY ----------
fig, axes = plt.subplots(1, 2, figsize=(10,5))

axes[0].imshow(img_resized)
axes[0].scatter(gt_resized[:,0], gt_resized[:,1], c="lime", s=30)
axes[0].set_title("Labeled (Ground Truth)")
axes[0].axis("off")

axes[1].imshow(img_resized)
axes[1].scatter(pred_pts[:,0], pred_pts[:,1], c="red", s=30)
axes[1].set_title("Predicted (Model Output)")
axes[1].axis("off")

plt.show()
