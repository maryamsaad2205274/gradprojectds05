# utils/inference.py
import os
import torch
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "hrnet_landmarks.ts")


def load_model():
    global _model
    if _model is None:
        _model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        _model.eval()
    return _model


def preprocess(pil_img, img_size=256):
    pil_img = pil_img.convert("RGB")
    w0, h0 = pil_img.size

    # resize
    pil_resized = pil_img.resize((img_size, img_size))
    img = np.array(pil_resized).astype(np.float32) / 255.0

    # ImageNet normalization (MATCHES COLAB)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,256,256)
    return x, (w0, h0)


def _heatmaps_to_points_soft(hm, input_size=256, radius=2):
    """
    hm: torch.Tensor on CPU, shape (K, H, W)
    Returns: xs, ys in input_size coordinate space (e.g., 256x256)
    """
    k, h, w = hm.shape
    xs = torch.zeros(k, dtype=torch.float32)
    ys = torch.zeros(k, dtype=torch.float32)

    for i in range(k):
        m = hm[i]  # (H,W)

        # coarse peak
        flat_idx = torch.argmax(m).item()
        py = flat_idx // w
        px = flat_idx % w

        # small window around peak (e.g., 5x5 if radius=2)
        y0 = max(0, py - radius)
        y1 = min(h, py + radius + 1)
        x0 = max(0, px - radius)
        x1 = min(w, px + radius + 1)

        patch = m[y0:y1, x0:x1]

        # weights must be positive
        patch = torch.relu(patch)
        s = patch.sum()

        if s.item() < 1e-8:
            fx = float(px)
            fy = float(py)
        else:
            yy = torch.arange(y0, y1, dtype=torch.float32).unsqueeze(1)  # (ph,1)
            xx = torch.arange(x0, x1, dtype=torch.float32).unsqueeze(0)  # (1,pw)

            fy = (patch * yy).sum().item() / s.item()
            fx = (patch * xx).sum().item() / s.item()

        # heatmap -> input coords (256)
        xs[i] = fx * (input_size / w)
        ys[i] = fy * (input_size / h)

    return xs, ys


def predict_landmarks(image_path):
    model = load_model()

    # absolute path safety
    if not os.path.isabs(image_path):
        image_path = os.path.join(BASE_DIR, image_path)

    pil_img = Image.open(image_path)
    x, (w0, h0) = preprocess(pil_img)
    x = x.to(DEVICE)

    with torch.no_grad():
        out = model(x)

    # TorchScript sometimes returns tuple/list
    if isinstance(out, (tuple, list)):
        out = out[0]

    # out expected: (1,17,64,64) or (17,64,64)
    hm = out.detach().cpu()
    if hm.dim() == 4:
        hm = hm[0]  # (17,64,64)

    k, h, w = hm.shape

    # ✅ Better decoding than argmax
    xs, ys = _heatmaps_to_points_soft(hm, input_size=256, radius=2)

    # Scale from model input (256x256) -> original image size
    sx = w0 / 256.0
    sy = h0 / 256.0
    xs = xs * sx
    ys = ys * sy

    return [{"x": float(xs[i].item()), "y": float(ys[i].item())} for i in range(k)]
