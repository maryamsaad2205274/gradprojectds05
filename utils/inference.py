import os
import json
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils.model_paths import MODEL_DIR, OUTPUTS_DIR, resolve_model_file, resolve_side_model_path
from utils.paths import resolve_project_path
from utils.side_model import HEATMAP_SIZE as SIDE_HEATMAP_SIZE
from utils.side_model import IMG_SIZE as SIDE_IMG_SIZE
from utils.side_model import NUM_LANDMARKS as SIDE_NUM_LANDMARKS
from utils.side_model import SimpleHRNet


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "SIDE": {
        "arch": "simple_hrnet",
        "path_candidates": [
            "best_hrnet_landmarks.pth",
        ],
        "num_keypoints": SIDE_NUM_LANDMARKS,
        "img_size": SIDE_IMG_SIZE,
        "heatmap_size": SIDE_HEATMAP_SIZE,
    },
    "FRONT_NS": {
        "arch": "hrnet_timm",
        "path_candidates": [
            "best_hrnet_front34.pth",
        ],
        "num_keypoints": 34,
        "img_size": 512,
        "heatmap_size": 128,
    },
}


def _resolved_path_for_variant(key: str) -> str:
    spec = _MODEL_SPECS[key]
    path = resolve_model_file(spec["path_candidates"])
    if not path:
        names = ", ".join(spec["path_candidates"])
        raise FileNotFoundError(
            f"No weights found for {key} in {MODEL_DIR}. Expected one of: {names}"
        )
    return path


def get_model_spec(variant: str) -> Dict[str, Any]:
    """Resolved spec including absolute weight path (for health checks / debugging)."""
    key = variant.upper()
    if key not in _MODEL_SPECS:
        raise ValueError(f"Unknown model variant: {variant}")
    spec = dict(_MODEL_SPECS[key])
    spec["path"] = _resolved_path_for_variant(key)
    spec["variant"] = key
    return spec


# Backward-compatible path constants (best-effort resolve)
try:
    MODEL_PATH_SIDE = _resolved_path_for_variant("SIDE")
except FileNotFoundError:
    MODEL_PATH_SIDE = os.path.join(MODEL_DIR, "best_hrnet_landmarks.pth")

try:
    MODEL_PATH_FRONT_NS = _resolved_path_for_variant("FRONT_NS")
except FileNotFoundError:
    MODEL_PATH_FRONT_NS = os.path.join(MODEL_DIR, "best_hrnet_front34.pth")


# =========================================================
# MODEL CLASS
# =========================================================
class HRNetKeypoint(nn.Module):
    def __init__(self, num_keypoints=20, heatmap_size=96):
        super().__init__()

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=False,
            features_only=True,
            out_indices=(3,)
        )

        ch = self.backbone.feature_info.channels()[0]

        self.head = nn.Sequential(
            nn.Conv2d(ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1),
        )

        self.heatmap_size = heatmap_size

    def forward(self, x):
        feat = self.backbone(x)[0]
        hm = self.head(feat)

        return F.interpolate(
            hm,
            size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear",
            align_corners=False
        )


# =========================================================
# LOAD MODELS
# =========================================================
_models = {}


def _load_checkpoint(path: str):
    """Load .pth safely on CPU/GPU."""
    return torch.load(path, map_location=DEVICE, weights_only=False)


def _load_state_dict(path: str):
    checkpoint = _load_checkpoint(path)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_model(variant: str = "SIDE"):
    key = variant.upper()

    if key not in _MODEL_SPECS:
        raise ValueError(f"Unknown model variant: {variant}. Use 'SIDE' or 'FRONT_NS'.")

    if key in _models:
        return _models[key]

    spec = get_model_spec(key)
    path = spec["path"]
    arch = spec["arch"]

    state_dict = _load_state_dict(path)

    if arch == "simple_hrnet":
        model = SimpleHRNet(
            num_keypoints=spec["num_keypoints"],
            heatmap_size=spec["heatmap_size"],
        )
        model.load_state_dict(state_dict, strict=True)
    elif arch == "hrnet_timm":
        model = HRNetKeypoint(
            num_keypoints=spec["num_keypoints"],
            heatmap_size=spec["heatmap_size"],
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.to(DEVICE)
    model.eval()

    _models[key] = model
    return model


def clear_model_cache() -> None:
    """Unload cached models (useful after swapping weight files)."""
    _models.clear()


# =========================================================
# HEATMAP TO POINTS
# =========================================================
@torch.no_grad()
def heatmaps_to_points_argmax(hm: torch.Tensor) -> torch.Tensor:
    B, K, H, W = hm.shape

    flat = hm.view(B, K, -1)
    idx = torch.argmax(flat, dim=-1)

    ys = (idx // W).float()
    xs = (idx % W).float()

    return torch.stack([xs, ys], dim=-1)


# =========================================================
# DRAW LANDMARKS
# =========================================================
def draw_points(image_bgr: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    output = image_bgr.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(output, (int(x), int(y)), 3, (0, 0, 255), -1)

        cv2.putText(
            output,
            str(i + 1),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return output


def draw_points_small(image_bgr: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    output = image_bgr.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), -1)

        cv2.putText(
            output,
            str(i + 1),
            (int(x) + 4, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return output


# Side inference debug (set SIDE_INFERENCE_DEBUG=0 to silence)
SIDE_INFERENCE_DEBUG = os.environ.get("SIDE_INFERENCE_DEBUG", "1") != "0"


def _side_debug(msg: str) -> None:
    if SIDE_INFERENCE_DEBUG:
        print(f"[SIDE] {msg}")


# =========================================================
# SIDE PREDICTION (Colab-identical pipeline)
# =========================================================
@torch.no_grad()
def predict_side_landmarks(image_path: str) -> Dict[str, Any]:
    """
    Colab-matched side inference:
    - /255 only (no ImageNet mean/std)
    - 384x384 resize
    - SimpleHRNet -> [1, 20, 96, 96]
    - argmax heatmaps -> scale to original image
    """
    image_path = resolve_project_path(image_path) or image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_path = resolve_side_model_path()
    _side_debug(f"model path used: {model_path}")

    model = load_model("SIDE")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_height, original_width = img_rgb.shape[:2]
    _side_debug(f"image original size: {original_width} x {original_height}")

    img_size = SIDE_IMG_SIZE
    heatmap_size = SIDE_HEATMAP_SIZE

    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    x = img_resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    _side_debug(f"input tensor shape: {tuple(x.shape)}")

    hm = model(x)
    _side_debug(f"output heatmap shape: {tuple(hm.shape)}")

    pts_hm = heatmaps_to_points_argmax(hm)[0].cpu().numpy()

    # Heatmap coords -> 384x384 space -> original image (Colab)
    pts = pts_hm.astype(np.float64).copy()
    pts[:, 0] = pts[:, 0] * (img_size / heatmap_size)
    pts[:, 1] = pts[:, 1] * (img_size / heatmap_size)
    pts[:, 0] *= original_width / img_size
    pts[:, 1] *= original_height / img_size

    points = [(int(round(px)), int(round(py))) for px, py in pts]

    for i, (px, py) in enumerate(points[:3]):
        _side_debug(f"predicted point {i + 1}: ({px}, {py})")

    heatmap_peaks: List[float] = []
    for k in range(hm.shape[1]):
        px = int(round(float(pts_hm[k, 0])))
        py = int(round(float(pts_hm[k, 1])))
        px = max(0, min(heatmap_size - 1, px))
        py = max(0, min(heatmap_size - 1, py))
        heatmap_peaks.append(float(hm[0, k, py, px].item()))

    overlay = draw_points(img_bgr, points)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    outputs_path = os.path.join(OUTPUTS_DIR, f"{base}_side_landmarks.jpg")
    save_overlay_image(overlay, outputs_path)
    _side_debug(f"saved output image: {outputs_path}")

    return {
        "landmarks": points,
        "overlay_image": overlay,
        "original_width": original_width,
        "original_height": original_height,
        "heatmap_peaks": heatmap_peaks,
        "variant": "SIDE",
        "outputs_path": outputs_path,
        "model_path": model_path,
    }


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================
@torch.no_grad()
def predict_landmarks(image_path: str, variant: str = "SIDE"):
    image_path = resolve_project_path(image_path) or image_path
    key = variant.upper()
    if key == "SIDE":
        return predict_side_landmarks(image_path)

    if key not in _MODEL_SPECS:
        raise ValueError(f"Unknown model variant: {variant}. Use 'SIDE' or 'FRONT_NS'.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    spec = get_model_spec(key)
    img_size = spec["img_size"]
    heatmap_size = spec["heatmap_size"]

    model = load_model(key)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]

    img_resized = cv2.resize(
        img_rgb,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )

    arr = img_resized.astype(np.float32) / 255.0

    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    hm = model(x)
    pts_hm = heatmaps_to_points_argmax(hm)[0].cpu().numpy()

    K = hm.shape[1]
    heatmap_peaks: List[float] = []
    for k in range(K):
        px = int(round(float(pts_hm[k, 0])))
        py = int(round(float(pts_hm[k, 1])))
        px = max(0, min(heatmap_size - 1, px))
        py = max(0, min(heatmap_size - 1, py))
        heatmap_peaks.append(float(hm[0, k, py, px].item()))

    pts_resized = pts_hm * (img_size / heatmap_size)
    pts_orig = pts_resized.copy()
    pts_orig[:, 0] *= (w0 / img_size)
    pts_orig[:, 1] *= (h0 / img_size)

    points = [(int(round(x)), int(round(y))) for x, y in pts_orig]
    overlay = draw_points(bgr, points)

    return {
        "landmarks": points,
        "overlay_image": overlay,
        "original_width": w0,
        "original_height": h0,
        "heatmap_peaks": heatmap_peaks,
        "variant": key,
    }


# =========================================================
# SAVE OVERLAY IMAGE
# =========================================================
def save_overlay_image(overlay_image: np.ndarray, save_path: str) -> str:
    folder = os.path.dirname(save_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    success = cv2.imwrite(save_path, overlay_image)

    if not success:
        raise IOError(f"Failed to save overlay image to: {save_path}")

    return save_path


# =========================================================
# JSON HELPERS
# =========================================================
def landmarks_to_json(points: List[Tuple[int, int]]) -> str:
    data = [{"x": int(x), "y": int(y)} for x, y in points]
    return json.dumps(data)


def json_to_landmarks(landmarks_json: str) -> List[Tuple[int, int]]:
    data = json.loads(landmarks_json)
    return [(int(p["x"]), int(p["y"])) for p in data]


# =========================================================
# FULL HELPER
# =========================================================
def run_inference_and_save(
    image_path: str,
    overlay_save_path: str,
    variant: str = "SIDE"
) -> Dict:
    image_path = resolve_project_path(image_path) or image_path
    overlay_save_path = resolve_project_path(overlay_save_path) or overlay_save_path
    result = predict_landmarks(image_path, variant=variant)

    saved_path = save_overlay_image(
        result["overlay_image"],
        overlay_save_path
    )

    return {
        "landmarks": result["landmarks"],
        "landmarks_json": landmarks_to_json(result["landmarks"]),
        "overlay_path": saved_path,
        "original_width": result["original_width"],
        "original_height": result["original_height"],
        "variant": result["variant"],
    }