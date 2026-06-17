"""
AI Orthodontic Decision Support System — X-Ray pipeline.

Architecture : SimpleHRNet19 (stem → high_res / mid_res / low_res → fuse → final)
Checkpoint   : model/xray/best_hrnet_19_landmarks.pth  (full checkpoint dict)
Landmarks    : 19 cephalometric landmarks
Features     : 217  (38 normalized coords + 171 pairwise distances + 8 angles)
Models       : 11 XGBoost classifiers  (4 Primary  + 7 Supportive)
Output       : versioned JSON  schema_version=2

Legacy compatibility
────────────────────
Old cases stored a flat dict with keys:
  skeletal_class, upper_lip, lower_lip, profile_class
Those records can still be rendered via parse_xray_diagnosis_json().
"""

from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — MUST match the training checkpoint exactly
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.skip  = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class SimpleHRNet19(nn.Module):
    def __init__(self, num_landmarks: int = 19) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.high_res = nn.Sequential(
            BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64),
        )
        self.mid_res = nn.Sequential(
            nn.Conv2d(64, 96,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            BasicBlock(96, 96), BasicBlock(96, 96),
        )
        self.low_res = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BasicBlock(128, 128), BasicBlock(128, 128),
        )
        self.fuse_mid = nn.Conv2d(96,  64, kernel_size=1)
        self.fuse_low = nn.Conv2d(128, 64, kernel_size=1)
        self.final = nn.Sequential(
            BasicBlock(64, 64), BasicBlock(64, 64),
            nn.Conv2d(64, num_landmarks, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = self.high_res(self.stem(x))
        mid  = self.mid_res(high)
        low  = self.low_res(mid)
        mid_up = F.interpolate(
            self.fuse_mid(mid), size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        low_up = F.interpolate(
            self.fuse_low(low), size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.final(high + mid_up + low_up)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — validated against checkpoint metadata
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE      = 384
HEATMAP_SIZE  = 96
NUM_LANDMARKS = 19
NUM_FEATURES  = 217

# ─────────────────────────────────────────────────────────────────────────────
# Paths — all relative to project root, never pointing to Downloads or Colab
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_XRAY_DIR     = _PROJECT_ROOT / "model" / "xray"

_HRNET_PATH        = _XRAY_DIR / "best_hrnet_19_landmarks.pth"
_FEATURE_COLS_PATH = _XRAY_DIR / "feature_columns.pkl"

# 11 diagnosis models: 4 Primary + 7 Supportive
# Keys must NOT include Skeletal Maxilla or Lower Lip Position (excluded by design).
XRAY_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    # ── Primary ──────────────────────────────────────────────────────────────
    "skeletal_class": {
        "display_name": "Skeletal Class",
        "model":   "skeletal_class_model.pkl",
        "encoder": "skeletal_class_label_encoder.pkl",
        "status":  "Primary",
    },
    "skeletal_mandible": {
        "display_name": "Skeletal Mandible",
        "model":   "skeletal_mandible_model.pkl",
        "encoder": "skeletal_mandible_label_encoder.pkl",
        "status":  "Primary",
    },
    "overjet": {
        "display_name": "Overjet",
        "model":   "overjet_model.pkl",
        "encoder": "overjet_label_encoder.pkl",
        "status":  "Primary",
    },
    "upper_lip_position": {
        "display_name": "Upper Lip Position",
        "model":   "upper_lip_position_model.pkl",
        "encoder": "upper_lip_position_label_encoder.pkl",
        "status":  "Primary",
    },
    # ── Supportive ────────────────────────────────────────────────────────────
    "dental_relationship": {
        "display_name": "Dental Relationship",
        "model":   "dental_relationship_model.pkl",
        "encoder": "dental_relationship_label_encoder.pkl",
        "status":  "Supportive",
    },
    "overbite": {
        "display_name": "Overbite",
        "model":   "overbite_model.pkl",
        "encoder": "overbite_label_encoder.pkl",
        "status":  "Supportive",
    },
    "upper_incisor_inclination": {
        "display_name": "Upper Incisor Inclination",
        "model":   "upper_incisor_inclination_model.pkl",
        "encoder": "upper_incisor_inclination_label_encoder.pkl",
        "status":  "Supportive",
    },
    "lower_incisor_inclination": {
        "display_name": "Lower Incisor Inclination",
        "model":   "lower_incisor_inclination_model.pkl",
        "encoder": "lower_incisor_inclination_label_encoder.pkl",
        "status":  "Supportive",
    },
    "interincisal_angle": {
        "display_name": "Interincisal Angle",
        "model":   "interincisal_angle_model.pkl",
        "encoder": "interincisal_angle_label_encoder.pkl",
        "status":  "Supportive",
    },
    "upper_incisal_display": {
        "display_name": "Upper Incisal Display",
        "model":   "upper_incisal_display_model.pkl",
        "encoder": "upper_incisal_display_label_encoder.pkl",
        "status":  "Supportive",
    },
    "profile_class": {
        "display_name": "Profile Class",
        "model":   "profile_class_model.pkl",
        "encoder": "profile_class_label_encoder.pkl",
        "status":  "Supportive",
    },
}

_PRIMARY_KEYS   = [k for k, v in XRAY_MODEL_SPECS.items() if v["status"] == "Primary"]
_SUPPORTIVE_KEYS = [k for k, v in XRAY_MODEL_SPECS.items() if v["status"] == "Supportive"]

# ─────────────────────────────────────────────────────────────────────────────
# Singleton cache
# ─────────────────────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_hrnet() -> SimpleHRNet19:
    if "hrnet19" not in _cache:
        device  = _get_device()
        ckpt_path = str(_HRNET_PATH)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"X-ray HRNet checkpoint not found: {ckpt_path}\n"
                "Expected: model/xray/best_hrnet_19_landmarks.pth"
            )
        # Checkpoint is a full dict {model_state_dict, img_size, heatmap_size, ...}
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Unexpected checkpoint format in {ckpt_path}. "
                "Expected a dict with key 'model_state_dict'."
            )
        # Validate metadata
        ckpt_img_size      = checkpoint.get("img_size",      IMG_SIZE)
        ckpt_heatmap_size  = checkpoint.get("heatmap_size",  HEATMAP_SIZE)
        ckpt_num_landmarks = checkpoint.get("num_landmarks", NUM_LANDMARKS)
        if ckpt_img_size != IMG_SIZE:
            raise ValueError(f"Checkpoint img_size={ckpt_img_size} but expected {IMG_SIZE}.")
        if ckpt_heatmap_size != HEATMAP_SIZE:
            raise ValueError(f"Checkpoint heatmap_size={ckpt_heatmap_size} but expected {HEATMAP_SIZE}.")
        if ckpt_num_landmarks != NUM_LANDMARKS:
            raise ValueError(f"Checkpoint num_landmarks={ckpt_num_landmarks} but expected {NUM_LANDMARKS}.")

        model = SimpleHRNet19(num_landmarks=NUM_LANDMARKS).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        _cache["hrnet19"] = model
        if DEBUG:
            print(f"[XRAY] HRNet loaded: img={ckpt_img_size} hm={ckpt_heatmap_size} lm={ckpt_num_landmarks}")
    return _cache["hrnet19"]


def _load_diagnosis_models() -> Tuple[Dict, Dict, List]:
    """Load all 11 XGBoost models, 11 encoders, and feature columns (singleton)."""
    if "xgb11" not in _cache:
        feat_path = str(_FEATURE_COLS_PATH)
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(
                f"feature_columns.pkl not found: {feat_path}\n"
                "Expected: model/xray/feature_columns.pkl"
            )
        feature_cols: List[str] = joblib.load(feat_path)
        if len(feature_cols) != NUM_FEATURES:
            raise ValueError(
                f"feature_columns.pkl has {len(feature_cols)} entries but expected {NUM_FEATURES}."
            )
        if len(set(feature_cols)) != NUM_FEATURES:
            raise ValueError("feature_columns.pkl contains duplicate column names.")

        xgbs: Dict[str, Any] = {}
        encs: Dict[str, Any] = {}
        for key, spec in XRAY_MODEL_SPECS.items():
            model_path   = str(_XRAY_DIR / spec["model"])
            encoder_path = str(_XRAY_DIR / spec["encoder"])
            for p, label in [(model_path, "model"), (encoder_path, "encoder")]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        f"X-ray {label} file not found for '{key}': {p}"
                    )
            xgbs[key] = joblib.load(model_path)
            encs[key]  = joblib.load(encoder_path)
            if DEBUG:
                print(f"[XRAY] Loaded {key}: n_features={xgbs[key].n_features_in_}")

        _cache["xgb11"]        = xgbs
        _cache["enc11"]        = encs
        _cache["feature_cols"] = feature_cols

    return _cache["xgb11"], _cache["enc11"], _cache["feature_cols"]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — exact Colab version (no ImageNet norm, no letterbox)
# ─────────────────────────────────────────────────────────────────────────────

def _predict_landmarks(
    image_path: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Returns
    -------
    pred_original : (19, 2) float64 — (x, y) in original image pixels
    pred_384      : (19, 2) float64 — (x, y) in 384×384 space
    original_w    : int
    original_h    : int
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read X-ray image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]

    # Resize directly (no aspect-ratio preservation — matches training)
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] — /255 ONLY, no ImageNet mean/std
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))                           # HWC → CHW
    device = _get_device()
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1,3,384,384)

    model = _load_hrnet()
    with torch.no_grad():
        heatmaps = model(tensor)                               # (1, 19, 96, 96)

    if heatmaps.shape != (1, NUM_LANDMARKS, HEATMAP_SIZE, HEATMAP_SIZE):
        raise RuntimeError(
            f"Unexpected HRNet output shape {tuple(heatmaps.shape)}. "
            f"Expected (1, {NUM_LANDMARKS}, {HEATMAP_SIZE}, {HEATMAP_SIZE})."
        )

    # Argmax decode
    B, K, H, W = heatmaps.shape
    flat    = heatmaps.view(B, K, -1)
    max_idx = torch.argmax(flat, dim=2)
    ys = (max_idx // W).float()
    xs = (max_idx % W).float()
    # Scale from heatmap space to 384-image space
    points_384 = torch.stack([xs, ys], dim=2).float() * (IMG_SIZE / HEATMAP_SIZE)
    pred_384   = points_384[0].cpu().numpy()                  # (19, 2)

    # Restore to original image coordinates using actual w/h (not assumed 1935×2400)
    pred_original        = pred_384.copy()
    pred_original[:, 0] *= original_w / IMG_SIZE
    pred_original[:, 1] *= original_h / IMG_SIZE

    if DEBUG:
        print(f"[XRAY] Image: {original_w}x{original_h}  HM: {heatmaps.shape}")
        for i, (px, py) in enumerate(pred_original[:3]):
            print(f"[XRAY]   LM{i+1}: ({px:.1f}, {py:.1f})")

    return pred_original, pred_384, original_w, original_h


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — 38 coords + 171 distances + 8 angles = 217
# ─────────────────────────────────────────────────────────────────────────────

# 8 clinical angle triplets (1-based landmark indices; middle = vertex)
_ANGLE_TRIPLETS: List[Tuple[int, int, int]] = [
    (2, 13, 14),
    (2, 15, 8),
    (7, 15, 8),
    (9, 15, 8),
    (2, 13, 8),
    (3, 13, 14),
    (13, 14, 8),
    (15, 8, 9),
]


def _generate_features(pred_original: np.ndarray) -> Dict[str, float]:
    """
    Build the 217-feature dict from 19 original-image landmarks.

    Feature order
    ─────────────
    1–38   : normalized (x, y) for each of 19 landmarks  → p1_x, p1_y … p19_x, p19_y
    39–209 : pairwise Euclidean distances on norm pts     → d_1_2, d_1_3 … d_18_19
    210–217: 8 clinical angles (degrees)                  → a_2_13_14 … a_15_8_9
    """
    pts = np.asarray(pred_original, dtype=np.float32)
    if pts.shape != (NUM_LANDMARKS, 2):
        raise ValueError(f"Expected ({NUM_LANDMARKS}, 2) landmarks; got {pts.shape}.")
    if not np.all(np.isfinite(pts)):
        raise ValueError("Landmark coordinates contain non-finite values.")

    # Bounding-box normalization
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    scale  = np.linalg.norm(max_xy - min_xy)
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError(f"Degenerate landmark bounding-box (scale={scale:.4g}). Check image quality.")
    norm_pts = (pts - center) / scale

    feats: Dict[str, float] = {}

    # 38 normalized coordinates
    for i, (xv, yv) in enumerate(norm_pts, start=1):
        feats[f"p{i}_x"] = float(xv)
        feats[f"p{i}_y"] = float(yv)

    # 171 pairwise Euclidean distances
    for i, j in combinations(range(NUM_LANDMARKS), 2):
        feats[f"d_{i+1}_{j+1}"] = float(np.linalg.norm(norm_pts[i] - norm_pts[j]))

    # 8 clinical angles
    for a_i, b_i, c_i in _ANGLE_TRIPLETS:
        A  = norm_pts[a_i - 1]
        B  = norm_pts[b_i - 1]
        C  = norm_pts[c_i - 1]
        BA = A - B
        BC = C - B
        cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        feats[f"a_{a_i}_{b_i}_{c_i}"] = float(np.degrees(np.arccos(cos_angle)))

    if len(feats) != NUM_FEATURES:
        raise RuntimeError(f"Feature count mismatch: generated {len(feats)}, expected {NUM_FEATURES}.")

    if DEBUG:
        print(f"[XRAY] Features generated: {len(feats)}")

    return feats


def _build_feature_matrix(feature_dict: Dict[str, float]) -> "pd.DataFrame":
    """Reorder features to match training column order, returning a (1, 217) DataFrame."""
    import pandas as pd

    _, _, feature_cols = _load_diagnosis_models()

    missing = [c for c in feature_cols if c not in feature_dict]
    extra   = [c for c in feature_dict if c not in feature_cols]
    if missing or extra:
        raise ValueError(
            f"Feature mismatch!\n"
            f"  Missing from generated: {missing[:10]}{'…' if len(missing)>10 else ''}\n"
            f"  Extra (not in training): {extra[:10]}{'…' if len(extra)>10 else ''}"
        )

    X = pd.DataFrame([feature_dict])[feature_cols]

    if X.shape != (1, NUM_FEATURES):
        raise RuntimeError(f"Feature matrix shape {X.shape}, expected (1, {NUM_FEATURES}).")
    if not np.all(np.isfinite(X.values)):
        raise ValueError("Feature matrix contains NaN or Inf. Check landmark quality.")

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
    (0,255,255),(128,0,255),(255,128,0),(0,128,255),(128,255,0),
    (255,0,128),(0,255,128),(200,100,50),(50,200,100),(100,50,200),
    (220,220,0),(0,220,220),(220,0,220),(180,180,180),
]


def _draw_landmarks(image_path: str, landmarks_orig: np.ndarray, overlay_path: str) -> None:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image for overlay: {image_path}")

    h, w = bgr.shape[:2]
    radius     = max(4, min(w, h) // 100)
    font_scale = max(0.3, min(w, h) / 1000)
    thickness  = max(1, radius // 3)

    for i, (xv, yv) in enumerate(landmarks_orig):
        color = _COLORS[i % len(_COLORS)]
        xi, yi = int(round(xv)), int(round(yv))
        cv2.circle(bgr, (xi, yi), radius, color, -1)
        cv2.circle(bgr, (xi, yi), radius + 1, (0, 0, 0), 1)
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lx = max(0, xi - tw // 2)
        ly = max(th, yi - radius - 3)
        cv2.putText(bgr, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 1)
        cv2.putText(bgr, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    cv2.imwrite(overlay_path, bgr)


# ─────────────────────────────────────────────────────────────────────────────
# 11-model prediction → versioned JSON structure
# ─────────────────────────────────────────────────────────────────────────────

def _probability_label(prob: float) -> str:
    """Human-readable model probability tier (not clinical certainty)."""
    if prob >= 0.85:
        return "High"
    if prob >= 0.70:
        return "Medium"
    return "Low"


def _run_diagnosis(feature_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Run all 11 XGBoost models and return a versioned schema_version=2 dict.

    The returned dict is what gets stored in OrthoCase.diagnosis_json.
    """
    xgbs, encs, _ = _load_diagnosis_models()
    X = _build_feature_matrix(feature_dict)   # (1, 217) pandas DataFrame

    primary_results:   Dict[str, Any] = {}
    supportive_results: Dict[str, Any] = {}
    all_results:        List[Dict[str, Any]] = []

    for key, spec in XRAY_MODEL_SPECS.items():
        clf     = xgbs[key]
        enc     = encs[key]
        status  = spec["status"]

        # Predict
        encoded_pred = int(clf.predict(X)[0])
        probabilities = clf.predict_proba(X)[0]

        # Validate
        if encoded_pred < 0 or encoded_pred >= len(enc.classes_):
            raise RuntimeError(
                f"Model '{key}' returned class id {encoded_pred} "
                f"outside encoder range [0, {len(enc.classes_)-1}]."
            )
        if len(probabilities) != len(enc.classes_):
            raise RuntimeError(
                f"Model '{key}': probability vector length {len(probabilities)} "
                f"!= encoder class count {len(enc.classes_)}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(f"Model '{key}' returned non-finite probabilities.")

        predicted_label    = str(enc.inverse_transform([encoded_pred])[0])
        model_probability  = float(probabilities[encoded_pred])
        probability_pct    = round(model_probability * 100, 1)

        class_probabilities = {
            str(cls): round(float(p) * 100, 2)
            for cls, p in zip(enc.classes_, probabilities)
        }

        requires_review = (status == "Supportive") or (model_probability < 0.70)

        entry: Dict[str, Any] = {
            "display_name":        spec["display_name"],
            "prediction":          predicted_label,
            "probability":         round(model_probability, 4),
            "probability_percent": probability_pct,
            # Legacy-compatible aliases used by existing template/PDF code
            "label":               predicted_label,
            "confidence":          probability_pct,
            "confidence_level":    _probability_label(model_probability),
            "status":              status,
            "requires_doctor_review": requires_review,
            "class_probabilities": class_probabilities,
        }

        if status == "Primary":
            primary_results[key]   = entry
        else:
            supportive_results[key] = entry

        all_results.append({"key": key, **entry})

        if DEBUG:
            print(f"[XRAY] {key}: {predicted_label} ({probability_pct}%)")

    return {
        "schema_version":  2,
        "pipeline":        "hrnet19_xgboost_11",
        "landmark_count":  NUM_LANDMARKS,
        "feature_count":   NUM_FEATURES,
        "primary":         primary_results,
        "supportive":      supportive_results,
        "all_results":     all_results,
        "disclaimer": (
            "AI decision-support output. "
            "Final interpretation must be confirmed by the clinician. "
            "Model probabilities are not validated clinical confidence scores."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy compatibility
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_KEY_LABELS = {
    "skeletal_class": "Skeletal Class",
    "upper_lip":      "Upper Lip Position",
    "lower_lip":      "Lower Lip Position",
    "profile_class":  "Profile Class",
}


def parse_xray_diagnosis_json(diagnosis_json: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse OrthoCase.diagnosis_json and return a normalized view for templates/PDF.

    Supports both:
    ─ schema_version=2  (new 11-model pipeline)
    ─ legacy flat dict  (old 4-model pipeline: skeletal_class / upper_lip / lower_lip / profile_class)

    Returns None if the JSON is absent or unparseable.
    Returns a dict with keys:
        schema_version  int   (1 for legacy, 2 for new)
        primary         dict  of {key: entry}
        supportive      dict  of {key: entry}
        all_results     list
        is_legacy       bool
    """
    if not diagnosis_json:
        return None
    try:
        data = json.loads(diagnosis_json)
    except Exception:
        return None

    if isinstance(data, dict) and data.get("schema_version") == 2:
        return {**data, "is_legacy": False}

    # ── Legacy: flat dict with 4 keys ────────────────────────────────────────
    if isinstance(data, dict) and any(k in data for k in _LEGACY_KEY_LABELS):
        primary: Dict[str, Any]   = {}
        supportive: Dict[str, Any] = {}
        all_results: List[Dict]   = []

        # Map old keys to normalized entries
        for key, display in _LEGACY_KEY_LABELS.items():
            entry_raw = data.get(key)
            if not entry_raw:
                continue
            entry = {
                "display_name":           display,
                "prediction":             entry_raw.get("label", "—"),
                "label":                  entry_raw.get("label", "—"),
                "confidence":             entry_raw.get("confidence", 0),
                "probability_percent":    entry_raw.get("confidence", 0),
                "confidence_level":       entry_raw.get("confidence_level", "—"),
                "status":                 "Primary" if key in ("skeletal_class", "profile_class") else "Supportive",
                "requires_doctor_review": True,  # legacy cases should always be re-reviewed
                "class_probabilities":    {},
            }
            if entry["status"] == "Primary":
                primary[key] = entry
            else:
                supportive[key] = entry
            all_results.append({"key": key, **entry})

        return {
            "schema_version": 1,
            "pipeline":       "legacy_4model",
            "primary":        primary,
            "supportive":     supportive,
            "all_results":    all_results,
            "is_legacy":      True,
            "disclaimer": (
                "Legacy AI result (4-model pipeline). "
                "Re-run X-ray analysis to use the updated 11-model pipeline."
            ),
        }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_ortho_analysis(image_path: str, overlay_path: str) -> Dict[str, Any]:
    """
    Full X-ray pipeline: landmarks → features → 11 diagnoses → overlay image.

    Returns
    -------
    On success::
        {
            "success":       True,
            "landmarks":     [(x,y), ...],   # 19 int tuples
            "diagnosis":     {...},            # versioned schema_version=2 dict
            "overlay_path":  str,
            "num_landmarks": 19,
        }

    On failure::
        {"success": False, "error": str}
    """
    try:
        pred_original, pred_384, orig_w, orig_h = _predict_landmarks(image_path)
        feature_dict = _generate_features(pred_original)
        diagnosis    = _run_diagnosis(feature_dict)
        _draw_landmarks(image_path, pred_original, overlay_path)

        landmarks_list = [(int(round(x)), int(round(y))) for x, y in pred_original]

        return {
            "success":       True,
            "landmarks":     landmarks_list,
            "diagnosis":     diagnosis,
            "overlay_path":  overlay_path,
            "num_landmarks": len(landmarks_list),
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(exc)}
