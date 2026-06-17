"""
utils/frontal_diagnosis.py
==========================
Six-model frontal facial diagnosis pipeline.

Models live in:
    <project_root>/model/frontal_diagnosis_models/

Files required (15):
    vertical_diagnosis_model.pkl   + vertical_diagnosis_encoder.pkl
    lip_diagnosis_model.pkl        + lip_diagnosis_encoder.pkl
    line_diagnosis_model.pkl       + line_diagnosis_encoder.pkl
    upper_lip_diagnosis_model.pkl  + upper_lip_diagnosis_encoder.pkl
    chin_diagnosis_model.pkl       + chin_diagnosis_encoder.pkl
    width_diagnosis_model.pkl      + width_diagnosis_encoder.pkl
    model_configurations.pkl
    selected_algorithms.pkl
    treatment_maps.pkl

Feature formulas exactly reproduce the Colab training notebook.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

from utils.frontal_measurements import _normalize_points, _pt, _dist

logger = logging.getLogger(__name__)

# ── Model directory ──────────────────────────────────────────────────────────
_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "model" / "frontal_diagnosis_models"
)

_REQUIRED_FILES = [
    "vertical_diagnosis_model.pkl",
    "vertical_diagnosis_encoder.pkl",
    "lip_diagnosis_model.pkl",
    "lip_diagnosis_encoder.pkl",
    "line_diagnosis_model.pkl",
    "line_diagnosis_encoder.pkl",
    "upper_lip_diagnosis_model.pkl",
    "upper_lip_diagnosis_encoder.pkl",
    "chin_diagnosis_model.pkl",
    "chin_diagnosis_encoder.pkl",
    "width_diagnosis_model.pkl",
    "width_diagnosis_encoder.pkl",
    "model_configurations.pkl",
    "selected_algorithms.pkl",
    "treatment_maps.pkl",
]

FALLBACK_TREATMENT = "Clinical assessment required before selecting treatment."

# ── Lazy-loaded singleton cache ──────────────────────────────────────────────
_cache: Dict[str, Any] = {}
_cache_loaded: bool = False
_cache_error: str = ""


def _load_all() -> None:
    """Load all 15 model files once, then run startup diagnostics."""
    global _cache_loaded, _cache_error

    if _cache_loaded:
        return
    if _cache_error:
        raise RuntimeError(_cache_error)

    missing = [f for f in _REQUIRED_FILES if not (_MODEL_DIR / f).is_file()]
    if missing:
        _cache_error = (
            f"Missing frontal diagnosis model files in {_MODEL_DIR}:\n"
            + "\n".join(f"  {f}" for f in missing)
        )
        raise FileNotFoundError(_cache_error)

    try:
        for stem in ("vertical", "lip", "line", "upper_lip", "chin", "width"):
            _cache[f"{stem}_model"]   = joblib.load(_MODEL_DIR / f"{stem}_diagnosis_model.pkl")
            _cache[f"{stem}_encoder"] = joblib.load(_MODEL_DIR / f"{stem}_diagnosis_encoder.pkl")

        _cache["configurations"] = joblib.load(_MODEL_DIR / "model_configurations.pkl")
        _cache["algorithms"]     = joblib.load(_MODEL_DIR / "selected_algorithms.pkl")
        _cache["treatment_maps"] = joblib.load(_MODEL_DIR / "treatment_maps.pkl")
    except Exception as exc:
        _cache_error = str(exc)
        raise

    # ── Startup validation ───────────────────────────────────────────────────
    logger.info("=== Frontal Diagnosis Models Loaded from %s ===", _MODEL_DIR)
    for key in ("vertical", "lip", "line", "upper_lip", "chin", "width"):
        enc  = _cache[f"{key}_encoder"]
        alg  = _cache["algorithms"].get(key, "unknown")
        cfg  = _cache["configurations"].get(key, {})
        tmap = _cache["treatment_maps"].get(key, {})
        classes = list(enc.classes_)
        logger.info(
            "  [%s] algorithm=%s  feature='%s'  classes=%s",
            key, alg, cfg.get("feature"), classes,
        )
        for cls in classes:
            if cls in tmap:
                logger.info("    OK  '%s'", cls)
            else:
                logger.warning("    MISSING TREATMENT for '%s' — fallback will be used", cls)

    _cache_loaded = True


# ── Feature computation ──────────────────────────────────────────────────────

def _line_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Exact Colab formula.  Uses atan(dy/dx), not atan2, to reproduce training.
    Returns 90.0 when the line is vertical (dx == 0).
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 90.0
    return math.degrees(math.atan(dy / dx))


def _compute_features(pts: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Compute the six model features from a 34-point list (1-indexed via _pt).
    No rounding is applied; raw floats are passed to models.

    Raises ValueError for degenerate inputs (zero denominators).
    """
    # ── 1. Middle/Lower Ratio ────────────────────────────────────────────────
    p11 = _pt(pts, 11)
    p16 = _pt(pts, 16)
    p24 = _pt(pts, 24)
    lower = _dist(p16, p24)
    if lower == 0:
        raise ValueError("Zero lower-facial-third distance (landmark 16 == landmark 24).")
    middle_lower_ratio = _dist(p11, p16) / lower

    # ── 2. Upper/Lower Lip Ratio ─────────────────────────────────────────────
    p18 = _pt(pts, 18)
    p21 = _pt(pts, 21)
    p23 = _pt(pts, 23)
    lower_lip = _dist(p21, p23)
    if lower_lip == 0:
        raise ValueError("Zero lower-lip-and-chin distance (landmark 21 == landmark 23).")
    upper_lower_lip_ratio = _dist(p18, p21) / lower_lip

    # ── 3. Line Difference (°) ───────────────────────────────────────────────
    p8  = _pt(pts, 8)
    p32 = _pt(pts, 32)
    p20 = _pt(pts, 20)
    p22 = _pt(pts, 22)
    interp_angle = _line_angle(p8[0],  p8[1],  p32[0], p32[1])
    comm_angle   = _line_angle(p20[0], p20[1], p22[0], p22[1])
    diff = abs(interp_angle - comm_angle) % 180.0
    line_difference = min(diff, 180.0 - diff)

    # ── 4. Upper-Lip Midline Offset ──────────────────────────────────────────
    p12 = _pt(pts, 12)
    p14 = _pt(pts, 14)
    ref_x_upper = (
        p11[0] + p12[0] + p14[0] + p16[0] + p21[0] + p23[0] + p24[0]
    ) / 7.0
    upper_lip_offset = abs(p18[0] - ref_x_upper)

    # ── 5. Chin Midline Offset ───────────────────────────────────────────────
    ref_x_chin = (
        p11[0] + p12[0] + p14[0] + p16[0] + p18[0] + p21[0] + p23[0]
    ) / 7.0
    chin_offset = abs(p24[0] - ref_x_chin)

    # ── 6. Facial/Mandibular Width Ratio ────────────────────────────────────
    p1  = _pt(pts, 1)
    p2  = _pt(pts, 2)
    p3  = _pt(pts, 3)
    p26 = _pt(pts, 26)
    p27 = _pt(pts, 27)
    p28 = _pt(pts, 28)
    facial_width     = max(_dist(p1, p28), _dist(p2, p27))
    mandibular_width = _dist(p3, p26)
    if mandibular_width == 0:
        raise ValueError("Zero mandibular width (landmark 3 == landmark 26).")
    facial_mandibular_ratio = facial_width / mandibular_width

    return {
        "Middle/Lower Ratio":            middle_lower_ratio,
        "Upper/Lower Lip Ratio":         upper_lower_lip_ratio,
        "Line Difference (°)":           line_difference,
        "Upper-Lip Midline Offset":      upper_lip_offset,
        "Chin Midline Offset":           chin_offset,
        "Facial/Mandibular Width Ratio": facial_mandibular_ratio,
    }


# ── Per-model prediction ─────────────────────────────────────────────────────

def _predict_one(key: str, feature_col: str, value: float) -> Dict[str, Any]:
    """
    Run a single model and return a result dict.

    Uses explicit class-position lookup for confidence so the index is always
    correct regardless of how the model stores its internal class order.
    All numeric values are cast to native Python types for JSON serialisation.
    """
    model   = _cache[f"{key}_model"]
    encoder = _cache[f"{key}_encoder"]
    tmap    = _cache["treatment_maps"].get(key, {})
    cfg     = _cache["configurations"].get(key, {})
    alg     = _cache["algorithms"].get(key, "unknown")

    input_df = pd.DataFrame({feature_col: [value]})

    # Encoded numeric prediction
    predicted_encoded = int(model.predict(input_df)[0])
    probabilities     = model.predict_proba(input_df)[0]
    class_positions   = list(model.classes_)
    probability_index = class_positions.index(predicted_encoded)
    confidence        = float(probabilities[probability_index])

    # Decode to human-readable label
    diagnosis = str(encoder.inverse_transform([predicted_encoded])[0])

    # Treatment lookup
    treatment = tmap.get(diagnosis, FALLBACK_TREATMENT)

    return {
        "measurement": cfg.get("feature", feature_col),
        "value":       float(value),
        "diagnosis":   diagnosis,
        "treatment":   str(treatment),
        "confidence":  confidence,
        "algorithm":   str(alg),
    }


# ── Public API ────────────────────────────────────────────────────────────────

#: Ordered list of (cache_key, feature_column_name) for the six models.
_MODEL_KEYS = [
    ("vertical",  "Middle/Lower Ratio"),
    ("lip",       "Upper/Lower Lip Ratio"),
    ("line",      "Line Difference (°)"),
    ("upper_lip", "Upper-Lip Midline Offset"),
    ("chin",      "Chin Midline Offset"),
    ("width",     "Facial/Mandibular Width Ratio"),
]


def predict_frontal_diagnosis(landmarks: List[Any]) -> Dict[str, Any]:
    """
    Run all six frontal diagnosis models from 34 stored landmarks.

    Parameters
    ----------
    landmarks : list of {"x": int, "y": int} dicts (or [x, y] sequences),
                exactly 34 entries.

    Returns
    -------
    On success::

        {
            "success": True,
            "results": {
                "vertical":  {"measurement", "value", "diagnosis",
                               "treatment", "confidence", "algorithm"},
                "lip":       {...},
                "line":      {...},
                "upper_lip": {...},
                "chin":      {...},
                "width":     {...},
            }
        }

    On failure::

        {"success": False, "error": str}
    """
    try:
        _load_all()
    except Exception as exc:
        return {"success": False, "error": f"Model loading failed: {exc}"}

    if not isinstance(landmarks, list) or len(landmarks) < 34:
        n = len(landmarks) if isinstance(landmarks, list) else 0
        return {"success": False, "error": f"Expected 34 landmarks, got {n}."}

    try:
        pts = _normalize_points(landmarks)
    except Exception as exc:
        return {"success": False, "error": f"Invalid landmark coordinates: {exc}"}

    try:
        features = _compute_features(pts)
    except Exception as exc:
        return {"success": False, "error": f"Feature computation failed: {exc}"}

    results: Dict[str, Any] = {}
    errors: List[str] = []

    for key, feature_col in _MODEL_KEYS:
        try:
            results[key] = _predict_one(key, feature_col, features[feature_col])
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            results[key] = {"error": str(exc)}

    if errors:
        return {
            "success": False,
            "error": "; ".join(errors),
            "partial_results": results,
        }

    return {"success": True, "results": results}
