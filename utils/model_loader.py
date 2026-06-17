"""
model_loader.py — Public API surface for DentAlign model management.

All heavy lifting lives in utils/inference.py (HRNet) and
utils/side_diagnosis.py (XGBoost pipeline).  This module re-exports
the functions that external code (api routes, tests, scripts) should
call so they don't need to import from multiple internal modules.

Usage
-----
    from utils.model_loader import (
        load_hrnet,          # warm up HRNet (optional; called lazily on first predict)
        predict_side,        # image path → landmarks + heatmap peaks
        run_diagnosis,       # landmarks + growth_stage → XGBoost result dict
        full_pipeline,       # image path + growth_stage → one-shot result
    )
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

# ── HRNet layer ───────────────────────────────────────────────────────────────
from utils.inference import (
    load_model,
    predict_landmarks,
)

# ── XGBoost layer ────────────────────────────────────────────────────────────
from utils.side_diagnosis import run_side_diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hrnet(variant: str = "SIDE") -> None:
    """
    Pre-load the HRNet model into the singleton cache.

    Call this at app startup to avoid first-request latency.
    Safe to call multiple times — loads only once.
    """
    load_model(variant)


def predict_side(image_path: str) -> Dict[str, Any]:
    """
    Run the side-view HRNet model on *image_path*.

    Returns the raw inference dict (same keys as predict_landmarks):
        landmarks       – list of (x, y) tuples, original-image pixels
        heatmap_peaks   – list of floats in [0, 1], one per landmark
        overlay_image   – np.ndarray (BGR) with dot overlay
        original_width  – int
        original_height – int
        variant         – "SIDE"

    Raises FileNotFoundError if the image or weights are missing.
    Raises ValueError if the image cannot be read.
    """
    return predict_landmarks(image_path, variant="SIDE")


def run_diagnosis(
    landmarks: Any,
    growth_stage: str,
) -> Dict[str, Any]:
    """
    Run the XGBoost diagnosis pipeline on pre-computed landmarks.

    Parameters
    ----------
    landmarks   : list of [x, y] or {"x": ..., "y": ...} dicts
    growth_stage: "adult" or "growing" (case-insensitive)

    Returns
    -------
    {
        "success":      bool,
        "angles":       {"nasiolabial": float, "profile_convexity": float,
                         "total_convexity": float, "mentolabial": float},
        "growth_stage": str,
        "diagnosis":    {"label": str, "confidence": float,
                         "confidence_level": str, "breakdown": list},
        "treatment":    {...} | None,   # present only when treatment model exists
        "error":        str,            # present only when success is False
    }
    """
    return run_side_diagnosis(landmarks, growth_stage)


def full_pipeline(
    image_path: str,
    growth_stage: str,
    min_landmarks: int = 17,
) -> Dict[str, Any]:
    """
    One-shot convenience function: HRNet → angles → XGBoost.

    Parameters
    ----------
    image_path   : absolute path to the side-view facial photo
    growth_stage : "adult" | "growing"
    min_landmarks: minimum landmark count required (default 17)

    Returns
    -------
    On success:
    {
        "success":      True,
        "landmarks":    [[x, y], ...],
        "heatmap_peaks": [float, ...],
        "angles":       {...},
        "growth_stage": str,
        "diagnosis":    {"label": str, "confidence": float, ...},
        "treatment":    {...} | None,
    }

    On failure:
    {
        "success": False,
        "error":   str,
    }
    """
    # ── Step 1: HRNet landmark detection ─────────────────────────────────────
    try:
        hrnet_out = predict_side(image_path)
    except Exception as exc:
        return {"success": False, "error": f"HRNet inference failed: {exc}"}

    landmarks = hrnet_out.get("landmarks", [])
    if len(landmarks) < min_landmarks:
        return {
            "success": False,
            "error": (
                f"HRNet returned only {len(landmarks)} landmarks "
                f"(need ≥ {min_landmarks}). Check that the image contains "
                "a clear side-view face."
            ),
        }

    # ── Step 2: XGBoost diagnosis ─────────────────────────────────────────────
    diag_out = run_diagnosis(landmarks, growth_stage)
    if not diag_out.get("success"):
        return diag_out  # already has "error" key

    return {
        "success":       True,
        "landmarks":     [[int(x), int(y)] for x, y in landmarks],
        "heatmap_peaks": hrnet_out.get("heatmap_peaks", []),
        "angles":        diag_out["angles"],
        "growth_stage":  diag_out["growth_stage"],
        "diagnosis":     diag_out["diagnosis"],
        "treatment":     diag_out.get("treatment"),
    }


def full_pipeline_from_bytes(
    image_bytes: bytes,
    growth_stage: str,
    suffix: str = ".jpg",
) -> Dict[str, Any]:
    """
    Like full_pipeline() but accepts raw image bytes (e.g. from request.files).

    Writes a temp file, runs the pipeline, deletes the file.
    """
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)
        return full_pipeline(tmp_path, growth_stage)
    except Exception as exc:
        return {"success": False, "error": f"Pipeline error: {exc}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
