"""
Run inference first, then validate model output (landmark count and coordinates).
No pre-inference face detection.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from utils.image_validation import MSG_ANALYSIS_FAILED
from utils.inference import predict_landmarks, save_overlay_image
from utils.landmark_validation import validate_landmarks_for_view
from utils.model_paths import OUTPUTS_DIR
from utils.paths import (
    normalize_stored_path,
    overlay_rel,
    resolve_project_path,
    results_abs,
    ensure_dir,
)


def _peaks_to_confidences(heatmap_peaks: Optional[List[float]]) -> Optional[str]:
    """
    Convert raw heatmap peak activations → 0-100% confidence scores (JSON string).

    WHY NO SIGMOID
    ──────────────
    Both HRNet models (SimpleHRNet for SIDE, HRNetKeypoint for FRONT_NS) end with
    a plain Conv2d and no explicit activation.  During training with MSE loss against
    Gaussian heatmap targets (peak ≈ 1.0, background ≈ 0.0), the weights learn to
    produce channel-peak values that naturally cluster in [0, 1]:

        confident landmark  → peak ≈ 0.8–1.0  → 80–100 %
        uncertain landmark  → peak ≈ 0.0–0.2  → 0–20 %

    Colab uses:  confidence = heatmap_max × 100
    (i.e. the raw peak value, already in [0, 1], scaled to a percentage).

    Applying sigmoid AGAIN to these already-bounded values crushes the distribution
    toward 50 % and breaks the clinical signal.  The fix is to clamp to [0, 1] and
    multiply by 100 — exactly matching the Colab pipeline.

    Negative peaks (landmark not found / suppressed activation) are clamped to 0 %.
    Peaks marginally above 1.0 (rare numerical overshoot) are clamped to 100 %.
    """
    if not heatmap_peaks:
        return None
    confidences = [
        round(min(max(float(v), 0.0), 1.0) * 100.0, 1)
        for v in heatmap_peaks
    ]
    return json.dumps(confidences)


def run_view_analysis(
    case_id: int,
    image_path: str,
    view_type: str,
    results_dir: str | None = None,
) -> Dict[str, Any]:
    view_type = view_type.upper()
    variant = view_type

    image_path = resolve_project_path(image_path) or image_path
    if results_dir is None:
        results_dir = results_abs()

    if not image_path or not os.path.isfile(image_path):
        return {
            "success": False,
            "message": MSG_ANALYSIS_FAILED,
            "reason": "missing image file",
            "landmarks": None,
            "failed_stage": "pre",
        }

    # 1) Model inference (no face-detection gate)
    try:
        pred = predict_landmarks(image_path, variant=variant)
    except FileNotFoundError as exc:
        return {
            "success": False,
            "message": MSG_ANALYSIS_FAILED,
            "reason": f"model missing: {exc}",
            "landmarks": None,
            "failed_stage": "inference",
        }
    except Exception as exc:
        return {
            "success": False,
            "message": MSG_ANALYSIS_FAILED,
            "reason": f"inference: {exc}",
            "landmarks": None,
            "failed_stage": "inference",
        }

    landmarks = pred.get("landmarks") or []
    heatmap_peaks = pred.get("heatmap_peaks")

    # 2) Post-inference: landmark count and coordinate sanity
    ok, user_msg, tech = validate_landmarks_for_view(
        image_path,
        landmarks,
        variant,
        heatmap_peaks=heatmap_peaks,
    )
    if not ok:
        return {
            "success": False,
            "message": user_msg or MSG_ANALYSIS_FAILED,
            "reason": f"post: {tech}",
            "landmarks": None,
            "failed_stage": "post",
        }

    overlay_filename = f"{case_id}_{view_type.lower()}_overlay.jpg"
    overlay_full = os.path.join(results_dir, overlay_filename)
    ensure_dir(results_dir)
    save_overlay_image(pred["overlay_image"], overlay_full)

    # Side-view: also save numbered landmark overlay under outputs/ (Colab-style export)
    if view_type == "SIDE":
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        outputs_path = os.path.join(OUTPUTS_DIR, overlay_filename)
        save_overlay_image(pred["overlay_image"], outputs_path)

    return {
        "success": True,
        "message": "",
        "reason": "",
        "landmarks": landmarks,
        "overlay_path": normalize_stored_path(overlay_rel(case_id, view_type)),
        "landmarks_json": json.dumps(
            [{"x": int(x), "y": int(y)} for x, y in landmarks]
        ),
        # Sigmoid-scaled confidence per landmark (0-100%), stored as JSON string.
        # Built from the raw peak activation of each heatmap channel.
        "confidence_json": _peaks_to_confidences(heatmap_peaks),
        "failed_stage": None,
    }
