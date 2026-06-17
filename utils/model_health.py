"""
Model health check: load weights, run sample inference, report landmark counts.
Used by /model-health page and `py check_models.py`.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from utils.inference import clear_model_cache, get_model_spec, load_model, predict_landmarks
from utils.landmark_validation import EXPECTED_KP, validate_landmarks_for_view
from utils.model_paths import BASE_DIR, MODEL_DIR, list_model_dir

UPLOADS_DIR = os.path.join(BASE_DIR, "static", "uploads")


def _find_sample_image(keyword: str) -> Optional[str]:
    if not os.path.isdir(UPLOADS_DIR):
        return None
    keyword = keyword.lower()
    for name in sorted(os.listdir(UPLOADS_DIR)):
        low = name.lower()
        if not low.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        if keyword in low:
            return os.path.join(UPLOADS_DIR, name)
    return None


def _check_variant(variant: str, sample_path: Optional[str] = None) -> Dict[str, Any]:
    key = variant.upper()
    label = "Side" if key == "SIDE" else "Frontal non-smile"
    row: Dict[str, Any] = {
        "variant": key,
        "label": label,
        "ok": False,
        "messages": [],
        "errors": [],
    }

    try:
        spec = get_model_spec(key)
        row["weight_file"] = os.path.basename(spec["path"])
        row["architecture"] = spec["arch"]
        row["expected_landmarks"] = EXPECTED_KP[key]
        row["messages"].append(f"{label} model weights found: {row['weight_file']}")
    except Exception as exc:
        row["errors"].append(str(exc))
        return row

    try:
        clear_model_cache()
        load_model(key)
        row["messages"].append(f"{label} model loaded successfully")
    except Exception as exc:
        row["errors"].append(f"Load failed: {exc}")
        return row

    img = sample_path or _find_sample_image("side" if key == "SIDE" else "front")
    if not img or not os.path.isfile(img):
        row["messages"].append("No sample image in static/uploads — load-only check passed")
        row["ok"] = True
        return row

    row["sample_image"] = os.path.basename(img)
    try:
        result = predict_landmarks(img, variant=key)
        landmarks = result.get("landmarks") or []
        count = len(landmarks)
        row["landmark_count"] = count
        row["messages"].append("Prediction successful")
        row["messages"].append(f"Landmark count: {count}")

        ok, msg, tech = validate_landmarks_for_view(img, landmarks, key)
        if not ok:
            row["errors"].append(f"Validation: {msg} ({tech})")
            return row

        row["ok"] = True
    except Exception as exc:
        row["errors"].append(f"Inference failed: {exc}")

    return row


def _check_xray() -> Dict[str, Any]:
    """Health check for the X-ray 11-model pipeline."""
    row: Dict[str, Any] = {
        "variant": "XRAY",
        "label":   "X-Ray Cephalometric (HRNet19 + 11×XGBoost)",
        "ok":      False,
        "messages": [],
        "errors":   [],
    }
    try:
        import joblib
        from utils.orthodontic_ai_inference import (
            XRAY_MODEL_SPECS, NUM_FEATURES, NUM_LANDMARKS,
            _HRNET_PATH, _FEATURE_COLS_PATH, _XRAY_DIR,
        )
        # HRNet checkpoint
        hrnet_path = str(_HRNET_PATH)
        if not os.path.isfile(hrnet_path):
            row["errors"].append(f"HRNet checkpoint missing: {hrnet_path}")
            return row
        sz = os.path.getsize(hrnet_path)
        row["messages"].append(f"HRNet checkpoint: {os.path.basename(hrnet_path)} ({sz:,} B)")

        # Feature columns
        feat_path = str(_FEATURE_COLS_PATH)
        if not os.path.isfile(feat_path):
            row["errors"].append(f"feature_columns.pkl missing: {feat_path}")
            return row
        cols = joblib.load(feat_path)
        if len(cols) != NUM_FEATURES:
            row["errors"].append(f"feature_columns.pkl: {len(cols)} cols, expected {NUM_FEATURES}")
            return row
        if len(set(cols)) != len(cols):
            row["errors"].append("feature_columns.pkl: duplicate column names")
            return row
        row["messages"].append(f"feature_columns.pkl: {len(cols)} unique columns OK")

        # 11 models + encoders
        for key, spec in XRAY_MODEL_SPECS.items():
            mp = str(_XRAY_DIR / spec["model"])
            ep = str(_XRAY_DIR / spec["encoder"])
            if not os.path.isfile(mp):
                row["errors"].append(f"Model missing: {spec['model']}")
                continue
            if not os.path.isfile(ep):
                row["errors"].append(f"Encoder missing: {spec['encoder']}")
                continue
            clf = joblib.load(mp)
            nf  = getattr(clf, "n_features_in_", None)
            if nf != NUM_FEATURES:
                row["errors"].append(f"{key}: n_features_in_={nf}, expected {NUM_FEATURES}")
            else:
                enc    = joblib.load(ep)
                n_cls  = len(getattr(enc, "classes_", []))
                row["messages"].append(f"{key}: n_features={nf}, classes={n_cls}")

        if not row["errors"]:
            row["ok"] = True
            row["messages"].append(f"All {len(XRAY_MODEL_SPECS)} XGBoost models verified OK")

    except Exception as exc:
        row["errors"].append(f"X-ray health check failed: {exc}")

    return row


def run_model_health_check() -> Dict[str, Any]:
    return {
        "model_dir": MODEL_DIR,
        "files_in_model_dir": list_model_dir(),
        "device": str(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")),
        "checks": [
            _check_variant("SIDE"),
            _check_variant("FRONT_NS"),
            _check_xray(),
        ],
    }


def print_health_report(report: Optional[Dict[str, Any]] = None) -> int:
    report = report or run_model_health_check()
    print("=== Model Health Check ===")
    print(f"Model directory: {report['model_dir']}")
    print(f"Weight files: {', '.join(report['files_in_model_dir']) or '(none)'}")
    print(f"Device: {report['device']}")
    exit_code = 0
    for check in report["checks"]:
        print()
        print(f"--- {check['label']} ({check['variant']}) ---")
        for msg in check.get("messages", []):
            print(f"  OK: {msg}")
        for err in check.get("errors", []):
            print(f"  ERROR: {err}")
            exit_code = 1
        if check.get("ok"):
            print("  Status: PASS")
        elif check.get("errors"):
            print("  Status: FAIL")
        else:
            print("  Status: PARTIAL")
    return exit_code


if __name__ == "__main__":
    sys.exit(print_health_report())
