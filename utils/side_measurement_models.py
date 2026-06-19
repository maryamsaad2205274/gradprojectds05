"""
utils/side_measurement_models.py
=================================
Measurement-level ML analysis service for side-profile angles.

Nine models (in model/side_profile_measurement_models/) provide one
diagnosis and one treatment consideration per cephalometric angle.
Results are merged so duplicate labels across measurements are grouped
with a list of the measurements that support them.

Entry point
-----------
    predict_side_measurement_analysis(
        nasiolabial, profile_convexity, total_convexity, mentolabial, growth_stage
    )

All models are loaded once at first call and cached for the process lifetime.

Model layout
------------
Diagnosis models (Random Forest, single-angle feature each):
    nasio_diagnosis_random_forest.pkl    — feature: "nasiolabial"
    profile_diagnosis_random_forest.pkl  — feature: "profile_convexity"
    total_diagnosis_random_forest.pkl    — feature: "total_convexity"
    mento_diagnosis_random_forest.pkl    — feature: "mentolabial"

Treatment models:
    nasio_treatment_random_forest.pkl    — features: ["nasiolabial", "growth_stage"]
    profile_treatment_random_forest.pkl  — features: ["profile_convexity", "growth_stage"]
    total_treatment_random_forest.pkl    — features: ["total_convexity", "growth_stage"]
    mento_treatment_xgboost.pkl          — features: ["mentolabial", "growth_stage"]
                                           (uses mento_treatment_label_encoder.pkl)

Safety rules
------------
- No model file is modified or retrained.
- No landmarks are re-detected or angles re-computed here.
  Callers must pass the angles already computed by the existing pipeline.
- growth_stage must be exactly "adult" or "growing" after strip+lower.
- All angle values are validated: must be numeric, non-NaN, non-infinite.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ── Model directory ───────────────────────────────────────────────────────────
_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "side_profile_measurement_models"
)

_MODEL_PATHS: Dict[str, Path] = {
    # Diagnosis — Random Forest, single-angle feature
    "nasio_diag":      _MODEL_DIR / "nasio_diagnosis_random_forest.pkl",
    "profile_diag":    _MODEL_DIR / "profile_diagnosis_random_forest.pkl",
    "total_diag":      _MODEL_DIR / "total_diagnosis_random_forest.pkl",
    "mento_diag":      _MODEL_DIR / "mento_diagnosis_random_forest.pkl",
    # Treatment — Random Forest (nasio, profile, total)
    "nasio_treat":     _MODEL_DIR / "nasio_treatment_random_forest.pkl",
    "profile_treat":   _MODEL_DIR / "profile_treatment_random_forest.pkl",
    "total_treat":     _MODEL_DIR / "total_treatment_random_forest.pkl",
    # Treatment — XGBoost + LabelEncoder (mentolabial)
    "mento_treat":     _MODEL_DIR / "mento_treatment_xgboost.pkl",
    "mento_treat_enc": _MODEL_DIR / "mento_treatment_label_encoder.pkl",
}

_DISPLAY_NAMES: Dict[str, str] = {
    "nasiolabial":       "Nasolabial angle",
    "profile_convexity": "Profile convexity",
    "total_convexity":   "Total facial convexity",
    "mentolabial":       "Mentolabial angle",
}

_DISCLAIMER = (
    "These outputs are clinical decision-support considerations "
    "and must be reviewed by a qualified orthodontist."
)

# ── Singleton model cache ─────────────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}


def _load(key: str) -> Any:
    """Load and cache a model file from _MODEL_PATHS[key].

    Raises FileNotFoundError with a clear message if the file is absent.
    """
    if key not in _model_cache:
        path = _MODEL_PATHS[key]
        if not path.is_file():
            raise FileNotFoundError(
                f"Required measurement model file missing: {path}\n"
                f"All nine model files must be present in: {_MODEL_DIR}"
            )
        _model_cache[key] = joblib.load(path)
    return _model_cache[key]


def _validate_angle(name: str, value: Any) -> float:
    """Return the angle as float, or raise ValueError for invalid input.

    Rejects: None, non-numeric strings, NaN, and infinite values.
    """
    if value is None:
        raise ValueError(f"Angle '{name}' is missing (received None).")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Angle '{name}' is not numeric: {value!r}")
    if math.isnan(f):
        raise ValueError(f"Angle '{name}' is NaN.")
    if math.isinf(f):
        raise ValueError(f"Angle '{name}' is infinite.")
    return f


def _rf_classes(model: Any) -> List[str]:
    """Return class labels from a Random Forest model or sklearn Pipeline.

    Tries model.classes_ first; falls back to checking each pipeline step
    in reverse (so the final estimator is checked before preprocessors).

    Raises AttributeError if no classes_ attribute is found anywhere.
    """
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return list(step.classes_)
    raise AttributeError(
        "Cannot find .classes_ on model or any pipeline step. "
        f"Model type: {type(model).__name__}"
    )


def _predict_rf_diag(
    key: str,
    feature_name: str,
    value: float,
) -> Tuple[str, float]:
    """Run one Random Forest diagnosis model.

    Input DataFrame: single column ``feature_name`` with one row.
    Returns (predicted_label, confidence_0_to_1).
    """
    model = _load(key)
    df = pd.DataFrame({feature_name: [value]})
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    classes = _rf_classes(model)
    idx = list(classes).index(prediction)
    confidence = round(float(probabilities[idx]), 4)
    return str(prediction), confidence


def _predict_rf_treat(
    key: str,
    feature_name: str,
    value: float,
    growth_stage: str,
    measurement_key: str,
) -> Tuple[str, float, Dict[str, Any]]:
    """Run one Random Forest treatment model + build its SHAP explanation.

    Input DataFrame: columns [feature_name, "growth_stage"] with one row.
    Returns (predicted_label, confidence_0_to_1, treatment_explanation).
    The treatment label always comes from the model; SHAP only explains it and
    never blocks the prediction.
    """
    model = _load(key)
    df = pd.DataFrame({feature_name: [value], "growth_stage": [growth_stage]})
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    classes = _rf_classes(model)
    idx = list(classes).index(prediction)
    confidence = round(float(probabilities[idx]), 4)

    explanation = _explain_treatment(
        pipeline=model, input_df=df, predicted_class=prediction,
        predicted_treatment=str(prediction), measurement_key=measurement_key,
        angle=value, growth_stage=growth_stage,
    )
    return str(prediction), confidence, explanation


def _predict_mento_treat(
    value: float,
    growth_stage: str,
) -> Tuple[str, float, Dict[str, Any]]:
    """Run the mentolabial XGBoost treatment model with its LabelEncoder.

    Input DataFrame: columns ["mentolabial", "growth_stage"] with one row.
    The XGBoost model predicts an integer class index; the LabelEncoder
    converts it back to the treatment text string.
    Returns (predicted_label, confidence_0_to_1, treatment_explanation).
    """
    model = _load("mento_treat")
    enc = _load("mento_treat_enc")
    df = pd.DataFrame({"mentolabial": [value], "growth_stage": [growth_stage]})
    encoded_prediction = model.predict(df).astype(int)
    treatment = enc.inverse_transform(encoded_prediction)[0]
    probabilities = model.predict_proba(df)[0]
    predicted_index = int(encoded_prediction[0])
    confidence = round(float(probabilities[predicted_index]), 4)

    # SHAP must use the ENCODED integer class; the display label is decoded.
    explanation = _explain_treatment(
        pipeline=model, input_df=df, predicted_class=predicted_index,
        predicted_treatment=str(treatment), measurement_key="mentolabial",
        angle=value, growth_stage=growth_stage,
    )
    return str(treatment), confidence, explanation


def _explain_treatment(**kwargs) -> Dict[str, Any]:
    """Thin wrapper so SHAP stays optional and never breaks treatment output."""
    try:
        from utils.side_treatment_shap import explain_treatment_prediction
        return explain_treatment_prediction(**kwargs)
    except Exception:
        return {
            "available": False,
            "short_summary": "Explanation unavailable.",
            "summary": (
                "The treatment prediction was generated successfully, but its "
                "model explanation is currently unavailable."
            ),
            "features": [],
            "error_code": "SHAP_EXPLANATION_FAILED",
        }


def _merge_labels(
    pairs: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Group duplicate labels together, accumulating their supporting display names.

    Preserves the insertion order of the first occurrence of each label.

    Args:
        pairs: List of (label, display_name) in measurement order.

    Returns:
        [{"label": str, "supported_by": [str, ...]}, ...]
    """
    seen: Dict[str, List[str]] = {}
    order: List[str] = []
    for label, display_name in pairs:
        if label not in seen:
            seen[label] = []
            order.append(label)
        seen[label].append(display_name)
    return [{"label": lbl, "supported_by": seen[lbl]} for lbl in order]


# ── Public API ────────────────────────────────────────────────────────────────

def predict_side_measurement_analysis(
    nasiolabial: Any,
    profile_convexity: Any,
    total_convexity: Any,
    mentolabial: Any,
    growth_stage: str,
) -> Dict[str, Any]:
    """Run measurement-level ML diagnosis and treatment for four side-profile angles.

    Parameters
    ----------
    nasiolabial, profile_convexity, total_convexity, mentolabial:
        Angle values in degrees.  Must be numeric, non-NaN, non-infinite.
        Pass the values already computed by the existing pipeline — do NOT
        re-detect landmarks or re-compute angles inside this function.
    growth_stage:
        "adult" or "growing".  Stripped and lowercased before use.
        growth_stage is used only by treatment models, not diagnosis models.

    Returns
    -------
    On success::

        {
            "success": True,
            "source": "measurement_ml",
            "growth_stage": str,
            "measurements": {
                "nasiolabial":       {"display_name", "angle", "diagnosis",
                                      "diagnosis_confidence", "treatment",
                                      "treatment_confidence"},
                "profile_convexity": {...},
                "total_convexity":   {...},
                "mentolabial":       {...},
            },
            "diagnosis_summary":  [{"label": str, "supported_by": [str, ...]}, ...],
            "treatment_summary":  [{"label": str, "supported_by": [str, ...]}, ...],
            "requires_doctor_review": True,
            "disclaimer": str,
        }

    On failure::

        {"success": False, "error": str}
    """
    try:
        # ── 1. Validate and normalise growth_stage ────────────────────────────
        growth_stage = str(growth_stage).strip().lower()
        if growth_stage not in ("adult", "growing"):
            raise ValueError(
                f"growth_stage must be 'adult' or 'growing', got {growth_stage!r}."
            )

        # ── 2. Validate all angle inputs ──────────────────────────────────────
        nasio_f   = _validate_angle("nasiolabial",       nasiolabial)
        profile_f = _validate_angle("profile_convexity", profile_convexity)
        total_f   = _validate_angle("total_convexity",   total_convexity)
        mento_f   = _validate_angle("mentolabial",       mentolabial)

        # ── 3. Diagnosis models (no growth_stage) ─────────────────────────────
        nasio_diag,   nasio_diag_conf   = _predict_rf_diag(
            "nasio_diag",   "nasiolabial",       nasio_f)
        profile_diag, profile_diag_conf = _predict_rf_diag(
            "profile_diag", "profile_convexity", profile_f)
        total_diag,   total_diag_conf   = _predict_rf_diag(
            "total_diag",   "total_convexity",   total_f)
        mento_diag,   mento_diag_conf   = _predict_rf_diag(
            "mento_diag",   "mentolabial",       mento_f)

        # ── 4. Treatment models (angle + growth_stage) + SHAP explanation ─────
        nasio_treat,   nasio_treat_conf,   nasio_treat_expl   = _predict_rf_treat(
            "nasio_treat",   "nasiolabial",       nasio_f,   growth_stage, "nasiolabial")
        profile_treat, profile_treat_conf, profile_treat_expl = _predict_rf_treat(
            "profile_treat", "profile_convexity", profile_f, growth_stage, "profile_convexity")
        total_treat,   total_treat_conf,   total_treat_expl   = _predict_rf_treat(
            "total_treat",   "total_convexity",   total_f,   growth_stage, "total_convexity")
        mento_treat,   mento_treat_conf,   mento_treat_expl   = _predict_mento_treat(
            mento_f, growth_stage)

        # ── 5. Per-measurement results ────────────────────────────────────────
        measurements: Dict[str, Any] = {
            "nasiolabial": {
                "display_name":         _DISPLAY_NAMES["nasiolabial"],
                "angle":                nasio_f,
                "diagnosis":            nasio_diag,
                "diagnosis_confidence": nasio_diag_conf,
                "treatment":            nasio_treat,
                "treatment_confidence": nasio_treat_conf,
                "treatment_explanation": nasio_treat_expl,
            },
            "profile_convexity": {
                "display_name":         _DISPLAY_NAMES["profile_convexity"],
                "angle":                profile_f,
                "diagnosis":            profile_diag,
                "diagnosis_confidence": profile_diag_conf,
                "treatment":            profile_treat,
                "treatment_confidence": profile_treat_conf,
                "treatment_explanation": profile_treat_expl,
            },
            "total_convexity": {
                "display_name":         _DISPLAY_NAMES["total_convexity"],
                "angle":                total_f,
                "diagnosis":            total_diag,
                "diagnosis_confidence": total_diag_conf,
                "treatment":            total_treat,
                "treatment_confidence": total_treat_conf,
                "treatment_explanation": total_treat_expl,
            },
            "mentolabial": {
                "display_name":         _DISPLAY_NAMES["mentolabial"],
                "angle":                mento_f,
                "diagnosis":            mento_diag,
                "diagnosis_confidence": mento_diag_conf,
                "treatment":            mento_treat,
                "treatment_confidence": mento_treat_conf,
                "treatment_explanation": mento_treat_expl,
            },
        }

        # ── 6. Merged summaries (duplicate labels grouped) ────────────────────
        diag_pairs: List[Tuple[str, str]] = [
            (nasio_diag,   _DISPLAY_NAMES["nasiolabial"]),
            (profile_diag, _DISPLAY_NAMES["profile_convexity"]),
            (total_diag,   _DISPLAY_NAMES["total_convexity"]),
            (mento_diag,   _DISPLAY_NAMES["mentolabial"]),
        ]
        treat_pairs: List[Tuple[str, str]] = [
            (nasio_treat,   _DISPLAY_NAMES["nasiolabial"]),
            (profile_treat, _DISPLAY_NAMES["profile_convexity"]),
            (total_treat,   _DISPLAY_NAMES["total_convexity"]),
            (mento_treat,   _DISPLAY_NAMES["mentolabial"]),
        ]

        return {
            "success":               True,
            "source":                "measurement_ml",
            "growth_stage":          growth_stage,
            "measurements":          measurements,
            "diagnosis_summary":     _merge_labels(diag_pairs),
            "treatment_summary":     _merge_labels(treat_pairs),
            "requires_doctor_review": True,
            "disclaimer":            _DISCLAIMER,
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(exc)}
