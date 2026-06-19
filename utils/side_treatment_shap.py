"""
utils/side_treatment_shap.py
============================
SHAP explainability for the FOUR side-profile *treatment* models only.

This module never retrains, modifies, or reloads model files, never changes the
model inputs, and never selects or alters the treatment label.  The treatment
label always comes from the trained model; SHAP only *explains* that already-made
prediction.  SHAP failure is non-fatal — callers always keep the prediction.

Each treatment model is a sklearn Pipeline:
    Pipeline(steps=[("preprocessor", ColumnTransformer), ("model", estimator)])
where the ColumnTransformer emits:
    num__<angle>, cat__growth_stage_adult, cat__growth_stage_growing
and the estimator is a RandomForestClassifier (text classes) or an
XGBClassifier (integer classes decoded via a LabelEncoder).

Public API
----------
    describe_measurement_status(measurement_key, angle) -> dict
    explain_treatment_prediction(pipeline, input_df, predicted_class,
                                 predicted_treatment, measurement_key,
                                 angle, growth_stage) -> dict
    build_treatment_explanation(...) -> (short_summary, summary)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Clinical reference ranges — USED ONLY FOR DESCRIPTION, never for choosing
#    or altering the treatment. ───────────────────────────────────────────────
REFERENCE_RANGES: Dict[str, Tuple[float, float]] = {
    "nasiolabial":       (90.0, 110.0),
    "profile_convexity": (151.0, 171.0),
    "total_convexity":   (127.0, 137.0),
    "mentolabial":       (110.0, 130.0),
}

# Doctor-friendly measurement display names.
MEASUREMENT_DISPLAY: Dict[str, str] = {
    "nasiolabial":       "Nasolabial angle",
    "profile_convexity": "Profile convexity angle",
    "total_convexity":   "Total facial convexity angle",
    "mentolabial":       "Mentolabial angle",
}

GROWTH_FEATURE_NAME = "Growth stage"

# Lazy per-model SHAP TreeExplainer cache (one explainer per treatment model).
_EXPLAINER_CACHE: Dict[str, Any] = {}

# Near-zero SHAP magnitude treated as no meaningful direction.
_EPS = 1e-6


# ── JSON safety ────────────────────────────────────────────────────────────────
def _to_py(value: Any) -> Any:
    """Convert NumPy scalars/arrays to plain JSON-safe Python types."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_to_py(v) for v in value.tolist()]
    return value


# ── Measurement status (description only) ──────────────────────────────────────
def describe_measurement_status(measurement_key: str, angle: Any) -> Dict[str, Any]:
    """Describe an angle as decreased / normal / increased vs its reference range.

    This NEVER selects or changes a treatment — it is descriptive only.
    Returns ``status: "unknown"`` when the range or angle is unusable.
    """
    rng = REFERENCE_RANGES.get(measurement_key)
    try:
        a = float(angle)
        if math.isnan(a) or math.isinf(a):
            raise ValueError("non-finite angle")
    except (TypeError, ValueError):
        a = None

    if rng is None or a is None:
        return {
            "status": "unknown",
            "description": "could not be compared to a reference range",
            "normal_min": rng[0] if rng else None,
            "normal_max": rng[1] if rng else None,
        }

    lo, hi = rng
    if a < lo:
        status, desc = "decreased", "below the normal range"
    elif a > hi:
        status, desc = "increased", "above the normal range"
    else:
        status, desc = "normal", "within the normal range"

    return {
        "status": status,
        "description": desc,
        "normal_min": float(lo),
        "normal_max": float(hi),
    }


# ── SHAP plumbing ──────────────────────────────────────────────────────────────
def _get_explainer(measurement_key: str, estimator: Any):
    """Return a cached shap.TreeExplainer for this treatment model."""
    cached = _EXPLAINER_CACHE.get(measurement_key)
    if cached is not None:
        return cached
    import shap  # imported lazily so SHAP stays optional
    explainer = shap.TreeExplainer(estimator)
    _EXPLAINER_CACHE[measurement_key] = explainer
    return explainer


def _class_index(estimator: Any, predicted_class: Any) -> int:
    """Find the column index of ``predicted_class`` in estimator.classes_.

    Works for text (RF) and integer (XGB) class arrays.  Raises ValueError if
    the class is not present.
    """
    classes = list(estimator.classes_)
    if predicted_class in classes:
        return classes.index(predicted_class)
    # Integer fallback (XGB encoded labels may arrive as numpy ints / strings).
    try:
        pc_int = int(predicted_class)
        int_classes = [int(c) for c in classes]
        if pc_int in int_classes:
            return int_classes.index(pc_int)
    except (TypeError, ValueError):
        pass
    raise ValueError(f"predicted_class {predicted_class!r} not in estimator.classes_")


def _shap_for_class(shap_values: Any, class_idx: int, n_features: int) -> np.ndarray:
    """Extract the 1-D per-feature SHAP vector for one sample + one class.

    Supports both common formats:
      * list of (samples, features) arrays, one per class
      * ndarray of shape (samples, features, classes) or (samples, features)
    Shapes are validated before indexing.
    """
    # Format A: list of per-class arrays.
    if isinstance(shap_values, list):
        if class_idx >= len(shap_values):
            raise ValueError(f"SHAP class index {class_idx} out of range "
                             f"(list len {len(shap_values)})")
        arr = np.asarray(shap_values[class_idx])
        vec = arr[0] if arr.ndim == 2 else arr
        vec = np.asarray(vec).ravel()
        if vec.shape[0] != n_features:
            raise ValueError(f"SHAP vector len {vec.shape[0]} != {n_features}")
        return vec

    arr = np.asarray(shap_values)
    # Format B: (samples, features, classes)
    if arr.ndim == 3:
        if class_idx >= arr.shape[2]:
            raise ValueError(f"SHAP class index {class_idx} out of range "
                             f"(classes {arr.shape[2]})")
        vec = arr[0, :, class_idx]
    # Binary / single-output: (samples, features)
    elif arr.ndim == 2:
        vec = arr[0]
    else:
        raise ValueError(f"Unexpected SHAP ndarray ndim {arr.ndim}")

    vec = np.asarray(vec).ravel()
    if vec.shape[0] != n_features:
        raise ValueError(f"SHAP vector len {vec.shape[0]} != {n_features}")
    return vec


def _group_features(
    feature_names: List[str],
    shap_vec: np.ndarray,
    measurement_key: str,
    angle: float,
    growth_stage: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Collapse raw transformed features into the angle factor + growth-stage factor.

    All one-hot growth-stage columns are summed into a single Growth-stage factor.
    Returns (angle_factor, growth_factor) dicts with raw shap_value.
    """
    angle_shap = 0.0
    growth_shap = 0.0
    for name, val in zip(feature_names, shap_vec):
        v = float(val)
        if name.startswith("cat__growth_stage"):
            growth_shap += v
        else:  # num__<angle> (and any other numeric angle column)
            angle_shap += v

    angle_factor = {
        "feature": MEASUREMENT_DISPLAY.get(measurement_key, measurement_key),
        "value": float(angle),
        "shap_value": round(float(angle_shap), 4),
    }
    growth_factor = {
        "feature": GROWTH_FEATURE_NAME,
        "value": str(growth_stage),
        "shap_value": round(float(growth_shap), 4),
    }
    return angle_factor, growth_factor


def _direction(shap_value: float) -> str:
    if shap_value > _EPS:
        return "supports"
    if shap_value < -_EPS:
        return "opposes"
    return "neutral"


def _assign_strengths(factors: List[Dict[str, Any]]) -> None:
    """Add a relative 'strength' + 'direction' label to each grouped factor.

    Strength is a transparent local ranking of the two grouped factors only —
    NOT a calibrated clinical strength.
    """
    mags = [abs(f["shap_value"]) for f in factors]
    max_mag = max(mags) if mags else 0.0
    for f in factors:
        mag = abs(f["shap_value"])
        f["direction"] = _direction(f["shap_value"])
        if max_mag <= _EPS or mag <= _EPS:
            f["strength"] = "minimal"
            continue
        ratio = mag / max_mag
        if mag == max_mag:
            f["strength"] = "strong"
        elif ratio >= 0.4:
            f["strength"] = "moderate"
        elif ratio >= 0.1:
            f["strength"] = "small"
        else:
            f["strength"] = "minimal"


# ── Sentence generation ────────────────────────────────────────────────────────
def build_treatment_explanation(
    measurement_name: str,
    angle: float,
    measurement_status: Dict[str, Any],
    growth_stage: str,
    predicted_treatment: str,
    grouped_shap_features: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Generate (short_summary, summary) from the ACTUAL SHAP result.

    grouped_shap_features: [angle_factor, growth_factor] with direction/strength.
    The sentence is selected by the real SHAP importance order + directions.
    """
    angle_f = grouped_shap_features[0]
    growth_f = grouped_shap_features[1]

    status = measurement_status.get("status", "unknown")
    range_desc = measurement_status.get("description", "")
    nmin = measurement_status.get("normal_min")
    nmax = measurement_status.get("normal_max")
    rng_txt = (f"{_fmt(nmin)}°–{_fmt(nmax)}°"
               if nmin is not None and nmax is not None else "the reference range")
    a_txt = _fmt(angle)
    m_low = measurement_name[0].lower() + measurement_name[1:]

    # ── Measurement is within the normal range ──────────────────────────────
    if status == "normal":
        summary = (
            f"The {measurement_name} measured {a_txt}°, which is within the "
            f"normal range of {rng_txt}. SHAP indicates how the measurement and "
            f"the patient's {growth_stage} growth stage contributed to "
            f"{predicted_treatment} as the model-predicted treatment consideration."
        )
        short = (
            f"A normal {m_low} and the patient's {growth_stage} growth stage "
            f"contributed to {predicted_treatment} as the model-predicted "
            f"treatment consideration."
        )
        return short, summary

    angle_primary = abs(angle_f["shap_value"]) >= abs(growth_f["shap_value"])
    angle_supports = angle_f["direction"] == "supports"
    growth_supports = growth_f["direction"] == "supports"

    if angle_primary:
        if angle_supports and growth_f["direction"] == "opposes":
            # Angle supports and growth stage opposes
            summary = (
                f"The {measurement_name} measured {a_txt}°, which is {range_desc} "
                f"of {rng_txt} and supported the model prediction of "
                f"{predicted_treatment}. The patient's {growth_stage} growth stage "
                f"reduced support for this prediction, while the angle remained the "
                f"stronger model factor."
            )
            short = (
                f"A {status} {m_low} supported {predicted_treatment}, while the "
                f"{growth_stage} growth stage reduced support for it."
            )
        else:
            # Angle is strongest and supports (default angle-primary phrasing)
            summary = (
                f"The {measurement_name} measured {a_txt}°, which is {range_desc} "
                f"of {rng_txt} and is therefore classified as {status}. SHAP "
                f"indicates that this measurement was the strongest factor "
                f"supporting the model prediction. The patient's {growth_stage} "
                f"growth stage also contributed to {predicted_treatment} as the "
                f"model-predicted treatment consideration."
            )
            short = (
                f"A {status} {m_low} and the patient's {growth_stage} growth stage "
                f"supported {predicted_treatment} as the model-predicted treatment "
                f"consideration."
            )
    else:
        if growth_supports and angle_f["direction"] == "opposes":
            # Growth stage supports and angle opposes
            summary = (
                f"The patient's {growth_stage} growth stage supported the model "
                f"prediction of {predicted_treatment}. The {measurement_name} "
                f"measured {a_txt}°, but its SHAP contribution reduced support for "
                f"this result. Growth stage remained the stronger supporting factor."
            )
            short = (
                f"The patient's {growth_stage} growth stage supported "
                f"{predicted_treatment}, while the {m_low} reduced support for it."
            )
        else:
            # Growth stage is strongest and supports
            summary = (
                f"The patient is in the {growth_stage} growth stage, which was the "
                f"strongest factor supporting the model prediction. The "
                f"{measurement_name} measured {a_txt}°, placing it {range_desc} of "
                f"{rng_txt}. Together, these factors supported {predicted_treatment} "
                f"as the model-predicted treatment consideration."
            )
            short = (
                f"The patient's {growth_stage} growth stage and a {status} {m_low} "
                f"supported {predicted_treatment} as the model-predicted treatment "
                f"consideration."
            )

    return short, summary


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.1f}"


def _fallback(reason: str) -> Dict[str, Any]:
    """Standard non-fatal SHAP failure payload (no internals leaked)."""
    return {
        "available": False,
        "short_summary": "Explanation unavailable.",
        "summary": (
            "The treatment prediction was generated successfully, but its model "
            "explanation is currently unavailable."
        ),
        "features": [],
        "error_code": "SHAP_EXPLANATION_FAILED",
    }


# ── Public entry point ─────────────────────────────────────────────────────────
def explain_treatment_prediction(
    pipeline: Any,
    input_df: Any,
    predicted_class: Any,
    predicted_treatment: str,
    measurement_key: str,
    angle: Any,
    growth_stage: str,
) -> Dict[str, Any]:
    """Explain ONE treatment model's prediction with SHAP.

    Parameters
    ----------
    pipeline:
        The trained treatment Pipeline (preprocessor + model).  Not modified.
    input_df:
        The exact 1-row DataFrame already used for prediction.
    predicted_class:
        The estimator-space class (text label for RF, encoded int for XGB).
    predicted_treatment:
        The doctor-facing treatment label (decoded for XGB).
    measurement_key, angle, growth_stage:
        Descriptive context for the explanation text.

    Returns a JSON-safe ``treatment_explanation`` dict.  On ANY failure returns
    the non-fatal fallback (available=False) — never raises to the caller.
    """
    try:
        status = describe_measurement_status(measurement_key, angle)
        angle_f = float(angle)

        pre = pipeline.named_steps["preprocessor"]
        est = pipeline.named_steps["model"]

        transformed = pre.transform(input_df)
        if hasattr(transformed, "toarray"):           # sparse → dense
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)

        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(transformed.shape[1])]
        n_features = transformed.shape[1]

        class_idx = _class_index(est, predicted_class)

        explainer = _get_explainer(measurement_key, est)
        shap_values = explainer.shap_values(transformed)
        shap_vec = _shap_for_class(shap_values, class_idx, n_features)

        angle_factor, growth_factor = _group_features(
            feat_names, shap_vec, measurement_key, angle_f, growth_stage
        )
        factors = [angle_factor, growth_factor]
        _assign_strengths(factors)

        measurement_name = MEASUREMENT_DISPLAY.get(measurement_key, measurement_key)
        short_summary, summary = build_treatment_explanation(
            measurement_name=measurement_name,
            angle=angle_f,
            measurement_status=status,
            growth_stage=str(growth_stage),
            predicted_treatment=predicted_treatment,
            grouped_shap_features=factors,
        )

        # Primary = larger absolute contribution.
        ordered = sorted(factors, key=lambda f: abs(f["shap_value"]), reverse=True)

        explanation = {
            "available": True,
            "measurement_status": {
                "status": status["status"],
                "description": status["description"],
                "normal_range": {
                    "minimum": status["normal_min"],
                    "maximum": status["normal_max"],
                },
            },
            "predicted_treatment": str(predicted_treatment),
            "main_factor": {
                "feature": ordered[0]["feature"],
                "value": ordered[0]["value"],
                "direction": ordered[0]["direction"],
                "strength": ordered[0]["strength"],
            },
            "secondary_factor": {
                "feature": ordered[1]["feature"],
                "value": ordered[1]["value"],
                "direction": ordered[1]["direction"],
                "strength": ordered[1]["strength"],
            },
            "short_summary": short_summary,
            "summary": summary,
            "features": [
                {
                    "feature": f["feature"],
                    "value": f["value"],
                    "shap_value": f["shap_value"],
                    "direction": f["direction"],
                    "strength": f["strength"],
                }
                for f in factors
            ],
        }
        # Guarantee JSON safety end-to-end.
        return _json_safe(explanation)

    except Exception as exc:  # SHAP must never break the prediction
        logger.exception(
            "SHAP treatment explanation failed for measurement_key=%s", measurement_key
        )
        return _fallback(str(exc))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return _to_py(obj)
