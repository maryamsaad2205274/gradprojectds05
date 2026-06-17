"""
xray_shap.py — Explainable AI (SHAP) service for the X-ray cephalometric pipeline.

Wraps the existing orthodontic_ai_inference.py without duplicating any model-loading,
preprocessing, feature engineering, or prediction logic.

One cached shap.TreeExplainer is kept per XGBoost model (singleton pattern).
Patient-specific SHAP values are computed per-request and never stored globally.

Thread safety: _shap_cache uses module-level dict.  Python's GIL makes dict
key lookups atomic at the bytecode level.  In the unlikely event that two
concurrent requests initialize the same explainer simultaneously the worst
result is harmless duplicate construction.  For production deployments behind
a multi-worker server each worker has its own memory space.

Requires: shap==0.52.0
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Lazy shap import — fail clearly if not installed
# ─────────────────────────────────────────────────────────────────────────────
try:
    import shap as _shap_lib
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    _shap_lib = None  # type: ignore[assignment]


def _require_shap() -> None:
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed.  Run: pip install shap==0.52.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Doctor-explanation constants — thresholds describe model influence strength
# only, NOT medical severity.  Change these values to recalibrate wording.
# ─────────────────────────────────────────────────────────────────────────────
SHAP_STRONG_THRESHOLD   = 0.30   # abs(shap) >= this → "strongly"
SHAP_MODERATE_THRESHOLD = 0.10   # abs(shap) >= this → "moderately"; below → "slightly"

# Path to the pre-computed reference statistics file
_PROJECT_ROOT_SHAP = Path(__file__).resolve().parent.parent
_REFERENCE_STATS_PATH = (
    _PROJECT_ROOT_SHAP / "model" / "xray" / "feature_reference_statistics.json"
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (re-imported from existing module — no duplication)
# ─────────────────────────────────────────────────────────────────────────────
from utils.orthodontic_ai_inference import (
    NUM_FEATURES,
    NUM_LANDMARKS,
    XRAY_MODEL_SPECS,
    _build_feature_matrix,
    _generate_features,
    _load_diagnosis_models,
    _predict_landmarks,
)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton SHAP explainer cache (explainers only — no patient data)
# ─────────────────────────────────────────────────────────────────────────────
_shap_cache: Dict[str, Any] = {}


def _get_explainer(key: str, clf: Any) -> Any:
    """
    Return a cached shap.TreeExplainer for the given XGBoost model.

    Explainers are created once per model per process and reused for
    every subsequent request.  Patient-specific SHAP arrays are computed
    inside explain_one_diagnosis() and never stored in this cache.
    """
    _require_shap()
    cache_key = f"explainer_{key}"
    if cache_key not in _shap_cache:
        _shap_cache[cache_key] = _shap_lib.TreeExplainer(clf)
    return _shap_cache[cache_key]


# ─────────────────────────────────────────────────────────────────────────────
# Feature description — no anatomical names (no verified mapping exists)
# ─────────────────────────────────────────────────────────────────────────────

def describe_feature(name: str) -> str:
    """
    Return a human-readable description for a cephalometric feature name.

    Patterns recognised:
        p{i}_x    → Horizontal position of landmark {i}
        p{i}_y    → Vertical position of landmark {i}
        d_{i}_{j} → Relative distance between landmarks {i} and {j}
        a_{a}_{b}_{c} → Angle between landmarks {a}-{b}-{c}
                        (landmark {b} is the angle vertex)

    No anatomical landmark names are assigned because no verified mapping
    exists in this project.
    """
    # Coordinate features
    m = re.fullmatch(r"p(\d+)_([xy])", name)
    if m:
        idx, axis = m.group(1), m.group(2)
        direction = "Horizontal" if axis == "x" else "Vertical"
        return f"{direction} position of landmark {idx}"

    # Pairwise distance features
    m = re.fullmatch(r"d_(\d+)_(\d+)", name)
    if m:
        return f"Relative distance between landmarks {m.group(1)} and {m.group(2)}"

    # Angle features
    m = re.fullmatch(r"a_(\d+)_(\d+)_(\d+)", name)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        return f"Angle between landmarks {a}–{b}–{c}"

    # Fallback: return the raw name unchanged
    return name


def _angle_vertex(name: str) -> Optional[str]:
    """Return the vertex landmark index for an angle feature, or None."""
    m = re.fullmatch(r"a_(\d+)_(\d+)_(\d+)", name)
    return m.group(2) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Reference statistics — loaded once, cached for process lifetime
# ─────────────────────────────────────────────────────────────────────────────

# Minimum number of cases required before higher/lower/within wording is shown.
# Below this threshold the comparison is suppressed and a clear note is shown.
REFERENCE_MIN_CASES = 20

_ref_stats_cache: Optional[Dict[str, Dict[str, float]]] = None
_ref_meta_cache:  Optional[Dict[str, Any]] = None   # metadata block from JSON
_ref_stats_loaded: bool = False   # True once we have attempted a load


def load_reference_stats() -> Dict[str, Dict[str, float]]:
    """
    Load feature reference statistics (median, Q1, Q3 for all 217 features).

    Returns a dict keyed by feature name.  Returns an empty dict if the
    statistics file does not exist or has fewer than REFERENCE_MIN_CASES cases
    (graceful degradation — SHAP sentences will still show without comparison).

    Results are cached for the lifetime of the process.
    Call load_reference_meta() to retrieve the accompanying metadata block.
    """
    _load_reference_file()
    return _ref_stats_cache or {}


def load_reference_meta() -> Dict[str, Any]:
    """
    Return the metadata block from feature_reference_statistics.json.

    Keys (when available):
        reference_source : "available_system_cases" (or replacement identifier)
        sample_count     : int — number of cases used
        disclaimer       : str — canonical disclaimer text
        generated_at     : ISO-8601 timestamp
        feature_count    : int

    Returns an empty dict when the file has not been loaded or failed.
    """
    _load_reference_file()
    return _ref_meta_cache or {}


def _load_reference_file() -> None:
    """
    Internal one-time loader.  Populates _ref_stats_cache and _ref_meta_cache.
    Enforces REFERENCE_MIN_CASES — if n_cases < threshold, statistics are
    discarded and _ref_stats_cache is left empty so comparison wording is
    suppressed automatically.
    """
    global _ref_stats_cache, _ref_meta_cache, _ref_stats_loaded
    if _ref_stats_loaded:
        return

    _ref_stats_loaded = True
    stats_path = str(_REFERENCE_STATS_PATH)

    if not _REFERENCE_STATS_PATH.is_file():
        warnings.warn(
            f"[xray_shap] Reference statistics file not found: {stats_path}\n"
            "  Run scripts/generate_xray_reference_stats.py to create it.\n"
            "  Reference comparisons will be omitted from SHAP explanations.",
            RuntimeWarning,
            stacklevel=2,
        )
        _ref_stats_cache = {}
        _ref_meta_cache  = {}
        return

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_meta  = data.get("metadata", {})
        n_cases   = int(raw_meta.get("n_cases", raw_meta.get("sample_count", 0)))

        # Normalise metadata to the canonical schema expected by the API/UI.
        # Older files used "n_cases"; new files use "sample_count" and
        # "reference_source".  Either format is accepted here.
        _ref_meta_cache = {
            "reference_source": raw_meta.get(
                "reference_source", "available_system_cases"
            ),
            "sample_count":  n_cases,
            "disclaimer":    raw_meta.get(
                "disclaimer",
                (
                    "Reference comparisons are based on available system cases. "
                    "They are not validated clinical normal ranges and do not "
                    "represent the original model-training dataset."
                ),
            ),
            "generated_at":  raw_meta.get("generated_at", ""),
            "feature_count": int(raw_meta.get("feature_count", 0)),
        }

        if n_cases < REFERENCE_MIN_CASES:
            warnings.warn(
                f"[xray_shap] Reference statistics contain only {n_cases} cases "
                f"(minimum required: {REFERENCE_MIN_CASES}). "
                "Higher/lower/within comparison wording will be suppressed.",
                RuntimeWarning,
                stacklevel=2,
            )
            _ref_stats_cache = {}   # disable comparison wording
        else:
            _ref_stats_cache = data.get("statistics", {})

    except Exception as exc:
        warnings.warn(
            f"[xray_shap] Could not load reference statistics: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        _ref_stats_cache = {}
        _ref_meta_cache  = {}


# ─────────────────────────────────────────────────────────────────────────────
# Doctor-explanation helper functions
# ─────────────────────────────────────────────────────────────────────────────

def format_feature_value(feature_name: str, value: float) -> str:
    """
    Format a feature value for display to a doctor.

    Angles   → 1 decimal place + degree symbol  (e.g. "146.8°")
    Distances → 3 decimal places                (e.g. "0.845")
    Coordinates → 3 decimal places              (e.g. "-0.213")
    Unknown   → 4 decimal places (safe fallback)
    """
    if re.fullmatch(r"a_\d+_\d+_\d+", feature_name):
        return f"{value:.1f}°"          # degree symbol
    if re.fullmatch(r"d_\d+_\d+", feature_name):
        return f"{value:.3f}"
    if re.fullmatch(r"p\d+_[xy]", feature_name):
        return f"{value:.3f}"
    return f"{value:.4f}"


def classify_reference_position(
    value: float,
    q1: float,
    q3: float,
) -> str:
    """
    Classify a patient value relative to the available system-case reference
    range (Q1–Q3 of previously analysed system cases).

    Returns one of: "higher", "within", "lower"

    IMPORTANT: This is NOT a clinical normal range and does NOT represent
    the original XGBoost model-training dataset.  The IQR is derived from
    the system cases stored in app.db.
    """
    if value > q3:
        return "higher"
    if value < q1:
        return "lower"
    return "within"


def get_influence_strength(abs_shap: float) -> str:
    """
    Convert absolute SHAP magnitude to plain-English influence strength.

    Thresholds (SHAP_STRONG_THRESHOLD, SHAP_MODERATE_THRESHOLD) describe
    model influence only, NOT medical severity.
    """
    if abs_shap >= SHAP_STRONG_THRESHOLD:
        return "strongly"
    if abs_shap >= SHAP_MODERATE_THRESHOLD:
        return "moderately"
    return "slightly"


def build_doctor_explanation(
    feat: Dict[str, Any],
    predicted_label: str,
    ref_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Enrich a single feature record with doctor-friendly interpretation fields.

    Adds to the existing record (does not replace any existing keys):
        display_name        : Short label suitable for the UI heading
        formatted_value     : Patient value with appropriate units/decimals
        reference_position  : "higher" | "within" | "lower" | "unavailable"
        reference_sentence  : One sentence about the reference comparison
        influence_strength  : "strongly" | "moderately" | "slightly"
        doctor_sentence     : Full plain-English sentence about prediction effect

    Safe rules enforced:
        - No anatomical names invented
        - No clinical causation claimed ("caused", "proved", "confirmed")
        - No clinical normal ranges stated
        - Reference labelled as "available system-case reference range" only.
          The terms "model-training reference" and "training-data reference"
          are intentionally excluded from all user-facing output.
    """
    name  = feat["feature_name"]
    value = feat["feature_value"]
    sv    = feat["shap_value"]
    abs_sv = feat["absolute_shap_value"]
    direction = feat["direction"]   # "supports" | "opposes" | "neutral"

    # ── Display name ────────────────────────────────────────────────────────
    base_desc = describe_feature(name)
    vertex = _angle_vertex(name)
    if vertex:
        display_name = f"{base_desc} (vertex: landmark {vertex})"
    else:
        display_name = base_desc

    # ── Formatted value ─────────────────────────────────────────────────────
    formatted_value = format_feature_value(name, value)

    # ── Reference comparison ────────────────────────────────────────────────
    # ref_stats is empty either when the file is missing OR when sample_count
    # is below REFERENCE_MIN_CASES — both cases land in the else branch.
    stat = ref_stats.get(name)
    if stat and "q1" in stat and "q3" in stat:
        ref_pos = classify_reference_position(value, stat["q1"], stat["q3"])
        if ref_pos == "higher":
            reference_sentence = (
                "Higher than the available system-case reference range."
            )
        elif ref_pos == "lower":
            reference_sentence = (
                "Lower than the available system-case reference range."
            )
        else:
            reference_sentence = (
                "Within the available system-case reference range."
            )
    else:
        ref_pos = "unavailable"
        reference_sentence = (
            "Insufficient reference cases for comparison."
        )

    # ── Influence strength ──────────────────────────────────────────────────
    strength = get_influence_strength(abs_sv)

    # ── Doctor sentence ─────────────────────────────────────────────────────
    safe_label = str(predicted_label)
    if direction == "supports":
        doctor_sentence = (
            f"This measurement {strength} supported "
            f"the {safe_label} prediction."
        )
    elif direction == "opposes":
        doctor_sentence = (
            f"This measurement {strength} opposed "
            f"the {safe_label} prediction."
        )
    else:
        doctor_sentence = "This measurement had little or no effect on the prediction."

    enriched = dict(feat)   # copy all existing keys
    enriched.update(
        {
            "display_name":        display_name,
            "formatted_value":     formatted_value,
            "reference_position":  ref_pos,
            "reference_sentence":  reference_sentence,
            "influence_strength":  strength,
            "doctor_sentence":     doctor_sentence,
        }
    )
    return enriched


def _enrich_feature_list(
    feats: List[Dict[str, Any]],
    predicted_label: str,
    ref_stats: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Apply build_doctor_explanation() to every record in a feature list."""
    return [build_doctor_explanation(f, predicted_label, ref_stats) for f in feats]


# ─────────────────────────────────────────────────────────────────────────────
# SHAP extraction — handles all three 3-D orientations + list-of-arrays
# ─────────────────────────────────────────────────────────────────────────────

def extract_predicted_class_shap(
    shap_values: Any,
    predicted_class_id: int,
) -> np.ndarray:
    """
    Extract the (217,) SHAP vector for the predicted class from a
    multi-class shap.TreeExplainer.shap_values() output.

    Supported output orientations:
    ─ list of arrays          → [array(n_samples, 217), ...] per class
    ─ 3-D array (1, 217, K)  → first sample, predicted class column
    ─ 3-D array (K, 1, 217)  → predicted class row, first sample
    ─ 3-D array (1, K, 217)  → first sample, predicted class row
    ─ 2-D array (1, 217)     → binary / single-class fallback

    After extraction, the result is asserted to be exactly shape (217,).
    """
    if isinstance(shap_values, list):
        # Most common with SHAP 0.52.0 + XGBoost multiclass:
        # shap_values[class_id] has shape (n_samples, n_features)
        arr = np.asarray(shap_values[predicted_class_id])
        if arr.ndim == 1:
            result = arr                # already (217,)
        elif arr.ndim == 2:
            result = arr[0]            # (1, 217) → (217,)
        else:
            raise ValueError(
                f"Unexpected shape in SHAP list element: {arr.shape}"
            )
    else:
        arr = np.asarray(shap_values)
        ndim = arr.ndim

        if ndim == 1:
            result = arr

        elif ndim == 2:
            # (1, 217) — binary or single-class
            result = arr[0]

        elif ndim == 3:
            shape = arr.shape
            # Identify which dimension equals NUM_FEATURES (217)
            if shape[1] == NUM_FEATURES:
                # (1, 217, K) orientation
                result = arr[0, :, predicted_class_id]
            elif shape[2] == NUM_FEATURES and shape[0] == 1:
                # (1, K, 217) orientation
                result = arr[0, predicted_class_id, :]
            elif shape[2] == NUM_FEATURES:
                # (K, 1, 217) orientation
                result = arr[predicted_class_id, 0, :]
            elif shape[0] == NUM_FEATURES:
                # (217, 1, K) — unusual but handle
                result = arr[:, 0, predicted_class_id]
            else:
                raise ValueError(
                    f"Cannot determine SHAP orientation from shape {shape}. "
                    f"Expected one dimension to equal {NUM_FEATURES} (n_features)."
                )
        else:
            raise ValueError(
                f"Unexpected SHAP values ndim={ndim}, shape={arr.shape}"
            )

    result = np.asarray(result, dtype=np.float64).ravel()
    assert result.shape == (NUM_FEATURES,), (
        f"SHAP extraction produced shape {result.shape}; "
        f"expected ({NUM_FEATURES},)"
    )
    return result


def _get_base_value(explainer: Any, predicted_class_id: int) -> float:
    """
    Return the SHAP base value (expected model output) for the predicted class.
    Handles both scalar (binary) and list/array (multiclass) expected_value.
    """
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev_list = list(ev)
        if predicted_class_id < len(ev_list):
            return float(ev_list[predicted_class_id])
        # Fallback: return the only element or the first one
        return float(ev_list[0])
    return float(ev)


# ─────────────────────────────────────────────────────────────────────────────
# Single-diagnosis SHAP explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_one_diagnosis(
    key: str,
    clf: Any,
    X: "pd.DataFrame",
    feature_cols: List[str],
    predicted_class_id: int,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Compute a local SHAP explanation for one XGBoost diagnosis model.

    Parameters
    ----------
    key              : Model key from XRAY_MODEL_SPECS (used for explainer cache)
    clf              : Fitted XGBoost classifier
    X                : (1, 217) pandas DataFrame in training-column order
    feature_cols     : Ordered list of 217 feature names
    predicted_class_id : Integer class index for the predicted class
    top_n            : Number of top supporting / opposing features to return

    Returns
    -------
    dict with keys:
        raw_shap_shape, top_features, supporting_features, opposing_features,
        full_explanation, base_value, reconstructed_margin, model_raw_margin,
        additivity_difference, additivity_valid
    """
    _require_shap()
    explainer = _get_explainer(key, clf)

    # Compute SHAP values — suppress verbose internal warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values_raw = explainer.shap_values(X)

    # Record raw shape for diagnostics
    if isinstance(shap_values_raw, list):
        raw_shap_shape = str([v.shape for v in shap_values_raw])
    else:
        raw_shap_shape = str(np.asarray(shap_values_raw).shape)

    # Extract (217,) vector for the predicted class
    predicted_class_shap = extract_predicted_class_shap(
        shap_values_raw, predicted_class_id
    )
    # Guard asserted inside extract_predicted_class_shap; double-check here
    assert predicted_class_shap.shape == (NUM_FEATURES,)

    # Base value for the predicted class
    base_value = _get_base_value(explainer, predicted_class_id)

    # ── Additivity validation in raw margin space ────────────────────────────
    # XGBoost multiclass output_margin returns (n_samples, n_classes)
    margin_output = clf.predict(X, output_margin=True)
    raw_margin = float(np.asarray(margin_output)[0, predicted_class_id])

    reconstructed_margin = float(base_value + predicted_class_shap.sum())
    additivity_valid = bool(
        np.isclose(reconstructed_margin, raw_margin, rtol=1e-4, atol=1e-4)
    )
    additivity_difference = float(abs(reconstructed_margin - raw_margin))

    if not additivity_valid:
        warnings.warn(
            f"[xray_shap] SHAP additivity check failed for model '{key}': "
            f"reconstructed={reconstructed_margin:.6f}, "
            f"actual_margin={raw_margin:.6f}, "
            f"diff={additivity_difference:.6f}",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Build per-feature explanation records ────────────────────────────────
    feature_values = X.values[0]  # (217,)
    full_explanation: List[Dict[str, Any]] = []

    for feat_name, feat_val, shap_val in zip(
        feature_cols, feature_values, predicted_class_shap
    ):
        sv = float(shap_val)
        fv = float(feat_val)
        abs_sv = abs(sv)
        if sv > 0:
            direction = "supports"
        elif sv < 0:
            direction = "opposes"
        else:
            direction = "neutral"

        full_explanation.append(
            {
                "feature_name":        feat_name,
                "feature_description": describe_feature(feat_name),
                "feature_value":       fv,
                "shap_value":          sv,
                "absolute_shap_value": abs_sv,
                "direction":           direction,
            }
        )

    # Top features by |SHAP| (regardless of direction)
    top_features = sorted(
        full_explanation, key=lambda x: x["absolute_shap_value"], reverse=True
    )[:top_n]

    # Supporting features: positive SHAP, sorted descending
    supporting_features = sorted(
        [f for f in full_explanation if f["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True,
    )[:top_n]

    # Opposing features: negative SHAP, sorted ascending (most negative first)
    opposing_features = sorted(
        [f for f in full_explanation if f["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )[:top_n]

    return {
        "raw_shap_shape":        raw_shap_shape,
        "top_features":          top_features,
        "supporting_features":   supporting_features,
        "opposing_features":     opposing_features,
        "full_explanation":      full_explanation,
        "base_value":            base_value,
        "reconstructed_margin":  reconstructed_margin,
        "model_raw_margin":      raw_margin,
        "additivity_difference": additivity_difference,
        "additivity_valid":      additivity_valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline with SHAP
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_xray_with_shap(
    image_path: str,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Complete X-ray pipeline with per-diagnosis SHAP explanations.

    Step 1 : HRNet landmark prediction → (19, 2) in original image pixels
    Step 2 : Feature engineering       → 217 features
    Step 3 : 11 XGBoost predictions
    Step 4 : 11 SHAP local explanations (one per model, cached explainers)

    All model assets are loaded via the singleton cache in
    orthodontic_ai_inference.py — no double-loading.

    Parameters
    ----------
    image_path : Absolute path to the cephalometric X-ray image
    top_n      : Number of top supporting / opposing features per diagnosis

    Returns
    -------
    {
        "image_path"       : str,
        "landmarks"        : list of (x, y) int tuples,
        "features"         : dict of {feature_name: float},
        "diagnosis_results": dict keyed by model key,
        "diagnosis_table"  : pandas DataFrame (n_diagnoses × 4),
        "shap_results"     : same as diagnosis_results (alias for clarity),
        "shap_summary"     : pandas DataFrame of top features per diagnosis,
    }

    Each entry in diagnosis_results / shap_results contains:
        diagnosis, status, predicted_class_id, predicted_label,
        predicted_probability, class_probabilities,
        raw_shap_shape, top_features, supporting_features, opposing_features,
        full_explanation, base_value, reconstructed_margin, model_raw_margin,
        additivity_difference, additivity_valid
    """
    import pandas as pd

    _require_shap()

    # ── Step 1: Landmark prediction ──────────────────────────────────────────
    pred_original, _pred_384, _orig_w, _orig_h = _predict_landmarks(image_path)
    if pred_original.shape != (NUM_LANDMARKS, 2):
        raise RuntimeError(
            f"HRNet returned shape {pred_original.shape}, "
            f"expected ({NUM_LANDMARKS}, 2)."
        )

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    feature_dict = _generate_features(pred_original)
    X = _build_feature_matrix(feature_dict)
    if X.shape != (1, NUM_FEATURES):
        raise RuntimeError(
            f"Feature matrix shape {X.shape}, expected (1, {NUM_FEATURES})."
        )

    # ── Steps 3 + 4: Predict + SHAP for each model ──────────────────────────
    xgbs, encs, feature_cols = _load_diagnosis_models()

    # Load reference statistics once for this request (cached after first call)
    ref_stats = load_reference_stats()
    ref_meta  = load_reference_meta()   # metadata for API/UI transparency

    shap_results: Dict[str, Any] = {}
    diagnosis_rows: List[Dict[str, Any]] = []

    for key, spec in XRAY_MODEL_SPECS.items():
        clf = xgbs[key]
        enc = encs[key]

        encoded_pred   = int(clf.predict(X)[0])
        probabilities  = clf.predict_proba(X)[0]
        predicted_label    = str(enc.inverse_transform([encoded_pred])[0])
        model_probability  = float(probabilities[encoded_pred])

        class_probs: Dict[str, float] = {
            str(cls): round(float(p) * 100, 2)
            for cls, p in zip(enc.classes_, probabilities)
        }

        shap_explanation = explain_one_diagnosis(
            key=key,
            clf=clf,
            X=X,
            feature_cols=feature_cols,
            predicted_class_id=encoded_pred,
            top_n=top_n,
        )

        # Enrich top/supporting/opposing feature lists with doctor-friendly fields
        enriched_top = _enrich_feature_list(
            shap_explanation["top_features"], predicted_label, ref_stats
        )
        enriched_supporting = _enrich_feature_list(
            shap_explanation["supporting_features"], predicted_label, ref_stats
        )
        enriched_opposing = _enrich_feature_list(
            shap_explanation["opposing_features"], predicted_label, ref_stats
        )

        shap_results[key] = {
            "diagnosis":             spec["display_name"],
            "status":                spec["status"],
            "predicted_class_id":    encoded_pred,
            "predicted_label":       predicted_label,
            "predicted_probability": model_probability,
            "class_probabilities":   class_probs,
            **shap_explanation,
            # Overwrite the three feature lists with enriched versions
            "top_features":        enriched_top,
            "supporting_features": enriched_supporting,
            "opposing_features":   enriched_opposing,
        }

        diagnosis_rows.append(
            {
                "diagnosis":   spec["display_name"],
                "status":      spec["status"],
                "prediction":  predicted_label,
                "probability": round(model_probability * 100, 1),
            }
        )

    # ── Build convenience DataFrames ─────────────────────────────────────────
    diagnosis_table = pd.DataFrame(diagnosis_rows)

    summary_rows: List[Dict[str, Any]] = []
    for key, res in shap_results.items():
        for feat in res.get("top_features", []):
            summary_rows.append(
                {
                    "diagnosis":           res["diagnosis"],
                    "feature_name":        feat["feature_name"],
                    "feature_description": feat["feature_description"],
                    "feature_value":       feat["feature_value"],
                    "shap_value":          feat["shap_value"],
                    "absolute_shap_value": feat["absolute_shap_value"],
                    "direction":           feat["direction"],
                }
            )
    shap_summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    return {
        "image_path":        image_path,
        "landmarks":         [(int(round(x)), int(round(y))) for x, y in pred_original],
        "features":          feature_dict,
        "diagnosis_results": shap_results,
        "diagnosis_table":   diagnosis_table,
        "shap_results":      shap_results,
        "shap_summary":      shap_summary,
        # Reference metadata — forwarded to the API response for UI transparency
        "reference_meta":    ref_meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper (used by Flask route)
# ─────────────────────────────────────────────────────────────────────────────

def shap_results_to_json_safe(
    shap_results: Dict[str, Any],
    reference_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a shap_results dict (from diagnose_xray_with_shap) into a
    structure that is safe to pass to flask.jsonify().

    Converts numpy scalars → Python float/int/bool.
    Excludes full_explanation (217 entries × 11 models = 2387 rows) to
    keep the response lightweight; the top-N lists are included.

    Parameters
    ----------
    shap_results   : dict returned by diagnose_xray_with_shap()["shap_results"]
    reference_meta : optional metadata dict from diagnose_xray_with_shap()
                     ["reference_meta"].  When supplied, the keys
                     ``reference_source`` and ``sample_count`` are added to
                     every per-model output record so the UI can display the
                     provenance of comparison ranges without an extra request.
    """

    def _safe(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _safe(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_safe(vv) for vv in v]
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    # Normalise reference_meta for embedding in the response.
    # The UI needs exactly two fields per the spec; everything else is optional.
    ref_source = "available_system_cases"
    ref_count  = 0
    if reference_meta:
        ref_source = str(reference_meta.get("reference_source", ref_source))
        ref_count  = int(reference_meta.get("sample_count", 0))

    # Doctor-friendly fields included per feature record.
    # These keys are added by build_doctor_explanation() and forwarded here:
    #   display_name, formatted_value, reference_position,
    #   reference_sentence, influence_strength, doctor_sentence
    # _safe() handles them automatically (all are str/float/Python types).

    out: Dict[str, Any] = {}
    for key, res in shap_results.items():
        out[key] = {
            "diagnosis":             str(res["diagnosis"]),
            "status":                str(res["status"]),
            "predicted_label":       str(res["predicted_label"]),
            "predicted_probability": float(res["predicted_probability"]),
            "class_probabilities":   _safe(res["class_probabilities"]),
            "top_features":          _safe(res["top_features"]),
            "supporting_features":   _safe(res["supporting_features"]),
            "opposing_features":     _safe(res["opposing_features"]),
            "additivity_valid":      bool(res["additivity_valid"]),
            "additivity_difference": float(res["additivity_difference"]),
            # Provenance fields — UI uses these to label the reference range
            "reference_source":      ref_source,
            "sample_count":          ref_count,
        }
    return out
