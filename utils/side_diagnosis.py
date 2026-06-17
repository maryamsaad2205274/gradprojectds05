"""
Side-view diagnosis + treatment pipeline.

TWO-STAGE ARCHITECTURE
══════════════════════

Stage 1 — Diagnosis (XGBoost, tree-based)
──────────────────────────────────────────
  Input  (5 raw features, NO scaling — XGBoost is scale-invariant):
    [nasiolabial, profile_convexity, total_convexity, mentolabial, growth_stage]

  Model  : diagnosis_xgboost.pkl
  Encoder: diagnosis_encoder.pkl  (5 classes)
  Output : diagnosis label + per-class probabilities

Stage 2 — Treatment (Random Forest, needs scaling)
──────────────────────────────────────────────────
  Input  (6 features, scaled with scaler.pkl):
    [nasiolabial, profile_convexity, total_convexity, mentolabial,
     growth_stage, final_diagnosis]

  Where `final_diagnosis` is the integer class-ID output from Stage 1.

  Scaler : scaler.pkl  — fitted on exactly these 6 columns in this order
           (feature_names_in_ = ['nasiolabial', 'profile_convexity',
            'total_convexity', 'mentolabial', 'growth_stage', 'final_diagnosis'])
  Model  : treatment_model.pkl  (Random Forest)
  Encoder: treatment_encoder.pkl  (8 classes)
  Output : treatment label + per-class probabilities

WHY SCALER IS NOT APPLIED TO STAGE 1
──────────────────────────────────────
The scaler.pkl has 6 features and includes `final_diagnosis`.  That column
only exists AFTER Stage 1 runs.  Therefore the scaler cannot be applied
before Stage 1.  XGBoost decision trees are intrinsically scale-invariant,
so applying (or omitting) a scaler has no effect on correctness — the model
simply needs the same raw feature values it saw during training.

LANDMARK INDICES (1-based)
──────────────────────────
  nasiolabial      : angle at pt 8  between pts 7  and 10 → ABC(pts[6], pts[7], pts[9])
  profile_convexity: angle at pt 8  between pts 3  and 17 → ABC(pts[2], pts[7], pts[16])
  total_convexity  : angle at pt 5  between pts 3  and 17 → ABC(pts[2], pts[4], pts[16])
  mentolabial      : angle at pt 16 between pts 15 and 17 → ABC(pts[14], pts[15], pts[16])
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from utils.measurements import angle_ABC, normalize_points

# ── Model file paths ──────────────────────────────────────────────────────────
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))

_PATHS = {
    # Stage 1 — Diagnosis
    "diagnosis_model":   os.path.join(_BASE, "diagnosis_xgboost.pkl"),
    "diagnosis_encoder": os.path.join(_BASE, "diagnosis_encoder.pkl"),
    "growth_encoder":    os.path.join(_BASE, "growth_encoder.pkl"),
    # Stage 2 — Treatment
    "scaler":            os.path.join(_BASE, "scaler.pkl"),
    "treatment_model":   os.path.join(_BASE, "treatment_model.pkl"),
    "treatment_encoder": os.path.join(_BASE, "treatment_encoder.pkl"),
}

# ── Singleton cache ───────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}


def _load_required(key: str) -> Any:
    """Load and cache a required .pkl.  Raises FileNotFoundError with a clear message."""
    if key not in _cache:
        path = _PATHS[key]
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required model file missing: {path}\n"
                f"Copy '{os.path.basename(path)}' into the project's model/ directory."
            )
        _cache[key] = joblib.load(path)
    return _cache[key]


def _load_optional(key: str) -> Optional[Any]:
    """Load and cache an optional .pkl.  Returns None if the file is missing."""
    if key not in _cache:
        path = _PATHS[key]
        _cache[key] = joblib.load(path) if os.path.isfile(path) else None
    return _cache[key]


# ── Angle computation ─────────────────────────────────────────────────────────
# Each tuple: (angle_name, (A_1based, B_1based, C_1based))
# B is the vertex; angle is measured at B between rays B→A and B→C.
_ANGLE_SPECS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("nasiolabial",       (7,  8,  10)),
    ("profile_convexity", (3,  8,  17)),
    ("total_convexity",   (3,  5,  17)),
    ("mentolabial",       (15, 16, 17)),
]


def _compute_angles(pts: np.ndarray) -> Dict[str, float]:
    """
    pts: (N, 2) float array of landmark (x, y) in original-image pixels.
    Returns a dict with the 4 angle keys.
    """
    angles: Dict[str, float] = {}
    for name, (a_i, b_i, c_i) in _ANGLE_SPECS:
        A = pts[a_i - 1]
        B = pts[b_i - 1]
        C = pts[c_i - 1]
        angles[name] = angle_ABC(A, B, C)
    return angles


# ── Stage 1 feature vector (5 raw features for XGBoost) ──────────────────────
# Column order must match the XGBoost training data exactly.
_DIAG_FEATURE_ORDER = [
    "nasiolabial",
    "profile_convexity",
    "total_convexity",
    "mentolabial",
    "growth_stage",   # replaced by its integer encoding at runtime
]


def _build_diag_row(angles: Dict[str, float], growth_encoded: int) -> np.ndarray:
    """Return a (1, 5) float64 array for Stage 1 (no scaling)."""
    row = [
        angles["nasiolabial"],
        angles["profile_convexity"],
        angles["total_convexity"],
        angles["mentolabial"],
        float(growth_encoded),
    ]
    return np.array([row], dtype=np.float64)


# ── Stage 2 feature vector (6 features for scaler → RandomForest) ────────────
# MUST match scaler's feature_names_in_ exactly:
#   ['nasiolabial', 'profile_convexity', 'total_convexity', 'mentolabial',
#    'growth_stage', 'final_diagnosis']
_TREAT_FEATURE_NAMES = [
    "nasiolabial",
    "profile_convexity",
    "total_convexity",
    "mentolabial",
    "growth_stage",
    "final_diagnosis",
]

_EXPECTED_SCALER_FEATURES  = 6
_EXPECTED_DIAG_FEATURES    = 5
_EXPECTED_TREAT_FEATURES   = 6


def validate_pipeline_schemas() -> None:
    """
    Load every PKL and assert that feature counts match the expected schema.

    Call this once at app startup (e.g. from app.py after db.create_all()).
    Raises ValueError with a precise message if any mismatch is found,
    so misconfigured deployments fail loudly on boot instead of silently
    returning wrong results.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress sklearn version warnings

        errors: list = []

        # Diagnosis model: must expect 5 features
        try:
            dm = _load_required("diagnosis_model")
            if int(dm.n_features_in_) != _EXPECTED_DIAG_FEATURES:
                errors.append(
                    f"diagnosis_xgboost.pkl expects {dm.n_features_in_} features "
                    f"but pipeline sends {_EXPECTED_DIAG_FEATURES}."
                )
        except Exception as e:
            errors.append(f"Could not load diagnosis_model: {e}")

        # Scaler: must expect 6 features matching _TREAT_FEATURE_NAMES
        try:
            sc = _load_required("scaler")
            n = int(sc.n_features_in_)
            if n != _EXPECTED_SCALER_FEATURES:
                errors.append(
                    f"scaler.pkl has {n} features but pipeline sends "
                    f"{_EXPECTED_SCALER_FEATURES}.\n"
                    f"  Expected: {_TREAT_FEATURE_NAMES}\n"
                    f"  Got     : "
                    + (str(list(sc.feature_names_in_)) if hasattr(sc, "feature_names_in_") else "unknown")
                )
            elif hasattr(sc, "feature_names_in_"):
                got_names = list(sc.feature_names_in_)
                if got_names != _TREAT_FEATURE_NAMES:
                    errors.append(
                        f"scaler.pkl column order mismatch.\n"
                        f"  Expected: {_TREAT_FEATURE_NAMES}\n"
                        f"  Got     : {got_names}"
                    )
        except Exception as e:
            errors.append(f"Could not load scaler: {e}")

        # Treatment model: must expect 6 features
        try:
            tm = _load_optional("treatment_model")
            if tm is not None and int(tm.n_features_in_) != _EXPECTED_TREAT_FEATURES:
                errors.append(
                    f"treatment_model.pkl expects {tm.n_features_in_} features "
                    f"but pipeline sends {_EXPECTED_TREAT_FEATURES}."
                )
        except Exception as e:
            errors.append(f"Could not load treatment_model: {e}")

        if errors:
            raise ValueError(
                "DentAlign pipeline schema validation FAILED:\n" +
                "\n".join(f"  ✗ {e}" for e in errors)
            )

    print(
        f"[DentAlign] Pipeline schema validation passed. "
        f"diag={_EXPECTED_DIAG_FEATURES}f raw->XGBoost | "
        f"treat={_EXPECTED_TREAT_FEATURES}f scaled->RandomForest"
    )


def _build_treat_row(
    angles: Dict[str, float],
    growth_encoded: int,
    diag_class_id: int,
) -> np.ndarray:
    """Return a (1, 6) float64 array for Stage 2 (before scaling)."""
    row = [
        angles["nasiolabial"],
        angles["profile_convexity"],
        angles["total_convexity"],
        angles["mentolabial"],
        float(growth_encoded),
        float(diag_class_id),   # final_diagnosis — the integer output of Stage 1
    ]
    return np.array([row], dtype=np.float64)


# ── Confidence tier ───────────────────────────────────────────────────────────
def _confidence_level(pct: float) -> str:
    if pct >= 85:
        return "High"
    if pct >= 65:
        return "Medium"
    return "Low"


# ── Public function ───────────────────────────────────────────────────────────
def run_side_diagnosis(
    landmarks: Any,
    growth_stage: str,
) -> Dict[str, Any]:
    """
    Run the full side-view diagnosis + treatment pipeline.

    Parameters
    ----------
    landmarks:
        Any format accepted by ``normalize_points``:
        list of [x, y] pairs, or list of {"x": …, "y": …} dicts.
        Must contain at least 17 landmarks (indices 0–16 used).
    growth_stage:
        'adult' or 'growing'  (case-insensitive; stripped).

    Returns
    -------
    On success::

        {
            "success":      True,
            "angles":       {"nasiolabial": float, ...},   # degrees
            "growth_stage": str,                            # normalised
            "diagnosis":    {
                "label":            str,
                "confidence":       float,   # 0-100 %
                "confidence_level": str,     # "High" | "Medium" | "Low"
                "breakdown":        [{"label": str, "probability": float}, ...]
            },
            "treatment":    {                               # always present
                "label":            str,
                "confidence":       float,
                "confidence_level": str,
                "breakdown":        [...]
            },
        }

    On failure::

        {"success": False, "error": str}
    """
    try:
        # ════════════════════════════════════════════════════════════════════
        # 0. Validate and encode growth_stage
        # ════════════════════════════════════════════════════════════════════
        growth_stage = growth_stage.strip().lower()
        growth_enc = _load_required("growth_encoder")
        valid_classes = list(growth_enc.classes_)
        if growth_stage not in valid_classes:
            raise ValueError(
                f"growth_stage must be one of {valid_classes!r}, got '{growth_stage}'"
            )
        growth_encoded: int = int(growth_enc.transform([growth_stage])[0])

        # ════════════════════════════════════════════════════════════════════
        # 1. Parse landmarks → (N, 2) float array
        # ════════════════════════════════════════════════════════════════════
        pts = normalize_points(landmarks)
        if len(pts) < 17:
            raise ValueError(
                f"Need at least 17 landmarks; got {len(pts)}. "
                "Run side-view analysis first."
            )

        # ════════════════════════════════════════════════════════════════════
        # 2. Compute the 4 clinical angles (degrees)
        # ════════════════════════════════════════════════════════════════════
        angles = _compute_angles(pts)

        # ════════════════════════════════════════════════════════════════════
        # STAGE 1 — Diagnosis (XGBoost, raw features, no scaler)
        # ════════════════════════════════════════════════════════════════════
        X_diag = _build_diag_row(angles, growth_encoded)   # (1, 5)

        diag_model = _load_required("diagnosis_model")
        diag_enc   = _load_required("diagnosis_encoder")

        diag_class_id = int(diag_model.predict(X_diag)[0])
        diag_label    = str(diag_enc.inverse_transform([diag_class_id])[0])
        diag_proba    = diag_model.predict_proba(X_diag)[0]
        diag_conf_pct = round(float(np.max(diag_proba)) * 100, 1)

        diag_all_classes = list(diag_enc.classes_)
        diag_breakdown = sorted(
            [
                {"label": cls, "probability": round(float(diag_proba[i]) * 100, 1)}
                for i, cls in enumerate(diag_all_classes)
            ],
            key=lambda x: x["probability"],
            reverse=True,
        )

        diagnosis: Dict[str, Any] = {
            "label":            diag_label,
            "confidence":       diag_conf_pct,
            "confidence_level": _confidence_level(diag_conf_pct),
            "breakdown":        diag_breakdown,
        }

        # ════════════════════════════════════════════════════════════════════
        # STAGE 2 — Treatment (RandomForest, 6 features → scaler → model)
        # ════════════════════════════════════════════════════════════════════
        # Feature vector order MUST match scaler's feature_names_in_:
        #   ['nasiolabial', 'profile_convexity', 'total_convexity',
        #    'mentolabial', 'growth_stage', 'final_diagnosis']
        treatment: Optional[Dict[str, Any]] = None

        treat_model = _load_optional("treatment_model")
        treat_enc   = _load_optional("treatment_encoder")

        if treat_model is not None and treat_enc is not None:
            X_treat_raw = _build_treat_row(angles, growth_encoded, diag_class_id)  # (1, 6)

            scaler = _load_required("scaler")

            # ── Schema guard ──────────────────────────────────────────────
            # Catch scaler/feature mismatches immediately with a precise
            # message instead of the opaque sklearn error.
            n_expected = int(scaler.n_features_in_)
            n_got      = X_treat_raw.shape[1]
            if n_got != n_expected:
                expected_cols = (
                    list(scaler.feature_names_in_)
                    if hasattr(scaler, "feature_names_in_")
                    else f"{n_expected} features"
                )
                raise ValueError(
                    f"Treatment scaler schema mismatch: "
                    f"built {n_got} features but scaler expects {n_expected}.\n"
                    f"  Scaler expects: {expected_cols}\n"
                    f"  Built vector  : {_TREAT_FEATURE_NAMES}\n"
                    f"Ensure model/scaler.pkl matches the 6-column schema: "
                    f"[nasiolabial, profile_convexity, total_convexity, "
                    f"mentolabial, growth_stage, final_diagnosis]."
                )

            X_treat_scaled = scaler.transform(X_treat_raw)   # (1, 6) → scaled

            treat_class_id = int(treat_model.predict(X_treat_scaled)[0])
            treat_label    = str(treat_enc.inverse_transform([treat_class_id])[0])
            treat_proba    = treat_model.predict_proba(X_treat_scaled)[0]
            treat_conf_pct = round(float(np.max(treat_proba)) * 100, 1)

            treat_all_classes = list(treat_enc.classes_)
            treat_breakdown = sorted(
                [
                    {"label": cls, "probability": round(float(treat_proba[i]) * 100, 1)}
                    for i, cls in enumerate(treat_all_classes)
                ],
                key=lambda x: x["probability"],
                reverse=True,
            )

            treatment = {
                "label":            treat_label,
                "confidence":       treat_conf_pct,
                "confidence_level": _confidence_level(treat_conf_pct),
                "breakdown":        treat_breakdown,
            }

        return {
            "success":      True,
            "angles":       {k: round(v, 2) for k, v in angles.items()},
            "growth_stage": growth_stage,
            "diagnosis":    diagnosis,
            "treatment":    treatment,
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error":   str(exc),
        }
