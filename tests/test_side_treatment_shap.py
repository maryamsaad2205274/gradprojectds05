"""
Tests for SHAP treatment explainability (utils/side_treatment_shap.py) and its
integration into utils/side_measurement_models.py.

Covers: the four treatment models, both growth stages, status thresholds,
invalid / missing / NaN / inf angles, SHAP list & ndarray formats, encoded
mentolabial class handling, one-hot growth-stage grouping, JSON serialisation,
explainer caching, SHAP-failure fallback, and that diagnosis / treatment output
is unchanged.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from utils import side_treatment_shap as STS
from utils.side_measurement_models import (
    _predict_rf_treat,
    _predict_mento_treat,
    predict_side_measurement_analysis,
)


# ── describe_measurement_status (description only) ──────────────────────────────
@pytest.mark.parametrize("key,angle,status,nmin,nmax", [
    ("nasiolabial",       85,  "decreased", 90.0, 110.0),
    ("nasiolabial",       100, "normal",    90.0, 110.0),
    ("nasiolabial",       120, "increased", 90.0, 110.0),
    ("profile_convexity", 175, "increased", 151.0, 171.0),
    ("total_convexity",   120, "decreased", 127.0, 137.0),
    ("mentolabial",       140, "increased", 110.0, 130.0),
])
def test_status_thresholds(key, angle, status, nmin, nmax):
    out = STS.describe_measurement_status(key, angle)
    assert out["status"] == status
    assert out["normal_min"] == nmin
    assert out["normal_max"] == nmax


@pytest.mark.parametrize("bad", [None, "abc", float("nan"), float("inf")])
def test_status_invalid_angle(bad):
    out = STS.describe_measurement_status("nasiolabial", bad)
    assert out["status"] == "unknown"


# ── End-to-end SHAP per model (treatment unchanged + explanation produced) ──────
def _treatment_only(model_key, feat, angle, gs):
    """Predict treatment WITHOUT SHAP, to compare label before/after."""
    from utils.side_measurement_models import _load, _rf_classes
    model = _load(model_key)
    df = pd.DataFrame({feat: [angle], "growth_stage": [gs]})
    return str(model.predict(df)[0])


@pytest.mark.parametrize("model_key,feat,key,angle,gs,exp_treat,exp_status,rng", [
    ("nasio_treat",   "nasiolabial",       "nasiolabial",       85,  "growing",
     "Headgear therapy", "decreased", (90.0, 110.0)),
    ("profile_treat", "profile_convexity", "profile_convexity", 175, "adult",
     "Fixed orthodontic treatment with extraction", "increased", (151.0, 171.0)),
    ("total_treat",   "total_convexity",   "total_convexity",   120, "growing",
     "Twin block therapy", "decreased", (127.0, 137.0)),
])
def test_rf_models_explanation(model_key, feat, key, angle, gs, exp_treat, exp_status, rng):
    before = _treatment_only(model_key, feat, angle, gs)
    label, conf, expl = _predict_rf_treat(model_key, feat, angle, gs, key)
    assert label == before == exp_treat            # treatment unchanged
    assert expl["available"] is True
    assert expl["measurement_status"]["status"] == exp_status
    assert expl["measurement_status"]["normal_range"]["minimum"] == rng[0]
    assert expl["measurement_status"]["normal_range"]["maximum"] == rng[1]
    assert exp_treat in expl["summary"]
    assert len(expl["features"]) == 2              # angle + grouped growth stage
    json.dumps(expl)                               # JSON-safe


def test_mento_xgb_encoded_handling():
    label, conf, expl = _predict_mento_treat(140, "growing")
    assert label == "Face mask or chin cup"        # decoded label unchanged
    assert expl["available"] is True
    assert expl["measurement_status"]["status"] == "increased"
    assert expl["measurement_status"]["normal_range"] == {"minimum": 110.0, "maximum": 130.0}
    # grouped to exactly the angle + growth-stage factors
    feats = {f["feature"] for f in expl["features"]}
    assert "Mentolabial angle" in feats
    assert "Growth stage" in feats
    json.dumps(expl)


@pytest.mark.parametrize("gs", ["adult", "growing"])
def test_both_growth_stages(gs):
    _, _, expl = _predict_rf_treat("nasio_treat", "nasiolabial", 95, gs, "nasiolabial")
    assert expl["available"] is True
    assert any(f["feature"] == "Growth stage" and f["value"] == gs
               for f in expl["features"])


# ── SHAP output-format handling ────────────────────────────────────────────────
def test_shap_ndarray_3d_format():
    sv = np.zeros((1, 3, 5))
    sv[0, 0, 2] = 0.5
    vec = STS._shap_for_class(sv, 2, 3)
    assert vec.shape == (3,)
    assert vec[0] == 0.5


def test_shap_list_format():
    sv = [np.zeros((1, 3)) for _ in range(5)]
    sv[2][0, 1] = 0.7
    vec = STS._shap_for_class(sv, 2, 3)
    assert vec.shape == (3,)
    assert vec[1] == 0.7


def test_shap_shape_validation():
    with pytest.raises(ValueError):
        STS._shap_for_class(np.zeros((1, 3, 5)), 9, 3)   # class idx out of range


def test_onehot_growth_grouping():
    names = ["num__nasiolabial", "cat__growth_stage_adult", "cat__growth_stage_growing"]
    vec = np.array([0.4, -0.1, 0.3])   # growth-stage cols summed → 0.2
    angle_f, growth_f = STS._group_features(names, vec, "nasiolabial", 85.0, "growing")
    assert angle_f["shap_value"] == 0.4
    assert growth_f["shap_value"] == pytest.approx(0.2)
    assert growth_f["value"] == "growing"


# ── Explainer caching ──────────────────────────────────────────────────────────
def test_explainer_cache():
    STS._EXPLAINER_CACHE.clear()
    _predict_rf_treat("nasio_treat", "nasiolabial", 85, "growing", "nasiolabial")
    assert "nasiolabial" in STS._EXPLAINER_CACHE
    first = STS._EXPLAINER_CACHE["nasiolabial"]
    _predict_rf_treat("nasio_treat", "nasiolabial", 99, "adult", "nasiolabial")
    assert STS._EXPLAINER_CACHE["nasiolabial"] is first   # reused, not rebuilt


# ── SHAP failure must never break the prediction ───────────────────────────────
def test_shap_failure_fallback():
    class Boom:
        named_steps = {"preprocessor": None, "model": None}
    out = STS.explain_treatment_prediction(
        pipeline=Boom(), input_df=pd.DataFrame({"nasiolabial": [85], "growth_stage": ["adult"]}),
        predicted_class="Headgear therapy", predicted_treatment="Headgear therapy",
        measurement_key="nasiolabial", angle=85, growth_stage="adult",
    )
    assert out["available"] is False
    assert out["error_code"] == "SHAP_EXPLANATION_FAILED"
    assert out["features"] == []
    json.dumps(out)


# ── Full pipeline: diagnosis/treatment unchanged + explanation attached ─────────
def test_full_analysis_attaches_explanation_without_changing_outputs():
    out = predict_side_measurement_analysis(
        nasiolabial=85, profile_convexity=175, total_convexity=120,
        mentolabial=140, growth_stage="growing",
    )
    assert out["success"] is True
    for key in ("nasiolabial", "profile_convexity", "total_convexity", "mentolabial"):
        m = out["measurements"][key]
        # diagnosis fields still present & untouched
        assert "diagnosis" in m and "diagnosis_confidence" in m
        # treatment fields still present & untouched
        assert "treatment" in m and "treatment_confidence" in m
        # new explanation present and JSON-safe
        assert "treatment_explanation" in m
        json.dumps(m["treatment_explanation"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None])
def test_invalid_angles_handled_by_full_pipeline(bad):
    out = predict_side_measurement_analysis(
        nasiolabial=bad, profile_convexity=160, total_convexity=130,
        mentolabial=120, growth_stage="adult",
    )
    # invalid angle is rejected by the existing validation layer (unchanged)
    assert out["success"] is False
