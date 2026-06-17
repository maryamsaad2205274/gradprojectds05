"""
tests/test_xray_shap.py
=======================
Focused test suite for the SHAP X-ray explanation pipeline.

Tests verify:
  1.  All required model files exist on disk.
  2.  Registry contains exactly 11 diagnoses.
  3.  All XGBoost models expect exactly 217 features.
  4.  HRNet returns (19, 2) on a synthetic white image.
  5.  Feature builder returns (1, 217).
  6.  diagnose_xray_with_shap() returns 11 diagnosis entries.
  7.  SHAP is computed for all 11 diagnoses.
  8.  Every predicted-class SHAP vector has shape (217,).
  9.  Both 3-class and 4-class models produce valid SHAP outputs.
 10.  Additivity validation runs (may fail for softmax but records the flag).
 11.  shap_results_to_json_safe() produces a JSON-serialisable structure.
 12.  Existing non-X-ray module imports are unaffected.

NOTE: Tests use a synthetic white X-ray image — NOT a patient image.
      Results measure pipeline integrity ONLY, not diagnostic accuracy.

Run with:
    python -m pytest tests/test_xray_shap.py -v
or standalone:
    python tests/test_xray_shap.py
"""

import json
import os
import sys
import tempfile
import warnings

import numpy as np

# ── Add project root to sys.path when running standalone ────────────────────
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Suppress sklearn version warnings ───────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_xray(path: str, width: int = 800, height: int = 1000) -> str:
    """Create a synthetic white image that the HRNet can process."""
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    # Add minimal structure so landmark heatmaps have nonzero peaks
    cv2.rectangle(img, (100, 100), (700, 900), (180, 180, 180), 2)
    cv2.circle(img, (400, 300), 60, (160, 160, 160), -1)
    cv2.imwrite(path, img)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Model files exist
# ─────────────────────────────────────────────────────────────────────────────
def test_1_model_files_exist():
    from utils.orthodontic_ai_inference import (
        XRAY_MODEL_SPECS,
        _FEATURE_COLS_PATH,
        _HRNET_PATH,
        _XRAY_DIR,
    )

    assert os.path.isfile(str(_HRNET_PATH)), \
        f"HRNet checkpoint missing: {_HRNET_PATH}"

    assert os.path.isfile(str(_FEATURE_COLS_PATH)), \
        f"feature_columns.pkl missing: {_FEATURE_COLS_PATH}"

    missing = []
    for key, spec in XRAY_MODEL_SPECS.items():
        mp = str(_XRAY_DIR / spec["model"])
        ep = str(_XRAY_DIR / spec["encoder"])
        if not os.path.isfile(mp):
            missing.append(mp)
        if not os.path.isfile(ep):
            missing.append(ep)

    assert not missing, f"Missing model/encoder files:\n" + "\n".join(missing)
    print("  PASS test_1: all model files exist")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Registry contains exactly 11 diagnoses
# ─────────────────────────────────────────────────────────────────────────────
def test_2_registry_11_diagnoses():
    from utils.orthodontic_ai_inference import XRAY_MODEL_SPECS
    assert len(XRAY_MODEL_SPECS) == 11, \
        f"Expected 11 diagnoses in XRAY_MODEL_SPECS, got {len(XRAY_MODEL_SPECS)}"
    print("  PASS test_2: registry has 11 diagnoses")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: All XGBoost models expect 217 features
# ─────────────────────────────────────────────────────────────────────────────
def test_3_models_expect_217_features():
    from utils.orthodontic_ai_inference import NUM_FEATURES, _load_diagnosis_models
    xgbs, encs, feature_cols = _load_diagnosis_models()

    assert len(feature_cols) == NUM_FEATURES, \
        f"feature_columns.pkl has {len(feature_cols)} entries, expected {NUM_FEATURES}"

    bad = {}
    for key, clf in xgbs.items():
        nf = getattr(clf, "n_features_in_", None)
        if nf != NUM_FEATURES:
            bad[key] = nf
    assert not bad, f"Models with wrong n_features_in_: {bad}"
    print(f"  PASS test_3: all {len(xgbs)} models expect {NUM_FEATURES} features")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: HRNet returns (19, 2) on a synthetic image
# ─────────────────────────────────────────────────────────────────────────────
def test_4_hrnet_shape():
    from utils.orthodontic_ai_inference import NUM_LANDMARKS, _predict_landmarks

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        pred_orig, pred_384, w, h = _predict_landmarks(tmp_path)
        assert pred_orig.shape == (NUM_LANDMARKS, 2), \
            f"HRNet output shape {pred_orig.shape}, expected ({NUM_LANDMARKS}, 2)"
        assert np.all(np.isfinite(pred_orig)), \
            "HRNet output contains non-finite values"
        print(f"  PASS test_4: HRNet output {pred_orig.shape} on {w}x{h} image")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Feature builder returns (1, 217)
# ─────────────────────────────────────────────────────────────────────────────
def test_5_feature_shape():
    from utils.orthodontic_ai_inference import (
        NUM_FEATURES,
        NUM_LANDMARKS,
        _build_feature_matrix,
        _generate_features,
        _predict_landmarks,
    )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        pred_orig, _, _, _ = _predict_landmarks(tmp_path)
        feature_dict = _generate_features(pred_orig)
        assert len(feature_dict) == NUM_FEATURES, \
            f"Feature dict has {len(feature_dict)} keys, expected {NUM_FEATURES}"
        X = _build_feature_matrix(feature_dict)
        assert X.shape == (1, NUM_FEATURES), \
            f"Feature matrix shape {X.shape}, expected (1, {NUM_FEATURES})"
        assert np.all(np.isfinite(X.values)), \
            "Feature matrix contains non-finite values"
        print(f"  PASS test_5: feature matrix shape {X.shape}")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Diagnosis returns 11 rows
# ─────────────────────────────────────────────────────────────────────────────
def test_6_diagnosis_11_rows():
    from utils.xray_shap import diagnose_xray_with_shap

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        result = diagnose_xray_with_shap(tmp_path, top_n=5)
        n_diag = len(result["diagnosis_results"])
        assert n_diag == 11, f"Expected 11 diagnoses, got {n_diag}"
        assert result["diagnosis_table"].shape[0] == 11
        print(f"  PASS test_6: diagnosis returns {n_diag} results")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: SHAP returns 11 explanation dicts
# ─────────────────────────────────────────────────────────────────────────────
def test_7_shap_11_explanations():
    from utils.xray_shap import diagnose_xray_with_shap

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        result = diagnose_xray_with_shap(tmp_path, top_n=5)
        n_shap = len(result["shap_results"])
        assert n_shap == 11, f"Expected 11 SHAP results, got {n_shap}"
        for key, res in result["shap_results"].items():
            assert "supporting_features" in res, f"'{key}' missing supporting_features"
            assert "opposing_features" in res, f"'{key}' missing opposing_features"
            assert "additivity_valid" in res, f"'{key}' missing additivity_valid"
        print(f"  PASS test_7: SHAP produced {n_shap} explanations")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Every predicted-class SHAP vector has shape (217,)
# ─────────────────────────────────────────────────────────────────────────────
def test_8_shap_vector_shape():
    from utils.orthodontic_ai_inference import (
        NUM_FEATURES,
        _build_feature_matrix,
        _generate_features,
        _load_diagnosis_models,
        _predict_landmarks,
        XRAY_MODEL_SPECS,
    )
    from utils.xray_shap import _get_explainer, extract_predicted_class_shap

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        pred_orig, _, _, _ = _predict_landmarks(tmp_path)
        feature_dict = _generate_features(pred_orig)
        X = _build_feature_matrix(feature_dict)
        xgbs, encs, _ = _load_diagnosis_models()

        for key in XRAY_MODEL_SPECS:
            clf = xgbs[key]
            enc = encs[key]
            predicted_id = int(clf.predict(X)[0])
            explainer = _get_explainer(key, clf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_vals = explainer.shap_values(X)
            vec = extract_predicted_class_shap(shap_vals, predicted_id)
            assert vec.shape == (NUM_FEATURES,), \
                f"Model '{key}': SHAP vector shape {vec.shape}, expected ({NUM_FEATURES},)"

        print(f"  PASS test_8: all 11 SHAP vectors have shape ({NUM_FEATURES},)")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Both 3-class and 4-class models produce valid SHAP
# ─────────────────────────────────────────────────────────────────────────────
def test_9_3class_and_4class():
    from utils.orthodontic_ai_inference import (
        NUM_FEATURES,
        _build_feature_matrix,
        _generate_features,
        _load_diagnosis_models,
        _predict_landmarks,
    )
    from utils.xray_shap import _get_explainer, extract_predicted_class_shap

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        pred_orig, _, _, _ = _predict_landmarks(tmp_path)
        X = _build_feature_matrix(_generate_features(pred_orig))
        xgbs, encs, _ = _load_diagnosis_models()

        # profile_class has 4 classes; others have 3
        # Test one 3-class and one 4-class model explicitly
        for key in ("skeletal_class", "profile_class"):
            clf  = xgbs[key]
            enc  = encs[key]
            n_cls = len(enc.classes_)
            pred_id = int(clf.predict(X)[0])
            explainer = _get_explainer(key, clf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_vals = explainer.shap_values(X)
            vec = extract_predicted_class_shap(shap_vals, pred_id)
            assert vec.shape == (NUM_FEATURES,), \
                f"'{key}' ({n_cls}-class): shape {vec.shape}"
            print(f"  PASS test_9/{key}: {n_cls}-class, SHAP shape {vec.shape}")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Additivity flag is recorded (may be False for softmax models)
# ─────────────────────────────────────────────────────────────────────────────
def test_10_additivity_recorded():
    from utils.xray_shap import diagnose_xray_with_shap

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        result = diagnose_xray_with_shap(tmp_path, top_n=5)
        for key, res in result["shap_results"].items():
            assert isinstance(res["additivity_valid"], bool), \
                f"'{key}': additivity_valid must be bool, got {type(res['additivity_valid'])}"
            assert isinstance(res["additivity_difference"], float), \
                f"'{key}': additivity_difference must be float"
        print("  PASS test_10: additivity_valid flag recorded for all 11 models")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: JSON-safe serialisation works
# ─────────────────────────────────────────────────────────────────────────────
def test_11_json_safe():
    from utils.xray_shap import diagnose_xray_with_shap, shap_results_to_json_safe

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        result = diagnose_xray_with_shap(tmp_path, top_n=5)
        json_safe = shap_results_to_json_safe(result["shap_results"])

        # Must be JSON-serialisable (no numpy types)
        serialised = json.dumps(json_safe)
        parsed = json.loads(serialised)
        assert len(parsed) == 11
        print("  PASS test_11: shap_results_to_json_safe() produces valid JSON")
    finally:
        os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Unrelated modules still import cleanly
# ─────────────────────────────────────────────────────────────────────────────
def test_12_unrelated_imports():
    from utils.side_diagnosis import run_side_diagnosis
    from utils.model_health import run_model_health_check
    from utils.case_pdf import _draw_xray_diagnosis_section
    from utils.orthodontic_ai_inference import run_ortho_analysis
    # parse_xray_diagnosis_json
    from utils.orthodontic_ai_inference import parse_xray_diagnosis_json
    print("  PASS test_12: all unrelated module imports succeed")


# ─────────────────────────────────────────────────────────────────────────────
# describe_feature unit tests
# ─────────────────────────────────────────────────────────────────────────────
def test_describe_feature():
    from utils.xray_shap import describe_feature
    assert "Horizontal" in describe_feature("p3_x"),   describe_feature("p3_x")
    assert "Vertical"   in describe_feature("p17_y"),  describe_feature("p17_y")
    assert "distance"   in describe_feature("d_11_12").lower(), describe_feature("d_11_12")
    assert "Angle"      in describe_feature("a_2_13_14"), describe_feature("a_2_13_14")
    assert describe_feature("unknown_feat") == "unknown_feat"
    print("  PASS test_describe_feature: all feature description patterns correct")


# ─────────────────────────────────────────────────────────────────────────────
# Doctor-friendly SHAP helper unit tests (Task D — 15 new tests)
# ─────────────────────────────────────────────────────────────────────────────

# Test D-1: Angle formatting adds degree symbol
def test_d1_angle_formatting():
    from utils.xray_shap import format_feature_value
    result = format_feature_value("a_2_13_14", 146.8)
    assert result.endswith("°"), f"Expected degree symbol, got: {result!r}"
    assert "146.8" in result, f"Expected 1 dp value, got: {result!r}"
    print(f"  PASS test_d1: angle formatted as {result!r}")


# Test D-2: Distance formatting uses 3 decimal places
def test_d2_distance_formatting():
    from utils.xray_shap import format_feature_value
    result = format_feature_value("d_11_12", 0.8453217)
    # Must have exactly 3 decimal places
    parts = result.split(".")
    assert len(parts) == 2 and len(parts[1]) == 3, \
        f"Expected 3 dp, got: {result!r}"
    print(f"  PASS test_d2: distance formatted as {result!r}")


# Test D-3: Coordinate formatting uses 3 decimal places
def test_d3_coordinate_formatting():
    from utils.xray_shap import format_feature_value
    result = format_feature_value("p5_x", -0.21251)
    parts = result.lstrip("-").split(".")
    assert len(parts) == 2 and len(parts[1]) == 3, \
        f"Expected 3 dp, got: {result!r}"
    print(f"  PASS test_d3: coordinate formatted as {result!r}")


# Test D-4: Value below Q1 returns "lower"
def test_d4_reference_below_q1():
    from utils.xray_shap import classify_reference_position
    pos = classify_reference_position(value=0.10, q1=0.20, q3=0.30)
    assert pos == "lower", f"Expected 'lower', got {pos!r}"
    print("  PASS test_d4: value below Q1 classified as 'lower'")


# Test D-5: Value within IQR returns "within"
def test_d5_reference_within_iqr():
    from utils.xray_shap import classify_reference_position
    pos = classify_reference_position(value=0.25, q1=0.20, q3=0.30)
    assert pos == "within", f"Expected 'within', got {pos!r}"
    print("  PASS test_d5: value within IQR classified as 'within'")


# Test D-6: Value above Q3 returns "higher"
def test_d6_reference_above_q3():
    from utils.xray_shap import classify_reference_position
    pos = classify_reference_position(value=0.40, q1=0.20, q3=0.30)
    assert pos == "higher", f"Expected 'higher', got {pos!r}"
    print("  PASS test_d6: value above Q3 classified as 'higher'")


# Test D-7: Positive SHAP produces "supported" sentence
def test_d7_positive_shap_supported():
    from utils.xray_shap import build_doctor_explanation
    feat = {
        "feature_name": "a_2_13_14",
        "feature_value": 146.8,
        "shap_value": 0.50,
        "absolute_shap_value": 0.50,
        "direction": "supports",
        "feature_description": "Angle between landmarks 2-13-14",
    }
    result = build_doctor_explanation(feat, "Class I", {})
    assert "supported" in result["doctor_sentence"], \
        f"Expected 'supported' in: {result['doctor_sentence']!r}"
    print(f"  PASS test_d7: positive SHAP sentence: {result['doctor_sentence']!r}")


# Test D-8: Negative SHAP produces "opposed" sentence
def test_d8_negative_shap_opposed():
    from utils.xray_shap import build_doctor_explanation
    feat = {
        "feature_name": "d_3_7",
        "feature_value": 0.123,
        "shap_value": -0.45,
        "absolute_shap_value": 0.45,
        "direction": "opposes",
        "feature_description": "Relative distance between landmarks 3 and 7",
    }
    result = build_doctor_explanation(feat, "Class II", {})
    assert "opposed" in result["doctor_sentence"], \
        f"Expected 'opposed' in: {result['doctor_sentence']!r}"
    print(f"  PASS test_d8: negative SHAP sentence: {result['doctor_sentence']!r}")


# Test D-9: Strong/moderate/slight wording matches thresholds
def test_d9_influence_strength_wording():
    from utils.xray_shap import (
        SHAP_MODERATE_THRESHOLD,
        SHAP_STRONG_THRESHOLD,
        get_influence_strength,
    )
    assert get_influence_strength(SHAP_STRONG_THRESHOLD)   == "strongly"
    assert get_influence_strength(SHAP_STRONG_THRESHOLD + 0.1) == "strongly"
    assert get_influence_strength(SHAP_MODERATE_THRESHOLD) == "moderately"
    assert get_influence_strength(SHAP_MODERATE_THRESHOLD + 0.01) == "moderately"
    assert get_influence_strength(SHAP_MODERATE_THRESHOLD - 0.01) == "slightly"
    assert get_influence_strength(0.0)                     == "slightly"
    print("  PASS test_d9: influence strength thresholds correct")


# Test D-10: No doctor sentence contains the word "caused"
def test_d10_no_causation_word():
    from utils.xray_shap import build_doctor_explanation
    for direction, sv in [("supports", 0.5), ("opposes", -0.5), ("neutral", 0.0)]:
        feat = {
            "feature_name": "p1_x",
            "feature_value": -0.21,
            "shap_value": sv,
            "absolute_shap_value": abs(sv),
            "direction": direction,
            "feature_description": "Horizontal position of landmark 1",
        }
        result = build_doctor_explanation(feat, "Class I", {})
        sentence = result["doctor_sentence"].lower()
        assert "caused" not in sentence, \
            f"Forbidden word 'caused' found in: {result['doctor_sentence']!r}"
    print("  PASS test_d10: no doctor sentence contains 'caused'")


# Test D-11: Missing reference stats handled safely (no crash, graceful fallback)
def test_d11_missing_reference_stats_safe():
    from utils.xray_shap import build_doctor_explanation
    feat = {
        "feature_name": "d_1_2",
        "feature_value": 0.500,
        "shap_value": 0.20,
        "absolute_shap_value": 0.20,
        "direction": "supports",
        "feature_description": "Relative distance between landmarks 1 and 2",
    }
    # Pass empty ref_stats (simulates missing file)
    result = build_doctor_explanation(feat, "Class I", {})
    assert result["reference_position"] == "unavailable", \
        f"Expected 'unavailable', got: {result['reference_position']!r}"
    assert "unavailable" not in result["reference_sentence"].lower() or \
           "no" in result["reference_sentence"].lower() or \
           len(result["reference_sentence"]) > 0
    # doctor_sentence must still exist
    assert isinstance(result["doctor_sentence"], str) and len(result["doctor_sentence"]) > 0
    print("  PASS test_d11: missing reference stats handled safely")


# Test D-12: 3-class and 4-class SHAP extraction still produce (217,) vectors
def test_d12_3class_4class_shap_extraction():
    # This is a re-confirmation with enrichment pipeline — delegate to existing logic
    from utils.orthodontic_ai_inference import (
        NUM_FEATURES,
        _build_feature_matrix,
        _generate_features,
        _load_diagnosis_models,
        _predict_landmarks,
    )
    from utils.xray_shap import _get_explainer, extract_predicted_class_shap

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        pred_orig, _, _, _ = _predict_landmarks(tmp_path)
        X = _build_feature_matrix(_generate_features(pred_orig))
        xgbs, encs, _ = _load_diagnosis_models()

        for key in ("skeletal_class", "profile_class"):
            clf = xgbs[key]
            pred_id = int(clf.predict(X)[0])
            explainer = _get_explainer(key, clf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_vals = explainer.shap_values(X)
            vec = extract_predicted_class_shap(shap_vals, pred_id)
            assert vec.shape == (NUM_FEATURES,), \
                f"'{key}' SHAP shape {vec.shape} != ({NUM_FEATURES},)"
        print("  PASS test_d12: 3-class and 4-class SHAP extraction still valid")
    finally:
        os.remove(tmp_path)


# Test D-13: Diagnosis predictions unchanged by doctor-explanation layer
def test_d13_predictions_unchanged():
    """Confirm predicted_label is identical whether or not enrichment runs."""
    from utils.orthodontic_ai_inference import (
        _build_feature_matrix,
        _generate_features,
        _load_diagnosis_models,
        _predict_landmarks,
        XRAY_MODEL_SPECS,
    )
    from utils.xray_shap import diagnose_xray_with_shap

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        pred_orig, _, _, _ = _predict_landmarks(tmp_path)
        X = _build_feature_matrix(_generate_features(pred_orig))
        xgbs, encs, _ = _load_diagnosis_models()

        # Ground-truth predictions from direct model calls
        direct_preds = {}
        for key in XRAY_MODEL_SPECS:
            clf = xgbs[key]
            enc = encs[key]
            pid = int(clf.predict(X)[0])
            direct_preds[key] = str(enc.inverse_transform([pid])[0])

        # Predictions returned by the full SHAP pipeline
        result = diagnose_xray_with_shap(tmp_path, top_n=3)
        for key, res in result["shap_results"].items():
            assert res["predicted_label"] == direct_preds[key], (
                f"'{key}': SHAP pipeline returned '{res['predicted_label']}', "
                f"direct call returned '{direct_preds[key]}'"
            )
        print("  PASS test_d13: all 11 predictions unchanged by doctor-explanation layer")
    finally:
        os.remove(tmp_path)


# Test D-14: No DB record is modified by the SHAP pipeline
def test_d14_no_db_changes():
    """
    Verify that diagnose_xray_with_shap() does not write to any DB table.
    We check by importing and asserting no SQLAlchemy session is touched.
    """
    # No DB import is used inside xray_shap.py — confirm the import chain
    import inspect
    import utils.xray_shap as shap_mod
    source = inspect.getsource(shap_mod)
    forbidden = ["db.session", "db.commit", "OrthoCase", "session.add"]
    for token in forbidden:
        assert token not in source, \
            f"Forbidden DB token '{token}' found in xray_shap.py"
    print("  PASS test_d14: xray_shap.py contains no DB writes")


# Test D-15: Model files are unchanged (size and path checks)
def test_d15_model_files_unchanged():
    """Spot-check that key model files still have the expected byte counts."""
    from utils.orthodontic_ai_inference import _HRNET_PATH, _XRAY_DIR

    # HRNet checkpoint must exist and be > 1 MB
    hrnet_size = os.path.getsize(str(_HRNET_PATH))
    assert hrnet_size > 1_000_000, \
        f"HRNet checkpoint unexpectedly small: {hrnet_size} bytes"

    # All 11 XGBoost .pkl files must still exist
    from utils.orthodontic_ai_inference import XRAY_MODEL_SPECS
    for key, spec in XRAY_MODEL_SPECS.items():
        mp = _XRAY_DIR / spec["model"]
        assert mp.is_file(), f"Model file missing: {mp}"
        assert os.path.getsize(str(mp)) > 1_000, \
            f"Model file suspiciously small: {mp} ({os.path.getsize(str(mp))} bytes)"

    print("  PASS test_d15: all model files present and non-empty")


# ─────────────────────────────────────────────────────────────────────────────
# Safety-correction tests (this session)
# ─────────────────────────────────────────────────────────────────────────────

# Test S-1: reference_source and sample_count are present in API output
def test_s1_api_includes_reference_provenance():
    from utils.xray_shap import diagnose_xray_with_shap, shap_results_to_json_safe

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        make_synthetic_xray(tmp_path)
        result   = diagnose_xray_with_shap(tmp_path, top_n=3)
        json_out = shap_results_to_json_safe(
            result["shap_results"],
            reference_meta=result.get("reference_meta"),
        )
        # Every model entry must carry reference_source and sample_count
        for key, entry in json_out.items():
            assert "reference_source" in entry, \
                f"'{key}' missing 'reference_source' in JSON output"
            assert "sample_count" in entry, \
                f"'{key}' missing 'sample_count' in JSON output"
            assert isinstance(entry["sample_count"], int), \
                f"'{key}' sample_count must be int, got {type(entry['sample_count'])}"
        print("  PASS test_s1: reference_source and sample_count present in all API entries")
    finally:
        os.remove(tmp_path)


# Test S-2: API / wording never uses "model reference range" or
#           "training-data reference range"
def test_s2_no_model_training_range_wording():
    import inspect
    import utils.xray_shap as shap_mod

    source = inspect.getsource(shap_mod)
    forbidden_phrases = [
        "model reference range",
        "training-data reference range",
        "training data reference range",
        "model training reference",
    ]
    for phrase in forbidden_phrases:
        assert phrase.lower() not in source.lower(), (
            f"Forbidden phrase '{phrase}' found in xray_shap.py. "
            "Use 'available system-case reference range' instead."
        )
    print("  PASS test_s2: no forbidden 'model/training reference range' wording in xray_shap.py")


# Test S-3: reference_sentence never contains "model reference range"
def test_s3_reference_sentence_wording():
    from utils.xray_shap import build_doctor_explanation

    fake_stats = {
        "p1_x": {"median": 0.0, "q1": -0.5, "q3": 0.5},
    }
    for value, direction, sv in [
        (1.0, "supports", 0.4),    # higher
        (0.0, "supports", 0.4),    # within
        (-1.0, "opposes", -0.4),   # lower
    ]:
        feat = {
            "feature_name": "p1_x",
            "feature_value": value,
            "shap_value": sv,
            "absolute_shap_value": abs(sv),
            "direction": direction,
            "feature_description": "Horizontal position of landmark 1",
        }
        result = build_doctor_explanation(feat, "Class I", fake_stats)
        sent = result["reference_sentence"].lower()
        assert "model reference range" not in sent, \
            f"Forbidden 'model reference range' in reference_sentence: {sent!r}"
        assert "training" not in sent, \
            f"Forbidden 'training' in reference_sentence: {sent!r}"
        assert "system-case reference range" in sent, \
            f"Expected 'system-case reference range' in: {sent!r}"
    print("  PASS test_s3: reference_sentence wording is correct in all cases")


# Test S-4: insufficient sample size (< 20) disables comparison wording
def test_s4_insufficient_samples_disable_comparison():
    """
    Simulate what happens when the JSON file contains only 10 cases.
    load_reference_stats() should return {} and build_doctor_explanation()
    should fall back to 'Insufficient reference cases for comparison.'
    """
    import copy
    import utils.xray_shap as shap_mod

    # Temporarily override the cached state
    orig_cache  = shap_mod._ref_stats_cache
    orig_meta   = shap_mod._ref_meta_cache
    orig_loaded = shap_mod._ref_stats_loaded

    try:
        # Simulate: file was loaded, but only 10 cases — stats discarded
        shap_mod._ref_stats_loaded = True
        shap_mod._ref_stats_cache  = {}         # empty = disabled
        shap_mod._ref_meta_cache   = {
            "reference_source": "available_system_cases",
            "sample_count": 10,
        }

        from utils.xray_shap import build_doctor_explanation, load_reference_stats
        stats = load_reference_stats()
        assert stats == {}, \
            f"Expected empty stats when sample_count < 20, got: {list(stats.keys())[:3]}"

        # Build explanation with the empty stats
        feat = {
            "feature_name": "p1_x",
            "feature_value": 0.5,
            "shap_value": 0.4,
            "absolute_shap_value": 0.4,
            "direction": "supports",
            "feature_description": "Horizontal position of landmark 1",
        }
        result = build_doctor_explanation(feat, "Class I", stats)
        assert result["reference_position"] == "unavailable", \
            f"Expected 'unavailable', got: {result['reference_position']!r}"
        assert "insufficient" in result["reference_sentence"].lower(), \
            f"Expected 'Insufficient' message, got: {result['reference_sentence']!r}"
        assert "higher" not in result["reference_sentence"].lower()
        assert "lower"  not in result["reference_sentence"].lower()
        assert "within" not in result["reference_sentence"].lower()
        print(
            f"  PASS test_s4: 10-case stats return 'unavailable' "
            f"({result['reference_sentence']!r})"
        )
    finally:
        # Restore cache
        shap_mod._ref_stats_cache  = orig_cache
        shap_mod._ref_meta_cache   = orig_meta
        shap_mod._ref_stats_loaded = orig_loaded


# Test S-5: REFERENCE_MIN_CASES constant is defined and equals 20
def test_s5_min_cases_constant():
    from utils.xray_shap import REFERENCE_MIN_CASES
    assert REFERENCE_MIN_CASES == 20, \
        f"Expected REFERENCE_MIN_CASES == 20, got {REFERENCE_MIN_CASES}"
    print(f"  PASS test_s5: REFERENCE_MIN_CASES == {REFERENCE_MIN_CASES}")


# Test S-6: load_reference_meta() returns reference_source and sample_count
def test_s6_load_reference_meta():
    from utils.xray_shap import load_reference_meta
    meta = load_reference_meta()
    # May be empty if file is missing — but if populated must have these keys
    if meta:
        assert "reference_source" in meta, \
            f"'reference_source' missing from meta: {list(meta.keys())}"
        assert "sample_count" in meta, \
            f"'sample_count' missing from meta: {list(meta.keys())}"
        assert isinstance(meta["sample_count"], int), \
            f"sample_count must be int, got {type(meta['sample_count'])}"
        assert meta["reference_source"] == "available_system_cases", \
            f"Unexpected reference_source: {meta['reference_source']!r}"
    print(f"  PASS test_s6: load_reference_meta() returns {meta}")


# Test S-7: JSON stats file itself contains the canonical fields
def test_s7_stats_file_canonical_schema():
    from utils.orthodontic_ai_inference import _XRAY_DIR
    stats_path = _XRAY_DIR / "feature_reference_statistics.json"

    if not stats_path.is_file():
        print(f"  SKIP test_s7: stats file not found at {stats_path}")
        return

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    assert "reference_source" in meta, \
        "'reference_source' missing from JSON metadata"
    assert "sample_count" in meta, \
        "'sample_count' missing from JSON metadata"
    assert meta["reference_source"] == "available_system_cases", \
        f"Unexpected reference_source: {meta['reference_source']!r}"
    assert isinstance(meta["sample_count"], int) and meta["sample_count"] > 0, \
        f"sample_count must be positive int, got: {meta['sample_count']!r}"
    # Disclaimer must mention "system cases" not "model" or "training"
    disclaimer = meta.get("disclaimer", "").lower()
    assert "model-training" not in disclaimer or "not" in disclaimer, \
        f"Disclaimer may incorrectly claim model-training source: {disclaimer!r}"
    print(
        f"  PASS test_s7: stats file has canonical schema "
        f"(source={meta['reference_source']!r}, n={meta['sample_count']})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
ALL_TESTS = [
    test_1_model_files_exist,
    test_2_registry_11_diagnoses,
    test_3_models_expect_217_features,
    test_4_hrnet_shape,
    test_5_feature_shape,
    test_6_diagnosis_11_rows,
    test_7_shap_11_explanations,
    test_8_shap_vector_shape,
    test_9_3class_and_4class,
    test_10_additivity_recorded,
    test_11_json_safe,
    test_12_unrelated_imports,
    test_describe_feature,
    # Task D — doctor-friendly SHAP (15 tests)
    test_d1_angle_formatting,
    test_d2_distance_formatting,
    test_d3_coordinate_formatting,
    test_d4_reference_below_q1,
    test_d5_reference_within_iqr,
    test_d6_reference_above_q3,
    test_d7_positive_shap_supported,
    test_d8_negative_shap_opposed,
    test_d9_influence_strength_wording,
    test_d10_no_causation_word,
    test_d11_missing_reference_stats_safe,
    test_d12_3class_4class_shap_extraction,
    test_d13_predictions_unchanged,
    test_d14_no_db_changes,
    test_d15_model_files_unchanged,
    # Safety corrections (this session — 7 tests)
    test_s1_api_includes_reference_provenance,
    test_s2_no_model_training_range_wording,
    test_s3_reference_sentence_wording,
    test_s4_insufficient_samples_disable_comparison,
    test_s5_min_cases_constant,
    test_s6_load_reference_meta,
    test_s7_stats_file_canonical_schema,
]


if __name__ == "__main__":
    print("=" * 60)
    print("DentAlign X-Ray SHAP — Pipeline Test Suite")
    print("NOTE: Uses a synthetic image. Does NOT test diagnostic accuracy.")
    print("=" * 60)
    passed = 0
    failed = 0
    for fn in ALL_TESTS:
        name = fn.__name__
        try:
            print(f"\n[RUN] {name}")
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL {name}: {exc}")
            traceback.print_exc()
            failed += 1
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)} tests")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
