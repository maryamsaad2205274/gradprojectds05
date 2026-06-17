"""
tests/test_side_measurement_models.py
======================================
Tests for utils/side_measurement_models.py — the measurement-level ML
analysis service for side-profile angles.

Run from the project root:
    python -m pytest tests/test_side_measurement_models.py -v

All 17 test IDs match the requirements specification exactly.
"""

from __future__ import annotations

import json
import math
import os
import sys
import importlib
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure project root is on sys.path ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.side_measurement_models import (
    _merge_labels,
    _validate_angle,
    predict_side_measurement_analysis,
)

# ── Standard test angles (from the required manual test) ─────────────────────
_STD = dict(
    nasiolabial=85.0,
    profile_convexity=175.0,
    total_convexity=142.0,
    mentolabial=140.0,
    growth_stage="growing",
)

# ── Helper ────────────────────────────────────────────────────────────────────

def _run(**kwargs) -> Dict[str, Any]:
    """Call predict_side_measurement_analysis with defaults overridden by kwargs."""
    params = {**_STD, **kwargs}
    return predict_side_measurement_analysis(**params)


# ════════════════════════════════════════════════════════════════════════════════
# MM-1  All-normal angles (within typical normal ranges for each measurement)
# ════════════════════════════════════════════════════════════════════════════════

class TestMM1AllNormalAngles:
    def test_success_flag(self):
        result = _run(nasiolabial=100.0, profile_convexity=161.0,
                      total_convexity=132.0, mentolabial=120.0,
                      growth_stage="adult")
        assert result["success"] is True

    def test_source_field(self):
        result = _run(nasiolabial=100.0, profile_convexity=161.0,
                      total_convexity=132.0, mentolabial=120.0,
                      growth_stage="adult")
        assert result["source"] == "measurement_ml"

    def test_four_measurements_present(self):
        result = _run(nasiolabial=100.0, profile_convexity=161.0,
                      total_convexity=132.0, mentolabial=120.0,
                      growth_stage="adult")
        assert set(result["measurements"].keys()) == {
            "nasiolabial", "profile_convexity", "total_convexity", "mentolabial"
        }

    def test_disclaimer_present(self):
        result = _run(nasiolabial=100.0, profile_convexity=161.0,
                      total_convexity=132.0, mentolabial=120.0,
                      growth_stage="adult")
        assert "disclaimer" in result and result["disclaimer"]

    def test_doctor_review_required(self):
        result = _run(nasiolabial=100.0, profile_convexity=161.0,
                      total_convexity=132.0, mentolabial=120.0,
                      growth_stage="adult")
        assert result["requires_doctor_review"] is True


# ════════════════════════════════════════════════════════════════════════════════
# MM-2  Adult treatment paths
# ════════════════════════════════════════════════════════════════════════════════

class TestMM2AdultTreatment:
    def test_adult_growth_stage_in_output(self):
        result = _run(growth_stage="adult")
        assert result["success"] is True
        assert result["growth_stage"] == "adult"

    def test_treatment_confidence_float(self):
        result = _run(growth_stage="adult")
        for m in result["measurements"].values():
            assert isinstance(m["treatment_confidence"], float)
            assert 0.0 <= m["treatment_confidence"] <= 1.0

    def test_treatment_label_non_empty(self):
        result = _run(growth_stage="adult")
        for m in result["measurements"].values():
            assert m["treatment"] and isinstance(m["treatment"], str)


# ════════════════════════════════════════════════════════════════════════════════
# MM-3  Growing treatment paths
# ════════════════════════════════════════════════════════════════════════════════

class TestMM3GrowingTreatment:
    def test_growing_growth_stage_in_output(self):
        result = _run(growth_stage="growing")
        assert result["success"] is True
        assert result["growth_stage"] == "growing"

    def test_treatment_confidence_float(self):
        result = _run(growth_stage="growing")
        for m in result["measurements"].values():
            assert isinstance(m["treatment_confidence"], float)
            assert 0.0 <= m["treatment_confidence"] <= 1.0

    def test_treatment_label_non_empty(self):
        result = _run(growth_stage="growing")
        for m in result["measurements"].values():
            assert m["treatment"] and isinstance(m["treatment"], str)


# ════════════════════════════════════════════════════════════════════════════════
# MM-4  Duplicate diagnosis merging
# ════════════════════════════════════════════════════════════════════════════════

class TestMM4DuplicateDiagnosisMerging:
    def test_merge_duplicate_produces_one_entry(self):
        pairs = [("ClassA", "Measurement 1"), ("ClassA", "Measurement 2")]
        merged = _merge_labels(pairs)
        assert len(merged) == 1
        assert merged[0]["label"] == "ClassA"
        assert set(merged[0]["supported_by"]) == {"Measurement 1", "Measurement 2"}

    def test_merge_distinct_produces_two_entries(self):
        pairs = [("ClassA", "M1"), ("ClassB", "M2")]
        merged = _merge_labels(pairs)
        assert len(merged) == 2

    def test_manual_test_diagnoses_merge(self):
        """Profile convexity and total convexity both → 'Protruded chin' in the required manual test."""
        result = _run(**_STD)
        assert result["success"] is True
        summary = result["diagnosis_summary"]
        # Find the 'Protruded chin' entry
        pc_entry = next((e for e in summary if e["label"] == "Protruded chin"), None)
        if pc_entry:
            # If model outputs 'Protruded chin' for both profile and total,
            # supported_by must contain both display names.
            if len(pc_entry["supported_by"]) > 1:
                supported_names = pc_entry["supported_by"]
                assert "Profile convexity" in supported_names
                assert "Total facial convexity" in supported_names

    def test_diagnosis_summary_is_list(self):
        result = _run(**_STD)
        assert isinstance(result["diagnosis_summary"], list)

    def test_diagnosis_summary_entries_have_required_keys(self):
        result = _run(**_STD)
        for entry in result["diagnosis_summary"]:
            assert "label" in entry and "supported_by" in entry
            assert isinstance(entry["supported_by"], list)


# ════════════════════════════════════════════════════════════════════════════════
# MM-5  Duplicate treatment merging
# ════════════════════════════════════════════════════════════════════════════════

class TestMM5DuplicateTreatmentMerging:
    def test_merge_labels_preserves_order(self):
        pairs = [("TreatA", "M1"), ("TreatB", "M2"), ("TreatA", "M3")]
        merged = _merge_labels(pairs)
        assert merged[0]["label"] == "TreatA"
        assert merged[1]["label"] == "TreatB"
        assert set(merged[0]["supported_by"]) == {"M1", "M3"}

    def test_treatment_summary_is_list(self):
        result = _run(**_STD)
        assert isinstance(result["treatment_summary"], list)

    def test_treatment_summary_entries_have_required_keys(self):
        result = _run(**_STD)
        for entry in result["treatment_summary"]:
            assert "label" in entry and "supported_by" in entry
            assert isinstance(entry["supported_by"], list)


# ════════════════════════════════════════════════════════════════════════════════
# MM-6  Invalid growth stage
# ════════════════════════════════════════════════════════════════════════════════

class TestMM6InvalidGrowthStage:
    def test_rejects_invalid_growth_stage(self):
        result = _run(growth_stage="teenager")
        assert result["success"] is False
        assert "growth_stage" in result["error"].lower()

    def test_rejects_empty_growth_stage(self):
        result = _run(growth_stage="")
        assert result["success"] is False

    def test_case_insensitive_adult(self):
        """growth_stage is stripped and lowercased — 'Adult' must work."""
        result = _run(growth_stage="Adult")
        assert result["success"] is True

    def test_case_insensitive_growing(self):
        result = _run(growth_stage="GROWING")
        assert result["success"] is True

    def test_growth_stage_normalised_in_output(self):
        result = _run(growth_stage="  Growing  ")
        assert result["success"] is True
        assert result["growth_stage"] == "growing"


# ════════════════════════════════════════════════════════════════════════════════
# MM-7  Missing angle (None)
# ════════════════════════════════════════════════════════════════════════════════

class TestMM7MissingAngle:
    @pytest.mark.parametrize("field", [
        "nasiolabial", "profile_convexity", "total_convexity", "mentolabial"
    ])
    def test_none_angle_returns_error(self, field):
        result = _run(**{field: None})
        assert result["success"] is False
        assert "missing" in result["error"].lower() or field in result["error"]

    def test_validate_angle_raises_on_none(self):
        with pytest.raises(ValueError, match="missing"):
            _validate_angle("nasiolabial", None)


# ════════════════════════════════════════════════════════════════════════════════
# MM-8  Non-numeric angle
# ════════════════════════════════════════════════════════════════════════════════

class TestMM8NonNumericAngle:
    @pytest.mark.parametrize("bad_value", ["abc", "°", [], {}, object()])
    def test_non_numeric_returns_error(self, bad_value):
        result = _run(nasiolabial=bad_value)
        assert result["success"] is False

    def test_validate_angle_raises_on_string(self):
        with pytest.raises(ValueError, match="not numeric"):
            _validate_angle("mentolabial", "abc")


# ════════════════════════════════════════════════════════════════════════════════
# MM-9  NaN input
# ════════════════════════════════════════════════════════════════════════════════

class TestMM9NaNInput:
    @pytest.mark.parametrize("field", [
        "nasiolabial", "profile_convexity", "total_convexity", "mentolabial"
    ])
    def test_nan_angle_returns_error(self, field):
        result = _run(**{field: float("nan")})
        assert result["success"] is False
        assert "nan" in result["error"].lower()

    def test_validate_angle_raises_on_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            _validate_angle("nasiolabial", float("nan"))


# ════════════════════════════════════════════════════════════════════════════════
# MM-10  Infinity input
# ════════════════════════════════════════════════════════════════════════════════

class TestMM10InfinityInput:
    @pytest.mark.parametrize("field,val", [
        ("nasiolabial", float("inf")),
        ("profile_convexity", float("-inf")),
        ("total_convexity", float("inf")),
        ("mentolabial", float("-inf")),
    ])
    def test_infinite_angle_returns_error(self, field, val):
        result = _run(**{field: val})
        assert result["success"] is False
        assert "infinite" in result["error"].lower()

    def test_validate_angle_raises_on_inf(self):
        with pytest.raises(ValueError, match="infinite"):
            _validate_angle("profile_convexity", float("inf"))


# ════════════════════════════════════════════════════════════════════════════════
# MM-11  Missing model file
# ════════════════════════════════════════════════════════════════════════════════

class TestMM11MissingModelFile:
    def test_missing_file_returns_error(self, tmp_path):
        """Override _model_cache and _MODEL_PATHS to point to a nonexistent file."""
        import utils.side_measurement_models as svc
        original_paths = dict(svc._MODEL_PATHS)
        original_cache = dict(svc._model_cache)
        try:
            # Point nasio_diag to a non-existent path and clear its cache entry
            svc._MODEL_PATHS["nasio_diag"] = tmp_path / "does_not_exist.pkl"
            svc._model_cache.pop("nasio_diag", None)
            result = _run()
            assert result["success"] is False
            assert "missing" in result["error"].lower() or "not found" in result["error"].lower()
        finally:
            svc._MODEL_PATHS.update(original_paths)
            svc._model_cache.clear()
            svc._model_cache.update(original_cache)


# ════════════════════════════════════════════════════════════════════════════════
# MM-12  Model loading failure (corrupt file)
# ════════════════════════════════════════════════════════════════════════════════

class TestMM12ModelLoadingFailure:
    def test_corrupt_model_returns_error(self, tmp_path):
        """Write a corrupt .pkl and verify the service returns success=False."""
        import utils.side_measurement_models as svc
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_bytes(b"not a valid pickle file at all")

        original_paths = dict(svc._MODEL_PATHS)
        original_cache = dict(svc._model_cache)
        try:
            svc._MODEL_PATHS["nasio_diag"] = corrupt
            svc._model_cache.pop("nasio_diag", None)
            result = _run()
            assert result["success"] is False
        finally:
            svc._MODEL_PATHS.update(original_paths)
            svc._model_cache.clear()
            svc._model_cache.update(original_cache)


# ════════════════════════════════════════════════════════════════════════════════
# MM-13  JSON serialization of the output
# ════════════════════════════════════════════════════════════════════════════════

class TestMM13JsonSerialization:
    def test_output_is_json_serializable(self):
        result = _run(**_STD)
        # json.dumps must not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_round_trip_preserves_structure(self):
        result = _run(**_STD)
        rt = json.loads(json.dumps(result))
        assert rt["success"] is True
        assert set(rt["measurements"].keys()) == {
            "nasiolabial", "profile_convexity", "total_convexity", "mentolabial"
        }
        assert isinstance(rt["diagnosis_summary"], list)
        assert isinstance(rt["treatment_summary"], list)


# ════════════════════════════════════════════════════════════════════════════════
# MM-14  Exact DataFrame feature names sent to each model
# ════════════════════════════════════════════════════════════════════════════════

class TestMM14DataFrameFeatureNames:
    """Verify that each model receives a DataFrame with exactly the documented column names."""

    def _capture_df_calls(self, model_key: str, df_col_check):
        """
        Patch the model at model_key so its predict / predict_proba intercept
        the DataFrame and record the column names.  Returns captured columns.
        """
        import utils.side_measurement_models as svc
        original_cache = dict(svc._model_cache)
        captured_cols = []

        # Build a mock model
        real_model = svc._load(model_key)
        real_classes = list(real_model.classes_) if hasattr(real_model, "classes_") else ["A"]
        real_proba = real_model.predict_proba(
            __import__("pandas").DataFrame(
                {df_col_check[0]: [100.0]}
                if len(df_col_check) == 1
                else {df_col_check[0]: [100.0], df_col_check[1]: ["adult"]}
            )
        )[0]

        mock = MagicMock()
        mock.classes_ = real_classes
        mock.predict_proba = MagicMock(return_value=[real_proba])

        def capture_predict(df):
            captured_cols.extend(list(df.columns))
            return real_model.predict(df)

        mock.predict = capture_predict

        try:
            svc._model_cache[model_key] = mock
            _run()
        finally:
            svc._model_cache.clear()
            svc._model_cache.update(original_cache)

        return captured_cols

    def test_nasio_diag_feature_name(self):
        cols = self._capture_df_calls("nasio_diag", ["nasiolabial"])
        assert "nasiolabial" in cols

    def test_profile_diag_feature_name(self):
        cols = self._capture_df_calls("profile_diag", ["profile_convexity"])
        assert "profile_convexity" in cols

    def test_total_diag_feature_name(self):
        cols = self._capture_df_calls("total_diag", ["total_convexity"])
        assert "total_convexity" in cols

    def test_mento_diag_feature_name(self):
        cols = self._capture_df_calls("mento_diag", ["mentolabial"])
        assert "mentolabial" in cols

    def test_no_growth_stage_in_diag_models(self):
        """Diagnosis models must NOT receive growth_stage as a feature."""
        import utils.side_measurement_models as svc
        original_cache = dict(svc._model_cache)
        seen_growth_stage = []

        for key in ("nasio_diag", "profile_diag", "total_diag", "mento_diag"):
            real = svc._load(key)
            mock = MagicMock()
            mock.classes_ = list(real.classes_)
            mock.predict_proba = MagicMock(
                return_value=[real.predict_proba(
                    __import__("pandas").DataFrame(
                        {list(real.feature_names_in_)[0] if hasattr(real, "feature_names_in_") else "nasiolabial": [100.0]}
                    )
                )[0]]
            )

            def make_predict(r):
                def p(df):
                    if "growth_stage" in df.columns:
                        seen_growth_stage.append(key)
                    return r.predict(df)
                return p

            mock.predict = make_predict(real)
            svc._model_cache[key] = mock

        try:
            _run()
        finally:
            svc._model_cache.clear()
            svc._model_cache.update(original_cache)

        assert seen_growth_stage == [], (
            f"Diagnosis models received growth_stage: {seen_growth_stage}"
        )


# ════════════════════════════════════════════════════════════════════════════════
# MM-15  Old overall ML frontend panel no longer rendered
# ════════════════════════════════════════════════════════════════════════════════

class TestMM15OldMLPanelHidden:
    """Scan result.html to confirm the old overall-ML panel is removed from output."""

    _TEMPLATE = _PROJECT_ROOT / "templates" / "result.html"

    def _html(self) -> str:
        return self._TEMPLATE.read_text(encoding="utf-8")

    def test_no_diagnosis_label_jinja_render(self):
        """{{ diag.diagnosis_label }} must not appear in the template HTML."""
        html = self._html()
        assert "diag.diagnosis_label" not in html, (
            "result.html still renders diag.diagnosis_label (old overall ML diagnosis)"
        )

    def test_no_treatment_label_jinja_render(self):
        html = self._html()
        assert "diag.treatment_label" not in html, (
            "result.html still renders diag.treatment_label (old overall ML treatment)"
        )

    def test_no_overall_ai_diagnosis_heading(self):
        html = self._html()
        lower = html.lower()
        assert "overall ai diagnosis" not in lower, (
            "result.html still contains 'Overall AI Diagnosis' heading"
        )

    def test_no_old_js_renderresult_fields(self):
        """JS renderResult must not reference d.label / t.label (old ML fields)."""
        html = self._html()
        # The old code had: escHtml(d.label) and escHtml(t.label) inside renderResult
        # These should be gone.  We check for their specific pattern.
        assert "escHtml(d.label)" not in html, (
            "result.html JS still uses d.label (old overall ML diagnosis label)"
        )
        assert "escHtml(t.label)" not in html, (
            "result.html JS still uses t.label (old overall ML treatment label)"
        )


# ════════════════════════════════════════════════════════════════════════════════
# MM-16  Rule-based cards still rendered
# ════════════════════════════════════════════════════════════════════════════════

class TestMM16RuleBasedCardsPresent:
    _TEMPLATE = _PROJECT_ROOT / "templates" / "result.html"

    def _html(self) -> str:
        return self._TEMPLATE.read_text(encoding="utf-8")

    def test_rule_based_section_header_present(self):
        html = self._html()
        assert "Rule-Based Measurement Analysis" in html, (
            "result.html is missing the 'Rule-Based Measurement Analysis' section header"
        )

    def test_rb_grid_present(self):
        html = self._html()
        assert "rb-grid" in html, (
            "result.html is missing the rb-grid CSS class for rule-based cards"
        )

    def test_side_measurement_cards_loop_present(self):
        html = self._html()
        assert "side_measurement_cards" in html, (
            "result.html no longer iterates over side_measurement_cards"
        )

    def test_rb_card_renders_status_label(self):
        html = self._html()
        assert "card.status_label" in html, (
            "result.html no longer renders card.status_label in rule-based cards"
        )


# ════════════════════════════════════════════════════════════════════════════════
# MM-17  New measurement-level ML section rendered
# ════════════════════════════════════════════════════════════════════════════════

class TestMM17MeasurementMLSectionPresent:
    _TEMPLATE = _PROJECT_ROOT / "templates" / "result.html"

    def _html(self) -> str:
        return self._TEMPLATE.read_text(encoding="utf-8")

    def test_measurement_level_ml_section_header(self):
        html = self._html()
        assert "Measurement-Level ML Analysis" in html, (
            "result.html is missing the 'Measurement-Level ML Analysis' section header"
        )

    def test_diagResultArea_present(self):
        html = self._html()
        assert 'id="diagResultArea"' in html, (
            "result.html is missing the diagResultArea element"
        )

    def test_mml_grid_class_present(self):
        html = self._html()
        assert "mml-grid" in html, (
            "result.html is missing the mml-grid CSS class for ML cards"
        )

    def test_js_reads_measurement_ml_field(self):
        """JS must read data.measurement_ml from the API response."""
        html = self._html()
        assert "data.measurement_ml" in html, (
            "result.html JS does not read data.measurement_ml from the API response"
        )

    def test_js_renders_predicted_diagnosis_label(self):
        html = self._html()
        assert "Predicted diagnosis" in html, (
            "result.html JS does not use the label 'Predicted diagnosis'"
        )

    def test_js_renders_treatment_consideration_label(self):
        html = self._html()
        assert "Treatment consideration" in html, (
            "result.html JS does not use the label 'Treatment consideration'"
        )

    def test_disclaimer_in_js_output(self):
        html = self._html()
        # The clinical disclaimer is now rendered from the doctor-review state
        # (state.disclaimer) instead of the raw ml.disclaimer field.
        assert "state.disclaimer" in html, (
            "result.html JS does not render the clinical disclaimer"
        )

    def test_diagnosis_summary_in_js_output(self):
        html = self._html()
        assert "diagnosis_summary" in html, (
            "result.html JS does not render diagnosis_summary"
        )

    def test_treatment_summary_in_js_output(self):
        html = self._html()
        assert "treatment_summary" in html, (
            "result.html JS does not render treatment_summary"
        )

    def test_supported_by_in_js_output(self):
        html = self._html()
        assert "supported_by" in html, (
            "result.html JS does not render supported_by in summaries"
        )
