"""
Tests for the Smile Adjustment feature:
  * utils/smile_adjustment_prompt.py  (dynamic prompt + validation)
  * services/smile_adjustment_service.py (orchestration, image I/O, errors)

The image-edit API is mocked so no network / API key is required.
"""

import io
import json
import os

import pytest
from PIL import Image

from utils import smile_adjustment_prompt as P
from services import smile_adjustment_service as S
from services.smile_adjustment_service import run_smile_adjustment, SmileImageError


def _png_bytes(color=(200, 180, 160)):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


# ── Prompt generation ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("style,frag", [
    ("natural", "soft ovoid contours"),
    ("hollywood", "bold square tooth contours"),
    ("triangular", "narrower cervical neck"),
])
def test_prompt_updates_per_style(style, frag):
    out = P.build_smile_prompt(style, "BL1", 1.0)
    assert frag in out["prompt"]


@pytest.mark.parametrize("shade", list(P.VENEER_SHADES.keys()))
def test_prompt_updates_per_shade(shade):
    out = P.build_smile_prompt("natural", shade, 1.0)
    assert f"Veneer shade: {shade}" in out["prompt"]


def test_prompt_gummy_zero_vs_nonzero():
    z = P.build_smile_prompt("natural", "A1", 0)["prompt"]
    nz = P.build_smile_prompt("natural", "A1", 2.5)["prompt"]
    assert "Do not significantly change the gingival display" in z
    assert "approximately 2.5 mm" in nz


@pytest.mark.parametrize("style,shade,mm", [
    ("", "BL1", 1), ("natural", "", 1), ("natural", "BL1", 9),
    ("natural", "BL1", "x"), ("bad", "BL1", 1), ("natural", "ZZ", 1),
])
def test_validation_rejects_bad_selections(style, shade, mm):
    with pytest.raises(P.SmileSelectionError):
        P.validate_selections(style, shade, mm)


# ── Service flow (mocked API) ───────────────────────────────────────────────────
@pytest.fixture
def mock_api(monkeypatch):
    monkeypatch.setattr(S.gemini_client, "is_available", lambda: True)
    monkeypatch.setattr(S.gemini_client, "generate_image",
                        lambda prompt, b, mime: (_png_bytes((255, 255, 255)), "image/png"))


def test_service_success(tmp_path, mock_api):
    res = run_smile_adjustment(
        image_bytes=_png_bytes(), filename="smile.png",
        tooth_style="hollywood", shade="BL2", gummy_mm=3.0,
        output_dir=str(tmp_path), case_id=42,
    )
    assert res["tooth_style"] == "hollywood"
    assert res["shade"] == "BL2"
    assert res["gummy_mm"] == 3.0
    assert "bold square" in res["prompt"]
    assert os.path.isfile(tmp_path / os.path.basename(res["result_url_path"]))
    assert os.path.isfile(tmp_path / "uploads" / os.path.basename(res["original_url_path"]))
    json.dumps({k: v for k, v in res.items()})


def test_service_missing_image(tmp_path, mock_api):
    with pytest.raises(SmileImageError):
        run_smile_adjustment(b"", "x.png", "natural", "A1", 1.0, str(tmp_path), 1)


def test_service_unsupported_extension(tmp_path, mock_api):
    with pytest.raises(SmileImageError):
        run_smile_adjustment(_png_bytes(), "evil.txt", "natural", "A1", 1.0, str(tmp_path), 1)


def test_service_unreadable_image(tmp_path, mock_api):
    with pytest.raises(SmileImageError):
        run_smile_adjustment(b"not-an-image", "x.png", "natural", "A1", 1.0, str(tmp_path), 1)


def test_service_invalid_selection(tmp_path, mock_api):
    with pytest.raises(P.SmileSelectionError):
        run_smile_adjustment(_png_bytes(), "x.png", "natural", "BL1", 99, str(tmp_path), 1)


def test_service_api_failure_propagates(tmp_path, monkeypatch):
    from simulation.gemini_client import GeminiGenerationError
    monkeypatch.setattr(S.gemini_client, "is_available", lambda: True)
    def boom(*a, **k):
        raise GeminiGenerationError("no image returned")
    monkeypatch.setattr(S.gemini_client, "generate_image", boom)
    with pytest.raises(GeminiGenerationError):
        run_smile_adjustment(_png_bytes(), "x.png", "natural", "A1", 1.0, str(tmp_path), 1)


def test_service_unavailable(tmp_path, monkeypatch):
    from simulation.gemini_client import GeminiUnavailableError
    monkeypatch.setattr(S.gemini_client, "is_available", lambda: False)
    with pytest.raises(GeminiUnavailableError):
        run_smile_adjustment(_png_bytes(), "x.png", "natural", "A1", 1.0, str(tmp_path), 1)
