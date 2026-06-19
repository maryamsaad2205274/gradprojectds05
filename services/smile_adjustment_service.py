"""
services/smile_adjustment_service.py
====================================
Orchestrator for the esthetic Smile Adjustment feature.

Flow:
  uploaded smiling image bytes
    → validate type / decode
    → resize to <= 1024x1024 (aspect preserved)
    → build the dynamic smile-edit prompt (utils.smile_adjustment_prompt)
    → reuse the existing backend-only Gemini image-edit client
    → save the original + generated image under a sanitized, unique filename
    → return before/after paths + prompt + resolved settings

Reuses simulation.gemini_client (backend-only key, retries, 5-min timeout) so
no new API integration is introduced.  The original upload is never overwritten.
"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path
from typing import Dict

from PIL import Image, UnidentifiedImageError

from utils.smile_adjustment_prompt import build_smile_prompt, SmileSelectionError
from simulation import gemini_client
from simulation.gemini_client import GeminiUnavailableError, GeminiGenerationError

_MAX_DIM = 1024
ALLOWED_CONTENT = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SmileImageError(ValueError):
    """User-facing image problem (missing / unsupported / unreadable)."""


def _sanitize(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "smile"


def _open_resized(image_bytes: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.load()
    except (UnidentifiedImageError, OSError):
        raise SmileImageError("The uploaded file is not a readable image.")
    im = im.convert("RGB")
    im.thumbnail((_MAX_DIM, _MAX_DIM), Image.LANCZOS)
    return im


def run_smile_adjustment(
    image_bytes: bytes,
    filename: str,
    tooth_style: str,
    shade: str,
    gummy_mm,
    output_dir: str,
    case_id: int,
) -> Dict[str, object]:
    """Validate, build the prompt, call the image-edit API, and save results.

    Returns::
        {
          "original_url_path": "generated_simulations/smile/uploads/<f>.jpg",
          "result_url_path":   "generated_simulations/smile/<f>.png",
          "prompt": "...",
          "tooth_style": "...", "tooth_style_label": "...",
          "shade": "...", "shade_description": "...",
          "gummy_mm": 1.5,
          "gemini_model": "gemini-2.5-flash-image",
        }

    Raises SmileSelectionError / SmileImageError for validation,
    GeminiUnavailableError / GeminiGenerationError for the API.
    """
    if not image_bytes:
        raise SmileImageError("Please upload a smiling patient photo first.")

    ext = Path(filename or "").suffix.lower()
    if ext and ext not in ALLOWED_EXT:
        raise SmileImageError(
            "Unsupported file type. Upload a JPG, PNG, WEBP, or BMP image."
        )

    # Build prompt first so an invalid selection fails before any API cost.
    built = build_smile_prompt(tooth_style, shade, gummy_mm)

    if not gemini_client.is_available():
        raise GeminiUnavailableError(
            "The smile-adjustment service is not configured. "
            "Please contact your administrator."
        )

    # ── Prepare + persist the (resized) original upload ──────────────────────
    src_im = _open_resized(image_bytes)
    uid = _sanitize(uuid.uuid4().hex)

    out_base = Path(output_dir)
    uploads_dir = out_base / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    out_base.mkdir(parents=True, exist_ok=True)

    orig_name = f"case_{int(case_id)}_{uid}_original.jpg"
    orig_path = uploads_dir / orig_name
    src_im.save(orig_path, format="JPEG", quality=92)

    buf = io.BytesIO()
    src_im.save(buf, format="JPEG", quality=92)
    src_bytes = buf.getvalue()

    # ── Image-edit API call (retries + timeout handled in gemini_client) ─────
    result_bytes, result_mime = gemini_client.generate_image(
        built["prompt"], src_bytes, "image/jpeg"
    )

    result_ext = ".png" if "png" in (result_mime or "") else ".jpg"
    result_name = f"case_{int(case_id)}_{uid}_smile{result_ext}"
    result_path = out_base / result_name
    with Image.open(io.BytesIO(result_bytes)) as out_im:
        out_im = out_im.convert("RGB")
        if result_ext == ".png":
            out_im.save(result_path, format="PNG")
        else:
            out_im.save(result_path, format="JPEG", quality=95)

    rel_orig = "generated_simulations/smile/uploads/" + orig_name
    rel_result = "generated_simulations/smile/" + result_name

    return {
        "original_url_path": rel_orig,
        "result_url_path": rel_result,
        "prompt": built["prompt"],
        "tooth_style": built["tooth_style"],
        "tooth_style_label": built["tooth_style_label"],
        "shade": built["shade"],
        "shade_description": built["shade_description"],
        "gummy_mm": built["gummy_mm"],
        "gemini_model": gemini_client.IMAGE_MODEL,
    }
