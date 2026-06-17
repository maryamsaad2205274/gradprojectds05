"""
simulation/image_generator.py
=============================
Image I/O for the profile-simulation service:

  * Load + resize the source side image to <= 1024x1024 (aspect preserved).
  * Hand the bytes + prompt to the Gemini client.
  * Save the returned image under a sanitized, unique, case-specific filename.

The original patient image is never overwritten.
"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path
from typing import Tuple

from PIL import Image

from . import gemini_client

_MAX_DIM = 1024


def _resize_to_bytes(image_path: Path) -> Tuple[bytes, str]:
    """Open, RGB-normalise and downscale the source image; return (bytes, mime)."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((_MAX_DIM, _MAX_DIM), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "image/jpeg"


def _sanitize_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "case"


def generate_and_save(
    source_image_path: str,
    prompt: str,
    output_dir: str,
    case_id: int,
) -> str:
    """Generate the simulation and save it; return the saved absolute file path.

    Raises FileNotFoundError if the source image is missing, and the Gemini
    client errors (GeminiUnavailableError / GeminiGenerationError) on failure.
    """
    src = Path(source_image_path)
    if not src.is_file():
        raise FileNotFoundError("The original patient side image could not be found.")

    img_bytes, mime = _resize_to_bytes(src)

    result_bytes, result_mime = gemini_client.generate_image(prompt, img_bytes, mime)

    ext = ".png" if "png" in (result_mime or "") else ".jpg"
    fname = f"case_{int(case_id)}_{_sanitize_stem(uuid.uuid4().hex)}{ext}"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    # Normalise whatever Gemini returns to a real image file we control.
    with Image.open(io.BytesIO(result_bytes)) as out_im:
        out_im = out_im.convert("RGB")
        if ext == ".png":
            out_im.save(out_path, format="PNG")
        else:
            out_im.save(out_path, format="JPEG", quality=95)

    return str(out_path)
