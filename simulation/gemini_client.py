"""
simulation/gemini_client.py
===========================
Backend-only Gemini access for the profile-simulation service.

* The API key is read ONLY from the .env file beside app.py.
* The key is never returned, logged, or exposed to the frontend.
* Image bytes and full prompts are never logged.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

# .env lives beside app.py (one directory up from this package).
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_DIR / ".env"

IMAGE_MODEL = "gemini-2.5-flash-image"

# Five-minute timeout (milliseconds for the google-genai HTTP options).
_TIMEOUT_MS = 5 * 60 * 1000
_MAX_RETRIES = 3


class GeminiUnavailableError(RuntimeError):
    """Raised when the Gemini key / SDK is unavailable."""


class GeminiGenerationError(RuntimeError):
    """Raised when generation fails after retries or returns no image."""


def _load_api_key() -> str:
    """Return the Gemini API key from .env / environment, or raise."""
    try:
        from dotenv import load_dotenv
        if _ENV_PATH.is_file():
            load_dotenv(dotenv_path=_ENV_PATH, override=False)
    except Exception:
        # python-dotenv missing is fine if the var is already in the env.
        pass

    key = os.getenv("GEMINI_API_KEY")
    if not key or not key.strip() or key.strip() == "PASTE_MY_PRIVATE_KEY_HERE":
        raise GeminiUnavailableError(
            "The image-generation service is not configured. "
            "Please contact your administrator."
        )
    return key.strip()


def is_available() -> bool:
    """True when a usable key is present (no value is exposed)."""
    try:
        _load_api_key()
        return True
    except Exception:
        return False


def _build_client():
    key = _load_api_key()
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise GeminiUnavailableError(
            "The image-generation library is not installed on the server."
        ) from exc
    try:
        return genai.Client(api_key=key)
    except Exception as exc:
        raise GeminiUnavailableError(
            "Could not initialise the image-generation service."
        ) from exc


def _extract_image_bytes(response) -> Optional[Tuple[bytes, str]]:
    """Return (image_bytes, mime_type) from a generate_content response, or None."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                mime = getattr(inline, "mime_type", "image/png") or "image/png"
                return inline.data, mime
    return None


def generate_image(prompt: str, image_bytes: bytes, image_mime: str) -> Tuple[bytes, str]:
    """Send the prompt + source image to Gemini and return (bytes, mime).

    Retries transient network errors up to three times.  Raises
    GeminiGenerationError if no image is returned, and GeminiUnavailableError
    if the service is not configured.
    """
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    client = _build_client()

    contents = [
        genai_types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
        prompt,
    ]
    config = genai_types.GenerateContentConfig(http_options={"timeout": _TIMEOUT_MS})

    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=contents,
                config=config,
            )
            result = _extract_image_bytes(response)
            if result is None:
                raise GeminiGenerationError(
                    "The simulation could not be generated for this image. "
                    "Please try again."
                )
            return result
        except genai_errors.APIError as exc:
            msg = str(exc)
            # Transient → retry; permanent → fail immediately.
            if any(t in msg for t in ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503", "500")):
                last_exc = exc
                time.sleep(2 * attempt)
                continue
            raise GeminiGenerationError(
                "The image-generation service rejected the request."
            ) from exc
        except OSError as exc:  # network-level error
            last_exc = exc
            time.sleep(2 * attempt)
            continue

    raise GeminiGenerationError(
        "The image-generation service is temporarily unavailable. "
        "Please try again in a moment."
    ) from last_exc
