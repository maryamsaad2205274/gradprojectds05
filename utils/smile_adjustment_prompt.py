"""
utils/smile_adjustment_prompt.py
================================
Dynamic prompt generation for the esthetic Smile Adjustment feature.

The doctor's UI selections (tooth style, veneer shade, gummy-smile mm) are
converted here into a single controlled image-edit prompt.  Nothing in this
module calls an API or touches the filesystem — it only builds text and
validates the selections, so it is trivially unit-testable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ── Tooth-style options (code → friendly label, subtitle, prompt wording) ───────
TOOTH_STYLES: Dict[str, Dict[str, str]] = {
    "natural": {
        "label": "Natural",
        "subtitle": "Ovoid Curves",
        "prompt": (
            "Natural tooth style with soft ovoid contours, balanced proportions, "
            "and a realistic esthetic appearance"
        ),
    },
    "hollywood": {
        "label": "Hollywood",
        "subtitle": "Bold Square",
        "prompt": (
            "Hollywood smile style with bold square tooth contours, fuller "
            "esthetic presentation, brighter cosmetic appearance, and a polished "
            "smile design"
        ),
    },
    "triangular": {
        "label": "Triangular",
        "subtitle": "Narrow Neck",
        "prompt": (
            "Triangular tooth style with narrower cervical neck contours and a "
            "more tapered esthetic tooth form"
        ),
    },
}

# ── Veneer shade options (code → label, descriptive expansion) ──────────────────
VENEER_SHADES: Dict[str, str] = {
    "BL1": "very bright bleach shade",
    "BL2": "bright bleach shade",
    "A1":  "light natural shade",
    "A2":  "slightly warmer natural shade",
    "A3":  "warm natural shade",
    "B1":  "light reddish-yellow shade",
    "B2":  "soft reddish-yellow shade",
    "C1":  "light greyish shade",
}

# Gummy-smile slider bounds (mm).
GUMMY_MIN_MM = 0.0
GUMMY_MAX_MM = 5.0
GUMMY_STEP_MM = 0.5


class SmileSelectionError(ValueError):
    """Raised when a doctor selection is missing or invalid (user-facing)."""


def tooth_style_options() -> List[Dict[str, str]]:
    """UI-ready list of tooth styles."""
    return [
        {"code": c, "label": v["label"], "subtitle": v["subtitle"]}
        for c, v in TOOTH_STYLES.items()
    ]


def shade_options() -> List[Dict[str, str]]:
    """UI-ready list of veneer shades."""
    return [{"code": c, "description": d} for c, d in VENEER_SHADES.items()]


def validate_selections(tooth_style: str, shade: str, gummy_mm) -> Tuple[str, str, float]:
    """Validate + normalise the three selections. Returns (style, shade, mm).

    Raises SmileSelectionError with a friendly message on any invalid input.
    """
    style = (tooth_style or "").strip().lower()
    if style not in TOOTH_STYLES:
        raise SmileSelectionError("Please select a cosmetic tooth style.")

    shade_code = (shade or "").strip().upper()
    if shade_code not in VENEER_SHADES:
        raise SmileSelectionError("Please select a veneer shade.")

    try:
        mm = float(gummy_mm)
    except (TypeError, ValueError):
        raise SmileSelectionError("Gummy smile correction must be a number.")
    if mm != mm or mm in (float("inf"), float("-inf")):
        raise SmileSelectionError("Gummy smile correction must be a valid number.")
    if mm < GUMMY_MIN_MM or mm > GUMMY_MAX_MM:
        raise SmileSelectionError(
            f"Gummy smile correction must be between {GUMMY_MIN_MM:g} and "
            f"{GUMMY_MAX_MM:g} mm."
        )

    return style, shade_code, mm


def _gummy_instruction(gummy_mm: float) -> str:
    if gummy_mm <= 0:
        return (
            "Do not significantly change the gingival display; keep the existing "
            "gum line natural"
        )
    return (
        f"Reduce visible gingival display by approximately {gummy_mm:g} mm while "
        "preserving a natural smile line"
    )


def build_smile_prompt(tooth_style: str, shade: str, gummy_mm) -> Dict[str, object]:
    """Build the full smile-adjustment edit prompt from validated selections.

    Returns a dict with the resolved fields and the final ``prompt`` string.
    """
    style, shade_code, mm = validate_selections(tooth_style, shade, gummy_mm)

    style_desc = TOOTH_STYLES[style]["prompt"]
    shade_desc = VENEER_SHADES[shade_code]
    gummy_line = _gummy_instruction(mm)

    prompt = (
        "Edit the provided smiling patient photo while preserving the patient's "
        "identity, face shape, skin, hair, eyes, eyebrows, nose, cheeks, chin, "
        "face proportions, pose, lighting, and background. Only modify the smile "
        "region.\n\n"
        "Apply the following esthetic smile adjustments:\n\n"
        f"- Cosmetic tooth style: {style_desc}\n"
        f"- Veneer shade: {shade_code} ({shade_desc})\n"
        f"- Gummy smile correction: {gummy_line}\n\n"
        "Requirements:\n"
        "- Keep the result realistic and clinically plausible\n"
        "- Preserve the overall facial appearance and the patient's identity\n"
        "- Do not alter non-smile facial features\n"
        "- Only adjust the visible teeth, tooth shade, smile line, and gingival "
        "display as needed\n"
        "- Improve the smile according to the selected style\n"
        "- Make the teeth look consistent, symmetrical, and natural for the "
        "selected style\n"
        "- Return one photorealistic edited image only\n\n"
        "This is an educational esthetic smile simulation and is not a guaranteed "
        "clinical outcome."
    )

    return {
        "tooth_style": style,
        "tooth_style_label": TOOTH_STYLES[style]["label"],
        "shade": shade_code,
        "shade_description": shade_desc,
        "gummy_mm": mm,
        "prompt": prompt,
    }
