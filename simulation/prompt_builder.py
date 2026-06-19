"""
simulation/prompt_builder.py
============================
Builds the controlled Gemini prompt on the backend from validated, doctor-
approved findings and selected internal change codes.

The browser never sends prompt text — only internal codes + strength.  This
module converts those into the controlled anatomical instructions that are
sent to the Gemini image-editing API.
"""

from __future__ import annotations

from typing import List

from . import simulation_rules as rules

_DISCLAIMER = (
    "Educational AI-generated visualization; not a guaranteed clinical outcome."
)

# Strength wording sent inside the prompt.
# "exaggerated" is absent — any unknown value is clamped to "moderate".
_STRENGTH_PROMPT: dict[str, str] = {
    "conservative": (
        "Apply a subtle but clearly visible correction. The improvement must be "
        "noticeable in a side-by-side comparison but remain minimal, "
        "clinically restrained, and anatomically conservative. "
        "Do not overstate the change."
    ),
    "moderate": (
        "Apply a clearly visible correction that directly improves the approved "
        "profile findings. The improvement must be easy to see in a side-by-side "
        "comparison while remaining realistic, anatomically plausible, and not "
        "overdone. Do not exaggerate or over-correct."
    ),
}


def _normalize_strength(strength: str) -> str:
    """Return 'conservative' or 'moderate'; anything else is clamped to 'moderate'."""
    if (strength or "").strip().lower() == "conservative":
        return "conservative"
    return "moderate"


def build_prompt(
    approved_abnormal_labels: List[str],
    selected_change_codes: List[str],
    strength: str,
) -> str:
    """Return the full controlled prompt string for the Gemini image-editing API.

    approved_abnormal_labels:
        Clean display labels of the doctor-approved ABNORMAL findings.
        Normal findings must be excluded before calling this function.
    selected_change_codes:
        Validated internal change codes (from simulation_rules.CHANGE_LABELS).
    strength:
        'conservative' or 'moderate'.  Any other value is treated as 'moderate'.
    """
    # ── Findings block ───────────────────────────────────────────────────────
    findings_block = (
        "\n".join(f"- {lbl}" for lbl in approved_abnormal_labels)
        or "- (none)"
    )

    # ── Change instructions block ────────────────────────────────────────────
    instructions: List[str] = []
    for code in selected_change_codes:
        instr = rules.change_instruction(code)
        if instr:
            instructions.append(f"- {instr}")
    instructions_block = "\n".join(instructions) or "- (none)"

    # ── Strength block ───────────────────────────────────────────────────────
    safe_strength = _normalize_strength(strength)
    strength_label = "Conservative" if safe_strength == "conservative" else "Moderate"
    strength_paragraph = _STRENGTH_PROMPT[safe_strength]

    # ── Determine whether a lip change is selected ───────────────────────────
    _LIP_CODES = {
        "upper_lip_retraction", "lower_lip_retraction",
        "lip_retraction", "reduce_lip_prominence",
        "improve_lip_support", "lip_advancement",
        "improve_maxillary_lip_support",
    }
    lip_selected = any(c in _LIP_CODES for c in selected_change_codes)
    lip_preservation = (
        "- Preserve the lips exactly as they appear in the source photograph, "
        "including shape, size, colour, and skin texture of the lip region, "
        "because no lip-specific change was selected."
        if not lip_selected
        else
        "- Preserve the lips except in the specific lip region targeted by "
        "the selected lip-change instruction above. Do not change lip colour, "
        "texture, or size beyond what the change instruction requires."
    )

    # ── Build final prompt ───────────────────────────────────────────────────
    return (
        # ── 1. Task statement ────────────────────────────────────────────────
        "Edit the provided patient side-profile photograph to create a realistic "
        "orthodontic/maxillofacial profile-treatment visualization. "
        "This must be a strict localized edit of the original photograph — "
        "not a regenerated portrait, not a beauty retouch, and not a cosmetic "
        "enhancement. Apply changes only to the specific anatomical regions "
        "described below. Every other pixel must remain unchanged.\n\n"

        # ── 2. Doctor-approved findings ──────────────────────────────────────
        "Doctor-approved findings:\n"
        f"{findings_block}\n\n"

        # ── 3. Doctor-selected simulation changes ────────────────────────────
        "Doctor-selected simulation changes:\n"
        f"{instructions_block}\n\n"

        # ── 4. Change strength ───────────────────────────────────────────────
        f"Change strength ({strength_label}):\n"
        f"{strength_paragraph}\n\n"

        # ── 5. CLINICAL VISUALIZATION GOAL ───────────────────────────────────
        "CLINICAL VISUALIZATION GOAL:\n"
        "The selected simulation changes must directly improve the approved "
        "profile findings listed above. The result must show a visible, "
        "targeted improvement in the side-view profile — especially in the "
        "selected anatomical regions — that is clearly noticeable when the "
        "original and the edited image are compared side by side. "
        "Do not produce a generic beauty edit or a cosmetically enhanced "
        "portrait. Produce a specific, localized orthodontic/maxillofacial "
        "profile correction that addresses the approved findings only.\n\n"

        # ── 6. STRICT REGION CONTROL ─────────────────────────────────────────
        "STRICT REGION CONTROL:\n"
        "Modify ONLY the specific soft-tissue profile regions described in the "
        "selected simulation changes above. "
        "Every other part of the photograph — including all facial features not "
        "listed for change, the background, clothing, lighting, crop, and camera "
        "angle — must remain pixel-identical or visually indistinguishable from "
        "the original. "
        "Do not change the overall face shape except in the selected treatment "
        "region. Do not bleed the change into adjacent regions such as the neck, "
        "jawline hair, ears, or forehead.\n\n"

        # ── 7. CRITICAL PRESERVATION RULES ───────────────────────────────────
        "CRITICAL PRESERVATION RULES:\n"
        "- Preserve the patient's exact identity and recognizable appearance. "
        "The result must be clearly the same person.\n"
        "- FACIAL HAIR — ABSOLUTE PROHIBITION: Do not add, remove, or alter "
        "facial hair in any way. "
        "This means: do not add a beard, moustache, goatee, stubble, sideburns, "
        "or jawline hair where none exists in the source photograph. "
        "Do not remove, thin, thicken, darken, lighten, reshape, blur, or "
        "otherwise alter any beard, moustache, goatee, stubble, or sideburns "
        "that exist in the source photograph. "
        "The density, length, colour, and distribution of every hair on the face "
        "must be absolutely identical to the source photograph in all regions — "
        "including the chin, jaw, upper lip, cheeks, and neck — regardless of "
        "which anatomical change is being applied.\n"
        "- Preserve the exact nose shape including the nasal bridge, tip, "
        "nostrils, columella, and alar base. "
        "Do not change the nose in any way.\n"
        "- Preserve the exact forehead, eyes, eyelids, eyebrows, periorbital "
        "region, and ears exactly.\n"
        "- Preserve the exact scalp hair and hairline.\n"
        f"{lip_preservation}\n"
        "- Preserve the exact skin tone, complexion, and ethnicity. "
        "Do not alter the skin colour in any region.\n"
        "- Preserve the exact skin texture, pores, fine lines, wrinkles, "
        "shadows, blemishes, and photographic grain exactly as they appear in "
        "the source photograph. Do not smooth, soften, or alter skin texture "
        "in any region including the region being modified.\n"
        "- Preserve the original lighting direction, intensity, specular "
        "highlights, shadows, and photographic depth of field exactly.\n"
        "- Preserve the exact background, clothing, accessories, crop, "
        "camera angle, and head position.\n"
        "- Preserve any watermark, overlay text, or measurement annotations "
        "present in the original photograph.\n"
        "- Preserve the patient's apparent age, expression, and emotion.\n\n"

        # ── 8. Explicit DO NOT list ───────────────────────────────────────────
        "DO NOT:\n"
        "- Do not add, remove, or alter facial hair in any way.\n"
        "- Do not add a beard, moustache, goatee, stubble, sideburns, or "
        "jawline hair where none exists in the source photograph.\n"
        "- Do not remove, reduce, reshape, darken, lighten, blur, or otherwise "
        "change any existing beard, moustache, goatee, stubble, or sideburns.\n"
        "- Do not smooth, blur, airbrush, soften, beautify, retouch, or apply "
        "any skin filter in any facial region, including the region being "
        "modified.\n"
        "- Do not change the nose shape, size, tip, bridge, nostrils, or "
        "columella.\n"
        "- Do not change the skin colour, tone, or apparent ethnicity.\n"
        "- Do not change the patient's apparent age or make the patient look "
        "younger or older.\n"
        "- Do not change the patient's expression or emotion.\n"
        "- Do not modify any facial region not listed in the selected simulation "
        "changes above.\n"
        "- Do not invent invisible teeth, bones, implants, or hidden anatomy.\n"
        "- Do not add braces, implants, scars, surgical marks, labels, arrows, "
        "or any text.\n"
        "- Do not generate a new portrait. Apply a localized edit to the "
        "original photograph only.\n"
        "- Do not make the patient look more handsome, more attractive, younger, "
        "or cosmetically enhanced in any way.\n"
        "- Do not change the background, lighting, crop, camera angle, or "
        "clothing.\n\n"

        # ── 9. Output instruction ─────────────────────────────────────────────
        "Return one photorealistic edited image only.\n\n"

        # ── 10. Disclaimer ────────────────────────────────────────────────────
        "This is an educational AI-generated visualization and is not a "
        "guaranteed clinical treatment outcome."
    )


def disclaimer() -> str:
    return _DISCLAIMER
