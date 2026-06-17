"""
simulation/prompt_builder.py
============================
Builds the controlled Gemini prompt on the backend from validated, doctor-
approved findings and selected internal change codes.

The browser never sends prompt text — only internal codes + strength.  This
module converts those into the controlled anatomical instructions.
"""

from __future__ import annotations

from typing import List

from . import simulation_rules as rules

_DISCLAIMER = (
    "Educational AI-generated visualization; not a guaranteed clinical outcome."
)


def build_prompt(
    approved_abnormal_labels: List[str],
    selected_change_codes: List[str],
    strength: str,
) -> str:
    """Return the full controlled prompt string.

    approved_abnormal_labels:
        Human-readable labels of the doctor-approved ABNORMAL findings
        (Normal findings are excluded from the modification section).
    selected_change_codes:
        Validated internal change codes.
    strength:
        "conservative" or "moderate".
    """
    findings_block = "\n".join(f"- {lbl}" for lbl in approved_abnormal_labels) or "- (none)"

    instructions = []
    for code in selected_change_codes:
        instr = rules.change_instruction(code)
        if instr:
            instructions.append(f"- {instr}")
    instructions_block = "\n".join(instructions) or "- (none)"

    strength_word = rules.strength_label(strength).lower()

    if strength == "exaggerated":
        intensity_block = (
            f"Apply all selected changes at an {strength_word}, clearly pronounced "
            "strength.\n\n"
            "The changes must be strongly and obviously visible when compared with the\n"
            "original image — noticeably more pronounced than a subtle correction —\n"
            "while still looking like a real photograph of the same person and never\n"
            "becoming a cartoon, caricature, or distorted face.\n\n"
        )
    else:
        intensity_block = (
            f"Apply all selected changes at a {strength_word} strength.\n\n"
            "The changes must be visibly clear when compared with the original image,\n"
            "while remaining anatomically restrained and realistic.\n\n"
        )

    return (
        "Create a realistic orthodontic profile-treatment visualization using the\n"
        "provided patient side-profile photograph.\n\n"
        "Doctor-approved findings:\n"
        f"{findings_block}\n\n"
        "Doctor-selected simulation changes:\n"
        f"{instructions_block}\n\n"
        f"{intensity_block}"
        "Strict preservation requirements:\n"
        "- Preserve the patient's exact identity.\n"
        "- Preserve the original skin texture, pores, shadows, and facial details.\n"
        "- Do not smooth, blur, airbrush, beautify, or retouch the skin.\n"
        "- Preserve the eyes, nose, ears, hair, expression, clothing, and background.\n"
        "- Preserve the original lighting, camera angle, image crop, and head position.\n"
        "- Do not modify unrelated facial regions.\n"
        "- Do not invent invisible teeth or hidden anatomy.\n"
        "- Do not add braces, implants, scars, labels, arrows, or text.\n"
        "- Return one photorealistic edited image only.\n\n"
        "The result must look like the same original photograph with controlled\n"
        "anatomical modifications, not like a newly generated or beautified portrait.\n\n"
        "This is an educational AI-generated visualization and is not a guaranteed\n"
        "clinical treatment outcome."
    )


def disclaimer() -> str:
    return _DISCLAIMER
