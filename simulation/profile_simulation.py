"""
simulation/profile_simulation.py
================================
Orchestrator for the doctor-reviewed profile-simulation workflow.

Receives ONLY backend-validated data:
  * the original patient side-image path (resolved from the case, never from
    the browser),
  * the doctor-approved findings (label + internal diagnosis code),
  * the validated selected internal change codes,
  * the selected shared strength.

Performs conflict + support validation, builds the controlled prompt, calls the
image service, and returns the saved path + prompt + disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from . import image_generator, prompt_builder
from . import simulation_rules as rules


class SimulationValidationError(ValueError):
    """User-facing validation failure (no Gemini call should be made)."""


@dataclass
class ApprovedFinding:
    label: str          # human-readable model diagnosis
    code: str           # stable internal diagnosis code


def validate(
    approved_findings: List[ApprovedFinding],
    selected_change_codes: List[str],
    strength: str,
) -> str:
    """Validate the request. Returns the normalised strength or raises.

    Order of checks mirrors the required validation behaviour.
    """
    strength_norm = rules.normalize_strength(strength)
    if strength_norm is None:
        raise SimulationValidationError("Please select a valid change strength.")

    approved_codes = [f.code for f in approved_findings]
    abnormal_codes = [c for c in approved_codes if rules.is_abnormal(c)]

    if not abnormal_codes:
        raise SimulationValidationError(
            "No abnormal doctor-approved finding is available for simulation."
        )

    if not selected_change_codes:
        raise SimulationValidationError(
            "Select at least one change to simulate."
        )

    supported = rules.valid_change_codes(approved_codes)
    unsupported = [c for c in selected_change_codes if c not in supported]
    if unsupported:
        labels = ", ".join(rules.change_label(c) for c in unsupported)
        raise SimulationValidationError(
            f"These changes are not supported by the approved diagnoses: {labels}."
        )

    diag_conflicts = rules.find_diagnosis_conflicts(approved_codes)
    if diag_conflicts:
        pairs = "; ".join(" vs ".join(p) for p in diag_conflicts)
        raise SimulationValidationError(
            f"Approved findings conflict and cannot be simulated together: {pairs}. "
            "Please decline one of the conflicting findings."
        )

    change_conflicts = rules.find_change_conflicts(selected_change_codes)
    if change_conflicts:
        pairs = "; ".join(
            " vs ".join(rules.change_label(c) for c in p) for p in change_conflicts
        )
        raise SimulationValidationError(
            f"Selected changes conflict: {pairs}. Please remove one of them."
        )

    return strength_norm


def run_profile_simulation(
    source_image_path: str,
    approved_findings: List[ApprovedFinding],
    selected_change_codes: List[str],
    strength: str,
    output_dir: str,
    case_id: int,
) -> Dict[str, object]:
    """Validate, build the prompt, generate and save the simulation image.

    Returns::
        {
          "image_path": "<abs path>",
          "prompt": "<full prompt>",
          "gemini_model": "gemini-2.5-flash-image",
          "strength": "moderate",
          "selected_changes": [...],
          "disclaimer": "...",
        }
    """
    strength_norm = validate(approved_findings, selected_change_codes, strength)

    # Use the clean display label (from the internal code) rather than the raw
    # model-output string, so misleading model labels such as
    # "Prominent chin / mandibular retrusion" are shown correctly as
    # "Mandibular retrusion / retruded lower jaw" in the Gemini prompt.
    abnormal_labels = [
        rules.diagnosis_display_label(f.code)
        for f in approved_findings
        if rules.is_abnormal(f.code)
    ]

    prompt = prompt_builder.build_prompt(
        approved_abnormal_labels=abnormal_labels,
        selected_change_codes=selected_change_codes,
        strength=strength_norm,
    )

    saved_path = image_generator.generate_and_save(
        source_image_path=source_image_path,
        prompt=prompt,
        output_dir=output_dir,
        case_id=case_id,
    )

    from .gemini_client import IMAGE_MODEL

    return {
        "image_path": saved_path,
        "prompt": prompt,
        "gemini_model": IMAGE_MODEL,
        "strength": strength_norm,
        "selected_changes": list(selected_change_codes),
        "disclaimer": prompt_builder.disclaimer(),
    }
