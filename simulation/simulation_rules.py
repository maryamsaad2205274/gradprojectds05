"""
simulation/simulation_rules.py
==============================
Central configuration / rules file for the doctor-review → profile-simulation
workflow.  This is the SINGLE source of truth for:

  * Diagnosis-label normalization (variations in case / punctuation / slashes
    map to a stable internal diagnosis code).
  * Diagnosis-code → available simulation-change options (code + label).
  * Simulation-change-code → controlled anatomical Gemini instruction.
  * Strength values.
  * Conflict rules between approved diagnoses and selected changes.

Nothing here re-detects landmarks, re-computes angles, or calls a model.
The mapping must NOT be duplicated in HTML / JavaScript — the frontend asks
the backend for the resolved options instead.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ── Diagnosis normalization ────────────────────────────────────────────────────
#
# The four measurement models emit these exact display labels:
#   nasiolabial  : Normal | Protruded maxilla | Retruded maxilla / high columella
#   profile      : Normal | Protruded chin | Retruded chin
#   total        : Normal | Protruded chin | Retruded chin
#   mentolabial  : Normal | Prominent chin / mandibular retrusion
#                         | Retruded chin / proclined lower teeth
#
# NOTE — "Prominent chin / mandibular retrusion" is the label emitted by the
# trained mento_diagnosis_random_forest.pkl model.  Clinically it describes a
# case where the mentolabial angle is decreased (<110°), indicating a RETRUDED
# MANDIBLE / retruded lower jaw — NOT a protruded chin.  The model label is
# therefore misleading; we remap it to the clearer internal code
# "mandibular_retrusion" and display it as "Mandibular retrusion / retruded
# lower jaw" in all user-facing and Gemini-facing contexts.
#
# We map each canonical label to a stable internal code.  A normalization
# layer collapses capitalization, punctuation, slashes and spacing so we never
# compare raw uncontrolled display strings throughout the application.

NORMAL_CODE = "normal"


def _norm_key(label: str) -> str:
    """Collapse a display label to a comparison key.

    Lowercases, replaces slashes with spaces, strips all non-alphanumeric
    characters to single spaces, and squeezes whitespace.
    """
    s = (label or "").strip().lower()
    s = s.replace("/", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# canonical display label -> stable internal code
# Keys are the exact strings emitted by the trained .pkl models.
_CANONICAL_DIAGNOSES: Dict[str, str] = {
    "Normal":                                  NORMAL_CODE,
    "Protruded maxilla":                       "protruded_maxilla",
    "Retruded maxilla / high columella":       "retruded_maxilla_high_columella",
    "Protruded chin":                          "protruded_chin",
    "Retruded chin":                           "retruded_chin",
    # Model emits "Prominent chin / mandibular retrusion" but the clinical
    # meaning is mandibular retrusion (retruded lower jaw).  Internal code is
    # renamed to "mandibular_retrusion" throughout.
    "Prominent chin / mandibular retrusion":   "mandibular_retrusion",
    "Retruded chin / proclined lower teeth":   "retruded_chin_proclined_lower_teeth",
}

# Clean display labels used in the Gemini prompt and any user-facing context
# where the raw model output label would be confusing or misleading.
# Maps internal diagnosis code → clean display string.
DIAGNOSIS_DISPLAY_LABELS: Dict[str, str] = {
    NORMAL_CODE:                         "Normal",
    "protruded_maxilla":                 "Protruded maxilla",
    "retruded_maxilla_high_columella":   "Retruded maxilla",
    "protruded_chin":                    "Protruded chin",
    "retruded_chin":                     "Retruded chin",
    "mandibular_retrusion":              "Mandibular retrusion / retruded lower jaw",
    "retruded_chin_proclined_lower_teeth": "Retruded chin / proclined lower teeth",
    "unknown":                           "Unknown finding",
}

# Pre-computed normalized-key -> code lookup (built once).
_NORM_TO_CODE: Dict[str, str] = {
    _norm_key(label): code for label, code in _CANONICAL_DIAGNOSES.items()
}


def normalize_diagnosis(label: str) -> str:
    """Map a (possibly messy) model diagnosis label to a stable internal code.

    Unknown labels fall back to ``"unknown"`` — treated as abnormal but
    contributing no simulation-change options.
    """
    return _NORM_TO_CODE.get(_norm_key(label), "unknown")


def is_abnormal(diagnosis_code: str) -> bool:
    """True for any code other than the Normal code."""
    return diagnosis_code != NORMAL_CODE


# ── Simulation-change labels (user-friendly) ───────────────────────────────────

CHANGE_LABELS: Dict[str, str] = {
    "improve_upper_profile_support":   "Improve upper-profile support",
    "improve_maxillary_lip_support":   "Improve upper-lip and maxillary support",
    "upper_profile_advancement":       "Upper-profile advancement",
    "reduce_upper_profile_prominence": "Reduce upper-profile prominence",
    "upper_lip_retraction":            "Upper-lip profile retraction",
    "chin_advancement":                "Chin advancement",
    "improve_chin_projection":         "Improve chin projection",
    "chin_reduction":                  "Chin reduction",
    "reduce_chin_prominence":          "Reduce chin prominence",
    "mandibular_advancement":          "Mandibular advancement",
    "improve_lower_jaw_projection":    "Improve lower-jaw projection",
    "improve_lower_facial_balance":    "Improve lower facial balance",
    "mandibular_setback":              "Mandibular setback",
    "reduce_lower_jaw_prominence":     "Reduce lower-jaw prominence",
    "lower_lip_retraction":            "Lower-lip profile retraction",
    "lip_retraction":                  "Lip-profile retraction",
    "reduce_lip_prominence":           "Reduce lip prominence",
    "improve_lip_support":             "Improve lip support",
    "lip_advancement":                 "Lip advancement",
}

# ── Diagnosis-code → ordered list of available change codes ─────────────────────

DIAGNOSIS_CHANGES: Dict[str, List[str]] = {
    "retruded_maxilla_high_columella": [
        "improve_upper_profile_support",
        "improve_maxillary_lip_support",
        "upper_profile_advancement",
    ],
    "protruded_maxilla": [
        "reduce_upper_profile_prominence",
        "upper_lip_retraction",
    ],
    "retruded_chin": [
        "chin_advancement",
        "improve_chin_projection",
    ],
    "protruded_chin": [
        "chin_reduction",
        "reduce_chin_prominence",
    ],
    "mandibular_retrusion": [
        "mandibular_advancement",
        "improve_lower_jaw_projection",
        "improve_lower_facial_balance",
    ],
    "retruded_chin_proclined_lower_teeth": [
        "chin_advancement",
        "mandibular_advancement",
        "lower_lip_retraction",
        "improve_lower_facial_balance",
    ],
    NORMAL_CODE: [],
    "unknown": [],
}

# ── Simulation-change-code → controlled anatomical Gemini instruction ───────────

SIMULATION_CHANGE_RULES: Dict[str, str] = {
    # ── Upper-face / maxillary changes ────────────────────────────────────────
    "improve_upper_profile_support": (
        "Improve visible upper-profile and maxillary soft-tissue support "
        "beneath the nose and around the upper lip region only. "
        "The improvement should read as better forward support in the "
        "side-profile view. "
        "Do not touch the nose, nostrils, columella, or nasal bridge. "
        "Do not touch the upper lip unless a lip-specific change is also "
        "selected. "
        "Do not alter skin texture or any unrelated anatomy. "
        "Do not add, remove, or alter facial hair in any way."
    ),
    "improve_maxillary_lip_support": (
        "Improve upper-lip support and maxillary soft-tissue profile naturally "
        "so the upper lip appears better supported in the side-profile view. "
        "Do not enlarge, thin, or reshape the lips beyond natural support. "
        "Do not change the nose, nostrils, or columella in any way. "
        "Do not alter facial hair, skin texture, or any unrelated anatomy."
    ),
    "upper_profile_advancement": (
        "Create a conservative forward improvement in the visible upper "
        "soft-tissue profile support so the upper-face profile reads as more "
        "forward in the side view. "
        "Restrict the change to the maxillary soft-tissue region only. "
        "Do not change the nose shape, nostrils, or columella. "
        "Do not alter facial hair, skin texture, or any unrelated anatomy."
    ),
    "reduce_upper_profile_prominence": (
        "Reduce the visible prominence of the upper-profile and maxillary "
        "soft-tissue region so the upper-face profile reads as less protruded "
        "in the side view. "
        "Restrict the change to the upper-profile soft-tissue region only. "
        "Do not change the nose. "
        "Do not alter facial hair, skin texture, or any unrelated anatomy."
    ),
    "upper_lip_retraction": (
        "Reduce visible upper-lip prominence slightly so the upper lip reads "
        "as less protruded in the side-profile view. "
        "Restrict the change to the upper-lip soft-tissue region only. "
        "Preserve the natural lip shape, philtrum, lip colour, and lip texture. "
        "Do not change the nose, nostrils, or columella. "
        "Do not alter facial hair, skin texture, or any unrelated anatomy."
    ),
    # ── Lower-face / chin / mandibular changes ────────────────────────────────
    "chin_advancement": (
        "Advance the visible soft-tissue chin profile slightly forward to "
        "improve chin projection and lower facial balance. "
        "The advancement should be clearly noticeable in the side view. "
        "Restrict the change to the chin soft-tissue region only — "
        "do not alter the lips, neck, or any other region. "
        "Preserve the nose, lips unless lip change is also selected, "
        "skin texture, and all unrelated facial structures. "
        "Do not add, remove, or alter facial hair in any way."
    ),
    "improve_chin_projection": (
        "Improve the visible soft-tissue chin projection so the chin reads as "
        "better projected and more balanced with the mid-face in the side view. "
        "Restrict the change to the chin soft-tissue profile region only. "
        "Do not add, remove, or alter facial hair, beard, or stubble on or "
        "around the chin in any way. "
        "Preserve the nose, lips unless lip change is also selected, "
        "skin texture, and all unrelated facial structures."
    ),
    "chin_reduction": (
        "Reduce the visible soft-tissue chin prominence conservatively so the "
        "chin reads as less protruded in the side view. "
        "Restrict the change to the chin soft-tissue profile region only. "
        "Do not add, remove, or alter facial hair, beard, or stubble on or "
        "around the chin or jawline in any way. "
        "Preserve the nose, lips unless lip change is also selected, "
        "natural jawline, skin texture, and all unrelated facial structures."
    ),
    "reduce_chin_prominence": (
        "Reduce the visible prominence of the chin in the side-profile view "
        "conservatively to improve lower facial balance. "
        "Restrict the change to the chin soft-tissue profile region only. "
        "Do not add, remove, or alter facial hair, beard, or stubble on or "
        "around the chin or jawline in any way. "
        "Preserve the nose, lips unless lip change is also selected, "
        "natural jawline, skin texture, and all unrelated facial structures."
    ),
    "mandibular_advancement": (
        "Advance the visible lower-jaw and chin soft-tissue profile forward as "
        "one coordinated lower-face change so the lower jaw and chin region "
        "appear more forward in the side view, improving lower facial balance "
        "and the chin-to-neck relationship. "
        "The change must be clearly visible without being overdone. "
        "Restrict the change strictly to the lower-jaw and chin soft-tissue "
        "profile — do not alter any region outside this zone. "
        "Preserve the nose, upper lip unless lip change is also selected, "
        "skin texture, and all unrelated facial structures. "
        "Do not add, remove, or alter facial hair in any way."
    ),
    "improve_lower_jaw_projection": (
        "Improve the visible lower-jaw projection in the side-profile view so "
        "the lower jaw reads as better projected and more balanced. "
        "Restrict the change to the lower-jaw soft-tissue profile region. "
        "Do not add, remove, or alter facial hair, beard, or stubble on or "
        "around the jaw or chin in any way. "
        "Preserve the lips unless lip change is also selected, the nose, "
        "skin texture, and all unrelated facial structures."
    ),
    "improve_lower_facial_balance": (
        "Improve lower facial profile balance by subtly harmonizing the lower "
        "lip, chin, and jawline soft-tissue profile according to the approved "
        "diagnosis. The improvement should read as more balanced lower-face "
        "proportions in the side view. "
        "Restrict the change to the lower-face soft-tissue profile region. "
        "Preserve the nose, upper lip unless lip change is also selected, "
        "skin texture, and all unrelated facial structures. "
        "Do not add, remove, or alter facial hair in any way."
    ),
    "mandibular_setback": (
        "Move the visible lower-jaw and chin soft-tissue profile slightly "
        "backward as one coordinated lower-face change to reduce excessive "
        "lower-jaw prominence. "
        "The setback should be clearly visible in the side view without being "
        "overdone. "
        "Restrict the change strictly to the lower-jaw and chin soft-tissue "
        "profile — do not alter any region outside this zone. "
        "Do not add, remove, or alter facial hair, beard, or stubble on or "
        "around the chin, jaw, or neck in any way. "
        "Preserve the nose, upper lip unless lip change is also selected, "
        "skin texture, and all unrelated facial structures."
    ),
    "reduce_lower_jaw_prominence": (
        "Reduce visible lower-jaw prominence in the side-profile view to "
        "improve lower facial balance. "
        "Restrict the change to the lower-jaw soft-tissue profile region. "
        "Do not add, remove, or alter facial hair, beard, or stubble on or "
        "around the jaw or chin in any way. "
        "Preserve the lips unless lip change is also selected, the nose, "
        "skin texture, and all unrelated facial structures."
    ),
    # ── Lip changes ───────────────────────────────────────────────────────────
    "lower_lip_retraction": (
        "Reduce visible lower-lip prominence slightly so the lower lip reads "
        "as less protruded in the side-profile view. "
        "Restrict the change to the lower-lip soft-tissue region only. "
        "Preserve the natural lip shape, colour, and lip-skin texture. "
        "Do not alter the chin, nose, facial hair, or any unrelated anatomy."
    ),
    "lip_retraction": (
        "Reduce visible lip prominence slightly so the lips read as less "
        "protruded in the side-profile view. "
        "Restrict the change to the lip soft-tissue region only. "
        "Preserve the natural lip shape, mouth closure, lip colour, "
        "and lip-skin texture. "
        "Do not alter the chin, nose, facial hair, or any unrelated anatomy."
    ),
    "reduce_lip_prominence": (
        "Reduce visible lip prominence so the lip profile reads as less "
        "protruded in the side view. "
        "Restrict the change to the lip soft-tissue region only. "
        "Preserve the natural lip shape, philtrum, lip colour, "
        "and lip-skin texture. "
        "Do not alter the chin, nose, facial hair, or any unrelated anatomy."
    ),
    "improve_lip_support": (
        "Improve visible lip support naturally so the lips read as better "
        "supported in the side-profile view. "
        "Do not enlarge, artificially plump, or cosmetically enhance the lips. "
        "Restrict the change to the lip soft-tissue region only. "
        "Preserve the natural lip shape, lip colour, and lip-skin texture. "
        "Do not alter the chin, nose, facial hair, or any unrelated anatomy."
    ),
    "lip_advancement": (
        "Advance the visible lip profile very slightly to improve lip support "
        "in the side-profile view. "
        "Restrict the change to the lip soft-tissue region only. "
        "Preserve the natural lip shape, philtrum, lip colour, "
        "and lip-skin texture. "
        "Do not alter the chin, nose, facial hair, or any unrelated anatomy."
    ),
}

# ── Strength ───────────────────────────────────────────────────────────────────

STRENGTHS: Dict[str, str] = {
    "conservative": "Conservative",
    "moderate":     "Moderate",
}
DEFAULT_STRENGTH = "conservative"


# ── Conflict rules ─────────────────────────────────────────────────────────────
#
# Each entry is a frozenset of items (diagnosis codes OR change codes) that are
# mutually contradictory.  A conflict exists when an approved-diagnosis set or a
# selected-change set is a superset of any conflict pair.

_DIAGNOSIS_CONFLICTS: List[frozenset] = [
    # True clinical opposites — cannot be approved simultaneously.
    frozenset({"retruded_chin", "protruded_chin"}),
    frozenset({"retruded_chin_proclined_lower_teeth", "protruded_chin"}),
    frozenset({"retruded_maxilla_high_columella", "protruded_maxilla"}),
    # mandibular_retrusion (retruded lower jaw) + protruded_chin are
    # anatomically contradictory at the jaw/chin level.
    frozenset({"mandibular_retrusion", "protruded_chin"}),
    # NOTE: retruded_chin + mandibular_retrusion is intentionally NOT a
    # conflict — both can co-exist in the same patient profile.
]

_CHANGE_CONFLICTS: List[frozenset] = [
    frozenset({"chin_advancement", "chin_reduction"}),
    frozenset({"chin_advancement", "reduce_chin_prominence"}),
    frozenset({"improve_chin_projection", "chin_reduction"}),
    frozenset({"improve_chin_projection", "reduce_chin_prominence"}),
    frozenset({"mandibular_advancement", "mandibular_setback"}),
    frozenset({"mandibular_advancement", "reduce_lower_jaw_prominence"}),
    frozenset({"improve_lower_jaw_projection", "mandibular_setback"}),
    frozenset({"lip_advancement", "lip_retraction"}),
    frozenset({"improve_lip_support", "lip_retraction"}),
    frozenset({"upper_lip_retraction", "improve_maxillary_lip_support"}),
    frozenset({"upper_profile_advancement", "reduce_upper_profile_prominence"}),
    frozenset({"improve_upper_profile_support", "reduce_upper_profile_prominence"}),
]


# ── Public helpers ─────────────────────────────────────────────────────────────

def changes_for_diagnosis(diagnosis_code: str) -> List[Dict[str, str]]:
    """Return [{"code", "label"}, ...] of changes for an internal diagnosis code."""
    out = []
    for code in DIAGNOSIS_CHANGES.get(diagnosis_code, []):
        out.append({"code": code, "label": CHANGE_LABELS.get(code, code)})
    return out


def available_changes(approved_diagnosis_codes: List[str]) -> List[Dict[str, str]]:
    """Build the de-duplicated, ordered list of change options.

    Only abnormal diagnoses contribute options; Normal / unknown contribute none.
    """
    seen = set()
    out: List[Dict[str, str]] = []
    for code in approved_diagnosis_codes:
        if not is_abnormal(code):
            continue
        for change in DIAGNOSIS_CHANGES.get(code, []):
            if change in seen:
                continue
            seen.add(change)
            out.append({"code": change, "label": CHANGE_LABELS.get(change, change)})
    return out


def valid_change_codes(approved_diagnosis_codes: List[str]) -> set:
    """Set of change codes supported by the approved abnormal diagnoses."""
    return {c["code"] for c in available_changes(approved_diagnosis_codes)}


def diagnosis_display_label(code: str) -> str:
    """Return the clean, user-facing display label for an internal diagnosis code.

    Use this instead of the raw model-output string when building the Gemini
    prompt or displaying the finding to the doctor, so that clinically misleading
    model labels (e.g. 'Prominent chin / mandibular retrusion') are replaced with
    their correct clinical description.
    """
    return DIAGNOSIS_DISPLAY_LABELS.get(code, code)


def change_label(code: str) -> str:
    return CHANGE_LABELS.get(code, code)


def change_instruction(code: str) -> Optional[str]:
    return SIMULATION_CHANGE_RULES.get(code)


def normalize_strength(value: str) -> Optional[str]:
    """Return a valid lowercase strength code, or None if invalid."""
    v = (value or "").strip().lower()
    return v if v in STRENGTHS else None


def strength_label(code: str) -> str:
    return STRENGTHS.get(code, code)


def _conflicts_in(items: List[str], pairs: List[frozenset]) -> List[Tuple[str, str]]:
    s = set(items)
    found = []
    for pair in pairs:
        if pair <= s:
            found.append(tuple(sorted(pair)))
    return found


def find_diagnosis_conflicts(approved_diagnosis_codes: List[str]) -> List[Tuple[str, str]]:
    return _conflicts_in(approved_diagnosis_codes, _DIAGNOSIS_CONFLICTS)


def find_change_conflicts(selected_change_codes: List[str]) -> List[Tuple[str, str]]:
    return _conflicts_in(selected_change_codes, _CHANGE_CONFLICTS)
