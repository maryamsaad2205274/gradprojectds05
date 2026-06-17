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
_CANONICAL_DIAGNOSES: Dict[str, str] = {
    "Normal":                                  NORMAL_CODE,
    "Protruded maxilla":                       "protruded_maxilla",
    "Retruded maxilla / high columella":       "retruded_maxilla_high_columella",
    "Protruded chin":                          "protruded_chin",
    "Retruded chin":                           "retruded_chin",
    "Prominent chin / mandibular retrusion":   "prominent_chin_mandibular_retrusion",
    "Retruded chin / proclined lower teeth":   "retruded_chin_proclined_lower_teeth",
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
    "prominent_chin_mandibular_retrusion": [
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
    "improve_upper_profile_support": (
        "Improve visible upper-profile and maxillary soft-tissue support "
        "while preserving the nose and unrelated facial structures."
    ),
    "improve_maxillary_lip_support": (
        "Improve upper-lip and maxillary soft-tissue support while preserving "
        "natural lip shape, the nose, and unrelated facial structures."
    ),
    "upper_profile_advancement": (
        "Advance the visible upper-profile and maxillary region forward while "
        "preserving the nose and unrelated facial structures."
    ),
    "reduce_upper_profile_prominence": (
        "Reduce visible upper-profile and maxillary prominence while preserving "
        "the nose and unrelated facial structures."
    ),
    "upper_lip_retraction": (
        "Reduce visible upper-lip prominence while preserving natural lip shape "
        "and facial identity."
    ),
    "chin_advancement": (
        "Advance the visible soft-tissue chin while preserving the lips, nose, "
        "and unrelated facial structures."
    ),
    "improve_chin_projection": (
        "Improve the visible soft-tissue chin projection while preserving the "
        "lips, nose, and unrelated facial structures."
    ),
    "chin_reduction": (
        "Reduce the visible soft-tissue chin prominence while preserving the "
        "lips, nose, and unrelated facial structures."
    ),
    "reduce_chin_prominence": (
        "Reduce the visible prominence of the chin profile while preserving the "
        "lips, nose, and unrelated facial structures."
    ),
    "mandibular_advancement": (
        "Advance the lower-jaw and soft-tissue chin profile forward to improve "
        "lower facial balance."
    ),
    "improve_lower_jaw_projection": (
        "Improve the visible lower-jaw projection while preserving the lips, "
        "nose, and unrelated facial structures."
    ),
    "improve_lower_facial_balance": (
        "Improve the overall lower-facial balance of the profile while keeping "
        "changes anatomically restrained and realistic."
    ),
    "mandibular_setback": (
        "Set the lower-jaw profile back to reduce lower-jaw prominence while "
        "preserving unrelated facial structures."
    ),
    "reduce_lower_jaw_prominence": (
        "Reduce visible lower-jaw prominence while preserving the lips, nose, "
        "and unrelated facial structures."
    ),
    "lower_lip_retraction": (
        "Reduce visible lower-lip prominence while preserving natural lip shape "
        "and facial identity."
    ),
    "lip_retraction": (
        "Reduce visible lip prominence while preserving natural lip shape and "
        "facial identity."
    ),
    "reduce_lip_prominence": (
        "Reduce visible lip prominence while preserving natural lip shape and "
        "facial identity."
    ),
    "improve_lip_support": (
        "Improve visible lip support while preserving natural lip shape and "
        "facial identity."
    ),
    "lip_advancement": (
        "Advance the visible lip profile slightly while preserving natural lip "
        "shape and facial identity."
    ),
}

# ── Strength ───────────────────────────────────────────────────────────────────

STRENGTHS: Dict[str, str] = {
    "conservative": "Conservative",
    "moderate":     "Moderate",
    "exaggerated":  "Exaggerated",
}
DEFAULT_STRENGTH = "conservative"


# ── Conflict rules ─────────────────────────────────────────────────────────────
#
# Each entry is a frozenset of items (diagnosis codes OR change codes) that are
# mutually contradictory.  A conflict exists when an approved-diagnosis set or a
# selected-change set is a superset of any conflict pair.

_DIAGNOSIS_CONFLICTS: List[frozenset] = [
    frozenset({"retruded_chin", "protruded_chin"}),
    frozenset({"retruded_chin_proclined_lower_teeth", "protruded_chin"}),
    frozenset({"retruded_maxilla_high_columella", "protruded_maxilla"}),
    frozenset({"prominent_chin_mandibular_retrusion", "retruded_chin"}),
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
