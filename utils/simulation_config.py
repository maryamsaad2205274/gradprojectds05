"""
Treatment Simulation — landmark group configuration.
Indices are 0-based (0–19 for the 20-landmark side model).

stable_points: always anchored — never displaced regardless of sliders.
Slider displacements are ADDITIVE across groups sharing the same index.
"""

LANDMARK_GROUPS: dict[str, list[int]] = {
    "upper_lip":     [7, 8, 9, 10],
    "lower_lip":     [11, 12, 13, 14],
    "both_lips":     [7, 8, 9, 10, 11, 12, 13, 14],
    "chin":          [15, 16, 17, 18],
    "jaw_profile":   [15, 16, 17, 18, 19],   # 19 is also stable → only 15-18 move
    "stable_points": [0, 2, 3, 4, 5, 6, 19],
}


def resolve_groups() -> dict[str, list[int]]:
    """Return a resolved copy (derive both_lips if empty)."""
    g = {k: list(v) for k, v in LANDMARK_GROUPS.items()}
    if not g["both_lips"]:
        g["both_lips"] = sorted(set(g["upper_lip"]) | set(g["lower_lip"]))
    return g
