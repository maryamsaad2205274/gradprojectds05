"""
Post-inference validation: landmark count and coordinate sanity only.
"""

from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple, Union

from utils.image_validation import MSG_ANALYSIS_FAILED

Point = Union[Sequence[float], Tuple[float, float], dict]

EXPECTED_KP = {
    "SIDE": 20,
    "FRONT_NS": 34,
}


def _normalize_points(points: List[Point]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points or []:
        if isinstance(p, dict):
            out.append((float(p.get("x", 0)), float(p.get("y", 0))))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    return out


def validate_landmarks_for_view(
    image_path: str,
    landmarks: List[Point],
    variant: str = "SIDE",
    *,
    heatmap_peaks: Sequence[float] | None = None,
) -> Tuple[bool, str, str]:
    _ = heatmap_peaks

    key = (variant or "SIDE").upper()
    expected = EXPECTED_KP.get(key)
    if expected is None:
        return False, MSG_ANALYSIS_FAILED, f"Unknown variant: {variant}"

    if not image_path or not os.path.isfile(image_path):
        return False, MSG_ANALYSIS_FAILED, "Missing image file"

    if not landmarks:
        return False, MSG_ANALYSIS_FAILED, "No landmarks returned"

    pts = _normalize_points(landmarks)
    if len(pts) != expected:
        return False, MSG_ANALYSIS_FAILED, f"expected {expected} landmarks, got {len(pts)}"

    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in pts):
        return False, MSG_ANALYSIS_FAILED, "non-finite coordinates"

    return True, "", ""
