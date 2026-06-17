"""
Numbered landmark preview generator.

Draws each predicted landmark on the original image with its 0-based index
clearly visible so the doctor can map indices to anatomical regions.

Usage (standalone):
    python -m utils.landmark_preview <image_path> <landmarks_json_str>

Usage (from Flask):
    from utils.landmark_preview import draw_numbered_preview
    png_bytes = draw_numbered_preview(image_bgr, points)
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import List, Tuple, Union

import cv2
import numpy as np


# ── Colour palette: 20 visually distinct colours (BGR) ──────────────────────
# Chosen to be distinguishable against both dark and light backgrounds.
_PALETTE = [
    (  0, 200, 255),  #  0 – yellow-ish cyan
    (  0, 128, 255),  #  1 – orange
    ( 50, 205,  50),  #  2 – lime green
    (255,   0, 128),  #  3 – hot pink
    (255, 200,   0),  #  4 – sky blue
    (  0,   0, 255),  #  5 – red
    (128,   0, 255),  #  6 – magenta
    (  0, 255, 128),  #  7 – spring green
    (255, 128,   0),  #  8 – deep blue
    (  0, 255, 255),  #  9 – bright yellow
    (180,   0, 180),  # 10 – purple
    ( 30, 180, 255),  # 11 – golden
    (  0, 180,   0),  # 12 – medium green
    (100, 100, 255),  # 13 – salmon
    (255,  60, 255),  # 14 – violet
    (  0, 100, 200),  # 15 – dark orange
    (200, 255,   0),  # 16 – aqua-lime
    (200,   0,   0),  # 17 – dark blue
    ( 60, 255, 200),  # 18 – mint
    (255, 180, 100),  # 19 – powder blue
]


def _parse_points(
    landmarks: Union[str, list],
) -> List[Tuple[int, int]]:
    """Accept JSON string, list of dicts, or list of [x,y] / (x,y) tuples."""
    if isinstance(landmarks, str):
        landmarks = json.loads(landmarks)
    pts: List[Tuple[int, int]] = []
    for p in landmarks:
        if isinstance(p, dict):
            pts.append((int(p["x"]), int(p["y"])))
        else:
            pts.append((int(p[0]), int(p[1])))
    return pts


def draw_numbered_preview(
    image_bgr: np.ndarray,
    landmarks: Union[str, list],
    *,
    dot_radius_frac: float = 0.012,   # fraction of min(W,H) for the dot
    font_scale_frac: float = 0.030,   # fraction of min(W,H) for text height
    min_dot_r: int = 8,
    max_dot_r: int = 28,
    outline_thickness: int = 2,
) -> np.ndarray:
    """
    Draw 0-based index numbers on every landmark.

    Returns a new BGR ndarray (original is never modified).
    """
    points = _parse_points(landmarks)
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()

    short = min(w, h)
    dot_r  = int(round(max(min_dot_r, min(max_dot_r, short * dot_radius_frac))))
    # font scale: OpenCV text height ≈ font_scale * 30px, calibrate empirically
    font_scale = max(0.45, min(1.4, short * font_scale_frac / 30.0))
    font_thick = max(1, int(font_scale * 2))

    for idx, (px, py) in enumerate(points):
        color = _PALETTE[idx % len(_PALETTE)]

        # White outline halo (makes dot visible on any background)
        cv2.circle(out, (px, py), dot_r + outline_thickness, (255, 255, 255), -1)
        # Filled colour dot
        cv2.circle(out, (px, py), dot_r, color, -1)

        label = str(idx)  # 0-based
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick
        )
        # Centre the label inside the dot
        tx = px - tw // 2
        ty = py + th // 2

        # Black shadow for readability
        cv2.putText(
            out, label,
            (tx + 1, ty + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thick + 1,
            cv2.LINE_AA,
        )
        # White text on top
        cv2.putText(
            out, label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    # ── Legend strip at the bottom ──────────────────────────────────────────
    n = len(points)
    legend_h = max(28, dot_r * 3)
    cols_per_row = 10
    rows = (n + cols_per_row - 1) // cols_per_row
    legend_total_h = legend_h * rows

    legend = np.full((legend_total_h, w, 3), 30, dtype=np.uint8)

    cell_w = w // cols_per_row
    leg_dot_r = max(6, dot_r // 2)
    leg_font_scale = max(0.35, font_scale * 0.7)
    leg_thick = max(1, font_thick - 1)

    for idx, (px, py) in enumerate(points):
        row = idx // cols_per_row
        col = idx % cols_per_row
        color = _PALETTE[idx % len(_PALETTE)]

        cx = col * cell_w + cell_w // 5
        cy = row * legend_h + legend_h // 2

        cv2.circle(legend, (cx, cy), leg_dot_r, color, -1)
        cv2.circle(legend, (cx, cy), leg_dot_r, (200, 200, 200), 1)

        label = str(idx)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, leg_font_scale, leg_thick
        )
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(
            legend, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, leg_font_scale,
            (255, 255, 255), leg_thick, cv2.LINE_AA,
        )

    out = np.vstack([out, legend])
    return out


def preview_to_png_bytes(
    image_bgr: np.ndarray,
    landmarks: Union[str, list],
) -> bytes:
    """Return preview image encoded as PNG bytes (for Flask send_file)."""
    preview = draw_numbered_preview(image_bgr, landmarks)
    ok, buf = cv2.imencode(".png", preview)
    if not ok:
        raise RuntimeError("cv2.imencode failed for preview PNG")
    return buf.tobytes()
