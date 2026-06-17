"""
Treatment Simulation Engine — landmark overlay / profile tracing.

Primary output: draw the original soft-tissue profile curve and the
doctor-edited curve on top of the *unchanged* patient photo.
No pixel warping.  No distortion.  Clinically honest.

Pipeline (overlay mode)
-----------------------
1. Parse original and edited landmark lists  (20 points each)
2. Draw original profile curve  (dashed gray-blue spline)
3. Draw edited profile curve    (solid green spline)
4. Draw movement arrows         (red, for moved points only)
5. Draw landmark dots           (editable = colored, stable = gray)
6. Add a warning banner at the bottom
7. Return the composite image (original pixels + vector overlay)

The warping engine (run_local_warp) is retained for reference but is
no longer the primary simulation output.

DO NOT change HRNet, XGBoost, preprocessing, feature generation or inference.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.interpolate import RBFInterpolator as _RBFInterpolator
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LANDMARKS   = 20       # side model landmark count
BORDER_MARGIN = 8        # px from image edge for border anchors
PREVIEW_MAX_W = 640      # max width for AJAX preview (speed)
SAVE_MAX_W    = 1600     # max width for high-quality save
_NUMPY_CHUNK  = 60_000   # pixel chunk size for numpy TPS


# ---------------------------------------------------------------------------
# Landmark parsing helpers
# ---------------------------------------------------------------------------

def parse_landmarks(raw) -> List[Tuple[int, int]]:
    """Accept JSON str, list-of-dicts {x,y}, or list-of-[x,y]."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    pts: List[Tuple[int, int]] = []
    for p in raw:
        if isinstance(p, dict):
            pts.append((int(p["x"]), int(p["y"])))
        else:
            pts.append((int(p[0]), int(p[1])))
    return pts


def _pts_to_float64(raw) -> np.ndarray:
    """Convert any landmark format to Nx2 float64 numpy array."""
    return np.array(parse_landmarks(raw), dtype=np.float64)


# ---------------------------------------------------------------------------
# Border anchors — fixed points spread around image boundary
# ---------------------------------------------------------------------------

def _border_anchors(w: int, h: int) -> np.ndarray:
    m = BORDER_MARGIN
    pts = []
    # Top + bottom rows
    for x in [m, w // 4, w // 2, 3 * w // 4, w - m]:
        pts.append([float(x), float(m)])
        pts.append([float(x), float(h - m)])
    # Left + right columns (skip corners)
    for y in [h // 4, h // 2, 3 * h // 4]:
        pts.append([float(m),     float(y)])
        pts.append([float(w - m), float(y)])
    return np.array(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Ring anchors — fixed circle of points around a moved landmark
# ---------------------------------------------------------------------------

def _ring_anchors(centre: np.ndarray, radius: float, n: int = 12) -> np.ndarray:
    """Return `n` evenly-spaced points at `radius` distance from `centre`."""
    angles = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]
    return centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])


# ---------------------------------------------------------------------------
# TPS — scipy backend (fast)
# ---------------------------------------------------------------------------

def _tps_scipy(image: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Warp `image` so that pixels at `src` positions come from `dst` positions."""
    h, w = image.shape[:2]
    rbf_x = _RBFInterpolator(dst, src[:, 0], kernel="thin_plate_spline", smoothing=0.0)
    rbf_y = _RBFInterpolator(dst, src[:, 1], kernel="thin_plate_spline", smoothing=0.0)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                          np.arange(h, dtype=np.float64))
    q = np.column_stack([gx.ravel(), gy.ravel()])
    map_x = rbf_x(q).reshape(h, w).astype(np.float32)
    map_y = rbf_y(q).reshape(h, w).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# TPS — numpy-only backend (fallback, chunked to limit RAM)
# ---------------------------------------------------------------------------

def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, r2 * np.log(np.maximum(r2, 1e-20)), 0.0)


def _solve_tps(ctrl: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = ctrl.shape[0]
    d = ctrl[:, None, :] - ctrl[None, :, :]
    r2 = (d ** 2).sum(-1)
    K = _tps_kernel(r2)
    P = np.hstack([np.ones((N, 1)), ctrl])
    Z = np.zeros((3, 3))
    M = np.vstack([np.hstack([K, P]), np.hstack([P.T, Z])])
    rhs = np.concatenate([values, [0.0, 0.0, 0.0]])
    try:
        c = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return c[:N], c[N:]


def _tps_numpy(image: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    w_x, a_x = _solve_tps(dst, src[:, 0])
    w_y, a_y = _solve_tps(dst, src[:, 1])
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                          np.arange(h, dtype=np.float64))
    q = np.column_stack([gx.ravel(), gy.ravel()])
    M_pts = q.shape[0]
    maps_x, maps_y = [], []
    for start in range(0, M_pts, _NUMPY_CHUNK):
        end = min(start + _NUMPY_CHUNK, M_pts)
        qc = q[start:end]
        d = qc[:, None, :] - dst[None, :, :]
        r2 = (d ** 2).sum(-1)
        K = _tps_kernel(r2)
        maps_x.append(a_x[0] + a_x[1] * qc[:, 0] + a_x[2] * qc[:, 1] + K @ w_x)
        maps_y.append(a_y[0] + a_y[1] * qc[:, 0] + a_y[2] * qc[:, 1] + K @ w_y)
    map_x = np.concatenate(maps_x).reshape(h, w).astype(np.float32)
    map_y = np.concatenate(maps_y).reshape(h, w).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Public API — local point-drag warp
# ---------------------------------------------------------------------------

def run_local_warp(
    image_bgr: np.ndarray,
    original_pts_raw,
    edited_pts_raw,
    *,
    influence_radius: float = 120.0,
    strength: float = 1.0,
    full_quality: bool = False,
) -> Dict:
    """
    Warp ``image_bgr`` using direct landmark point dragging (Liquify style).

    Parameters
    ----------
    image_bgr        : original BGR image (never modified in-place)
    original_pts_raw : 20 original HRNet landmark positions
                       (JSON str, list-of-{x,y}, or list-of-[x,y])
    edited_pts_raw   : 20 doctor-edited positions (same format)
    influence_radius : pixel radius of the soft deformation zone
                       (in original-image coordinates)
    strength         : blend factor 0.0–1.0  (1.0 = full warp)
    full_quality     : False → scale to PREVIEW_MAX_W first (fast AJAX)
                       True  → scale to SAVE_MAX_W (high-quality save)

    Returns
    -------
    {
      "warped_bgr"          : np.ndarray,
      "simulated_landmarks" : list[tuple[int,int]],   # original-image coords
      "backend"             : "scipy" | "numpy" | "none",
    }
    """
    orig_pts = _pts_to_float64(original_pts_raw)  # N × 2
    edit_pts = _pts_to_float64(edited_pts_raw)    # N × 2

    h0, w0 = image_bgr.shape[:2]

    # ── Detect moved landmarks ──────────────────────────────────────────────
    displacement = np.linalg.norm(orig_pts - edit_pts, axis=1)  # N
    moved_mask   = displacement > 0.5                            # bool N
    n_moved      = int(moved_mask.sum())

    sim_pts = [(int(round(x)), int(round(y))) for x, y in edit_pts]

    if n_moved == 0:
        return {
            "warped_bgr":          image_bgr.copy(),
            "simulated_landmarks": sim_pts,
            "backend":             "none",
        }

    # ── Scale image for processing ──────────────────────────────────────────
    max_w = SAVE_MAX_W if full_quality else PREVIEW_MAX_W
    if w0 > max_w:
        scale   = max_w / w0
        proc_w  = int(w0 * scale)
        proc_h  = int(h0 * scale)
        proc_img = cv2.resize(image_bgr, (proc_w, proc_h),
                              interpolation=cv2.INTER_AREA)
    else:
        scale    = 1.0
        proc_w, proc_h = w0, h0
        proc_img = image_bgr.copy()

    # Scale landmark coordinates and influence radius
    orig_s   = orig_pts * scale
    edit_s   = edit_pts * scale
    radius_s = influence_radius * scale

    # ── Build TPS control points ────────────────────────────────────────────
    # Start with all 20 landmarks: moved ones go to new position, others stay
    src_ctrl = list(orig_s)   # source positions (original image)
    dst_ctrl = list(edit_s)   # destination positions (edited by doctor)

    # Ring anchors around each moved point (identity — clamp displacement)
    ring_r = radius_s * 1.30
    for i in range(len(orig_pts)):
        if moved_mask[i]:
            ring = _ring_anchors(orig_s[i], ring_r, n=12)
            # Clamp ring points to image bounds
            ring[:, 0] = np.clip(ring[:, 0], 0.0, proc_w - 1)
            ring[:, 1] = np.clip(ring[:, 1], 0.0, proc_h - 1)
            for rp in ring:
                src_ctrl.append(rp)
                dst_ctrl.append(rp)  # identity: this point does not move

    # Border anchors (always fixed)
    for bp in _border_anchors(proc_w, proc_h):
        src_ctrl.append(bp)
        dst_ctrl.append(bp)

    src_ctrl = np.array(src_ctrl, dtype=np.float64)
    dst_ctrl = np.array(dst_ctrl, dtype=np.float64)

    # ── TPS warp ─────────────────────────────────────────────────────────────
    warp_fn = _tps_scipy if _SCIPY else _tps_numpy
    backend  = "scipy"  if _SCIPY else "numpy"
    warped   = warp_fn(proc_img, src_ctrl, dst_ctrl)

    # ── Influence mask: Gaussian soft circles around moved points ─────────────
    mask = np.zeros((proc_h, proc_w), dtype=np.float32)
    for i in range(len(orig_pts)):
        if moved_mask[i]:
            cx = int(round(orig_s[i, 0]))
            cy = int(round(orig_s[i, 1]))
            cv2.circle(mask, (cx, cy), int(radius_s), 1.0, -1)

    # Gaussian blur for smooth falloff at the mask boundary
    blur_k = max(3, int(radius_s * 0.55)) | 1   # must be odd
    mask   = cv2.GaussianBlur(mask, (blur_k, blur_k), radius_s / 2.5)
    mask   = np.clip(mask * float(strength), 0.0, 1.0)

    # ── Blend warped + original ───────────────────────────────────────────────
    m3      = mask[:, :, np.newaxis]
    blended = warped.astype(np.float32) * m3 + proc_img.astype(np.float32) * (1.0 - m3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # ── Scale back to original resolution ────────────────────────────────────
    if scale < 1.0:
        final = cv2.resize(blended, (w0, h0), interpolation=cv2.INTER_LINEAR)
    else:
        final = blended

    return {
        "warped_bgr":          final,
        "simulated_landmarks": sim_pts,
        "backend":             backend,
    }


# ---------------------------------------------------------------------------
# Utility — encode preview as base64 PNG
# ---------------------------------------------------------------------------

def encode_preview_png(image_bgr: np.ndarray) -> str:
    """Encode BGR image as base64 PNG string for JSON response."""
    import base64
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Utility — before / after comparison image (used when saving)
# ---------------------------------------------------------------------------

def draw_comparison(
    original_bgr: np.ndarray,
    warped_bgr:   np.ndarray,
    orig_pts:     List[Tuple[int, int]],
    sim_pts:      List[Tuple[int, int]],
    *,
    max_w: int = 600,
) -> np.ndarray:
    """Return a side-by-side BEFORE | AFTER image with landmark dots."""

    def _resize(img, mw):
        h, w = img.shape[:2]
        if w > mw:
            img = cv2.resize(img, (mw, int(h * mw / w)), interpolation=cv2.INTER_AREA)
        return img

    def _draw_dots(img, pts, color, radius=4):
        out = img.copy()
        sh, sw = out.shape[:2]
        oh, ow = img.shape[:2]
        for x, y in pts:
            sx = int(x * sw / ow)
            sy = int(y * sh / oh)
            cv2.circle(out, (sx, sy), radius, (255, 255, 255), -1)
            cv2.circle(out, (sx, sy), radius - 1, color, -1)
        return out

    orig_small = _resize(original_bgr, max_w)
    warp_small = _resize(warped_bgr,   max_w)

    orig_vis = _draw_dots(orig_small, orig_pts, (0, 140, 255))    # orange
    warp_vis = _draw_dots(warp_small, sim_pts,  (0, 200, 100))    # green

    # Equal height
    th = max(orig_vis.shape[0], warp_vis.shape[0])
    def _pad_h(img, target_h):
        dh = target_h - img.shape[0]
        if dh > 0:
            img = np.pad(img, ((0, dh), (0, 0), (0, 0)), mode="edge")
        return img
    orig_vis = _pad_h(orig_vis, th)
    warp_vis = _pad_h(warp_vis, th)

    div    = np.full((th, 3, 3), 200, dtype=np.uint8)
    canvas = np.hstack([orig_vis, div, warp_vis])

    def _label(img, text, x, y):
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    canvas = _label(canvas, "BEFORE", 10, 28)
    canvas = _label(canvas, "AFTER",  orig_vis.shape[1] + 6, 28)

    banner_h = 30
    banner   = np.full((banner_h, canvas.shape[1], 3), 30, dtype=np.uint8)
    warn     = "SIMULATION ONLY — NOT A FINAL TREATMENT PLAN"
    (tw, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    bx = max(0, (canvas.shape[1] - tw) // 2)
    cv2.putText(banner, warn, (bx, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (60, 180, 255), 1, cv2.LINE_AA)

    return np.vstack([canvas, banner])


# ---------------------------------------------------------------------------
# Primary API — landmark overlay / profile tracing (no pixel warping)
# ---------------------------------------------------------------------------

# Editable landmark indices (lip + chin + jaw)
_EDITABLE: set = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

# BGR colour palette for the overlay
_COL_ORIG_CURVE  = (200, 150,  80)   # dashed blue-gray  — original profile
_COL_EDIT_CURVE  = ( 60, 210,  90)   # solid green        — edited profile
_COL_ARROW       = ( 50,  80, 255)   # red-orange         — movement vectors
_COL_EDIT_DOT    = ( 60, 210,  90)   # green              — moved editable dot
_COL_ORIG_DOT    = (180, 150,  90)   # muted blue         — un-moved editable dot
_COL_STABLE_DOT  = (140, 140, 140)   # gray               — stable points
_COL_GHOST       = (180, 180, 180)   # light gray         — ghost at original pos


def _catmull_rom_chain(
    pts: List[Tuple[int, int]],
    steps: int = 20,
) -> np.ndarray:
    """
    Return a dense chain of (x, y) int points for a Catmull-Rom spline
    through `pts`.  Suitable for cv2.polylines.
    """
    n = len(pts)
    if n == 0:
        return np.empty((0, 2), dtype=np.int32)
    if n == 1:
        return np.array([[pts[0][0], pts[0][1]]], dtype=np.int32)

    result: List[Tuple[int, int]] = []
    for i in range(n - 1):
        p0 = pts[max(0, i - 1)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(n - 1, i + 2)]

        for s in range(steps + 1):
            t  = s / steps
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append((int(round(x)), int(round(y))))

    return np.array(result, dtype=np.int32).reshape(-1, 1, 2)


def _draw_dashed_polyline(
    img: np.ndarray,
    pts_chain: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 12,
    gap: int = 7,
) -> None:
    """Draw a dashed polyline through a dense chain of points."""
    if len(pts_chain) == 0:
        return
    flat = pts_chain.reshape(-1, 2)
    draw = True
    count = 0
    for i in range(len(flat) - 1):
        if draw:
            cv2.line(img, tuple(flat[i]), tuple(flat[i + 1]), color, thickness, cv2.LINE_AA)
        count += 1
        if draw and count >= dash:
            draw = False
            count = 0
        elif not draw and count >= gap:
            draw = True
            count = 0


def draw_overlay_image(
    image_bgr: np.ndarray,
    original_pts_raw,
    edited_pts_raw,
    *,
    show_arrows:    bool = True,
    show_indices:   bool = True,
    show_orig_curve: bool = True,
    max_w: int = 0,           # 0 = no scaling
) -> np.ndarray:
    """
    Draw a clinical profile-tracing overlay on `image_bgr`.

    The original patient photo is **never modified** (a copy is used).
    Vector elements drawn:
      • original soft-tissue profile curve  (dashed blue-gray)
      • edited soft-tissue profile curve    (solid green)
      • movement arrows                     (red, moved points only)
      • landmark dots                       (editable=colored, stable=gray)
      • index labels
      • warning banner at the bottom

    Parameters
    ----------
    image_bgr        : original BGR image (read-only)
    original_pts_raw : 20 original HRNet landmark positions
    edited_pts_raw   : 20 doctor-edited positions
    show_arrows      : draw displacement arrows
    show_indices     : draw index numbers beside dots
    show_orig_curve  : draw the original profile curve
    max_w            : if > 0, scale the output image to this width

    Returns
    -------
    BGR ndarray — original image pixels + vector overlay + banner
    """
    orig_pts = parse_landmarks(original_pts_raw)  # [(x,y), …]  N=20
    edit_pts = parse_landmarks(edited_pts_raw)

    h0, w0 = image_bgr.shape[:2]

    # Optional downscale for saved thumbnails
    if max_w > 0 and w0 > max_w:
        scale  = max_w / w0
        img    = cv2.resize(image_bgr, (int(w0 * scale), int(h0 * scale)),
                            interpolation=cv2.INTER_AREA)
        orig_s = [(int(x * scale), int(y * scale)) for x, y in orig_pts]
        edit_s = [(int(x * scale), int(y * scale)) for x, y in edit_pts]
    else:
        img    = image_bgr.copy()
        orig_s = [(int(x), int(y)) for x, y in orig_pts]
        edit_s = [(int(x), int(y)) for x, y in edit_pts]

    out = img.copy()

    # ── Original profile curve (dashed, blue-gray) ────────────────────────
    if show_orig_curve and len(orig_s) >= 2:
        chain = _catmull_rom_chain(orig_s, steps=18)
        _draw_dashed_polyline(out, chain, _COL_ORIG_CURVE, thickness=2)

    # ── Edited profile curve (solid, green) ───────────────────────────────
    if len(edit_s) >= 2:
        chain = _catmull_rom_chain(edit_s, steps=18)
        cv2.polylines(out, [chain], False, _COL_EDIT_CURVE, 2, cv2.LINE_AA)

    # ── Movement arrows ───────────────────────────────────────────────────
    if show_arrows:
        for i, (op, ep) in enumerate(zip(orig_s, edit_s)):
            dx, dy = ep[0] - op[0], ep[1] - op[1]
            if abs(dx) + abs(dy) < 2:
                continue
            cv2.arrowedLine(out, op, ep, _COL_ARROW, 1,
                            cv2.LINE_AA, tipLength=0.35)

    # ── Landmark dots ─────────────────────────────────────────────────────
    for i, (op, ep) in enumerate(zip(orig_s, edit_s)):
        dx, dy  = ep[0] - op[0], ep[1] - op[1]
        moved   = abs(dx) + abs(dy) >= 2
        editable = i in _EDITABLE

        # Ghost dot at original position (only if moved)
        if moved:
            cv2.circle(out, op, 3, _COL_GHOST, -1, cv2.LINE_AA)

        # Edited dot
        r = 5 if editable else 3
        if editable:
            color = _COL_EDIT_DOT if moved else _COL_ORIG_DOT
        else:
            color = _COL_STABLE_DOT

        cv2.circle(out, ep, r + 1, (255, 255, 255), -1, cv2.LINE_AA)  # white halo
        cv2.circle(out, ep, r,     color,            -1, cv2.LINE_AA)

        # Index label
        if show_indices:
            label = str(i)
            fs    = 0.38 if editable else 0.30
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            tx = ep[0] - tw // 2
            ty = ep[1] + th // 2
            cv2.putText(out, label, (tx + 1, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(out, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Warning banner ────────────────────────────────────────────────────
    banner_h = 28
    bw       = out.shape[1]
    banner   = np.full((banner_h, bw, 3), 22, dtype=np.uint8)
    warn     = (
        "SIMULATION ONLY — Profile tracing illustration. "
        "Not a clinical treatment prediction."
    )
    fs_b = 0.38
    (tw, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, fs_b, 1)
    bx = max(4, (bw - tw) // 2)
    cv2.putText(banner, warn, (bx, 19),
                cv2.FONT_HERSHEY_SIMPLEX, fs_b, (100, 220, 255), 1, cv2.LINE_AA)

    return np.vstack([out, banner])
