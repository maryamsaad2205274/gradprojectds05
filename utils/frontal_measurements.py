import math
from typing import Any, Dict, List, Sequence, Tuple, Union


Point = Union[Sequence[float], Dict[str, Any]]


def _normalize_points(points: List[Point]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points:
        if isinstance(p, dict):
            out.append((float(p["x"]), float(p["y"])))
        else:
            out.append((float(p[0]), float(p[1])))
    return out


def _pt(pts: List[Tuple[float, float]], idx_1based: int) -> Tuple[float, float]:
    # Landmark IDs are 1–34; python indices are 0–33.
    return pts[idx_1based - 1]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # Angle of line segment a->b vs +x axis (degrees).
    return float(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))


def _angle_diff_deg(theta1: float, theta2: float) -> float:
    # Minimal absolute difference between two angles in degrees (0..180).
    d = abs(theta1 - theta2) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return float(d)


def calculate_frontal_measurements(landmarks: List[Point], *, parallel_threshold_deg: float = 5.0) -> Dict[str, Any]:
    """
    Compute frontal non-smile measurements from 34 landmarks.

    Landmarks are expected as length-34 list of either:
      - [[x, y], ...] OR
      - [{"x": x, "y": y}, ...]
    """
    if not isinstance(landmarks, list):
        raise ValueError("landmarks must be a list")
    if len(landmarks) < 34:
        raise ValueError(f"Expected 34 landmarks, got {len(landmarks)}")

    pts = _normalize_points(landmarks[:34])

    # Required points (IDs)
    p8 = _pt(pts, 8)
    p32 = _pt(pts, 32)
    p20 = _pt(pts, 20)
    p22 = _pt(pts, 22)

    p11 = _pt(pts, 11)
    p16 = _pt(pts, 16)
    p18 = _pt(pts, 18)
    p21 = _pt(pts, 21)
    p23 = _pt(pts, 23)
    p24 = _pt(pts, 24)

    # Bizygomatic / facial width: choose wider of (1,28) vs (2,27)
    w_1_28 = _dist(_pt(pts, 1), _pt(pts, 28))
    w_2_27 = _dist(_pt(pts, 2), _pt(pts, 27))
    if w_1_28 >= w_2_27:
        facial_width = w_1_28
        facial_width_pair = (1, 28)
    else:
        facial_width = w_2_27
        facial_width_pair = (2, 27)

    mandibular_width = _dist(_pt(pts, 3), _pt(pts, 26))
    facial_height = _dist(p11, p24)  # 11 -> 24

    # Lip-related
    lip_length_at_rest = _dist(p16, p21)  # corrected: vertical upper lip length proxy
    mouth_width = _dist(p20, p22)  # keep separate measurement
    interlabial_gap = _dist(p21, p23)  # approximate, per user note

    # Vertical thirds proxies
    middle_third = _dist(p11, p16)
    lower_third = _dist(p16, p24)

    # Lip ratios
    upper_lip_height = _dist(p18, p21)
    lower_lip_height = _dist(p21, p23)

    # Line alignment / parallelism
    interpupillary_angle = _angle_deg(p8, p32)
    commissure_angle = _angle_deg(p20, p22)
    parallel_delta = _angle_diff_deg(interpupillary_angle, commissure_angle)

    interpupillary_length = _dist(p8, p32)
    commissure_length = _dist(p20, p22)

    # Facial midline (mean x of specified midline-related points)
    midline_ids = [11, 12, 14, 16, 18, 21, 23, 24]
    midline_xs = [_pt(pts, i)[0] for i in midline_ids]
    midline_x = float(sum(midline_xs) / len(midline_xs))
    midline_rms_dev = float(
        math.sqrt(sum((x - midline_x) ** 2 for x in midline_xs) / len(midline_xs))
    )

    # Rule of fifths (approximate components)
    fifth = facial_width / 5.0 if facial_width > 0 else None
    left_eye_width = _dist(_pt(pts, 6), _pt(pts, 10))
    right_eye_width = _dist(_pt(pts, 30), _pt(pts, 34))
    intercanthal = _dist(_pt(pts, 10), _pt(pts, 30))

    def _safe_ratio(a: float, b: float) -> float:
        if b == 0:
            return float("nan")
        return float(a / b)

    vertical = {
        "middle_third_px": middle_third,
        "lower_third_px": lower_third,
        "middle_to_lower_third_ratio": _safe_ratio(middle_third, lower_third),
        "upper_lip_height_px": upper_lip_height,
        "lower_lip_height_px": lower_lip_height,
        "upper_to_lower_lip_ratio": _safe_ratio(upper_lip_height, lower_lip_height),
        "lip_length_at_rest_px": lip_length_at_rest,
        "mouth_width_px": mouth_width,
        "interlabial_gap_px": {
            "value": interlabial_gap,
            "note": "Approximate: point 23 is lower-lip center, not necessarily the upper border of lower lip.",
        },
        "interpupillary_line_alignment": {
            "angle_deg": interpupillary_angle,
            "abs_angle_from_horizontal_deg": abs(interpupillary_angle),
            "length_px": interpupillary_length,
            "points": [8, 32],
        },
        "commissure_line_alignment": {
            "angle_deg": commissure_angle,
            "abs_angle_from_horizontal_deg": abs(commissure_angle),
            "length_px": commissure_length,
            "points": [20, 22],
        },
        "interpupillary_vs_commissure_parallel": {
            "delta_deg": parallel_delta,
            "threshold_deg": float(parallel_threshold_deg),
            "parallel": bool(parallel_delta <= parallel_threshold_deg),
        },
    }

    horizontal = {
        "facial_midline": {
            "midline_x": midline_x,
            "rms_deviation_px": midline_rms_dev,
            "upper_lip_center_offset_px": float(p18[0] - midline_x),
            "chin_offset_px": float(p24[0] - midline_x),
            "points_used": midline_ids,
        },
        "rule_of_fifths": {
            "facial_width_px": facial_width,
            "fifth_px": fifth,
            "left_eye_width_px": left_eye_width,
            "intercanthal_px": intercanthal,
            "right_eye_width_px": right_eye_width,
            "left_eye_over_fifth": _safe_ratio(left_eye_width, fifth) if fifth else float("nan"),
            "intercanthal_over_fifth": _safe_ratio(intercanthal, fifth) if fifth else float("nan"),
            "right_eye_over_fifth": _safe_ratio(right_eye_width, fifth) if fifth else float("nan"),
        },
        "facial_width_bizygomatic_px": {
            "value": facial_width,
            "chosen_pair": list(facial_width_pair),
            "candidates_px": {"1-28": w_1_28, "2-27": w_2_27},
        },
        "mandibular_width_bigonial_px": mandibular_width,
        "bizygomatic_to_bigonial_ratio": _safe_ratio(facial_width, mandibular_width),
        "facial_index_height_to_width": _safe_ratio(facial_height, facial_width),
        "facial_height_px": facial_height,
    }

    return {
        "model": "FRONT_NS",
        "num_landmarks": 34,
        "vertical": vertical,
        "horizontal": horizontal,
    }


def frontal_pdf_interpretation_rows(data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    PDF-friendly rows: (title, value string, short interpretation).
    Uses project thresholds where defined (e.g. interpupillary vs commissure parallelism).
    """
    v = data.get("vertical") or {}
    h = data.get("horizontal") or {}
    rows: List[Tuple[str, str, str]] = []

    def _fmt_px(x: Any) -> str:
        try:
            return f"{float(x):.1f} px"
        except (TypeError, ValueError):
            return "—"

    def _fmt_ratio(x: Any) -> str:
        try:
            if x != x:  # NaN
                return "—"
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "—"

    mt = v.get("middle_third_px")
    lt = v.get("lower_third_px")
    mlr = v.get("middle_to_lower_third_ratio")
    rows.append(
        (
            "Middle vs lower facial third (proxy)",
            f"Middle {_fmt_px(mt)}, lower {_fmt_px(lt)}, ratio {_fmt_ratio(mlr)}",
            "Approximate vertical proportions; interpret with clinical context and growth pattern.",
        )
    )

    ulr = v.get("upper_to_lower_lip_ratio")
    rows.append(
        (
            "Upper / lower lip height ratio",
            _fmt_ratio(ulr),
            "Values far from ~1.0 may suggest disproportion; correlate with dental and soft-tissue exam.",
        )
    )

    ip = v.get("interpupillary_line_alignment") or {}
    cm = v.get("commissure_line_alignment") or {}
    par = v.get("interpupillary_vs_commissure_parallel") or {}
    thr = float(par.get("threshold_deg") or 5.0)
    delta = par.get("delta_deg")
    try:
        d = float(delta)
        ok = d <= thr
        interp = (
            f"Parallel within {thr:.1f}° (lines appear level)."
            if ok
            else f"Difference {d:.1f}° exceeds typical cosmetic parallelism threshold ({thr:.1f}°); clinical correlation suggested."
        )
    except (TypeError, ValueError):
        interp = "Could not assess parallelism."
    rows.append(
        (
            "Interpupillary vs commissure line",
            f"Interpupillary {float(ip.get('angle_deg', 0)):.1f}°, commissure {float(cm.get('angle_deg', 0)):.1f}°",
            interp,
        )
    )

    fm = h.get("facial_midline") or {}
    rms = fm.get("rms_deviation_px")
    fw = (h.get("facial_width_bizygomatic_px") or {}).get("value")
    try:
        rms_f = float(rms)
        fw_f = float(fw) if fw is not None else 0.0
        rel = (rms_f / fw_f * 100.0) if fw_f > 0 else None
        if rel is not None and rel <= 2.5:
            mid_txt = "Midline landmarks cluster closely; gross asymmetry unlikely from this proxy."
        elif rel is not None:
            mid_txt = "Midline deviation index is elevated; assess chin, nose, and dental midlines clinically."
        else:
            mid_txt = "Midline deviation could not be normalized to face width."
    except (TypeError, ValueError):
        mid_txt = "Midline summary unavailable."

    rows.append(
        (
            "Facial midline (RMS deviation)",
            _fmt_px(rms),
            mid_txt,
        )
    )

    rof = h.get("rule_of_fifths") or {}
    fifth = rof.get("fifth_px")
    rows.append(
        (
            "Rule of fifths (approximate)",
            f"Ideal fifth width {_fmt_px(fifth)}",
            "Compare eye widths and intercanthal distance to one-fifth of bizygomatic width; rough aesthetic guide only.",
        )
    )

    rows.append(("Facial width (bizygomatic)", _fmt_px((h.get("facial_width_bizygomatic_px") or {}).get("value")), "Useful for proportional checks; correlate with arch form and transverse skeletal pattern."))
    rows.append(("Mandibular width (bigonial)", _fmt_px(h.get("mandibular_width_bigonial_px")), "Wide or narrow lower face relative to cheekbones informs transverse diagnosis."))

    bz = h.get("bizygomatic_to_bigonial_ratio")
    rows.append(
        (
            "Bizygomatic / bigonial ratio",
            _fmt_ratio(bz),
            "Typical values often near ~1.2–1.4 in many samples; extremes suggest disproportion worth clinical review.",
        )
    )

    fi = h.get("facial_index_height_to_width")
    rows.append(
        (
            "Facial index (height / width)",
            _fmt_ratio(fi),
            "Longer narrower vs shorter wider facial form; integrate with vertical skeletal and growth assessment.",
        )
    )

    gap = (v.get("interlabial_gap_px") or {}).get("value")
    rows.append(
        (
            "Interlabial gap (approx.)",
            _fmt_px(gap),
            "Approximate soft-tissue opening at rest; verify with clinical lip posture exam.",
        )
    )

    return rows

