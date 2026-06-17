"""
Build doctor-friendly measurement cards and AI insight rows for the analysis result page.
Uses existing side interpreters and frontal measurement dictionaries — no raw coordinates.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.measurements import (
    angle_ABC,
    interpret_mentolabial,
    interpret_nasiolabial,
    interpret_profile_convexity,
    interpret_total_facial_convexity,
    normalize_points,
)

SIDE_SPECS: List[Tuple[str, str, Tuple[int, int, int], Any, str]] = [
    ("nasiolabial", "Nasiolabial angle", (7, 8, 10), interpret_nasiolabial, "90° – 110°"),
    ("profile_convexity", "Profile convexity angle", (3, 8, 17), interpret_profile_convexity, "151° – 171°"),
    ("total_facial_convexity", "Total facial convexity angle", (3, 5, 17), interpret_total_facial_convexity, "127° – 137°"),
    ("mentolabial", "Mentolabial angle", (15, 16, 17), interpret_mentolabial, "110° – 130°"),
]


def _badge_tone(status_text: str) -> str:
    t = (status_text or "").lower()
    if "approximate" in t:
        return "approximate"
    if "normal" in t:
        return "normal"
    if "increased" in t:
        return "increased"
    if "decreased" in t:
        return "reduced"
    if "high" in t or "protrud" in t or "elevated" in t:
        return "high"
    if "low" in t or "retrud" in t or "narrow" in t:
        return "low"
    return "approximate"


def _short_status_label(status_text: str) -> str:
    t = (status_text or "").lower()
    if "approximate" in t:
        return "Approximate"
    if "normal" in t:
        return "Normal"
    if "increased" in t:
        return "Increased"
    if "decreased" in t:
        return "Reduced"
    if "high" in t or "protrud" in t or "elevated" in t:
        return "High"
    if "low" in t or "retrud" in t:
        return "Low"
    return "Approximate"


def _fmt_num(v: Any, decimals: int = 1) -> Optional[float]:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return None


def build_side_measurement_cards(side_points: Any) -> List[Dict[str, Any]]:
    if not side_points:
        return []
    try:
        pts = normalize_points(side_points)
    except Exception:
        return []

    cards: List[Dict[str, Any]] = []
    for key, title, idxs, interpreter, normal_range in SIDE_SPECS:
        try:
            ang = float(angle_ABC(pts[idxs[0] - 1], pts[idxs[1] - 1], pts[idxs[2] - 1]))
            interp = interpreter(ang)
            status = interp.get("status", "")
            cards.append(
                {
                    "vis_key": key,
                    "name": title,
                    "value": f"{ang:.1f}°",
                    "value_raw": ang,
                    "unit": "°",
                    "normal_range": normal_range,
                    "status_label": _short_status_label(status),
                    "badge": _badge_tone(status),
                    "note": (interp.get("meaning") or "")[:200],
                }
            )
        except Exception:
            continue
    return cards


def _front_badge_parallel(delta: float, threshold: float) -> Tuple[str, str]:
    if delta <= threshold:
        return "normal", "Normal"
    if delta <= threshold * 2:
        return "approximate", "Approximate"
    return "high", "High"


def _front_badge_ratio(ratio: float, low: float, high: float, ideal: float) -> Tuple[str, str]:
    if ratio is None or math.isnan(ratio):
        return "approximate", "Approximate"
    if low <= ratio <= high:
        return "normal", "Normal"
    if ratio > high:
        return "high", "High"
    if ratio < low:
        return "low", "Low"
    return "approximate", "Approximate"


def build_frontal_measurement_cards(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not data:
        return []

    v = data.get("vertical") or {}
    h = data.get("horizontal") or {}
    cards: List[Dict[str, Any]] = []

    def add(
        vis_key: str,
        name: str,
        value: str,
        *,
        unit: str = "",
        normal_range: str = "—",
        badge: str = "approximate",
        status_label: str = "Approximate",
        note: str = "",
    ) -> None:
        cards.append(
            {
                "vis_key": vis_key,
                "name": name,
                "value": value,
                "unit": unit,
                "normal_range": normal_range,
                "status_label": status_label,
                "badge": badge,
                "note": note[:220],
            }
        )

    ip = v.get("interpupillary_line_alignment") or {}
    ip_ang = _fmt_num(ip.get("angle_deg"))
    if ip_ang is not None:
        abs_h = abs(ip_ang)
        b, sl = ("normal", "Normal") if abs_h <= 3 else ("approximate", "Approximate")
        if abs_h > 6:
            b, sl = "high", "High"
        add(
            "interpupillary_line",
            "Interpupillary line angle",
            f"{ip_ang:.1f}°",
            unit="°",
            normal_range="Near horizontal",
            badge=b,
            status_label=sl,
            note="Tilt of the interpupillary line relative to horizontal; correlate with head posture.",
        )

    cm = v.get("commissure_line_alignment") or {}
    cm_ang = _fmt_num(cm.get("angle_deg"))
    if cm_ang is not None:
        abs_h = abs(cm_ang)
        b, sl = ("normal", "Normal") if abs_h <= 3 else ("approximate", "Approximate")
        if abs_h > 6:
            b, sl = "high", "High"
        add(
            "commissure_line",
            "Commissure line angle",
            f"{cm_ang:.1f}°",
            unit="°",
            normal_range="Near horizontal",
            badge=b,
            status_label=sl,
            note="Commissure line inclination; assess smile and occlusal cant clinically.",
        )

    par = v.get("interpupillary_vs_commissure_parallel") or {}
    try:
        delta = float(par.get("delta_deg"))
        thr = float(par.get("threshold_deg") or 5.0)
        b, sl = _front_badge_parallel(delta, thr)
        add(
            "interpupillary_parallel",
            "Interpupillary vs commissure parallelism",
            f"{delta:.1f}° difference",
            unit="°",
            normal_range=f"≤ {thr:.0f}°",
            badge=b,
            status_label=sl,
            note="Difference between pupillary and commissure line angles; large values may suggest occlusal or soft-tissue cant.",
        )
    except (TypeError, ValueError):
        pass

    fm = h.get("facial_midline") or {}
    rms = _fmt_num(fm.get("rms_deviation_px"))
    fw = (h.get("facial_width_bizygomatic_px") or {}).get("value")
    if rms is not None:
        try:
            rel = (rms / float(fw) * 100.0) if fw and float(fw) > 0 else None
        except (TypeError, ValueError):
            rel = None
        if rel is not None and rel <= 2.5:
            b, sl = "normal", "Normal"
            note = "Midline landmarks cluster closely; gross asymmetry unlikely from this proxy."
        elif rel is not None:
            b, sl = "high", "High"
            note = "Midline deviation index is elevated; assess chin, nose, and dental midlines clinically."
        else:
            b, sl = "approximate", "Approximate"
            note = "Midline deviation from landmark cluster; correlate with clinical midline exam."
        add(
            "facial_midline",
            "Facial midline deviation",
            f"{rms:.1f} px RMS",
            unit="px",
            normal_range="Lower is more symmetric",
            badge=b,
            status_label=sl,
            note=note,
        )

    mlr = _fmt_num(v.get("middle_to_lower_third_ratio"))
    if mlr is not None:
        b, sl = _front_badge_ratio(mlr, 0.85, 1.15, 1.0)
        add(
            "middle_lower_ratio",
            "Middle / lower third ratio",
            f"{mlr:.3f}",
            unit="ratio",
            normal_range="~0.9 – 1.1 (approx.)",
            badge=b,
            status_label=sl,
            note="Vertical facial proportion proxy; interpret with growth pattern and clinical exam.",
        )

    ulr = _fmt_num(v.get("upper_to_lower_lip_ratio"))
    if ulr is not None:
        b, sl = _front_badge_ratio(ulr, 0.85, 1.15, 1.0)
        add(
            "upper_to_lower_lip_ratio",
            "Upper / lower lip height ratio",
            f"{ulr:.3f}",
            unit="ratio",
            normal_range="~1.0 (approx.)",
            badge=b,
            status_label=sl,
            note="Values far from ~1.0 may suggest lip height disproportion.",
        )

    lip = _fmt_num(v.get("lip_length_at_rest_px"))
    if lip is not None:
        add(
            "lip_length_at_rest",
            "Lip length at rest",
            f"{lip:.1f} px",
            unit="px",
            normal_range="Clinical reference",
            badge="approximate",
            status_label="Approximate",
            note="Vertical lip length proxy at rest; verify with clinical lip posture.",
        )

    gap_obj = v.get("interlabial_gap_px") or {}
    gap = _fmt_num(gap_obj.get("value") if isinstance(gap_obj, dict) else gap_obj)
    if gap is not None:
        add(
            "interlabial_gap",
            "Interlabial gap",
            f"{gap:.1f} px",
            unit="px",
            normal_range="Clinical reference",
            badge="approximate",
            status_label="Approximate",
            note="Approximate soft-tissue opening at rest.",
        )

    rof = h.get("rule_of_fifths") or {}
    fifth = _fmt_num(rof.get("fifth_px"))
    if fifth is not None:
        add(
            "rule_of_fifths",
            "Rule of fifths",
            f"{fifth:.1f} px per fifth",
            unit="px",
            normal_range="≈ bizygomatic / 5",
            badge="approximate",
            status_label="Approximate",
            note="Compare eye widths and intercanthal distance to one-fifth of bizygomatic width.",
        )

    fw_obj = h.get("facial_width_bizygomatic_px") or {}
    fw_val = _fmt_num(fw_obj.get("value"))
    if fw_val is not None:
        add(
            "facial_width",
            "Facial width (bizygomatic)",
            f"{fw_val:.1f} px",
            unit="px",
            normal_range="Patient-specific",
            badge="approximate",
            status_label="Approximate",
            note="Transverse facial width for proportional checks.",
        )

    mand = _fmt_num(h.get("mandibular_width_bigonial_px"))
    if mand is not None:
        add(
            "mandibular_width",
            "Mandibular width (bigonial)",
            f"{mand:.1f} px",
            unit="px",
            normal_range="Patient-specific",
            badge="approximate",
            status_label="Approximate",
            note="Lower facial width relative to cheekbones.",
        )

    bz = _fmt_num(h.get("bizygomatic_to_bigonial_ratio"))
    if bz is not None:
        b, sl = _front_badge_ratio(bz, 1.15, 1.45, 1.3)
        add(
            "bizygomatic_to_bigonial_ratio",
            "Bizygomatic / bigonial ratio",
            f"{bz:.3f}",
            unit="ratio",
            normal_range="~1.2 – 1.4 (approx.)",
            badge=b,
            status_label=sl,
            note="Typical range varies; extremes suggest transverse disproportion worth review.",
        )

    fi = _fmt_num(h.get("facial_index_height_to_width"))
    if fi is not None:
        b, sl = _front_badge_ratio(fi, 1.15, 1.55, 1.35)
        add(
            "facial_index",
            "Facial index (height / width)",
            f"{fi:.3f}",
            unit="ratio",
            normal_range="~1.2 – 1.5 (approx.)",
            badge=b,
            status_label=sl,
            note="Longer/narrower vs shorter/wider facial form; integrate with vertical skeletal assessment.",
        )

    return cards


def _confidence_from_badge(badge: str) -> int:
    return {"normal": 82, "high": 74, "low": 72, "approximate": 58}.get(badge, 60)


def build_ai_insights(
    side_points: Any,
    front_data: Optional[Dict[str, Any]],
    doctor_comment: str = "",
) -> Dict[str, Any]:
    side_cards = build_side_measurement_cards(side_points)
    front_cards = build_frontal_measurement_cards(front_data)

    side_rows: List[Dict[str, Any]] = []
    if side_cards:
        convex = next((c for c in side_cards if c["vis_key"] == "profile_convexity"), None)
        total = next((c for c in side_cards if c["vis_key"] == "total_facial_convexity"), None)
        nasio = next((c for c in side_cards if c["vis_key"] == "nasiolabial"), None)
        mento = next((c for c in side_cards if c["vis_key"] == "mentolabial"), None)

        if convex and total:
            tone = _badge_tone(convex.get("status_label", ""))
            side_rows.append(
                {
                    "name": "Skeletal / profile pattern",
                    "confidence": _confidence_from_badge(tone),
                    "explanation": (
                        f"Profile convexity {convex['value']} and total facial convexity {total['value']} "
                        f"suggest a {convex['status_label'].lower()} profile tendency; requires clinical correlation."
                    ),
                }
            )
        if nasio:
            side_rows.append(
                {
                    "name": "Soft tissue profile (nose–lip)",
                    "confidence": _confidence_from_badge(nasio["badge"]),
                    "explanation": (
                        f"Nasiolabial angle {nasio['value']} is {nasio['status_label'].lower()} "
                        f"(reference {nasio['normal_range']}); may indicate maxillary soft-tissue position."
                    ),
                }
            )
        if mento:
            side_rows.append(
                {
                    "name": "Lip / chin relation",
                    "confidence": _confidence_from_badge(mento["badge"]),
                    "explanation": (
                        f"Mentolabial angle {mento['value']} appears {mento['status_label'].lower()}; "
                        "correlate with chin prominence and lower incisor inclination."
                    ),
                }
            )

    front_rows: List[Dict[str, Any]] = []
    if front_cards:
        mid = next((c for c in front_cards if c["vis_key"] == "facial_midline"), None)
        par = next((c for c in front_cards if c["vis_key"] == "interpupillary_parallel"), None)
        bz = next((c for c in front_cards if c["vis_key"] == "bizygomatic_to_bigonial_ratio"), None)
        lip = next((c for c in front_cards if c["vis_key"] == "upper_to_lower_lip_ratio"), None)
        thirds = next((c for c in front_cards if c["vis_key"] == "middle_lower_ratio"), None)

        if mid:
            front_rows.append(
                {
                    "name": "Frontal symmetry",
                    "confidence": _confidence_from_badge(mid["badge"]),
                    "explanation": (
                        f"Facial midline proxy ({mid['value']}) appears {mid['status_label'].lower()}; "
                        "assess dental and skeletal midlines clinically."
                    ),
                }
            )
        if par:
            front_rows.append(
                {
                    "name": "Eye / mouth line alignment",
                    "confidence": _confidence_from_badge(par["badge"]),
                    "explanation": (
                        f"Interpupillary vs commissure parallelism ({par['value']}) is {par['status_label'].lower()}; "
                        "may suggest transverse cant or head posture effect."
                    ),
                }
            )
        if bz:
            front_rows.append(
                {
                    "name": "Facial width / jaw width",
                    "confidence": _confidence_from_badge(bz["badge"]),
                    "explanation": (
                        f"Bizygomatic to bigonial ratio {bz['value']} is {bz['status_label'].lower()} "
                        f"(reference {bz['normal_range']}); informs transverse diagnosis."
                    ),
                }
            )
        if thirds:
            front_rows.append(
                {
                    "name": "Facial thirds balance",
                    "confidence": _confidence_from_badge(thirds["badge"]),
                    "explanation": (
                        f"Middle to lower third ratio {thirds['value']} appears {thirds['status_label'].lower()}; "
                        "approximate vertical proportion guide only."
                    ),
                }
            )
        if lip:
            front_rows.append(
                {
                    "name": "Lip balance",
                    "confidence": _confidence_from_badge(lip["badge"]),
                    "explanation": (
                        f"Upper to lower lip ratio {lip['value']} suggests {lip['status_label'].lower()} "
                        "lip height balance; verify at rest and in function."
                    ),
                }
            )

    summary_parts: List[str] = []
    if side_rows:
        summary_parts.append("Side-view cephalometric-style angles were evaluated for profile and soft-tissue relationships.")
    if front_rows:
        summary_parts.append("Frontal non-smile proportions and alignment metrics were assessed for symmetry and balance.")
    if not summary_parts:
        summary_parts.append("Upload and analyze side and/or frontal photos to generate AI-assisted clinical support.")
    summary = " ".join(summary_parts)

    recommendations: List[str] = []
    for c in side_cards:
        if c["badge"] in ("high", "low"):
            recommendations.append(
                f"Review {c['name']} ({c['value']}) — {c['note'] or 'correlate with full diagnosis.'}"
            )
    for c in front_cards:
        if c["badge"] in ("high", "low"):
            recommendations.append(
                f"Consider frontal finding: {c['name']} ({c['value']}). {c['note'] or ''}"
            )
    if not recommendations:
        recommendations.append(
            "Measurements are within typical reference ranges where defined; continue routine orthodontic assessment."
        )

    suggested = (
        "Use these AI-assisted findings as clinical support alongside examination, radiographs, and patient goals. "
        "Confirm any treatment direction after chairside evaluation; this is not a definitive diagnosis."
    )
    if doctor_comment:
        suggested += " Doctor review notes are recorded for this case."

    return {
        "side": side_rows,
        "frontal": front_rows,
        "summary": summary,
        "recommendations": recommendations[:6],
        "suggested_approach": suggested,
    }


DEFAULT_TREATMENT_REVIEW = (
    "This measurement should be reviewed by the orthodontist together with the "
    "patient's clinical records."
)


# ──────────────────────────────────────────────────────────────────────────────
# X-ray AI profile measurements
# 8 angles that are the exact features validated by the XGBoost training pipeline.
# No published normative ranges exist for these AI-predicted soft-tissue X-ray
# landmarks under these names, so no Normal/Abnormal badge is shown.
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: (feature_key, (a_1based, b_1based, c_1based), display_name, clinical_note)
# Angle convention: angle at vertex b, rays b→a and b→c.
XRAY_ANGLE_SPECS: List[tuple] = [
    (
        "a_2_13_14",
        (2, 13, 14),
        "Nose–Upper Lip–Lower Lip Angle",
        "Relates the nasal landmark to the upper and lower lip contour positions along the profile.",
    ),
    (
        "a_2_15_8",
        (2, 15, 8),
        "Nasal–Lip Root–Chin Angle",
        "Profile angle spanning from the nasal region through the upper lip root to the chin prominence.",
    ),
    (
        "a_7_15_8",
        (7, 15, 8),
        "Mid-Profile Support Angle",
        "Middle-face to chin profile angle; contributes to the overall profile convexity assessment.",
    ),
    (
        "a_9_15_8",
        (9, 15, 8),
        "Chin Contour–Lip Root–Chin Angle",
        "Lower profile shape between two chin-area landmarks and the upper lip root.",
    ),
    (
        "a_2_13_8",
        (2, 13, 8),
        "Facial Profile Convexity Angle",
        "Full profile span from the nasal region through the upper lip to the chin point.",
    ),
    (
        "a_3_13_14",
        (3, 13, 14),
        "Subnasale–Upper Lip–Lower Lip Angle",
        "Sub-nasal point to lip relationship; may reflect lip prominence relative to nasal base.",
    ),
    (
        "a_13_14_8",
        (13, 14, 8),
        "Lip–Chin Angle",
        "Angle from the upper lip through the lower lip to the chin; relates to lip-chin balance.",
    ),
    (
        "a_15_8_9",
        (15, 8, 9),
        "Chin Prominence Angle",
        "Angle at the chin point between the upper lip root and the chin contour landmark.",
    ),
]


def _xray_angle_at_b(
    pts: "np.ndarray",
    a_1based: int,
    b_1based: int,
    c_1based: int,
) -> float:
    """
    Angle in degrees at vertex b, rays b→a and b→c.
    Exact formula used in the Colab training pipeline.
    pts: numpy array shape [N, 2], 0-indexed internally.
    """
    A = pts[a_1based - 1]
    B = pts[b_1based - 1]
    C = pts[c_1based - 1]
    BA = A - B
    BC = C - B
    cos_angle = np.dot(BA, BC) / (
        np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8
    )
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def build_xray_measurement_cards(landmarks_json_str: Any) -> List[Dict[str, Any]]:
    """
    Build display-only measurement cards from the 19 stored X-ray landmarks.

    landmarks_json_str: JSON string "[[x, y], ...]" in original-image pixel
    coordinates, as saved in OrthoCase.landmarks_json.

    Returns a list of card dicts (same schema as side/frontal cards).
    Cards have vis_key=None — they are not linked to any overlay canvas.
    Safely returns [] on any error so the page never breaks.
    """
    if not landmarks_json_str:
        return []
    try:
        raw = json.loads(landmarks_json_str)
    except Exception:
        return []

    if not isinstance(raw, list) or len(raw) < 19:
        return []

    try:
        pts = np.array(
            [[float(p[0]), float(p[1])] for p in raw[:19]],
            dtype=np.float64,
        )
    except Exception:
        return []

    cards: List[Dict[str, Any]] = []
    for feature_key, (a_i, b_i, c_i), name, note in XRAY_ANGLE_SPECS:
        try:
            ang = _xray_angle_at_b(pts, a_i, b_i, c_i)
            cards.append(
                {
                    "vis_key":      None,
                    "feature_key":  feature_key,
                    "name":         name,
                    "value":        f"{ang:.1f}°",
                    "value_raw":    round(ang, 1),
                    "unit":         "°",
                    "normal_range": "Model-derived feature",
                    "status_label": "Derived",
                    "badge":        "approximate",
                    "note":         note,
                }
            )
        except Exception:
            continue
    return cards


def _frontal_treatment_text(key: str, data: Dict[str, Any]) -> str:
    v = data.get("vertical") or {}
    h = data.get("horizontal") or {}

    if key == "interpupillary_line":
        ip = v.get("interpupillary_line_alignment") or {}
        try:
            abs_h = abs(float(ip.get("angle_deg")))
        except (TypeError, ValueError):
            return DEFAULT_TREATMENT_REVIEW
        if abs_h > 6:
            return (
                "Significant interpupillary line tilt may reflect head posture or occlusal cant; "
                "correlate with clinical exam before planning mechanics."
            )
        if abs_h > 3:
            return (
                "Mild interpupillary line inclination; monitor with full soft-tissue and "
                "occlusal assessment."
            )
        return (
            "No specific orthodontic action from pupillary line tilt alone; integrate with "
            "overall facial and occlusal findings."
        )

    if key == "mandibular_width":
        try:
            ratio = float(h.get("bizygomatic_to_bigonial_ratio"))
            if ratio > 1.45:
                return (
                    "Relatively wide mandible versus cheekbones may suggest transverse fullness; "
                    "evaluate arch width and expansion needs."
                )
            if ratio < 1.15:
                return (
                    "Relatively narrow mandible versus cheekbones may suggest transverse deficiency; "
                    "consider expansion or surgical widening after full diagnosis."
                )
        except (TypeError, ValueError):
            pass
        return (
            "Lower facial width informs transverse skeletal pattern; correlate with arch form, "
            "Bolton analysis, and TMJ status."
        )

    if key == "rule_of_fifths":
        return (
            "Proportional deviations in horizontal facial fifths may guide aesthetic planning; "
            "orthodontic mechanics alone may not correct underlying skeletal disproportion."
        )

    if key == "facial_midline":
        fm = h.get("facial_midline") or {}
        fw = (h.get("facial_width_bizygomatic_px") or {}).get("value")
        try:
            rms_f = float(fm.get("rms_deviation_px"))
            fw_f = float(fw) if fw is not None else 0.0
            rel = (rms_f / fw_f * 100.0) if fw_f > 0 else None
            if rel is not None and rel <= 2.5:
                return (
                    "Midline landmarks cluster closely; continue routine assessment of dental "
                    "and skeletal midlines at chairside."
                )
            if rel is not None:
                return (
                    "Elevated midline deviation index may warrant evaluation of dental midline, "
                    "chin position, and asymmetry correction options."
                )
        except (TypeError, ValueError):
            pass
        return DEFAULT_TREATMENT_REVIEW

    return DEFAULT_TREATMENT_REVIEW


def build_measurement_treatment_advice(
    side_points: Any,
    front_data: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Map visual measurement keys to treatment recommendation copy for the result page."""
    advice: Dict[str, str] = {}

    if side_points:
        try:
            pts = normalize_points(side_points)
            for key, _title, idxs, interpreter, _normal in SIDE_SPECS:
                try:
                    ang = float(
                        angle_ABC(pts[idxs[0] - 1], pts[idxs[1] - 1], pts[idxs[2] - 1])
                    )
                    interp = interpreter(ang)
                    advice[key] = (interp.get("treatment") or "").strip() or DEFAULT_TREATMENT_REVIEW
                except Exception:
                    advice[key] = DEFAULT_TREATMENT_REVIEW
        except Exception:
            for key, _title, _idxs, _interp, _normal in SIDE_SPECS:
                advice[key] = DEFAULT_TREATMENT_REVIEW

    if front_data:
        for key in (
            "mandibular_width",
            "interpupillary_line",
            "rule_of_fifths",
            "facial_midline",
        ):
            advice[key] = _frontal_treatment_text(key, front_data)

    return advice
