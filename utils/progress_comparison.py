"""
utils/progress_comparison.py
=============================
Build a structured measurement comparison between a baseline case and a
new progress case.

Rules
-----
- Does NOT re-run any model.
- Does NOT modify the baseline case.
- Does NOT treat every increase as clinical improvement.
- Uses reference ranges from the existing DentAlign measurement system.
- Pixel-scale measurements are displayed but excluded from
  improved/worsened/stable counts (they are image-scale dependent).
- All results are returned as plain dicts suitable for JSON storage.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

# ── Numeric normal ranges for side-view angles ─────────────────────────────
SIDE_NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "nasiolabial":            (90.0,  110.0),
    "profile_convexity":      (151.0, 171.0),
    "total_facial_convexity": (127.0, 137.0),
    "mentolabial":            (110.0, 130.0),
}

# Ordered side spec: (key, display_name, point_indices_1based, normal_range_str)
SIDE_SPECS_FOR_COMPARE = [
    ("nasiolabial",            "Nasiolabial angle",              (7, 8, 10),   "90° – 110°"),
    ("profile_convexity",      "Profile convexity angle",        (3, 8, 17),   "151° – 171°"),
    ("total_facial_convexity", "Total facial convexity angle",   (3, 5, 17),   "127° – 137°"),
    ("mentolabial",            "Mentolabial angle",              (15, 16, 17), "110° – 130°"),
]

MEANINGFUL_ANGLE_DIFF = 0.5   # degrees — smaller change = "no meaningful change"
MEANINGFUL_RATIO_DIFF = 0.005  # ratio — smaller change = "no meaningful change"


# ── Helper: distance from a [low, high] range ──────────────────────────────

def _dist_from_range(val: float, low: float, high: float) -> float:
    if val < low:
        return low - val
    if val > high:
        return val - high
    return 0.0


# ── Core comparison functions ───────────────────────────────────────────────

def _compare_range(
    old_val: float,
    new_val: float,
    normal_low: float,
    normal_high: float,
    min_diff: float,
    unit: str,
) -> Dict[str, Any]:
    """Compare two numeric values against a reference range [low, high]."""
    diff = new_val - old_val
    abs_diff = abs(diff)

    if abs_diff < min_diff:
        direction = "unchanged"
        direction_label = "No meaningful change"
    elif diff > 0:
        direction = "increased"
        direction_label = f"Increased by {abs_diff:.2f}{unit}"
    else:
        direction = "decreased"
        direction_label = f"Decreased by {abs_diff:.2f}{unit}"

    old_in = normal_low <= old_val <= normal_high
    new_in = normal_low <= new_val <= normal_high
    old_dist = _dist_from_range(old_val, normal_low, normal_high)
    new_dist = _dist_from_range(new_val, normal_low, normal_high)

    if old_in and new_in:
        ref_movement = "remained_in_range"
        ref_label = "Remained within the reference range"
    elif old_in and not new_in:
        ref_movement = "moved_out_of_range"
        ref_label = "Moved farther from the reference range"
    elif not old_in and new_in:
        ref_movement = "moved_into_range"
        ref_label = "Moved closer to the reference range"
    else:
        if abs_diff < min_diff:
            ref_movement = "no_meaningful_change"
            ref_label = "No meaningful numerical change"
        elif new_dist < old_dist:
            ref_movement = "moved_closer"
            ref_label = "Moved closer to the reference range"
        elif new_dist > old_dist:
            ref_movement = "moved_further"
            ref_label = "Moved farther from the reference range"
        else:
            ref_movement = "no_meaningful_change"
            ref_label = "No meaningful change relative to the reference range"

    return {
        "direction":          direction,
        "direction_label":    direction_label,
        "diff":               round(diff, 3),
        "abs_diff":           round(abs_diff, 3),
        "ref_movement":       ref_movement,
        "ref_label":          ref_label,
        "baseline_in_range":  old_in,
        "new_in_range":       new_in,
    }


def _compare_threshold(
    old_val: float,
    new_val: float,
    threshold: float,
    min_diff: float,
    unit: str,
) -> Dict[str, Any]:
    """Compare two values where being below a threshold is within the reference (e.g. tilt angles)."""
    diff = new_val - old_val
    abs_diff = abs(diff)

    if abs_diff < min_diff:
        direction = "unchanged"
        direction_label = "No meaningful change"
    elif diff > 0:
        direction = "increased"
        direction_label = f"Increased by {abs_diff:.2f}{unit}"
    else:
        direction = "decreased"
        direction_label = f"Decreased by {abs_diff:.2f}{unit}"

    old_ok = old_val <= threshold
    new_ok = new_val <= threshold

    if old_ok and new_ok:
        ref_movement = "remained_in_range"
        ref_label = "Remained within the reference range"
    elif old_ok and not new_ok:
        ref_movement = "moved_out_of_range"
        ref_label = "Moved farther from the reference range"
    elif not old_ok and new_ok:
        ref_movement = "moved_into_range"
        ref_label = "Moved closer to the reference range"
    else:
        if abs_diff < min_diff:
            ref_movement = "no_meaningful_change"
            ref_label = "No meaningful numerical change"
        elif new_val < old_val:
            ref_movement = "moved_closer"
            ref_label = "Moved closer to the reference range"
        elif new_val > old_val:
            ref_movement = "moved_further"
            ref_label = "Moved farther from the reference range"
        else:
            ref_movement = "no_meaningful_change"
            ref_label = "No meaningful change relative to the reference range"

    return {
        "direction":          direction,
        "direction_label":    direction_label,
        "diff":               round(diff, 3),
        "abs_diff":           round(abs_diff, 3),
        "ref_movement":       ref_movement,
        "ref_label":          ref_label,
        "baseline_in_range":  old_ok,
        "new_in_range":       new_ok,
    }


def categorize_ref_movement(ref_movement: str) -> str:
    """Return 'improved', 'worsened', 'stable', or 'note'."""
    if ref_movement in ("remained_in_range", "moved_into_range", "moved_closer"):
        return "improved"
    if ref_movement in ("moved_out_of_range", "moved_further"):
        return "worsened"
    return "stable"


# ── Side measurements ───────────────────────────────────────────────────────

def _build_side_comparison(
    baseline_landmarks_json: str,
    new_landmarks_json: str,
) -> Dict[str, Any]:
    """Return side-view measurement comparison dict."""
    from utils.measurements import (
        normalize_points, angle_ABC,
        interpret_nasiolabial,
        interpret_profile_convexity,
        interpret_total_facial_convexity,
        interpret_mentolabial,
    )

    INTERPRETERS = {
        "nasiolabial":            interpret_nasiolabial,
        "profile_convexity":      interpret_profile_convexity,
        "total_facial_convexity": interpret_total_facial_convexity,
        "mentolabial":            interpret_mentolabial,
    }

    try:
        b_raw = json.loads(baseline_landmarks_json)
        n_raw = json.loads(new_landmarks_json)
    except Exception:
        return {"available": False, "reason": "Could not parse landmark data."}

    try:
        b_pts = normalize_points(b_raw)
        n_pts = normalize_points(n_raw)
    except Exception as exc:
        return {"available": False, "reason": f"Invalid landmark format: {exc}"}

    measurements: List[Dict[str, Any]] = []

    for key, title, idxs, normal_range_str in SIDE_SPECS_FOR_COMPARE:
        i0, i1, i2 = idxs[0] - 1, idxs[1] - 1, idxs[2] - 1
        try:
            b_ang = float(angle_ABC(b_pts[i0], b_pts[i1], b_pts[i2]))
            n_ang = float(angle_ABC(n_pts[i0], n_pts[i1], n_pts[i2]))
        except Exception:
            measurements.append({
                "key": key, "name": title, "unit": "°",
                "normal_range": normal_range_str,
                "available": False,
                "reason": "Could not compute angle from landmarks.",
            })
            continue

        interp_fn = INTERPRETERS.get(key)
        b_status = interp_fn(b_ang).get("status", "") if interp_fn else ""
        n_status = interp_fn(n_ang).get("status", "") if interp_fn else ""

        low, high = SIDE_NORMAL_RANGES[key]
        cmp = _compare_range(b_ang, n_ang, low, high, MEANINGFUL_ANGLE_DIFF, "°")

        measurements.append({
            "key":              key,
            "name":             title,
            "unit":             "°",
            "normal_range":     normal_range_str,
            "available":        True,
            "pixel_only":       False,
            "baseline_value":   round(b_ang, 1),
            "new_value":        round(n_ang, 1),
            "baseline_status":  b_status,
            "new_status":       n_status,
            **cmp,
        })

    return {"available": bool(measurements), "measurements": measurements}


# ── Frontal measurements ────────────────────────────────────────────────────

def _build_frontal_comparison(
    baseline_landmarks_json: str,
    new_landmarks_json: str,
) -> Dict[str, Any]:
    """Return frontal measurement comparison dict."""
    try:
        b_raw = json.loads(baseline_landmarks_json)
        n_raw = json.loads(new_landmarks_json)
    except Exception:
        return {"available": False, "reason": "Could not parse frontal landmark data."}

    try:
        from utils.frontal_measurements import calculate_frontal_measurements
        b_data = calculate_frontal_measurements(b_raw) or {}
        n_data = calculate_frontal_measurements(n_raw) or {}
    except Exception as exc:
        return {"available": False, "reason": f"Could not compute frontal measurements: {exc}"}

    b_v = b_data.get("vertical") or {}
    n_v = n_data.get("vertical") or {}
    b_h = b_data.get("horizontal") or {}
    n_h = n_data.get("horizontal") or {}

    measurements: List[Dict[str, Any]] = []

    def _try_ratio(key, name, b_src, n_src, low, high, normal_label):
        b_val = b_src
        n_val = n_src
        if b_val is None or n_val is None:
            return
        try:
            b_f, n_f = float(b_val), float(n_val)
            if math.isnan(b_f) or math.isnan(n_f):
                return
        except (TypeError, ValueError):
            return
        cmp = _compare_range(b_f, n_f, low, high, MEANINGFUL_RATIO_DIFF, "")
        measurements.append({
            "key": key, "name": name, "unit": "ratio",
            "normal_range": normal_label,
            "available": True, "pixel_only": False,
            "baseline_value": round(b_f, 3),
            "new_value":      round(n_f, 3),
            "baseline_status": "", "new_status": "",
            **cmp,
        })

    _try_ratio(
        "middle_lower_ratio", "Middle / lower third ratio",
        b_v.get("middle_to_lower_third_ratio"),
        n_v.get("middle_to_lower_third_ratio"),
        0.85, 1.15, "~0.9 – 1.1",
    )
    _try_ratio(
        "upper_to_lower_lip_ratio", "Upper / lower lip height ratio",
        b_v.get("upper_to_lower_lip_ratio"),
        n_v.get("upper_to_lower_lip_ratio"),
        0.85, 1.15, "~1.0",
    )

    # Angle-based frontal measurements (threshold style)
    for key, label, normal_label, threshold, data_path in [
        ("interpupillary_line",    "Interpupillary line angle",               "≤3°",  3.0, "interpupillary_line_alignment"),
        ("commissure_line",        "Commissure line angle",                    "≤3°",  3.0, "commissure_line_alignment"),
    ]:
        b_obj = b_v.get(data_path) or {}
        n_obj = n_v.get(data_path) or {}
        b_ang = b_obj.get("angle_deg")
        n_ang = n_obj.get("angle_deg")
        if b_ang is None or n_ang is None:
            continue
        try:
            b_f, n_f = abs(float(b_ang)), abs(float(n_ang))
        except (TypeError, ValueError):
            continue
        cmp = _compare_threshold(b_f, n_f, threshold, MEANINGFUL_ANGLE_DIFF, "°")
        measurements.append({
            "key": key, "name": label, "unit": "°",
            "normal_range": normal_label,
            "available": True, "pixel_only": False,
            "baseline_value": round(b_f, 1),
            "new_value":      round(n_f, 1),
            "baseline_status": "", "new_status": "",
            **cmp,
        })

    # Parallelism
    b_par = b_v.get("interpupillary_vs_commissure_parallel") or {}
    n_par = n_v.get("interpupillary_vs_commissure_parallel") or {}
    try:
        b_delta = float(b_par.get("delta_deg"))
        n_delta = float(n_par.get("delta_deg"))
        thr     = float(b_par.get("threshold_deg") or 5.0)
        cmp = _compare_threshold(b_delta, n_delta, thr, MEANINGFUL_ANGLE_DIFF, "°")
        measurements.append({
            "key": "interpupillary_parallel",
            "name": "Interpupillary vs commissure parallelism",
            "unit": "°",
            "normal_range": f"≤{thr:.0f}°",
            "available": True, "pixel_only": False,
            "baseline_value": round(b_delta, 1),
            "new_value":      round(n_delta, 1),
            "baseline_status": "", "new_status": "",
            **cmp,
        })
    except (TypeError, ValueError):
        pass

    # Pixel-based measurements — shown but excluded from clinical progress counts
    def _try_px(key, name, b_raw_val, n_raw_val):
        if b_raw_val is None or n_raw_val is None:
            return
        try:
            # Support both scalar and {"value": ...} dict
            b_f = float(b_raw_val.get("value") if isinstance(b_raw_val, dict) else b_raw_val)
            n_f = float(n_raw_val.get("value") if isinstance(n_raw_val, dict) else n_raw_val)
            if math.isnan(b_f) or math.isnan(n_f):
                return
        except (TypeError, ValueError):
            return
        diff = n_f - b_f
        abs_diff = abs(diff)
        direction = "increased" if diff > 0.5 else ("decreased" if diff < -0.5 else "unchanged")
        measurements.append({
            "key": key, "name": name, "unit": "px",
            "normal_range": "Patient-specific (pixel scale)",
            "available": True, "pixel_only": True,
            "baseline_value": round(b_f, 1),
            "new_value":      round(n_f, 1),
            "diff":           round(diff, 1),
            "abs_diff":       round(abs_diff, 1),
            "direction":      direction,
            "direction_label": f"Changed by {diff:+.1f} px",
            "ref_movement":   "pixel_only",
            "ref_label":      "Pixel-scale value — comparison depends on image scale and cannot determine clinical progress.",
            "baseline_in_range": None,
            "new_in_range":      None,
            "baseline_status":   "",
            "new_status":        "",
        })

    _try_px("facial_width",    "Facial width (bizygomatic)",
            b_h.get("facial_width_bizygomatic_px"),
            n_h.get("facial_width_bizygomatic_px"))
    _try_px("mandibular_width", "Mandibular width (bigonial)",
            b_h.get("mandibular_width_bigonial_px"),
            n_h.get("mandibular_width_bigonial_px"))
    _try_px("lip_length_at_rest", "Lip length at rest",
            b_v.get("lip_length_at_rest_px"),
            n_v.get("lip_length_at_rest_px"))
    gap_b = b_v.get("interlabial_gap_px")
    gap_n = n_v.get("interlabial_gap_px")
    _try_px("interlabial_gap", "Interlabial gap",
            gap_b.get("value") if isinstance(gap_b, dict) else gap_b,
            gap_n.get("value") if isinstance(gap_n, dict) else gap_n)

    return {"available": bool(measurements), "measurements": measurements}


# ── X-ray comparison ────────────────────────────────────────────────────────

def _build_xray_comparison(baseline_ortho, new_ortho) -> Dict[str, Any]:
    """
    Compare two OrthoCase records' diagnosis predictions.
    Returns a dict with available + measurements list (prediction-comparison rows).
    Does NOT rerun any model.
    """
    try:
        from utils.orthodontic_ai_inference import parse_xray_diagnosis_json
    except Exception as exc:
        return {"available": False, "reason": f"Could not import xray inference utilities: {exc}"}

    b_diag = parse_xray_diagnosis_json(
        baseline_ortho.diagnosis_json if baseline_ortho else None
    )
    n_diag = parse_xray_diagnosis_json(
        new_ortho.diagnosis_json if new_ortho else None
    )

    if not b_diag:
        return {"available": False, "reason": "Baseline X-ray diagnosis data is not available."}
    if not n_diag:
        return {"available": False, "reason": "Progress X-ray diagnosis data is not available."}

    b_results: Dict[str, Any] = {
        r["key"]: r for r in (b_diag.get("all_results") or [])
    }
    n_results: Dict[str, Any] = {
        r["key"]: r for r in (n_diag.get("all_results") or [])
    }

    # Use baseline key order, then any extra keys from new
    all_keys: List[str] = list(dict.fromkeys(
        list(b_results.keys()) + list(n_results.keys())
    ))

    rows: List[Dict[str, Any]] = []
    for key in all_keys:
        b_r = b_results.get(key)
        n_r = n_results.get(key)

        b_pred  = (b_r.get("prediction") or b_r.get("label") or "—") if b_r else "—"
        n_pred  = (n_r.get("prediction") or n_r.get("label") or "—") if n_r else "—"
        b_conf  = round(float(b_r.get("confidence") or b_r.get("probability_percent") or 0), 1) if b_r else None
        n_conf  = round(float(n_r.get("confidence") or n_r.get("probability_percent") or 0), 1) if n_r else None
        display = (b_r or n_r or {}).get("display_name") or key

        if b_r is None or n_r is None:
            movement       = "data_missing"
            movement_label = "Not enough comparable data"
        elif b_pred == n_pred:
            movement       = "unchanged"
            movement_label = "Prediction remained the same"
        else:
            movement       = "changed"
            movement_label = f"Prediction changed from '{b_pred}' to '{n_pred}'"

        conf_movement = None
        if b_conf is not None and n_conf is not None:
            diff = n_conf - b_conf
            if abs(diff) < 2.0:
                conf_movement = "stable"
            elif diff > 0:
                conf_movement = "increased"
            else:
                conf_movement = "decreased"

        rows.append({
            "key":                  key,
            "name":                 display,
            "baseline_prediction":  b_pred,
            "new_prediction":       n_pred,
            "baseline_confidence":  b_conf,
            "new_confidence":       n_conf,
            "prediction_changed":   (b_pred != n_pred and b_r is not None and n_r is not None),
            "movement":             movement,
            "movement_label":       movement_label,
            "conf_movement":        conf_movement,
            "xray_row":             True,
        })

    return {"available": bool(rows), "measurements": rows}


# ── Public API ──────────────────────────────────────────────────────────────

def safe_json_loads(value: Any) -> Any:
    """Parse JSON string → Python object; return None on any error."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value  # already parsed
    try:
        return json.loads(value)
    except Exception:
        return None


def normalize_comparison_rows(comparison_json_or_dict: Any, section_key: str) -> List[Dict[str, Any]]:
    """
    Safely extract a flat list of measurement/prediction row dicts from
    the structured comparison JSON.

    comparison_json_or_dict: str (stored JSON) or dict (already parsed)
    section_key: 'side', 'frontal', or 'xray'

    Returns [] on any failure — never raises.
    """
    data = safe_json_loads(comparison_json_or_dict) if isinstance(comparison_json_or_dict, str) else comparison_json_or_dict
    if not isinstance(data, dict):
        return []
    section = data.get(section_key)
    if not isinstance(section, dict):
        return []
    if not section.get("available"):
        return []
    rows = section.get("measurements", [])
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def build_xray_progress_comparison(baseline_case, new_case) -> Dict[str, Any]:
    """
    Build comparison dict for X-ray analysis type.
    Loads OrthoCase records; does not rerun any model.
    """
    from models import OrthoCase
    b_ortho = OrthoCase.query.filter_by(case_id=baseline_case.id).first()
    n_ortho = OrthoCase.query.filter_by(case_id=new_case.id).first()

    xray_section = _build_xray_comparison(b_ortho, n_ortho)
    return {
        "analysis_type":    "xray",
        "baseline_case_id": baseline_case.id,
        "new_case_id":      new_case.id,
        "xray":             xray_section,
        "unavailable_reasons": [] if xray_section.get("available") else [xray_section.get("reason", "")],
    }


def build_progress_comparison(baseline_case, new_case) -> Dict[str, Any]:
    """
    Build full structured comparison between two SQLAlchemy Case objects.
    Loads Result records from the DB; does not rerun any model.
    Returns a dict suitable for JSON storage.
    """
    from models import Result

    comparison: Dict[str, Any] = {
        "baseline_case_id":   baseline_case.id,
        "new_case_id":        new_case.id,
        "side":               {"available": False, "reason": "No side-view data."},
        "frontal":            {"available": False, "reason": "No frontal-view data."},
        "unavailable_reasons": [],
    }

    # ── Side ──
    b_side = Result.query.filter_by(case_id=baseline_case.id, view_type="SIDE").first()
    n_side = Result.query.filter_by(case_id=new_case.id,      view_type="SIDE").first()

    if b_side and n_side and b_side.landmarks_json and n_side.landmarks_json:
        comparison["side"] = _build_side_comparison(
            b_side.landmarks_json, n_side.landmarks_json
        )
    elif b_side and not n_side:
        msg = "The new progress case does not contain a side-view analysis."
        comparison["side"] = {"available": False, "reason": msg}
        comparison["unavailable_reasons"].append(msg)
    elif not b_side and n_side:
        msg = "The baseline case does not contain a side-view analysis."
        comparison["side"] = {"available": False, "reason": msg}
        comparison["unavailable_reasons"].append(msg)
    else:
        comparison["side"] = {"available": False, "reason": "Neither case contains side-view data."}

    # ── Frontal ──
    b_front = Result.query.filter_by(case_id=baseline_case.id, view_type="FRONT_NS").first()
    n_front = Result.query.filter_by(case_id=new_case.id,      view_type="FRONT_NS").first()

    if b_front and n_front and b_front.landmarks_json and n_front.landmarks_json:
        comparison["frontal"] = _build_frontal_comparison(
            b_front.landmarks_json, n_front.landmarks_json
        )
    elif b_front and not n_front:
        msg = "The new progress case does not contain a frontal analysis."
        comparison["frontal"] = {"available": False, "reason": msg}
        comparison["unavailable_reasons"].append(msg)
    elif not b_front and n_front:
        msg = "The baseline case does not contain a frontal analysis."
        comparison["frontal"] = {"available": False, "reason": msg}
        comparison["unavailable_reasons"].append(msg)
    else:
        comparison["frontal"] = {"available": False, "reason": "Neither case contains frontal-view data."}

    return comparison


def build_summary_xray(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Build summary dict for an X-ray comparison."""
    rows = normalize_comparison_rows(comparison, "xray")
    total     = len(rows)
    changed   = sum(1 for r in rows if r.get("movement") == "changed")
    unchanged = sum(1 for r in rows if r.get("movement") == "unchanged")
    missing   = sum(1 for r in rows if r.get("movement") == "data_missing")

    if total == 0:
        note = "No comparable X-ray diagnosis data found."
    elif changed == 0:
        note = f"All {unchanged} X-ray predictions remained unchanged between the two cases."
    elif changed == total:
        note = f"All {total} X-ray predictions changed. Clinical review is strongly recommended."
    else:
        note = (f"{changed} of {total} X-ray prediction(s) changed; "
                f"{unchanged} remained the same.")
    if missing:
        note += f" {missing} task(s) had missing data in one of the cases."

    return {
        "total":     total,
        "changed":   changed,
        "unchanged": unchanged,
        "missing":   missing,
        "overall_note": note,
        "disclaimer": (
            "X-ray AI predictions are a decision-support tool only and must be "
            "reviewed and confirmed by the treating clinician before any clinical "
            "conclusions are drawn."
        ),
    }


def build_summary(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Build a summary dict from a comparison dict."""
    improved = worsened = stable = unavailable = pixel_only = total = 0

    for section_key in ("side", "frontal"):
        section = comparison.get(section_key, {})
        if not section.get("available"):
            continue
        for m in section.get("measurements", []):
            if not m.get("available", True):
                unavailable += 1
                continue
            if m.get("pixel_only"):
                pixel_only += 1
                continue
            total += 1
            cat = categorize_ref_movement(m.get("ref_movement", ""))
            if cat == "improved":
                improved += 1
            elif cat == "worsened":
                worsened += 1
            else:
                stable += 1

    if total == 0 and unavailable == 0 and pixel_only == 0:
        note = "No comparable measurements found in these cases."
    elif total == 0:
        note = "All measurements are unavailable or pixel-scale only for this comparison."
    elif improved > 0 and worsened == 0:
        note = (f"{improved} of {total} comparable measurement(s) moved closer to or "
                f"remained within the reference range; {stable} showed no meaningful change.")
    elif worsened > 0 and improved == 0:
        note = (f"{worsened} of {total} comparable measurement(s) moved farther from the "
                f"reference range. Clinical review is recommended.")
    else:
        parts = []
        if improved:
            parts.append(f"{improved} moved closer to the reference range")
        if stable:
            parts.append(f"{stable} showed no meaningful change")
        if worsened:
            parts.append(f"{worsened} moved farther from the reference range")
        note = "; ".join(parts) + "." if parts else "Results are mixed."

    if pixel_only:
        note += (f" {pixel_only} pixel-scale measurement(s) are shown in the table "
                 "but excluded from this count because they depend on image scale.")

    return {
        "improved":        improved,
        "worsened":        worsened,
        "stable":          stable,
        "unavailable":     unavailable,
        "pixel_only":      pixel_only,
        "total_compared":  total,
        "overall_note":    note,
        "disclaimer": (
            "This summary is a decision-support tool only. "
            "It must be reviewed and confirmed by the treating clinician "
            "before any clinical conclusions are drawn."
        ),
    }
