"""
Clinical PDF report: patient header, annotated images, measurement values,
interpretations from existing side interpreters + frontal heuristics.
No landmark coordinate tables or raw JSON.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas

from utils.frontal_measurements import calculate_frontal_measurements, frontal_pdf_interpretation_rows
from utils.measurements import (
    angle_ABC,
    interpret_mentolabial,
    interpret_nasiolabial,
    interpret_profile_convexity,
    interpret_total_facial_convexity,
    normalize_points,
)


def _wrap_lines(text: str, max_len: int = 92) -> List[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_len and cur:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines


def _draw_paragraph(c: pdf_canvas.Canvas, text: str, x: float, y: float, line_h: float = 12) -> float:
    for line in _wrap_lines(text, max_len=90):
        c.drawString(x, y, line)
        y -= line_h
    return y


def _draw_angle_on_bgr(img_bgr: np.ndarray, pts: np.ndarray, i1: int, i2: int, i3: int) -> np.ndarray:
    """Angle at i2 between i1–i2 and i2–i3; no numbered landmark clutter."""
    out = img_bgr.copy()
    A = pts[i1 - 1].astype(int)
    B = pts[i2 - 1].astype(int)
    C = pts[i3 - 1].astype(int)
    cv2.line(out, (A[0], A[1]), (B[0], B[1]), (40, 90, 220), 2)
    cv2.line(out, (C[0], C[1]), (B[0], B[1]), (40, 90, 220), 2)
    for p in (A, B, C):
        cv2.circle(out, (int(p[0]), int(p[1])), 5, (30, 144, 255), -1)
    ang = angle_ABC(pts[i1 - 1], pts[i2 - 1], pts[i3 - 1])
    tx, ty = 12, 28
    label = f"{ang:.1f} deg"
    cv2.rectangle(out, (tx - 4, ty - 22), (tx + 120, ty + 6), (32, 32, 32), -1)
    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def _draw_frontal_overlays(
    img_bgr: np.ndarray, pts: np.ndarray, h: Dict[str, Any]
) -> np.ndarray:
    """Light measurement lines on frontal overlay (no coordinate text)."""
    out = img_bgr.copy()

    def line(i: int, j: int, col: Tuple[int, int, int]) -> None:
        p1 = tuple(pts[i - 1].astype(int))
        p2 = tuple(pts[j - 1].astype(int))
        cv2.line(out, p1, p2, col, 2)

    line(11, 16, (40, 90, 220))
    line(16, 24, (40, 90, 220))
    line(8, 32, (180, 120, 40))
    line(20, 22, (180, 120, 40))
    line(16, 21, (60, 180, 80))
    pair = (h.get("facial_width_bizygomatic_px") or {}).get("chosen_pair") or [1, 28]
    if len(pair) == 2:
        line(int(pair[0]), int(pair[1]), (200, 60, 200))
        y_ids = [6, 11, 16, 21, 24]
        ys = [float(pts[i - 1][1]) for i in y_ids]
        y_top, y_bot = int(min(ys)), int(max(ys))
        x_min = float(min(pts[int(pair[0]) - 1][0], pts[int(pair[1]) - 1][0]))
        x_max = float(max(pts[int(pair[0]) - 1][0], pts[int(pair[1]) - 1][0]))
        w_span = x_max - x_min
        for i in range(6):
            xi = int(x_min + w_span * i / 5.0)
            cv2.line(out, (xi, y_top), (xi, y_bot), (140, 140, 220), 1)
    line(3, 26, (120, 60, 200))
    mid = [11, 12, 14, 16, 18, 21, 23, 24]
    for k in range(len(mid) - 1):
        line(mid[k], mid[k + 1], (100, 100, 100))
    return out


def render_case_pdf(
    base_dir: str,
    case: Any,
    side: Any,
    side_points: Any,
    patient: Any,
    doctor_name: Optional[str],
    front_ns: Any,
    front_ns_points: Any,
) -> str:
    os.makedirs(os.path.join(base_dir, "static", "reports"), exist_ok=True)
    pdf_path = os.path.join(base_dir, "static", "reports", f"case_{case.id}_report.pdf")

    c = pdf_canvas.Canvas(pdf_path, pagesize=A4)
    H = A4[1]

    doctor_name = doctor_name or "—"
    patient_name = getattr(patient, "name", None) or "—"
    patient_age = getattr(patient, "age", None)
    patient_gender = getattr(patient, "gender", None)
    patient_code = getattr(patient, "patient_code", None) or getattr(patient, "code", None) or "—"
    age_str = str(patient_age) if patient_age is not None else "—"
    gender_str = patient_gender if patient_gender else "—"
    doctor_comment = (getattr(case, "doctor_comment", None) or "").strip()

    y = H - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, f"Orthodontic Analysis Report (Case #{case.id})")

    y -= 26
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Patient information")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Doctor: {doctor_name}")
    y -= 14
    c.drawString(40, y, f"Patient: {patient_name}   Code: {patient_code}")
    y -= 14
    c.drawString(40, y, f"Age: {age_str}   Gender: {gender_str}")
    y -= 14
    c.drawString(40, y, f"Case status: {case.status}")
    y -= 14
    c.drawString(40, y, f"Created: {case.created_at.strftime('%Y-%m-%d %H:%M')}")
    y -= 20

    if doctor_comment:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "Clinical notes / diagnosis")
        y -= 14
        c.setFont("Helvetica", 10)
        y = _draw_paragraph(c, doctor_comment, 40, y, 12)
        y -= 8

    side_specs: List[Tuple[str, Tuple[int, int, int], str, Callable[[float], dict]]] = [
        ("nasiolabial", (7, 8, 10), "Nasiolabial angle", interpret_nasiolabial),
        ("profile_convexity", (3, 8, 17), "Profile convexity angle", interpret_profile_convexity),
        ("total_facial_convexity", (3, 5, 17), "Total facial convexity angle", interpret_total_facial_convexity),
        ("mentolabial", (15, 16, 17), "Mentolabial angle", interpret_mentolabial),
    ]

    side_img_path = None
    if side and getattr(side, "overlay_path", None):
        side_img_path = os.path.join(base_dir, "static", side.overlay_path).replace("\\", "/")

    if side and side_img_path and os.path.exists(side_img_path) and side_points:
        try:
            pts = normalize_points(side_points)
            bgr = cv2.imread(side_img_path)
            if bgr is not None:
                c.showPage()
                y = H - 50
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, y, "Side view — measurement visualizations and values")
                y -= 24

                img_w, img_h = 240.0, 240.0
                for key, idxs, title, interpreter in side_specs:
                    if y < 120:
                        c.showPage()
                        y = H - 50
                    try:
                        ang = float(angle_ABC(pts[idxs[0] - 1], pts[idxs[1] - 1], pts[idxs[2] - 1]))
                        vis = _draw_angle_on_bgr(bgr, pts, idxs[0], idxs[1], idxs[2])
                        tmp = os.path.join(
                            base_dir, "static", "reports", f"_pdf_{case.id}_side_{key}.jpg"
                        )
                        cv2.imwrite(tmp, vis)
                        img_bottom = y - img_h
                        c.drawImage(
                            ImageReader(tmp), 40, img_bottom, width=img_w, height=img_h, mask="auto"
                        )
                        interp = interpreter(ang)
                        txt_y = img_bottom - 10
                        c.setFont("Helvetica-Bold", 10)
                        c.drawString(
                            40,
                            txt_y,
                            f"{title}: {ang:.1f}° — {interp.get('status', '')}",
                        )
                        txt_y -= 12
                        c.setFont("Helvetica", 9)
                        txt_y = _draw_paragraph(
                            c,
                            f"{interp.get('meaning', '')} {interp.get('treatment', '')}",
                            40,
                            txt_y,
                            11,
                        )
                        y = txt_y - 20
                    except Exception:
                        continue
        except Exception:
            c.setFont("Helvetica", 10)
            c.drawString(40, y, "Side view data could not be fully rendered for this PDF.")

    front_path = None
    if front_ns and getattr(front_ns, "overlay_path", None):
        front_path = os.path.join(base_dir, "static", front_ns.overlay_path).replace("\\", "/")

    if front_path and os.path.exists(front_path) and front_ns_points and len(front_ns_points) >= 34:
        try:
            fm = calculate_frontal_measurements(front_ns_points)
            bgr = cv2.imread(front_path)
            if bgr is not None:
                pts = normalize_points(front_ns_points)
                h = fm.get("horizontal") or {}
                vis = _draw_frontal_overlays(bgr, pts, h)
                tmp = os.path.join(base_dir, "static", "reports", f"_pdf_{case.id}_front.jpg")
                cv2.imwrite(tmp, vis)

                c.showPage()
                y = H - 50
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, y, "Frontal non-smile — measurement visualization")
                y -= 22
                c.drawImage(ImageReader(tmp), 40, y - 280, width=280, height=280, mask="auto")
                y = y - 300

                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, "Frontal measurements and interpretation")
                y -= 16
                c.setFont("Helvetica", 9)
                for title, val, note in frontal_pdf_interpretation_rows(fm):
                    if y < 60:
                        c.showPage()
                        y = H - 50
                    c.setFont("Helvetica-Bold", 9)
                    c.drawString(40, y, f"{title}: {val}")
                    y -= 10
                    c.setFont("Helvetica", 8)
                    y = _draw_paragraph(c, note, 52, y, 10)
                    y -= 6
        except Exception:
            c.showPage()
            c.setFont("Helvetica", 10)
            c.drawString(40, H - 60, "Frontal analysis could not be fully rendered for this PDF.")

    c.save()
    return pdf_path
