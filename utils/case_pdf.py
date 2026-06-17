"""
Clinical PDF report — professional DentAlign layout with headers, footers,
wrapped tables, and paginated measurement cards.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import Paragraph

from utils.frontal_measurements import calculate_frontal_measurements, frontal_pdf_interpretation_rows
from utils.measurements import (
    angle_ABC,
    interpret_mentolabial,
    interpret_nasiolabial,
    interpret_profile_convexity,
    interpret_total_facial_convexity,
    normalize_points,
)
from utils.paths import join_stored, resolve_overlay_path, static_dir, ensure_dir

# Page layout
PAGE_W, PAGE_H = A4
MARGIN_L = 42
MARGIN_R = 42
MARGIN_TOP = 36
HEADER_H = 56
FOOTER_H = 32
HEADER_GAP = 20
SECTION_SPACING = 12
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
CONTENT_TOP = PAGE_H - MARGIN_TOP - HEADER_H - HEADER_GAP
CONTENT_BOTTOM = MARGIN_TOP + FOOTER_H

# DentAlign medical theme
COLOR_PRIMARY = colors.HexColor("#1a6b8a")
COLOR_PRIMARY_DARK = colors.HexColor("#0f4d63")
COLOR_PRIMARY_LIGHT = colors.HexColor("#e8f4f8")
COLOR_ACCENT = colors.HexColor("#2aa8c4")
COLOR_TEXT = colors.HexColor("#1e293b")
COLOR_MUTED = colors.HexColor("#64748b")
COLOR_BORDER = colors.HexColor("#cbd5e1")
COLOR_WHITE = colors.white
COLOR_ROW_ALT = colors.HexColor("#f8fafc")

SIDE_SPECS: List[Tuple[str, Tuple[int, int, int], str, Callable[[float], dict], str]] = [
    ("nasiolabial", (7, 8, 10), "Nasiolabial angle", interpret_nasiolabial, "90° – 110°"),
    ("profile_convexity", (3, 8, 17), "Profile convexity angle", interpret_profile_convexity, "151° – 171°"),
    ("total_facial_convexity", (3, 5, 17), "Total facial convexity angle", interpret_total_facial_convexity, "127° – 137°"),
    ("mentolabial", (15, 16, 17), "Mentolabial angle", interpret_mentolabial, "110° – 130°"),
]

_CELL_STYLE = ParagraphStyle(
    "table_cell",
    fontName="Helvetica",
    fontSize=7,
    leading=9,
    textColor=COLOR_TEXT,
)
_CELL_BOLD = ParagraphStyle(
    "table_cell_bold",
    fontName="Helvetica-Bold",
    fontSize=7.5,
    leading=10,
    textColor=COLOR_WHITE,
)
_NOTE_STYLE = ParagraphStyle(
    "table_note",
    fontName="Helvetica",
    fontSize=6.5,
    leading=8.5,
    textColor=COLOR_TEXT,
)


def _resolve_patient(patient: Any, case: Any) -> Any:
    if patient is not None:
        return patient
    if case is not None:
        linked = getattr(case, "patient", None)
        if linked is not None:
            return linked
    return None


def _format_age(patient: Any, case: Any) -> str:
    p = _resolve_patient(patient, case)
    if p is None:
        return "—"
    age = getattr(p, "age", None)
    if age is None or age == "":
        return "—"
    try:
        return f"{int(age)} years"
    except (TypeError, ValueError):
        return str(age)


def _format_gender(patient: Any, case: Any) -> str:
    p = _resolve_patient(patient, case)
    if p is None:
        return "—"
    raw = (getattr(p, "gender", None) or "").strip().upper()
    if raw == "MALE":
        return "Male"
    if raw == "FEMALE":
        return "Female"
    if raw:
        return raw.replace("_", " ").title()
    return "—"


def _escape_xml(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _wrap_lines(text: str, max_len: int = 88) -> List[str]:
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


def _paragraph_height(text: str, width: float, style: ParagraphStyle) -> float:
    p = Paragraph(_escape_xml(text or "—"), style)
    _w, h = p.wrap(max(width - 8, 24), 10000)
    return h + 6


def _draw_paragraph_on_canvas(
    c: pdf_canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    style: ParagraphStyle,
) -> float:
    """Draw wrapped paragraph; y is top baseline area. Returns new y below block."""
    p = Paragraph(_escape_xml(text or ""), style)
    w, h = p.wrap(max(width - 4, 24), 10000)
    p.drawOn(c, x, y - h)
    return y - h


def _draw_paragraph(
    c: pdf_canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font: str = "Helvetica",
    size: float = 9,
    line_h: float = 11,
    color=COLOR_TEXT,
) -> float:
    c.setFont(font, size)
    c.setFillColor(color)
    for line in _wrap_lines(text, max_len=int(max_width / (size * 0.52))):
        c.drawString(x, y, line)
        y -= line_h
    return y


def draw_report_header(
    c: pdf_canvas.Canvas,
    case_id: int,
    report_date: str,
) -> None:
    y_top = PAGE_H - MARGIN_TOP
    c.setFillColor(COLOR_PRIMARY)
    c.rect(0, y_top - HEADER_H, PAGE_W, HEADER_H + 6, fill=1, stroke=0)

    c.setFillColor(COLOR_WHITE)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MARGIN_L, y_top - 20, "DentAlign")
    c.setFont("Helvetica", 7.5)
    c.drawString(MARGIN_L, y_top - 32, "ORTHODONTIC SUITE")

    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(PAGE_W / 2, y_top - 22, "AI-Based Orthodontic Facial Analysis Report")

    c.setFont("Helvetica", 9)
    c.drawRightString(PAGE_W - MARGIN_R, y_top - 20, f"Case #{case_id}")
    c.drawRightString(PAGE_W - MARGIN_R, y_top - 32, report_date)


draw_header = draw_report_header


def draw_footer(c: pdf_canvas.Canvas, page_num: int) -> None:
    y = MARGIN_TOP + 8
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, y + FOOTER_H - 6, PAGE_W - MARGIN_R, y + FOOTER_H - 6)

    c.setFillColor(COLOR_MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN_L, y, "Generated by DentAlign")
    c.drawRightString(PAGE_W - MARGIN_R, y, f"Page {page_num}")


def draw_section_title(
    c: pdf_canvas.Canvas,
    title: str,
    y: float,
    pages: Optional["_ReportPages"] = None,
) -> float:
    y = ensure_space_or_new_page_y(c, y, 36, pages)
    c.setFillColor(COLOR_PRIMARY_DARK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN_L, y, title)
    y -= 6
    c.setStrokeColor(COLOR_ACCENT)
    c.setLineWidth(1.2)
    c.line(MARGIN_L, y, PAGE_W - MARGIN_R, y)
    return y - 16


def ensure_space_or_new_page_y(
    c: pdf_canvas.Canvas,
    y: float,
    needed: float,
    pages: Optional["_ReportPages"] = None,
) -> float:
    if y - needed < CONTENT_BOTTOM:
        if pages is not None:
            return pages.new_page()
        return CONTENT_TOP
    return y


def draw_info_card(
    c: pdf_canvas.Canvas,
    rows: List[Tuple[str, str]],
    y: float,
    title: str = "Patient & case information",
) -> float:
    card_w = CONTENT_W
    row_h = 18
    pad = 12
    card_h = pad * 2 + 20 + len(rows) * row_h

    c.setFillColor(COLOR_PRIMARY_LIGHT)
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN_L, y - card_h, card_w, card_h, 6, fill=1, stroke=1)

    ty = y - pad - 14
    c.setFillColor(COLOR_PRIMARY_DARK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN_L + pad, ty, title)
    ty -= 22

    col1_x = MARGIN_L + pad
    col2_x = MARGIN_L + card_w * 0.34
    for label, value in rows:
        c.setFillColor(COLOR_MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(col1_x, ty, label)
        c.setFillColor(COLOR_TEXT)
        c.setFont("Helvetica", 9)
        display_val = (str(value).strip() if value is not None else "") or "—"
        val_lines = _wrap_lines(display_val, max_len=52)
        c.drawString(col2_x, ty, val_lines[0])
        if len(val_lines) > 1:
            ty -= 10
            c.setFont("Helvetica", 8)
            c.drawString(col2_x, ty, val_lines[1][:70])
        ty -= row_h

    return y - card_h - SECTION_SPACING


def draw_wrapped_table(
    c: pdf_canvas.Canvas,
    headers: List[str],
    col_widths: List[float],
    data_rows: List[List[str]],
    y: float,
    pages: Optional["_ReportPages"] = None,
    *,
    section_title: Optional[str] = None,
    header_font_size: float = 7.5,
    body_style: Optional[ParagraphStyle] = None,
) -> float:
    """Draw a table with Paragraph-wrapped cell text and dynamic row heights."""
    if not data_rows:
        return y

    if section_title:
        y = ensure_space_or_new_page_y(c, y, 48, pages)
        y = draw_section_title(c, section_title, y, pages)
    else:
        y = ensure_space_or_new_page_y(c, y, 40, pages)

    body_style = body_style or _CELL_STYLE
    x0 = MARGIN_L
    table_w = sum(col_widths)
    header_h = 22
    pad_y = 5

    def _row_block_height(cells: List[str]) -> float:
        heights = [_paragraph_height(cell, col_widths[i], body_style) for i, cell in enumerate(cells)]
        return max(heights) + pad_y

    row_heights = [_row_block_height(row) for row in data_rows]
    table_body_h = sum(row_heights)
    needed = header_h + table_body_h + 12

    if y - needed < CONTENT_BOTTOM and pages is not None:
        y = pages.new_page()
        if section_title:
            y = draw_section_title(c, f"{section_title} (continued)", y, pages)

    # Header row
    c.setFillColor(COLOR_PRIMARY)
    c.rect(x0, y - header_h, table_w, header_h, fill=1, stroke=0)
    c.setFillColor(COLOR_WHITE)
    hdr_style = ParagraphStyle(
        "hdr",
        parent=_CELL_BOLD,
        fontSize=header_font_size,
        leading=header_font_size + 2,
    )
    x = x0 + 4
    for i, h in enumerate(headers):
        _draw_paragraph_on_canvas(c, h, x, y - 6, col_widths[i], hdr_style)
        x += col_widths[i]

    cy = y - header_h
    for ri, (row, rh) in enumerate(zip(data_rows, row_heights)):
        cy -= rh
        if ri % 2 == 0:
            c.setFillColor(COLOR_ROW_ALT)
            c.rect(x0, cy, table_w, rh, fill=1, stroke=0)
        x = x0 + 4
        for i, cell in enumerate(row):
            _draw_paragraph_on_canvas(c, cell, x, cy + rh - 4, col_widths[i], body_style)
            x += col_widths[i]

    top = y
    bottom = cy
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.5)
    c.rect(x0, bottom, table_w, top - bottom, fill=0, stroke=1)

    return bottom - SECTION_SPACING


def draw_measurement_table(
    c: pdf_canvas.Canvas,
    table_rows: List[Dict[str, str]],
    y: float,
    pages: Optional["_ReportPages"] = None,
) -> float:
    """Side-view summary: measurement, value, classification, normal range, clinical note."""
    if not table_rows:
        return y

    col_widths = [108, 46, 82, 64, CONTENT_W - 108 - 46 - 82 - 64]
    headers = ["Measurement", "Value", "Classification", "Normal range", "Clinical note"]
    data = [
        [
            row.get("name", ""),
            row.get("value", ""),
            row.get("classification", ""),
            row.get("normal_range", ""),
            row.get("note", ""),
        ]
        for row in table_rows
    ]
    return draw_wrapped_table(
        c,
        headers,
        col_widths,
        data,
        y,
        pages,
        section_title="Side view — measurement summary",
    )


def draw_frontal_summary_table(
    c: pdf_canvas.Canvas,
    frontal_rows: List[Tuple[str, str, str]],
    y: float,
    pages: Optional["_ReportPages"] = None,
) -> float:
    if not frontal_rows:
        return y

    col_widths = [118, 88, CONTENT_W - 118 - 88]
    headers = ["Measurement", "Value", "Interpretation"]
    data = [[title, val, note] for title, val, note in frontal_rows]
    return draw_wrapped_table(
        c,
        headers,
        col_widths,
        data,
        y,
        pages,
        section_title="Frontal non-smile — measurement summary",
        body_style=_NOTE_STYLE,
    )


def draw_measurement_card(
    c: pdf_canvas.Canvas,
    x: float,
    y: float,
    card_w: float,
    card_h: float,
    title: str,
    angle_val: float,
    interp: dict,
    normal_range: str,
    image_path: Optional[str],
) -> float:
    """Single measurement card; returns bottom y."""
    c.setFillColor(COLOR_WHITE)
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.6)
    c.roundRect(x, y - card_h, card_w, card_h, 5, fill=1, stroke=1)

    img_w, img_h = 98, 98
    img_x = x + 10
    img_y = y - 14 - img_h

    if image_path and os.path.isfile(image_path):
        try:
            c.drawImage(ImageReader(image_path), img_x, img_y, width=img_w, height=img_h, mask="auto")
        except Exception:
            _draw_image_placeholder(c, img_x, img_y, img_w, img_h)
    else:
        _draw_image_placeholder(c, img_x, img_y, img_w, img_h)

    tx = x + img_w + 18
    ty = y - 20
    text_w = card_w - img_w - 28

    c.setFillColor(COLOR_PRIMARY_DARK)
    c.setFont("Helvetica-Bold", 9)
    for line in _wrap_lines(title, max_len=int(text_w / 5.5))[:2]:
        c.drawString(tx, ty, line)
        ty -= 12

    c.setFillColor(COLOR_PRIMARY)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(tx, ty, f"{angle_val:.1f}°")
    ty -= 14

    c.setFillColor(COLOR_MUTED)
    c.setFont("Helvetica", 7)
    c.drawString(tx, ty, f"Normal: {normal_range}")
    ty -= 12

    c.setFillColor(COLOR_TEXT)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(tx, ty, (interp.get("status", "") or "")[:48])
    ty -= 11

    c.setFont("Helvetica", 7)
    meaning = interp.get("meaning", "")
    ty = _draw_paragraph(c, meaning, tx, ty, text_w, size=7, line_h=9)
    ty -= 4
    treatment = interp.get("treatment", "")
    if treatment:
        c.setFillColor(COLOR_MUTED)
        ty = _draw_paragraph(c, treatment, tx, ty, text_w, size=6.5, line_h=8.5, color=COLOR_MUTED)

    return y - card_h


def _draw_image_placeholder(
    c: pdf_canvas.Canvas, x: float, y: float, w: float, h: float
) -> None:
    c.setFillColor(COLOR_ROW_ALT)
    c.setStrokeColor(COLOR_BORDER)
    c.rect(x, y, w, h, fill=1, stroke=1)
    c.setFillColor(COLOR_MUTED)
    c.setFont("Helvetica", 7)
    c.drawCentredString(x + w / 2, y + h / 2 - 3, "Image not available")


def _image_display_size(path: str, max_w: float, max_h: float) -> Tuple[float, float]:
    try:
        ir = ImageReader(path)
        iw, ih = ir.getSize()
        if iw <= 0 or ih <= 0:
            return max_w, max_h
        scale = min(max_w / iw, max_h / ih)
        return iw * scale, ih * scale
    except Exception:
        return max_w, max_h


def _draw_frontal_visualization(
    c: pdf_canvas.Canvas,
    pages: "_ReportPages",
    y: float,
    image_path: str,
    frontal_rows: List[Tuple[str, str, str]],
) -> float:
    """Overlay image left, interpretation list right; paginate when needed."""
    img_max_w = 210
    img_max_h = 240
    col_gap = 18
    text_x = MARGIN_L + img_max_w + col_gap
    text_w = PAGE_W - MARGIN_R - text_x

    img_w, img_h = _image_display_size(image_path, img_max_w, img_max_h)

    y = ensure_space_or_new_page_y(c, y, img_h + 48, pages)
    y = draw_section_title(c, "Frontal visualization & interpretations", y, pages)

    img_bottom = y - img_h
    text_col_x = MARGIN_L + img_w + col_gap
    text_col_w = PAGE_W - MARGIN_R - text_col_x
    if os.path.isfile(image_path):
        try:
            c.drawImage(ImageReader(image_path), MARGIN_L, img_bottom, width=img_w, height=img_h, mask="auto")
        except Exception:
            _draw_image_placeholder(c, MARGIN_L, img_bottom, img_w, img_h)
    else:
        _draw_image_placeholder(c, MARGIN_L, img_bottom, img_w, img_h)

    c.setFillColor(COLOR_PRIMARY_DARK)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, img_bottom - 12, "Annotated frontal overlay")

    ty = y
    text_x = text_col_x
    text_w = text_col_w
    for title, val, note in frontal_rows:
        block_need = 42 + _paragraph_height(note, text_w, _NOTE_STYLE)
        if ty - block_need < CONTENT_BOTTOM:
            y = pages.new_page()
            y = draw_section_title(c, "Frontal interpretations (continued)", y, pages)
            ty = y
            text_x = MARGIN_L
            text_w = CONTENT_W

        c.setFillColor(COLOR_PRIMARY_DARK)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(text_x, ty, (title or "")[:55])
        ty -= 11
        c.setFillColor(COLOR_TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(text_x, ty, (val or "—")[:80])
        ty -= 10
        ty = _draw_paragraph_on_canvas(c, note, text_x, ty, text_w, _NOTE_STYLE)
        ty -= 10

    return ty - SECTION_SPACING


def _draw_angle_on_bgr(img_bgr: np.ndarray, pts: np.ndarray, i1: int, i2: int, i3: int) -> np.ndarray:
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


class _ReportPages:
    def __init__(self, c: pdf_canvas.Canvas, case_id: int, report_date: str):
        self.c = c
        self.case_id = case_id
        self.report_date = report_date
        self.page_num = 0

    def new_page(self) -> float:
        if self.page_num > 0:
            draw_footer(self.c, self.page_num)
            self.c.showPage()
        self.page_num += 1
        draw_report_header(self.c, self.case_id, self.report_date)
        return CONTENT_TOP

    def finish(self) -> None:
        if self.page_num > 0:
            draw_footer(self.c, self.page_num)


def _prepare_side_measurements(
    case_id: int,
    bgr: np.ndarray,
    pts: np.ndarray,
    reports_dir: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    measurements: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, str]] = []

    for key, idxs, title, interpreter, normal_range in SIDE_SPECS:
        try:
            ang = float(angle_ABC(pts[idxs[0] - 1], pts[idxs[1] - 1], pts[idxs[2] - 1]))
            interp = interpreter(ang)
            vis = _draw_angle_on_bgr(bgr, pts, idxs[0], idxs[1], idxs[2])
            tmp = os.path.join(reports_dir, f"_pdf_{case_id}_side_{key}.jpg")
            cv2.imwrite(tmp, vis)
            note = f"{interp.get('meaning', '')} {interp.get('treatment', '')}".strip()
            measurements.append(
                {
                    "key": key,
                    "title": title,
                    "angle": ang,
                    "interp": interp,
                    "normal_range": normal_range,
                    "image_path": tmp,
                }
            )
            table_rows.append(
                {
                    "name": title,
                    "value": f"{ang:.1f}°",
                    "classification": interp.get("status", ""),
                    "normal_range": normal_range,
                    "note": note,
                }
            )
        except Exception:
            continue

    return measurements, table_rows


def _draw_side_measurement_cards(
    c: pdf_canvas.Canvas,
    pages: _ReportPages,
    y: float,
    measurements: List[Dict[str, Any]],
) -> float:
    if not measurements:
        return y

    card_gap = 14
    card_w = (CONTENT_W - card_gap) / 2
    card_h = 178

    y = ensure_space_or_new_page_y(c, y, card_h + 50, pages)
    y = draw_section_title(c, "Side view — measurement visualizations", y, pages)

    i = 0
    while i < len(measurements):
        if y - card_h < CONTENT_BOTTOM:
            y = pages.new_page()
            y = draw_section_title(c, "Side view — measurement visualizations (continued)", y, pages)

        m0 = measurements[i]
        draw_measurement_card(
            c, MARGIN_L, y, card_w, card_h,
            m0["title"], m0["angle"], m0["interp"], m0["normal_range"], m0.get("image_path"),
        )
        if i + 1 < len(measurements):
            m1 = measurements[i + 1]
            draw_measurement_card(
                c, MARGIN_L + card_w + card_gap, y, card_w, card_h,
                m1["title"], m1["angle"], m1["interp"], m1["normal_range"], m1.get("image_path"),
            )
            i += 2
        else:
            i += 1
        y -= card_h + card_gap

    return y


def _draw_xray_overlay_image(
    c: pdf_canvas.Canvas,
    pages: "_ReportPages",
    y: float,
    ortho_rec: Any,
    base_dir: str,
) -> float:
    """Draw the X-ray landmark overlay image in the PDF."""
    overlay_path = getattr(ortho_rec, "overlay_path", None)
    if not overlay_path:
        return y

    # Resolve stored path (e.g. "static/results/ortho/overlay_xxx.jpg") to absolute
    norm = overlay_path.replace("\\", "/")
    if not os.path.isabs(norm):
        abs_path = os.path.join(base_dir, norm)
    else:
        abs_path = norm

    if not os.path.isfile(abs_path):
        return y

    y = ensure_space_or_new_page_y(c, y, 60, pages)
    y = draw_section_title(c, "X-Ray AI — Landmark Detection", y, pages)

    max_w = CONTENT_W * 0.55
    max_h = 260.0
    img_w, img_h = _image_display_size(abs_path, max_w, max_h)

    y = ensure_space_or_new_page_y(c, y, img_h + 28, pages)
    img_x = MARGIN_L
    img_y = y - img_h

    try:
        c.drawImage(ImageReader(abs_path), img_x, img_y, width=img_w, height=img_h, mask="auto")
    except Exception:
        _draw_image_placeholder(c, img_x, img_y, img_w, img_h)

    # Caption beside the image
    cap_x = img_x + img_w + 16
    cap_w = PAGE_W - MARGIN_R - cap_x
    if cap_w > 60:
        c.setFillColor(COLOR_PRIMARY_DARK)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(cap_x, y - 14, "19 Predicted Landmarks")
        c.setFillColor(COLOR_MUTED)
        c.setFont("Helvetica", 7.5)
        caption_lines = [
            "Cephalometric landmarks predicted",
            "by HRNet AI from the X-ray image.",
            "",
            "These landmarks are used to compute",
            "the profile angle measurements and",
            "the AI orthodontic diagnosis.",
        ]
        ty = y - 28
        for line in caption_lines:
            c.drawString(cap_x, ty, line)
            ty -= 11

    c.setFillColor(COLOR_PRIMARY_DARK)
    c.setFont("Helvetica", 7)
    c.drawString(img_x, img_y - 10, "Annotated X-ray overlay — landmark positions are AI-predicted.")

    return img_y - 20


def _draw_xray_diagnosis_section(
    c: pdf_canvas.Canvas,
    pages: "_ReportPages",
    y: float,
    ortho_rec: Any,
) -> float:
    """Draw X-ray AI diagnosis table and optional doctor review block.

    Supports both:
    - schema_version=2  (new 11-model pipeline, Primary + Supportive sections)
    - legacy flat dict  (old 4-model pipeline: skeletal_class / upper_lip / lower_lip / profile_class)
    """
    from utils.orthodontic_ai_inference import parse_xray_diagnosis_json

    diagnosis_json = getattr(ortho_rec, "diagnosis_json", None)
    if not diagnosis_json:
        return y

    diag = parse_xray_diagnosis_json(diagnosis_json)
    if not diag:
        return y

    is_legacy       = diag.get("is_legacy", False)
    schema_version  = diag.get("schema_version", 1)
    all_results     = diag.get("all_results", [])
    primary_keys    = set(diag.get("primary", {}).keys())
    supportive_keys = set(diag.get("supportive", {}).keys())

    col_widths = [110, 130, 65, 65, CONTENT_W - 110 - 130 - 65 - 65]
    headers    = ["Category", "Finding", "Conf %", "Level", "Status"]

    # ── Primary findings table ────────────────────────────────────────────────
    primary_rows = []
    for entry in all_results:
        if entry.get("key") not in primary_keys:
            continue
        confidence  = entry.get("confidence", 0)
        conf_str    = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else "—"
        level       = (entry.get("confidence_level") or "—").capitalize()
        review_flag = "⚠ Review" if entry.get("requires_doctor_review") else "OK"
        primary_rows.append([
            entry.get("display_name", entry.get("key", "—")),
            str(entry.get("label", "—")),
            conf_str,
            level,
            review_flag,
        ])

    if primary_rows:
        title = "X-Ray AI — Primary Findings"
        if is_legacy:
            title += " (Legacy 4-model)"
        y = draw_wrapped_table(
            c, headers, col_widths, primary_rows, y, pages,
            section_title=title,
        )

    # ── Supportive findings table ─────────────────────────────────────────────
    supportive_rows = []
    for entry in all_results:
        if entry.get("key") not in supportive_keys:
            continue
        confidence  = entry.get("confidence", 0)
        conf_str    = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else "—"
        level       = (entry.get("confidence_level") or "—").capitalize()
        review_flag = "⚠ Review" if entry.get("requires_doctor_review") else "OK"
        supportive_rows.append([
            entry.get("display_name", entry.get("key", "—")),
            str(entry.get("label", "—")),
            conf_str,
            level,
            review_flag,
        ])

    if supportive_rows:
        y = draw_wrapped_table(
            c, headers, col_widths, supportive_rows, y, pages,
            section_title="X-Ray AI — Supportive Findings",
        )

    if not primary_rows and not supportive_rows:
        return y

    # ── Disclaimer / pipeline note ────────────────────────────────────────────
    y = ensure_space_or_new_page_y(c, y, 28, pages)
    c.setFillColor(COLOR_MUTED)
    c.setFont("Helvetica-Oblique", 7.5)
    note = (
        f"Pipeline: {diag.get('pipeline', 'unknown')} | "
        f"Landmarks: {diag.get('landmark_count', '?')} | "
        f"Features: {diag.get('feature_count', '?')} | "
        "AI decision-support only — must be reviewed by the orthodontist."
    )
    c.drawString(MARGIN_L, y, note)
    y -= 18

    # ── Doctor review block (if reviewed) ────────────────────────────────────
    reviewed     = getattr(ortho_rec, "reviewed", False)
    reviewed_at  = getattr(ortho_rec, "reviewed_at", None)
    doctor_final = (getattr(ortho_rec, "doctor_final_diagnosis", None) or "").strip()
    doctor_notes = (getattr(ortho_rec, "doctor_review_notes", None) or "").strip()

    if reviewed or doctor_final or doctor_notes:
        y = ensure_space_or_new_page_y(c, y, 56, pages)
        y = draw_section_title(c, "Doctor Review", y, pages)
        review_rows = [
            ("Reviewed",              "Yes" if reviewed else "No"),
            ("Reviewed at",           reviewed_at.strftime("%d %b %Y · %H:%M") if reviewed_at else "—"),
            ("Doctor final diagnosis", doctor_final or "—"),
            ("Doctor review notes",    doctor_notes or "—"),
        ]
        y = draw_info_card(c, review_rows, y, title="Doctor review details")

    return y


def _draw_xray_measurements_section(
    c: pdf_canvas.Canvas,
    pages: "_ReportPages",
    y: float,
    ortho_rec: Any,
) -> float:
    """Draw X-ray AI profile measurement angles as a table."""
    import json as _json
    import numpy as _np

    landmarks_json = getattr(ortho_rec, "landmarks_json", None)
    if not landmarks_json:
        return y

    try:
        raw = _json.loads(landmarks_json)
    except Exception:
        return y

    if not isinstance(raw, list) or len(raw) < 19:
        return y

    try:
        pts = _np.array([[float(p[0]), float(p[1])] for p in raw[:19]], dtype=_np.float64)
    except Exception:
        return y

    SPECS = [
        ("a_2_13_14",  2, 13, 14, "Nose–Upper Lip–Lower Lip Angle",
         "Relates the nasal landmark to upper and lower lip contour positions along the profile."),
        ("a_2_15_8",   2, 15,  8, "Nasal–Lip Root–Chin Angle",
         "Profile angle spanning from the nasal region through the upper lip root to the chin prominence."),
        ("a_7_15_8",   7, 15,  8, "Mid-Profile Support Angle",
         "Middle-face to chin profile angle; contributes to overall profile convexity assessment."),
        ("a_9_15_8",   9, 15,  8, "Chin Contour–Lip Root–Chin Angle",
         "Lower profile shape between two chin-area landmarks and the upper lip root."),
        ("a_2_13_8",   2, 13,  8, "Facial Profile Convexity Angle",
         "Full profile span from the nasal region through the upper lip to the chin point."),
        ("a_3_13_14",  3, 13, 14, "Subnasale–Upper Lip–Lower Lip Angle",
         "Sub-nasal point to lip relationship; may reflect lip prominence relative to nasal base."),
        ("a_13_14_8", 13, 14,  8, "Lip–Chin Angle",
         "Angle from the upper lip through the lower lip to the chin; relates to lip-chin balance."),
        ("a_15_8_9",  15,  8,  9, "Chin Prominence Angle",
         "Angle at the chin point between the upper lip root and the chin contour landmark."),
    ]

    data_rows = []
    for feat_key, a_i, b_i, c_i, name, note in SPECS:
        try:
            A = pts[a_i - 1]; B = pts[b_i - 1]; C = pts[c_i - 1]
            BA = A - B; BC = C - B
            cos_a = _np.dot(BA, BC) / (_np.linalg.norm(BA) * _np.linalg.norm(BC) + 1e-8)
            ang = float(_np.degrees(_np.arccos(float(_np.clip(cos_a, -1.0, 1.0)))))
            data_rows.append([feat_key, name, f"{ang:.1f}°", note])
        except Exception:
            continue

    if not data_rows:
        return y

    col_widths = [72, 130, 44, CONTENT_W - 72 - 130 - 44]
    headers    = ["Feature", "Measurement", "Value", "Note"]

    y = draw_wrapped_table(
        c, headers, col_widths, data_rows, y, pages,
        section_title="X-Ray AI Profile Measurements",
        body_style=_NOTE_STYLE,
    )

    y = ensure_space_or_new_page_y(c, y, 20, pages)
    c.setFillColor(COLOR_MUTED)
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(
        MARGIN_L, y,
        "These are AI landmark-derived profile features, not official cephalometric normative measurements.",
    )
    y -= 16

    return y


def render_case_pdf(
    base_dir: str,
    case: Any,
    side: Any,
    side_points: Any,
    patient: Any,
    doctor_name: Optional[str],
    front_ns: Any,
    front_ns_points: Any,
    ortho_rec: Any = None,
) -> str:
    reports_dir = static_dir("reports", base_dir=base_dir)
    ensure_dir(reports_dir)
    pdf_path = os.path.join(reports_dir, f"case_{case.id}_report.pdf")

    patient = _resolve_patient(patient, case)

    doctor_name = doctor_name or "—"
    patient_name = getattr(patient, "name", None) if patient else "—"
    patient_code = "—"
    if patient:
        patient_code = (
            getattr(patient, "patient_code", None)
            or getattr(patient, "code", None)
            or "—"
        )
    age_str = _format_age(patient, case)
    gender_str = _format_gender(patient, case)
    doctor_comment = (getattr(case, "doctor_comment", None) or "").strip()
    report_date = datetime.now().strftime("%d %b %Y")
    created_str = case.created_at.strftime("%d %b %Y · %H:%M") if case.created_at else "—"
    case_date_str = "—"
    if getattr(case, "display_case_date", None):
        try:
            case_date_str = case.display_case_date.strftime("%d %b %Y")
        except Exception:
            case_date_str = str(case.display_case_date)
    elif case.created_at:
        case_date_str = case.created_at.strftime("%d %b %Y")

    c = pdf_canvas.Canvas(pdf_path, pagesize=A4)
    pages = _ReportPages(c, case.id, report_date)
    y = pages.new_page()

    # --- Cover / summary page ---
    y = draw_section_title(c, "Clinical analysis report", y, pages)

    info_rows = [
        ("Doctor", doctor_name),
        ("Patient", patient_name),
        ("Patient code", patient_code),
        ("Age", age_str),
        ("Gender", gender_str),
        ("Case number", f"#{case.id}"),
        ("Case date", case_date_str),
        ("Created", created_str),
    ]
    y = draw_info_card(c, info_rows, y)

    if doctor_comment:
        y = ensure_space_or_new_page_y(c, y, 60, pages)
        y = draw_section_title(c, "Clinical notes / diagnosis", y, pages)
        y = _draw_paragraph(c, doctor_comment, MARGIN_L, y, CONTENT_W, size=9, line_h=11)
        y -= SECTION_SPACING

    # --- Side view measurements ---
    side_measurements: List[Dict[str, Any]] = []
    side_table_rows: List[Dict[str, str]] = []

    side_img_path = None
    if side and getattr(side, "overlay_path", None):
        side_img_path = resolve_overlay_path(side.overlay_path, base_dir)

    if side and side_img_path and os.path.isfile(side_img_path) and side_points:
        try:
            pts = normalize_points(side_points)
            bgr = cv2.imread(side_img_path)
            if bgr is not None:
                side_measurements, side_table_rows = _prepare_side_measurements(
                    case.id, bgr, pts, reports_dir
                )
        except Exception:
            pass

    if side_table_rows:
        y = ensure_space_or_new_page_y(c, y, 80, pages)
        y = draw_measurement_table(c, side_table_rows, y, pages)

    if side_measurements:
        y = _draw_side_measurement_cards(c, pages, y, side_measurements)

    elif side and side_points and not side_measurements:
        y = ensure_space_or_new_page_y(c, y, 30, pages)
        c.setFillColor(COLOR_MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN_L, y, "Side view measurements could not be computed for this case.")
        y -= 20

    # --- Frontal section ---
    front_path = None
    if front_ns and getattr(front_ns, "overlay_path", None):
        front_path = resolve_overlay_path(front_ns.overlay_path, base_dir)

    if front_path and os.path.isfile(front_path) and front_ns_points and len(front_ns_points) >= 34:
        try:
            fm = calculate_frontal_measurements(front_ns_points)
            bgr = cv2.imread(front_path)
            if bgr is not None:
                pts = normalize_points(front_ns_points)
                h = fm.get("horizontal") or {}
                vis = _draw_frontal_overlays(bgr, pts, h)
                tmp = os.path.join(reports_dir, f"_pdf_{case.id}_front.jpg")
                cv2.imwrite(tmp, vis)

                frontal_rows = frontal_pdf_interpretation_rows(fm)

                y = ensure_space_or_new_page_y(c, y, 56, pages)
                y = draw_section_title(c, "Frontal non-smile analysis", y, pages)

                if frontal_rows:
                    y = draw_frontal_summary_table(c, frontal_rows, y, pages)

                y = _draw_frontal_visualization(c, pages, y, tmp, frontal_rows)

        except Exception:
            y = pages.new_page()
            c.setFillColor(COLOR_MUTED)
            c.setFont("Helvetica", 9)
            c.drawString(MARGIN_L, CONTENT_TOP - 24, "Frontal analysis could not be fully rendered for this PDF.")

    # --- X-ray overlay image ---
    if ortho_rec and getattr(ortho_rec, "overlay_path", None):
        y = _draw_xray_overlay_image(c, pages, y, ortho_rec, base_dir)

    # --- X-ray AI diagnosis ---
    if ortho_rec and getattr(ortho_rec, "diagnosis_json", None):
        y = _draw_xray_diagnosis_section(c, pages, y, ortho_rec)

    # --- X-ray AI profile measurements ---
    if ortho_rec and getattr(ortho_rec, "landmarks_json", None):
        y = _draw_xray_measurements_section(c, pages, y, ortho_rec)

    pages.finish()
    c.save()
    return join_stored("static", "reports", f"case_{case.id}_report.pdf")
