"""Doctor availability and patient appointment booking helpers."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from extensions import db  # noqa: F401 — shared SQLAlchemy instance (models register on this)

WEEKDAY_LABELS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

ACTIVE_APPOINTMENT_STATUSES = ("SCHEDULED", "REQUESTED", "BOOKED")


def format_time_12h(t: time) -> str:
    return datetime.combine(date.today(), t).strftime("%I:%M %p").lstrip("0")


def weekday_label(weekday: int) -> str:
    if 0 <= weekday <= 6:
        return WEEKDAY_LABELS[weekday]
    return "Unknown"


def _slot_taken(
    appointments: List[Any],
    appt_date: date,
    slot_time: time,
    exclude_id: Optional[int] = None,
) -> bool:
    for a in appointments:
        if exclude_id and a.id == exclude_id:
            continue
        if a.appointment_date == appt_date and a.appointment_time == slot_time:
            if a.status in ACTIVE_APPOINTMENT_STATUSES:
                return True
    return False


def get_doctor_active_appointments(doctor_id: int, from_date: Optional[date] = None) -> List[Any]:
    from models import Appointment

    q = Appointment.query.filter_by(doctor_id=doctor_id).filter(
        Appointment.status.in_(ACTIVE_APPOINTMENT_STATUSES)
    )
    if from_date:
        q = q.filter(Appointment.appointment_date >= from_date)
    return q.all()


def get_available_slots_for_date(
    doctor_id: int,
    appt_date: date,
    availability_rows: List[Any],
    appointments: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Slots on a given calendar date based on weekday availability."""
    if appointments is None:
        appointments = get_doctor_active_appointments(doctor_id, from_date=date.today())

    weekday = appt_date.weekday()
    day_slots = [s for s in availability_rows if s.weekday == weekday and s.is_active]
    day_slots.sort(key=lambda s: s.slot_time)

    result: List[Dict[str, Any]] = []
    for slot in day_slots:
        taken = _slot_taken(appointments, appt_date, slot.slot_time)
        result.append(
            {
                "time": slot.slot_time,
                "label": format_time_12h(slot.slot_time),
                "available": not taken,
                "availability_id": slot.id,
            }
        )
    return result


def get_bookable_dates(
    doctor_id: int,
    availability_rows: List[Any],
    days_ahead: int = 56,
) -> List[date]:
    """Dates that have at least one free slot in the doctor's schedule."""
    if not availability_rows:
        return []

    active_weekdays = {s.weekday for s in availability_rows if s.is_active}
    if not active_weekdays:
        return []

    start = date.today()
    appointments = get_doctor_active_appointments(doctor_id, from_date=start)
    bookable: List[date] = []

    for offset in range(days_ahead):
        d = start + timedelta(days=offset)
        if d.weekday() not in active_weekdays:
            continue
        slots = get_available_slots_for_date(doctor_id, d, availability_rows, appointments)
        if any(s["available"] for s in slots):
            bookable.append(d)

    return bookable


def group_availability_by_weekday(rows: List[Any]) -> Dict[int, List[Any]]:
    grouped: Dict[int, List[Any]] = {i: [] for i in range(7)}
    for row in rows:
        if row.is_active:
            grouped[row.weekday].append(row)
    for wd in grouped:
        grouped[wd].sort(key=lambda r: r.slot_time)
    return grouped
