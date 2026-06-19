"""Remove cases and patients with related DB rows and stored files."""

from __future__ import annotations

import os
from typing import Optional

from extensions import db
from models import (
    Appointment,
    Case,
    Image,
    Patient,
    PatientMessage,
    PatientUploadCode,
    ProgressComparison,
    Report,
    Result,
)
from utils.paths import resolve_project_path


def _safe_remove_file(stored_path: Optional[str]) -> None:
    if not stored_path:
        return
    abs_path = resolve_project_path(stored_path)
    if abs_path and os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass


def purge_case(case: Case) -> None:
    """Delete a case, its results/images/reports, and unlink appointments.

    Progress comparisons that reference this case (as baseline or as the new
    progress case) are also deleted — a comparison cannot remain valid when
    either of its constituent cases is gone.
    """
    case_id = case.id

    for img in Image.query.filter_by(case_id=case_id).all():
        _safe_remove_file(img.file_path)
    for res in Result.query.filter_by(case_id=case_id).all():
        _safe_remove_file(res.overlay_path)
    report = Report.query.filter_by(case_id=case_id).first()
    if report:
        _safe_remove_file(report.file_path)

    # Delete ProgressComparison rows that reference this case before deleting
    # the case itself.  Both FK columns are NOT NULL so SQLAlchemy must not
    # attempt to SET them to NULL; we remove the rows explicitly instead.
    ProgressComparison.query.filter(
        db.or_(
            ProgressComparison.baseline_case_id == case_id,
            ProgressComparison.new_case_id      == case_id,
        )
    ).delete(synchronize_session=False)

    Appointment.query.filter_by(case_id=case_id).update(
        {Appointment.case_id: None},
        synchronize_session=False,
    )
    Report.query.filter_by(case_id=case_id).delete(synchronize_session=False)
    Result.query.filter_by(case_id=case_id).delete(synchronize_session=False)
    Image.query.filter_by(case_id=case_id).delete(synchronize_session=False)
    db.session.delete(case)


def _purge_patient_related_rows(patient: Patient, doctor_id: Optional[int] = None) -> None:
    """
    Remove all rows that reference this patient (never NULL-out NOT NULL FKs).
    Order: cases (with images/results/reports) → upload codes → messages → appointments → patient.
    """
    if doctor_id is not None and patient.doctor_id != doctor_id:
        raise PermissionError("Not allowed to delete this patient.")

    case_query = Case.query.filter_by(patient_id=patient.id)
    if doctor_id is not None:
        case_query = case_query.filter_by(doctor_id=doctor_id)
    for case in case_query.all():
        purge_case(case)

    upload_code_query = PatientUploadCode.query.filter_by(patient_id=patient.id)
    message_query = PatientMessage.query.filter_by(patient_id=patient.id)
    appointment_query = Appointment.query.filter_by(patient_id=patient.id)
    if doctor_id is not None:
        upload_code_query = upload_code_query.filter_by(doctor_id=doctor_id)
        message_query = message_query.filter_by(doctor_id=doctor_id)
        appointment_query = appointment_query.filter_by(doctor_id=doctor_id)

    upload_code_query.delete(synchronize_session=False)
    message_query.delete(synchronize_session=False)
    appointment_query.delete(synchronize_session=False)

    # Flush explicit DELETEs before removing patient so ORM never NULLs NOT NULL FKs.
    db.session.flush()
    db.session.expire(
        patient,
        ["upload_codes", "cases", "messages", "appointments"],
    )
    db.session.delete(patient)


def purge_patient_for_doctor(patient: Patient, doctor_id: int) -> None:
    """Delete a patient and all data owned by this doctor for that patient."""
    _purge_patient_related_rows(patient, doctor_id=doctor_id)


def purge_patient_completely(patient: Patient) -> None:
    """Delete a patient and every dependent record (admin or full teardown)."""
    _purge_patient_related_rows(patient, doctor_id=None)
