from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sqlalchemy import cast, String, or_
from sqlalchemy.orm import joinedload
from datetime import datetime, date, time, timedelta
from utils.calibration import calculate_measurement
from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from utils.calibration import calculate_measurement
from utils.inference import predict_landmarks, save_overlay_image, landmarks_to_json

from utils.measurements import analyze_measurement
from utils.frontal_measurements import calculate_frontal_measurements
from utils.result_ui_data import (
    build_ai_insights,
    build_frontal_measurement_cards,
    build_measurement_treatment_advice,
    build_side_measurement_cards,
    build_xray_measurement_cards,
)
from utils.case_cleanup import purge_case, purge_patient_completely, purge_patient_for_doctor
from utils.case_pdf import render_case_pdf
from utils.analysis_pipeline import run_view_analysis
from utils.scheduling import (
    WEEKDAY_LABELS,
    ACTIVE_APPOINTMENT_STATUSES,
    format_time_12h,
    weekday_label,
    get_available_slots_for_date,
    get_bookable_dates,
    get_doctor_active_appointments,
)
from utils.paths import (
    normalize_stored_path,
    resolve_project_path,
    static_url_filename,
    join_stored,
    ensure_static_subdirs,
    upload_abs,
    upload_rel,
    results_abs,
    overlay_rel,
    resolve_overlay_path,
    static_dir,
)
from utils.image_validation import FRIENDLY_FAIL, MSG_ANALYSIS_FAILED
from utils.model_health import run_model_health_check
from utils.orthodontic_ai_inference import run_ortho_analysis, parse_xray_diagnosis_json
from utils.side_diagnosis import run_side_diagnosis, validate_pipeline_schemas
from utils.frontal_diagnosis import predict_frontal_diagnosis
from utils.progress_comparison import (
    build_progress_comparison,
    build_summary as build_comparison_summary,
    build_xray_progress_comparison,
    build_summary_xray as build_comparison_summary_xray,
    normalize_comparison_rows,
)
from utils.side_measurement_models import predict_side_measurement_analysis
from utils.model_loader import full_pipeline_from_bytes
from utils.landmark_preview import preview_to_png_bytes

# Doctor-reviewed Gemini profile-simulation service (backend-only)
from simulation import simulation_rules as sim_rules
from simulation import gemini_client as sim_gemini
from simulation.profile_simulation import (
    run_profile_simulation,
    ApprovedFinding,
    SimulationValidationError,
)
from simulation.gemini_client import GeminiUnavailableError, GeminiGenerationError

# Esthetic Smile Adjustment feature
from services.smile_adjustment_service import run_smile_adjustment, SmileImageError
from utils.smile_adjustment_prompt import SmileSelectionError

# Load environment variables from the .env beside this file (Gemini key, etc.)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
                 override=False)
except Exception:
    pass

import io
import os
import uuid
import json
import re
import secrets
import string
import cv2

from utils.inference import predict_landmarks
from utils.inference import draw_points

#verification
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def is_valid_email(email: str) -> bool:
    if not email:
        return False
    return bool(EMAIL_RE.match(email.strip().lower()))

def is_valid_password(pw: str, min_len: int = 8) -> bool:
    if not pw:
        return False
    return len(pw) >= min_len

def clean_email(email: str) -> str:
    return (email or "").strip().lower()

def clean_name(name: str) -> str:
    return " ".join((name or "").strip().split())

def generate_link_code(length=10):
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


# ── Medical history – valid condition keys (Phase 1) ──────────────
VALID_MEDICAL_CONDITIONS = {
    "no_known_conditions",
    "diabetes",
    "high_blood_pressure",
    "low_blood_pressure",
    "heart_disease",
    "arrhythmia",
    "previous_heart_surgery",
    "asthma_respiratory",
    "bleeding_disorders",
    "anemia",
    "kidney_disease",
    "liver_disease",
    "thyroid_problems",
    "epilepsy_seizures",
    "medication_allergies",
    "other_allergies",
    "regular_medications",
    "previous_major_surgeries",
    "pregnancy",
    "other",
}

MEDICAL_CONDITION_LABELS = {
    "no_known_conditions": "No known medical conditions",
    "diabetes": "Diabetes",
    "high_blood_pressure": "High blood pressure",
    "low_blood_pressure": "Low blood pressure",
    "heart_disease": "Heart disease",
    "arrhythmia": "Irregular heart rate / arrhythmia",
    "previous_heart_surgery": "Previous heart surgery",
    "asthma_respiratory": "Asthma or respiratory problems",
    "bleeding_disorders": "Blood-clotting or bleeding disorders",
    "anemia": "Anemia",
    "kidney_disease": "Kidney disease",
    "liver_disease": "Liver disease",
    "thyroid_problems": "Thyroid problems",
    "epilepsy_seizures": "Epilepsy or seizures",
    "medication_allergies": "Medication allergies",
    "other_allergies": "Other allergies",
    "regular_medications": "Currently taking regular medications",
    "previous_major_surgeries": "Previous major surgeries",
    "pregnancy": "Pregnancy",
    "other": "Other",
}


def _parse_medical_history_form(form):
    """Parse and validate medical history form data. Returns a dict of clean values."""
    import html as _html

    raw_conditions = form.getlist("medical_conditions")
    conditions = [c for c in raw_conditions if c in VALID_MEDICAL_CONDITIONS]

    # Mutual exclusivity: no_known_conditions trumps everything
    if "no_known_conditions" in conditions:
        conditions = ["no_known_conditions"]

    other_details = _html.escape((form.get("medical_other_details") or "").strip())[:2000]
    if "other" not in conditions:
        other_details = ""

    medications = _html.escape((form.get("current_medications") or "").strip())[:2000]
    allergies_text = _html.escape((form.get("allergies") or "").strip())[:2000]
    surgeries = _html.escape((form.get("previous_surgeries") or "").strip())[:2000]
    notes = _html.escape((form.get("additional_medical_notes") or "").strip())[:5000]

    return {
        "conditions": conditions,
        "other_details": other_details,
        "medications": medications,
        "allergies": allergies_text,
        "surgeries": surgeries,
        "notes": notes,
    }


def _apply_medical_history(patient, parsed):
    """Apply parsed medical history data to a Patient model instance."""
    patient.medical_conditions_json = json.dumps(parsed["conditions"]) if parsed["conditions"] else "[]"
    patient.medical_other_details = parsed["other_details"] or None
    patient.current_medications = parsed["medications"] or None
    patient.allergies = parsed["allergies"] or None
    patient.previous_surgeries = parsed["surgeries"] or None
    patient.additional_medical_notes = parsed["notes"] or None
    patient.medical_history_recorded = True


def patient_login_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if "patient_auth_id" not in session:
            return redirect(url_for("patient_login"))
        return route_function(*args, **kwargs)
    return wrapper

app = Flask(__name__)
app.secret_key = "change-this-to-any-random-string"

@app.template_filter("from_json")
def from_json_filter(value):
    """Jinja2 filter: parse a JSON string into a Python object."""
    try:
        return json.loads(value) if value else {}
    except Exception:
        return {}

# Database config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

from extensions import db

db.init_app(app)

from models import (  # noqa: E402 — register models after db.init_app
    Appointment,
    Case,
    DoctorAvailability,
    Image,
    Patient,
    PatientAuth,
    PatientMessage,
    PatientUploadCode,
    OrthoCase,
    Report,
    Result,
    FrontalDiagnosis,
    ProgressComparison,
    SideDiagnosis,
    TreatmentSimulation,
    MeasurementReview,
    ProfileSimulation,
    User,
)


def _ensure_sqlite_columns():
    """Lightweight migration for columns added after first deploy."""
    from sqlalchemy import inspect, text

    try:
        insp = inspect(db.engine)
        if insp.has_table("case"):
            cols = {c["name"] for c in insp.get_columns("case")}
            with db.engine.begin() as conn:
                if "failure_message" not in cols:
                    conn.execute(text("ALTER TABLE \"case\" ADD COLUMN failure_message TEXT"))
                if "case_date" not in cols:
                    conn.execute(text('ALTER TABLE "case" ADD COLUMN case_date DATE'))
                if "follow_up_requested" not in cols:
                    conn.execute(
                        text('ALTER TABLE "case" ADD COLUMN follow_up_requested BOOLEAN DEFAULT 0')
                    )
                if "patient_id" not in cols:
                    conn.execute(text('ALTER TABLE "case" ADD COLUMN patient_id INTEGER'))
        if insp.has_table("appointment"):
            acols = {c["name"] for c in insp.get_columns("appointment")}
            with db.engine.begin() as conn:
                if "case_id" not in acols:
                    conn.execute(text("ALTER TABLE appointment ADD COLUMN case_id INTEGER"))
                if "source" not in acols:
                    conn.execute(
                        text("ALTER TABLE appointment ADD COLUMN source VARCHAR(20) DEFAULT 'doctor'")
                    )
        if not insp.has_table("doctor_availability"):
            db.create_all()
        if insp.has_table("patient"):
            pcols = {c["name"] for c in insp.get_columns("patient")}
            with db.engine.begin() as conn:
                if "private_notes" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN private_notes TEXT"))
                if "private_notes_updated_at" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN private_notes_updated_at DATETIME"))
                # ── Medical history (Phase 1) ──
                if "medical_conditions_json" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN medical_conditions_json TEXT"))
                if "medical_other_details" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN medical_other_details TEXT"))
                if "current_medications" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN current_medications TEXT"))
                if "allergies" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN allergies TEXT"))
                if "previous_surgeries" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN previous_surgeries TEXT"))
                if "additional_medical_notes" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN additional_medical_notes TEXT"))
                if "medical_history_recorded" not in pcols:
                    conn.execute(text("ALTER TABLE patient ADD COLUMN medical_history_recorded BOOLEAN"))
        if not insp.has_table("patient_message"):
            db.create_all()
        if not insp.has_table("ortho_case"):
            db.create_all()
        elif insp.has_table("ortho_case"):
            ocols = {c["name"] for c in insp.get_columns("ortho_case")}
            with db.engine.begin() as conn:
                if "case_id" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN case_id INTEGER"))
                if "landmarks_json" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN landmarks_json TEXT"))
                if "diagnosis_json" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN diagnosis_json TEXT"))
                if "overlay_path" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN overlay_path VARCHAR(300)"))
                if "status" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN status VARCHAR(20) DEFAULT 'PENDING'"))
                if "error_message" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN error_message TEXT"))
                if "reviewed" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN reviewed BOOLEAN DEFAULT 0"))
                if "reviewed_at" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN reviewed_at DATETIME"))
                if "doctor_final_diagnosis" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN doctor_final_diagnosis TEXT"))
                if "doctor_review_notes" not in ocols:
                    conn.execute(text("ALTER TABLE ortho_case ADD COLUMN doctor_review_notes TEXT"))
        # result.confidence_json — added when landmark confidence feature was introduced
        if insp.has_table("result"):
            rcols = {c["name"] for c in insp.get_columns("result")}
            with db.engine.begin() as conn:
                if "confidence_json" not in rcols:
                    conn.execute(text("ALTER TABLE result ADD COLUMN confidence_json TEXT"))
        # treatment_simulation table (created via db.create_all if missing)
        if not insp.has_table("treatment_simulation"):
            db.create_all()
        # side_diagnosis — XGBoost angles pipeline result table
        if not insp.has_table("side_diagnosis"):
            db.create_all()
        # frontal_diagnosis — six-model frontal pipeline result table
        if not insp.has_table("frontal_diagnosis"):
            db.create_all()
        # measurement_review — doctor approve/decline state per measurement
        if not insp.has_table("measurement_review"):
            db.create_all()
        elif insp.has_table("measurement_review"):
            mrcols = {c["name"] for c in insp.get_columns("measurement_review")}
            with db.engine.begin() as conn:
                if "treatment_explanation_json" not in mrcols:
                    conn.execute(text(
                        "ALTER TABLE measurement_review "
                        "ADD COLUMN treatment_explanation_json TEXT"))
        # profile_simulation — saved Gemini profile-simulation results
        if not insp.has_table("profile_simulation"):
            db.create_all()
        # progress_comparison — saved progress comparison records (Phase 2)
        if not insp.has_table("progress_comparison"):
            db.create_all()
        elif insp.has_table("progress_comparison"):
            pccols = {c["name"] for c in insp.get_columns("progress_comparison")}
            with db.engine.begin() as conn:
                if "analysis_type" not in pccols:
                    conn.execute(text(
                        "ALTER TABLE progress_comparison ADD COLUMN analysis_type VARCHAR(10)"
                    ))
    except Exception:
        pass


def _normalize_database_paths():
    """Normalize legacy Windows-style backslashes in stored paths."""
    changed = False
    for img in Image.query.all():
        norm = normalize_stored_path(img.file_path)
        if norm and norm != img.file_path:
            img.file_path = norm
            changed = True
    for res in Result.query.all():
        norm = normalize_stored_path(res.overlay_path)
        if norm and norm != res.overlay_path:
            res.overlay_path = norm
            changed = True
    for rep in Report.query.all():
        norm = normalize_stored_path(rep.file_path)
        if norm and norm != rep.file_path:
            rep.file_path = norm
            changed = True
    if changed:
        db.session.commit()


def _parse_case_date(form_value: str):
    """Parse case date from form; default to today."""
    if not form_value or not str(form_value).strip():
        return date.today()
    try:
        return datetime.strptime(str(form_value).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _next_patient_appointment(doctor_id: int, patient_id: int):
    return (
        Appointment.query.filter_by(
            doctor_id=doctor_id,
            patient_id=patient_id,
        )
        .filter(Appointment.status.in_(ACTIVE_APPOINTMENT_STATUSES))
        .filter(Appointment.appointment_date >= date.today())
        .order_by(Appointment.appointment_date.asc(), Appointment.appointment_time.asc())
        .first()
    )


def _patient_portal_reports(patient_id, doctor_id):
    """PDF reports available for a patient (cases with generated reports)."""
    rows = []
    cases = (
        Case.query.filter_by(patient_id=patient_id, doctor_id=doctor_id)
        .order_by(Case.created_at.desc())
        .all()
    )
    for c in cases:
        report = Report.query.filter_by(case_id=c.id).first()
        if not report:
            continue
        rows.append(
            {
                "case": c,
                "report": report,
                "title": f"Case #{c.id} — {case_type_label(c.case_type)} analysis",
            }
        )
    return rows


def _month_start():
    now = datetime.now()
    return datetime(now.year, now.month, 1)


def _week_start():
    return datetime.now() - timedelta(days=7)


def format_relative_time(dt):
    if not dt:
        return ""
    now = datetime.now()
    if getattr(dt, "tzinfo", None):
        dt = dt.replace(tzinfo=None)
    diff = now - dt
    if diff.days == 0:
        return f"Today, {dt.strftime('%H:%M')}"
    if diff.days == 1:
        return "Yesterday"
    if diff.days < 7:
        return f"{diff.days} days ago"
    return dt.strftime("%Y-%m-%d")


def case_status_display(status):
    st = (status or "").upper()
    if st in ("REVIEWED", "REVIWED", "COMPLETED"):
        return "Analyzed", "analyzed"
    if st == "PENDING_REVIEW":
        return "In review", "review"
    if st in ("PROCESSING", "PENDING"):
        return "Pending", "pending"
    if st in ("NEEDS_REUPLOAD", "FAILED"):
        return "Needs attention", "warn"
    return st.replace("_", " ").title(), "default"


def case_type_label(case_type):
    ct = (case_type or "INITIAL").upper()
    if ct == "FOLLOW_UP":
        return "Follow-up"
    return "Initial"


def get_or_create_patient_access_code(patient_id: int, doctor_id: int) -> PatientUploadCode:
    """One stable portal code per patient per doctor."""
    existing = (
        PatientUploadCode.query.filter_by(
            patient_id=patient_id,
            doctor_id=doctor_id,
        )
        .order_by(PatientUploadCode.created_at.asc())
        .first()
    )
    if existing:
        return existing

    code_value = generate_link_code()
    while PatientUploadCode.query.filter_by(code=code_value).first():
        code_value = generate_link_code()

    row = PatientUploadCode(
        code=code_value,
        doctor_id=doctor_id,
        patient_id=patient_id,
    )
    db.session.add(row)
    db.session.commit()
    return row


def regenerate_patient_access_code(patient_id: int, doctor_id: int) -> PatientUploadCode:
    """Replace portal code; old code stops working immediately."""
    existing = (
        PatientUploadCode.query.filter_by(
            patient_id=patient_id,
            doctor_id=doctor_id,
        )
        .order_by(PatientUploadCode.created_at.asc())
        .first()
    )

    code_value = generate_link_code()
    while PatientUploadCode.query.filter_by(code=code_value).first():
        code_value = generate_link_code()

    if existing:
        existing.code = code_value
        existing.created_at = datetime.utcnow()
        db.session.commit()
        return existing

    row = PatientUploadCode(
        code=code_value,
        doctor_id=doctor_id,
        patient_id=patient_id,
    )
    db.session.add(row)
    db.session.commit()
    return row


def _failure_status_for_outcomes(failures: list) -> str:
    """Use FAILED for hard inference errors; NEEDS_REUPLOAD for fixable upload issues."""
    stages = {f.get("failed_stage") for f in failures if f.get("uploaded")}
    if stages == {"inference"}:
        return "FAILED"
    return "NEEDS_REUPLOAD"


def build_case_view_summary(case: Case):
    """Per-view status for doctor UI: uploaded image vs successful result."""
    images = {img.view_type: img for img in (case.images or [])}
    results = {r.view_type: r for r in (case.results or [])}
    rows = []
    for view_type, label in (("SIDE", "Side view"), ("FRONT_NS", "Frontal non-smile")):
        if view_type not in images:
            continue
        if view_type in results:
            status = "analyzed"
            status_label = "Analyzed"
        else:
            status = "needs_reupload"
            status_label = "Needs re-upload"
        rows.append(
            {
                "view_type": view_type,
                "label": label,
                "status": status,
                "status_label": status_label,
            }
        )
    return rows


def _finalize_case_after_analysis(case: Case, outcomes: list) -> None:
    """
    outcomes: list of dicts with keys uploaded (bool), success (bool), message (str)
    """
    uploaded = [o for o in outcomes if o.get("uploaded")]
    successes = [o for o in uploaded if o.get("success")]
    failures = [o for o in uploaded if not o.get("success")]

    if not uploaded:
        case.status = "FAILED"
        case.failure_message = "No images were uploaded."
        return

    if failures and not successes:
        case.status = _failure_status_for_outcomes(failures)
        case.failure_message = failures[0].get("message") or MSG_ANALYSIS_FAILED
        return

    if failures:
        case.status = "NEEDS_REUPLOAD"
        case.failure_message = failures[0].get("message") or MSG_ANALYSIS_FAILED
        return

    case.status = "PENDING_REVIEW"
    case.failure_message = None


def generate_case_pdf(
    case,
    side,
    side_points,
    patient=None,
    doctor_name=None,
    front_ns=None,
    front_ns_points=None,
    ortho_rec=None,
):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return render_case_pdf(
        base_dir,
        case,
        side,
        side_points,
        patient,
        doctor_name,
        front_ns,
        front_ns_points,
        ortho_rec=ortho_rec,
    )


def login_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return route_function(*args, **kwargs)
    return wrapper


def admin_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "ADMIN":
            flash("Admin access only.", "error")
            return redirect(url_for("dashboard"))
        return route_function(*args, **kwargs)
    return wrapper


# Create tables once
with app.app_context():
    db.create_all()
    _ensure_sqlite_columns()

# Validate ML pipeline schemas at startup — fails loudly if PKL files are wrong
try:
    validate_pipeline_schemas()
except ValueError as _schema_err:
    import sys
    print(f"\n{'='*70}")
    print("STARTUP ERROR — ML pipeline schema mismatch detected:")
    print(str(_schema_err))
    print(f"{'='*70}\n")
    # Do NOT sys.exit() — let the server start so other routes still work
    # and the error is visible in logs.

with app.app_context():
    db.create_all()
    _ensure_sqlite_columns()

    # ---- Seed Admin User (only if not exists) ----
    admin_email = "admin@gmail.com"
    admin_password = "12345678"

    existing_admin = User.query.filter_by(email=admin_email).first()
    if not existing_admin:
        admin_user = User(
            name="Admin",
            email=admin_email,
            password_hash=generate_password_hash(admin_password),
            role="ADMIN"
        )
        admin_user.set_password(admin_password)
        db.session.add(admin_user)
        db.session.commit()

# ------------------------
# Routes
# ------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = clean_email(request.form.get("email"))
#         password = request.form.get("password", "")
#         # VERIFICATION (format)
#         if not is_valid_email(email):
#             flash("Please enter a valid email.", "error")
#             return redirect(url_for("login"))

#         if not password:
#             flash("Password is required.", "error")
#             return redirect(url_for("login"))

#         # VALIDATION (correct credentials)
#         user = User.query.filter_by(email=email).first()
#         if user and check_password_hash(user.password_hash, password):
#             session.clear()
#             session["user_id"] = user.id
#             session["user_name"] = user.name
#             session["role"] = user.role
#             session["account_type"] = "STAFF"

#             if user.role == "ADMIN":
#                 return redirect(url_for("admin_dashboard"))

#             return redirect(url_for("dashboard"))
        
#         patient_user= PatientAuth.query.filter_by(email=email).first()
#         if patient_user and patient_user.check_password(password):
#             session.clear()
#             session["patient_auth_id"]= patient_user.id
#             session["patient_name"]= patient_user.name
#             session["account_type"]= "PATIENT"

#             return redirect(url_for("patient_dashboard"))
        
#         flash("Invalid email or password." , "error")
#         return redirect(url_for('login'))
    
#     return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = clean_email(request.form.get("email"))
        password = request.form.get("password", "")

        if not is_valid_email(email):
            flash("Please enter a valid email.", "error")
            return redirect(url_for("login"))

        if not password:
            flash("Password is required.", "error")
            return redirect(url_for("login"))

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        session.clear()
        session["user_id"] = user.id
        session["user_name"] = user.name
        session["role"] = user.role

        if user.role == "ADMIN":
            return redirect(url_for("admin_dashboard"))

        return redirect(url_for("dashboard"))

    return render_template("login.html")

    
    
        



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = clean_name(request.form.get("name"))
        email = clean_email(request.form.get("email"))
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        # -------------------
        # VERIFICATION (format)
        # -------------------
        if not name:
            flash("Name is required.", "error")
            return redirect(url_for("register"))

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("register"))

        if not is_valid_password(password, min_len=8):
            flash("Password must be at least 8 characters.", "error")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        # -------------------
        # VALIDATION (business rules)
        # -------------------
        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("login"))

        # Default role for new accounts = DOCTOR
        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role="DOCTOR"
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# @app.route("/register", methods=["GET", "POST"])
# def register():
#     flash("Doctor registration is restricted. Please contact the admin.", "error")
#     return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    doctor_id = session["user_id"]
    now = datetime.now()
    today = now.date()

    total_patients = Patient.query.filter_by(doctor_id=doctor_id).count()
    total_cases = Case.query.filter_by(doctor_id=doctor_id).count()

    needs_review_statuses = ("PENDING", "PENDING_REVIEW", "PROCESSING")
    pending_review = Case.query.filter(
        Case.doctor_id == doctor_id,
        Case.status.in_(needs_review_statuses),
    ).count()

    today_appointments = (
        Appointment.query.filter_by(doctor_id=doctor_id, appointment_date=today)
        .filter(Appointment.status != "CANCELLED")
        .order_by(Appointment.appointment_time.asc())
        .all()
    )
    appointments_today = len(today_appointments)
    appointments_remaining = sum(
        1
        for a in today_appointments
        if a.status == "SCHEDULED" and a.starts_at >= now
    )

    recent_cases = (
        Case.query.filter_by(doctor_id=doctor_id)
        .order_by(Case.created_at.desc())
        .limit(5)
        .all()
    )

    recent_case_rows = []
    for c in recent_cases:
        label, tone = case_status_display(c.status)
        recent_case_rows.append(
            {
                "case": c,
                "patient": c.patient,
                "status_label": label,
                "status_tone": tone,
                "relative_time": format_relative_time(c.created_at),
                "type_label": case_type_label(c.case_type),
            }
        )

    patients_for_appt = Patient.query.filter_by(doctor_id=doctor_id).order_by(Patient.name).all()

    unread_messages = PatientMessage.query.filter_by(
        doctor_id=doctor_id, read=False
    ).count()
    recent_messages = (
        PatientMessage.query.filter_by(doctor_id=doctor_id)
        .order_by(PatientMessage.created_at.desc())
        .limit(5)
        .all()
    )

    return render_template(
        "dashboard.html",
        name=session.get("user_name"),
        active="dashboard",
        total_patients=total_patients,
        total_cases=total_cases,
        pending_review=pending_review,
        appointments_today=appointments_today,
        appointments_remaining=appointments_remaining,
        today_appointments=today_appointments,
        recent_case_rows=recent_case_rows,
        patients_for_appt=patients_for_appt,
        today_iso=today.isoformat(),
        unread_messages=unread_messages,
        recent_messages=recent_messages,
    )


@app.route("/messages/<int:message_id>/read", methods=["POST"])
@login_required
def mark_message_read(message_id):
    msg = PatientMessage.query.filter_by(
        id=message_id, doctor_id=session["user_id"]
    ).first_or_404()
    msg.read = True
    db.session.commit()
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/appointments", methods=["GET"])
@login_required
def appointments():
    doctor_id = session["user_id"]
    patients_list = Patient.query.filter_by(doctor_id=doctor_id).order_by(Patient.name).all()

    upcoming = (
        Appointment.query.filter_by(doctor_id=doctor_id)
        .filter(Appointment.status != "CANCELLED")
        .filter(Appointment.appointment_date >= date.today())
        .order_by(Appointment.appointment_date.asc(), Appointment.appointment_time.asc())
        .limit(50)
        .all()
    )

    past = (
        Appointment.query.filter_by(doctor_id=doctor_id)
        .filter(
            (Appointment.appointment_date < date.today())
            | (Appointment.status == "COMPLETED")
        )
        .order_by(Appointment.appointment_date.desc(), Appointment.appointment_time.desc())
        .limit(30)
        .all()
    )

    return render_template(
        "appointments.html",
        name=session.get("user_name"),
        active="appointments",
        patients=patients_list,
        upcoming=upcoming,
        past=past,
        today_iso=date.today().isoformat(),
    )


@app.route("/appointments/add", methods=["POST"])
@login_required
def add_appointment():
    doctor_id = session["user_id"]
    reason = (request.form.get("reason") or "").strip()
    date_str = (request.form.get("appointment_date") or "").strip()
    time_str = (request.form.get("appointment_time") or "").strip()
    patient_id = (request.form.get("patient_id") or "").strip()
    patient_code_input = (request.form.get("patient_code") or "").strip().upper()
    notes = (request.form.get("notes") or "").strip()
    redirect_to = request.form.get("redirect_to") or "appointments"
    case_id = request.form.get("case_id", type=int)
    if redirect_to == "dashboard":
        back = url_for("dashboard")
    elif redirect_to == "result" and case_id:
        back = url_for("view_result", case_id=case_id)
    elif redirect_to == "appointments":
        back = url_for("appointments")
    else:
        back = url_for("appointments")

    if not reason:
        flash("Appointment reason is required.", "error")
        return redirect(back)

    try:
        appt_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        appt_time = datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        flash("Please enter a valid date and time.", "error")
        return redirect(back)

    pid = None
    if patient_id and patient_id.isdigit():
        patient = Patient.query.filter_by(id=int(patient_id), doctor_id=doctor_id).first()
        if not patient:
            flash("Patient not found.", "error")
            return redirect(back)
        pid = patient.id
    elif patient_code_input:
        patient = Patient.query.filter_by(
            patient_code=patient_code_input, doctor_id=doctor_id
        ).first()
        if not patient:
            flash(f"No patient found with code {patient_code_input}.", "error")
            return redirect(back)
        pid = patient.id

    case_id = request.form.get("case_id", type=int)
    if case_id:
        linked = Case.query.filter_by(id=case_id, doctor_id=doctor_id).first()
        if not linked:
            case_id = None

    # Duplicate guard: don't create a second SCHEDULED appointment for the same
    # doctor + patient + case + date + time.
    dup_filter = Appointment.query.filter_by(
        doctor_id=doctor_id,
        appointment_date=appt_date,
        appointment_time=appt_time,
        status="SCHEDULED",
    )
    if pid:
        dup_filter = dup_filter.filter_by(patient_id=pid)
    if case_id:
        dup_filter = dup_filter.filter_by(case_id=case_id)
    if dup_filter.first():
        flash(
            "An appointment for this patient at that date and time is already scheduled.",
            "warning",
        )
        return redirect(back)

    appt = Appointment(
        doctor_id=doctor_id,
        patient_id=pid,
        case_id=case_id,
        reason=reason,
        appointment_date=appt_date,
        appointment_time=appt_time,
        notes=notes or None,
        status="SCHEDULED",
        source="doctor",
    )
    db.session.add(appt)
    db.session.commit()

    flash("Appointment scheduled.", "success")
    return redirect(back)


@app.route("/appointments/<int:appointment_id>/complete", methods=["POST"])
@login_required
def complete_appointment(appointment_id):
    appt = Appointment.query.filter_by(
        id=appointment_id, doctor_id=session["user_id"]
    ).first_or_404()
    appt.status = "COMPLETED"
    db.session.commit()
    flash("Appointment marked complete.", "success")
    return redirect(request.referrer or url_for("appointments"))


@app.route("/appointments/<int:appointment_id>/cancel", methods=["POST"])
@login_required
def cancel_appointment(appointment_id):
    appt = Appointment.query.filter_by(
        id=appointment_id, doctor_id=session["user_id"]
    ).first_or_404()
    appt.status = "CANCELLED"
    db.session.commit()
    flash("Appointment cancelled.", "success")
    return redirect(request.referrer or url_for("appointments"))


@app.route("/appointments/<int:appointment_id>/delete", methods=["POST"])
@login_required
def delete_appointment(appointment_id):
    appt = Appointment.query.filter_by(
        id=appointment_id, doctor_id=session["user_id"]
    ).first_or_404()
    db.session.delete(appt)
    db.session.commit()
    flash("Appointment removed.", "success")
    return redirect(request.referrer or url_for("appointments"))


@app.route("/new-analysis", methods=["GET", "POST"])
@login_required
def new_analysis():

    # =========================
    # GET → show form + patients
    # =========================
    if request.method == "GET":
        patients = Patient.query.filter_by(
            doctor_id=session["user_id"]
        ).order_by(Patient.patient_code).all()

        return render_template(
            "new_analysis.html",
            name=session.get("user_name"),
            active="new_analysis",
            patients=patients,
            today_iso=date.today().isoformat(),
            medical_condition_labels=MEDICAL_CONDITION_LABELS,
        )

    # =========================
    # POST → run analysis (any combination of front / side / xray)
    # =========================
    side_file     = request.files.get("side")
    front_ns_file = request.files.get("front_ns")
    xray_file     = request.files.get("xray")

    def _upload_nonempty(f):
        return f and getattr(f, "filename", None) and str(f.filename).strip() != ""

    has_side     = _upload_nonempty(side_file)
    has_front_ns = _upload_nonempty(front_ns_file)
    has_xray     = _upload_nonempty(xray_file)

    if not has_side and not has_front_ns and not has_xray:
        flash("Please upload at least one image: frontal, side, or cephalometric X-ray.", "error")
        return redirect(url_for("new_analysis"))

    # Quick-add fields (top form)
    new_name   = (request.form.get("new_name") or "").strip()
    new_code   = (request.form.get("new_code") or "").strip()
    new_age    = (request.form.get("new_age") or "").strip()
    new_gender = (request.form.get("new_gender") or "").strip().upper()

    # Existing patient selection
    patient_id = (request.form.get("patient_id") or "").strip()

    using_new = bool(new_code or new_name or new_age or new_gender)
    using_existing = bool(patient_id)

    if using_new and using_existing:
        flash("Please choose only one option: select an existing patient or add a new one.", "error")
        return redirect(url_for("new_analysis"))

    # -------------------------
    # Decide which patient to use
    # -------------------------
    patient = None

    if using_new:
        if not new_name:
            flash("Please enter the patient name.", "error")
            return redirect(url_for("new_analysis"))

        if not new_code:
            new_code = f"PT-{uuid.uuid4().hex[:6].upper()}"
        else:
            new_code = new_code.upper()

        age_val = None
        if new_age:
            try:
                age_val = int(new_age)
                if age_val < 1 or age_val > 119:
                    raise ValueError()
            except ValueError:
                flash("Patient age must be a number between 1 and 119.", "error")
                return redirect(url_for("new_analysis"))

        if new_gender and new_gender not in ("MALE", "FEMALE"):
            flash("Please select Male, Female, or leave gender empty.", "error")
            return redirect(url_for("new_analysis"))

        patient = Patient.query.filter_by(
            patient_code=new_code,
            doctor_id=session["user_id"],
        ).first()

        if patient:
            # Existing patient — update basic fields only; do not touch medical history
            patient.name = new_name
            patient.age = age_val
            patient.gender = new_gender if new_gender else None
            db.session.commit()
        else:
            # Brand-new patient — apply medical history BEFORE add/commit so it
            # is included in the single INSERT (same pattern as doctor_add_patient).
            patient = Patient(
                patient_code=new_code,
                name=new_name,
                age=age_val,
                gender=new_gender if new_gender else None,
                doctor_id=session["user_id"],
            )
            _apply_medical_history(patient, _parse_medical_history_form(request.form))
            db.session.add(patient)
            db.session.commit()

    else:
        if not patient_id:
            flash("Please select a patient (or add a new one).", "error")
            return redirect(url_for("new_analysis"))

        patient = Patient.query.filter_by(
            id=patient_id,
            doctor_id=session["user_id"]
        ).first()

        if not patient:
            flash("Selected patient not found.", "error")
            return redirect(url_for("new_analysis"))

    upload_notes = (request.form.get("upload_private_notes") or "").strip()
    if upload_notes:
        patient.private_notes = upload_notes
        patient.private_notes_updated_at = datetime.utcnow()

    case_date_val = _parse_case_date(request.form.get("case_date") or "")
    if case_date_val is None:
        flash("Please enter a valid case date.", "error")
        return redirect(url_for("new_analysis"))

    # =========================
    # Create case
    # =========================
    ensure_static_subdirs("uploads", "results")
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    new_case = Case(
        doctor_id=session["user_id"],
        patient_id=patient.id,
        case_type="INITIAL",
        status="PROCESSING",
        case_date=case_date_val,
    )
    db.session.add(new_case)
    db.session.commit()

    outcomes = []
    try:
        if has_side:
            side_name = f"{new_case.id}_side_{uuid.uuid4().hex}.jpg"
            side_rel = upload_rel(side_name)
            side_abs = upload_abs(side_name, BASE_DIR)
            side_file.save(side_abs)
            db.session.add(Image(case_id=new_case.id, view_type="SIDE", file_path=side_rel))
            db.session.commit()

            side_out = run_view_analysis(new_case.id, side_abs, "SIDE")
            if side_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="SIDE",
                        landmarks_json=side_out["landmarks_json"],
                        overlay_path=normalize_stored_path(side_out["overlay_path"]),
                        confidence_json=side_out.get("confidence_json"),
                    )
                )
            outcomes.append(
                {
                    "uploaded": True,
                    "success": side_out["success"],
                    "message": side_out.get("message") or "",
                    "failed_stage": side_out.get("failed_stage"),
                }
            )

        if has_front_ns:
            front_name = f"{new_case.id}_front_ns_{uuid.uuid4().hex}.jpg"
            front_rel = upload_rel(front_name)
            front_abs = upload_abs(front_name, BASE_DIR)
            front_ns_file.save(front_abs)
            db.session.add(Image(case_id=new_case.id, view_type="FRONT_NS", file_path=front_rel))
            db.session.commit()

            front_out = run_view_analysis(new_case.id, front_abs, "FRONT_NS")
            if front_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="FRONT_NS",
                        landmarks_json=front_out["landmarks_json"],
                        overlay_path=normalize_stored_path(front_out["overlay_path"]),
                        confidence_json=front_out.get("confidence_json"),
                    )
                )
            outcomes.append(
                {
                    "uploaded": True,
                    "success": front_out["success"],
                    "message": front_out.get("message") or "",
                    "failed_stage": front_out.get("failed_stage"),
                }
            )

        # ── X-Ray (optional, independent) ──────────────────────
        if has_xray:
            xray_name = f"{new_case.id}_xray_{uuid.uuid4().hex}.jpg"
            xray_upload_dir = os.path.join(BASE_DIR, "static", "uploads", "ortho")
            os.makedirs(xray_upload_dir, exist_ok=True)
            xray_abs = os.path.join(xray_upload_dir, xray_name)
            xray_file.save(xray_abs)

            xray_overlay_dir = os.path.join(BASE_DIR, "static", "results", "ortho")
            os.makedirs(xray_overlay_dir, exist_ok=True)
            xray_overlay_abs = os.path.join(xray_overlay_dir, f"overlay_{xray_name}")

            ortho_record = OrthoCase(
                doctor_id=session["user_id"],
                patient_id=new_case.patient_id,
                case_id=new_case.id,
                image_path=os.path.join("static", "uploads", "ortho", xray_name),
                status="PENDING",
            )
            db.session.add(ortho_record)
            db.session.commit()

            xray_result = run_ortho_analysis(xray_abs, xray_overlay_abs)
            if xray_result["success"]:
                ortho_record.overlay_path = os.path.join("static", "results", "ortho", f"overlay_{xray_name}")
                ortho_record.landmarks_json = json.dumps(xray_result["landmarks"])
                ortho_record.diagnosis_json = json.dumps(xray_result["diagnosis"])
                ortho_record.status = "DONE"
            else:
                ortho_record.status = "FAILED"
                ortho_record.error_message = xray_result.get("error", "Unknown error")
            db.session.commit()

        _finalize_case_after_analysis(new_case, outcomes)
        db.session.commit()

        if new_case.status in ("NEEDS_REUPLOAD", "FAILED") and not has_xray:
            flash(new_case.failure_message or MSG_ANALYSIS_FAILED, "error")
            return redirect(url_for("view_result", case_id=new_case.id))

        parts = []
        if has_front_ns:
            parts.append("frontal")
        if has_side:
            parts.append("side")
        if has_xray:
            parts.append("X-ray")
        flash(f"Analysis complete ({' + '.join(parts)}).", "success")
        return redirect(url_for("view_result", case_id=new_case.id))

    except Exception:
        new_case.status = "FAILED"
        new_case.failure_message = MSG_ANALYSIS_FAILED
        db.session.commit()
        flash(MSG_ANALYSIS_FAILED, "error")
        return redirect(url_for("view_result", case_id=new_case.id))


@app.route("/history")
@login_required
def history():
    doctor_id = session["user_id"]
    q = request.args.get("q", "").strip()
    status = request.args.get("status", "ALL").strip().upper()

    query = Case.query.filter_by(doctor_id=doctor_id)

    if status != "ALL":
        query = query.filter(Case.status == status)

    cases = query.order_by(Case.created_at.desc()).all()

    if q:
        filtered_cases = []
        for case in cases:
            patient = case.patient
            matches_case_id = q.isdigit() and case.id == int(q)
            matches_patient_code = patient and patient.patient_code and q.lower() in str(patient.patient_code).lower()
            matches_patient_name = patient and patient.name and q.lower() in patient.name.lower()

            if matches_case_id or matches_patient_code or matches_patient_name:
                filtered_cases.append(case)
        cases = filtered_cases

    counts = {
        "ALL": Case.query.filter_by(doctor_id=doctor_id).count(),
        "COMPLETED": Case.query.filter_by(doctor_id=doctor_id, status="COMPLETED").count(),
        "PENDING": Case.query.filter_by(doctor_id=doctor_id, status="PENDING").count(),
        "FAILED": Case.query.filter_by(doctor_id=doctor_id, status="FAILED").count(),
    }

    return render_template(
        "history.html",
        name=session.get("user_name"),
        active="cases",
        cases=cases,
        q=q,
        status=status,
        counts=counts
    )
@app.route("/case/<int:case_id>/delete", methods=["POST"])
@login_required
def delete_case(case_id):
    case = Case.query.get_or_404(case_id)
    doctor_id = session["user_id"]

    if case.doctor_id != doctor_id:
        flash("Not allowed.", "error")
        return redirect(url_for("history"))

    patient_id = case.patient_id
    purge_case(case)
    db.session.commit()

    flash(f"Case #{case_id} deleted.", "success")
    next_url = request.form.get("next") or request.args.get("next")
    if next_url and next_url.startswith("/"):
        return redirect(next_url)
    if patient_id:
        return redirect(url_for("patient_cases", patient_id=patient_id))
    return redirect(url_for("history"))


@app.route("/patients/<int:patient_id>/delete", methods=["POST"])
@login_required
def delete_patient(patient_id):
    doctor_id = session["user_id"]
    patient = Patient.query.filter_by(id=patient_id, doctor_id=doctor_id).first_or_404()

    patient_name = patient.name
    try:
        purge_patient_for_doctor(patient, doctor_id)
        db.session.commit()
    except PermissionError:
        flash("Not allowed.", "error")
        return redirect(url_for("patients"))

    flash(f'Patient "{patient_name}" and all related cases were deleted.', "success")
    return redirect(url_for("patients"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = User.query.get(session["user_id"])

    if request.method == "POST":
        action = request.form.get("action", "")

        # Update name
        if action == "update_name":
            new_name = request.form.get("name", "").strip()
            if not new_name:
                flash("Name cannot be empty.", "error")
                return redirect(url_for("profile"))

            user.name = new_name
            db.session.commit()
            session["user_name"] = new_name
            flash("Profile updated.", "success")
            return redirect(url_for("profile"))

        # Change password
        if action == "change_password":
            current_pw = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            confirm_pw = request.form.get("confirm_password", "")

            if not check_password_hash(user.password_hash, current_pw):
                flash("Current password is incorrect.", "error")
                return redirect(url_for("profile"))

            if len(new_pw) < 6:
                flash("New password must be at least 6 characters.", "error")
                return redirect(url_for("profile"))

            if new_pw != confirm_pw:
                flash("New passwords do not match.", "error")
                return redirect(url_for("profile"))

            user.password_hash = generate_password_hash(new_pw)
            db.session.commit()
            flash("Password updated successfully.", "success")
            return redirect(url_for("profile"))

    return render_template(
        "profile.html",
        name=session.get("user_name"),
        active="settings",
        user=user
    )

from sqlalchemy import cast, String, or_




@app.route("/patients/<int:patient_id>/cases")
@login_required
def patient_cases(patient_id):
    doctor_id = session["user_id"]

    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=doctor_id
    ).first_or_404()

    cases = (
        Case.query
        .filter_by(
            doctor_id=doctor_id,
            patient_id=patient.id          # ✅ FIX
        )
        .order_by(Case.created_at.desc())
        .all()
    )

    case_summaries = []
    for c in cases:
        case_summaries.append(
            {
                "case": c,
                "views": build_case_view_summary(c),
            }
        )

    access_code_row = (
        PatientUploadCode.query.filter_by(
            patient_id=patient.id,
            doctor_id=doctor_id,
        )
        .order_by(PatientUploadCode.created_at.asc())
        .first()
    )

    # Progress comparisons for this patient (for the "Progress Comparisons" section)
    progress_comparisons = (
        ProgressComparison.query
        .filter_by(patient_id=patient.id, doctor_id=doctor_id)
        .order_by(ProgressComparison.created_at.desc())
        .all()
    )

    return render_template(
        "patient_cases.html",
        name=session.get("user_name"),
        active="patients",
        patient=patient,
        cases=cases,
        case_summaries=case_summaries,
        access_code=access_code_row.code if access_code_row else None,
        progress_comparisons=progress_comparisons,
    )



# ── Progress Comparison routes ──────────────────────────────────────────────

_CP_VALID_TYPES = {"side", "front", "xray"}


def _cp_overlay_url(result_row):
    """Safe overlay URL for a Result row."""
    if not result_row or not result_row.overlay_path:
        return None
    fp = static_url_filename(result_row.overlay_path)
    return url_for("static", filename=fp) if fp else None


def _cp_ortho_overlay_url(ortho_rec):
    """Safe overlay URL for an OrthoCase row."""
    if not ortho_rec or not ortho_rec.overlay_path:
        return None
    fp = ortho_rec.overlay_path.replace("\\", "/")
    for prefix in ("static/", "static\\"):
        if fp.startswith(prefix):
            fp = fp[len(prefix):]
            break
    return url_for("static", filename=fp)


def _cp_parse_pts(result_row):
    """Parse landmarks JSON from a Result row; return [] on any error."""
    if not result_row or not result_row.landmarks_json:
        return []
    try:
        return json.loads(result_row.landmarks_json)
    except Exception:
        return []


@app.route(
    "/patients/<int:patient_id>/cases/<int:case_id>/compare-progress",
    methods=["GET", "POST"],
)
@login_required
def compare_progress(patient_id, case_id):
    """
    GET  (no ?type)      – Analysis-type selection screen.
    GET  (?type=…)       – Baseline display + upload form for that analysis type.
    POST (analysis_type) – Run the chosen pipeline, save comparison, redirect to view.
    """
    doctor_id = session["user_id"]

    patient = Patient.query.filter_by(id=patient_id, doctor_id=doctor_id).first_or_404()
    baseline_case = Case.query.filter_by(
        id=case_id, doctor_id=doctor_id, patient_id=patient.id
    ).first_or_404()

    # What baseline data exists?
    b_side_result  = Result.query.filter_by(case_id=baseline_case.id, view_type="SIDE").first()
    b_front_result = Result.query.filter_by(case_id=baseline_case.id, view_type="FRONT_NS").first()
    b_ortho        = OrthoCase.query.filter_by(case_id=baseline_case.id).first()
    has_side_data  = bool(b_side_result and b_side_result.landmarks_json)
    has_front_data = bool(b_front_result and b_front_result.landmarks_json)
    has_xray_data  = bool(b_ortho and b_ortho.status == "DONE" and b_ortho.diagnosis_json)

    # ── POST: run new analysis ───────────────────────────────────────────────
    if request.method == "POST":
        analysis_type = (request.form.get("analysis_type") or "").strip().lower()
        if analysis_type not in _CP_VALID_TYPES:
            flash("Invalid analysis type. Please choose Side, Frontal, or X-ray.", "error")
            return redirect(url_for("compare_progress", patient_id=patient_id, case_id=case_id))

        case_date_val = _parse_case_date(request.form.get("case_date") or "")
        if case_date_val is None:
            flash("Please enter a valid date for the progress analysis.", "error")
            return redirect(url_for("compare_progress", patient_id=patient_id,
                                    case_id=case_id, type=analysis_type))

        def _nonempty(f):
            return f and getattr(f, "filename", None) and str(f.filename).strip() != ""

        new_case = Case(
            doctor_id=doctor_id,
            patient_id=patient.id,
            case_type="FOLLOW_UP",
            status="PROCESSING",
            case_date=case_date_val,
        )
        db.session.add(new_case)
        db.session.commit()

        try:
            ensure_static_subdirs("uploads", "results")

            if analysis_type == "side":
                new_file = request.files.get("side")
                if not _nonempty(new_file):
                    raise ValueError("Please upload a side profile image.")
                fname = f"{new_case.id}_side_{uuid.uuid4().hex}.jpg"
                frel  = upload_rel(fname)
                fabs  = upload_abs(fname, BASE_DIR)
                new_file.save(fabs)
                db.session.add(Image(case_id=new_case.id, view_type="SIDE", file_path=frel))
                db.session.commit()

                out = run_view_analysis(new_case.id, fabs, "SIDE")
                if out["success"]:
                    db.session.add(Result(
                        case_id=new_case.id, view_type="SIDE",
                        landmarks_json=out["landmarks_json"],
                        overlay_path=normalize_stored_path(out["overlay_path"]),
                        confidence_json=out.get("confidence_json"),
                    ))
                outcomes = [{"uploaded": True, "success": out["success"],
                             "message": out.get("message") or "",
                             "failed_stage": out.get("failed_stage")}]
                db.session.commit()
                _finalize_case_after_analysis(new_case, outcomes)
                db.session.commit()

                # Build side-only comparison
                comparison_data = build_progress_comparison(baseline_case, new_case)
                comparison_data["analysis_type"] = "side"
                summary_data = build_comparison_summary(comparison_data)

            elif analysis_type == "front":
                new_file = request.files.get("front_ns")
                if not _nonempty(new_file):
                    raise ValueError("Please upload a frontal image.")
                fname = f"{new_case.id}_front_ns_{uuid.uuid4().hex}.jpg"
                frel  = upload_rel(fname)
                fabs  = upload_abs(fname, BASE_DIR)
                new_file.save(fabs)
                db.session.add(Image(case_id=new_case.id, view_type="FRONT_NS", file_path=frel))
                db.session.commit()

                out = run_view_analysis(new_case.id, fabs, "FRONT_NS")
                if out["success"]:
                    db.session.add(Result(
                        case_id=new_case.id, view_type="FRONT_NS",
                        landmarks_json=out["landmarks_json"],
                        overlay_path=normalize_stored_path(out["overlay_path"]),
                        confidence_json=out.get("confidence_json"),
                    ))
                outcomes = [{"uploaded": True, "success": out["success"],
                             "message": out.get("message") or "",
                             "failed_stage": out.get("failed_stage")}]
                db.session.commit()
                _finalize_case_after_analysis(new_case, outcomes)
                db.session.commit()

                comparison_data = build_progress_comparison(baseline_case, new_case)
                comparison_data["analysis_type"] = "front"
                summary_data = build_comparison_summary(comparison_data)

            else:  # xray
                new_file = request.files.get("xray")
                if not _nonempty(new_file):
                    raise ValueError("Please upload an X-ray image.")
                xray_name = f"{new_case.id}_xray_{uuid.uuid4().hex}.jpg"
                xray_upload_dir = os.path.join(BASE_DIR, "static", "uploads", "ortho")
                os.makedirs(xray_upload_dir, exist_ok=True)
                xray_abs = os.path.join(xray_upload_dir, xray_name)
                new_file.save(xray_abs)

                xray_overlay_dir = os.path.join(BASE_DIR, "static", "results", "ortho")
                os.makedirs(xray_overlay_dir, exist_ok=True)
                xray_overlay_abs = os.path.join(xray_overlay_dir, f"overlay_{xray_name}")

                ortho_record = OrthoCase(
                    doctor_id=doctor_id,
                    patient_id=patient.id,
                    case_id=new_case.id,
                    image_path=os.path.join("static", "uploads", "ortho", xray_name),
                    status="PENDING",
                )
                db.session.add(ortho_record)
                db.session.commit()

                xray_result = run_ortho_analysis(xray_abs, xray_overlay_abs)
                if xray_result["success"]:
                    ortho_record.overlay_path = os.path.join(
                        "static", "results", "ortho", f"overlay_{xray_name}"
                    )
                    ortho_record.landmarks_json = json.dumps(xray_result["landmarks"])
                    ortho_record.diagnosis_json = json.dumps(xray_result["diagnosis"])
                    ortho_record.status = "DONE"
                else:
                    ortho_record.status = "FAILED"
                    ortho_record.error_message = xray_result.get("error", "Unknown error")

                outcomes = [{"uploaded": True, "success": xray_result["success"],
                             "message": xray_result.get("error") or "",
                             "failed_stage": None}]
                db.session.commit()
                _finalize_case_after_analysis(new_case, outcomes)
                db.session.commit()

                comparison_data = build_xray_progress_comparison(baseline_case, new_case)
                summary_data    = build_comparison_summary_xray(comparison_data)

            comp_rec = ProgressComparison(
                patient_id       = patient.id,
                doctor_id        = doctor_id,
                baseline_case_id = baseline_case.id,
                new_case_id      = new_case.id,
                analysis_type    = analysis_type,
                comparison_json  = json.dumps(comparison_data),
                summary_json     = json.dumps(summary_data),
            )
            db.session.add(comp_rec)
            db.session.commit()

            flash("Progress analysis complete. Comparison saved.", "success")
            return redirect(url_for("view_progress_comparison", comparison_id=comp_rec.id))

        except ValueError as ve:
            db.session.rollback()
            flash(str(ve), "error")
            try:
                new_case.status = "FAILED"
                db.session.commit()
            except Exception:
                pass
            return redirect(url_for("compare_progress", patient_id=patient_id,
                                    case_id=case_id, **{"type": analysis_type}))
        except Exception as exc:
            db.session.rollback()
            app.logger.error("compare_progress analysis error: %s", exc)
            try:
                new_case.status = "FAILED"
                new_case.failure_message = MSG_ANALYSIS_FAILED
                db.session.commit()
            except Exception:
                pass
            flash(MSG_ANALYSIS_FAILED, "error")
            return redirect(url_for("compare_progress", patient_id=patient_id,
                                    case_id=case_id, **{"type": analysis_type}))

    # ── GET ─────────────────────────────────────────────────────────────────
    analysis_type = (request.args.get("type") or "").strip().lower()

    if not analysis_type:
        # Step 1: Show analysis-type selection cards
        return render_template(
            "compare_progress.html",
            mode="select",
            name=session.get("user_name"),
            active="patients",
            patient=patient,
            baseline_case=baseline_case,
            has_side_data=has_side_data,
            has_front_data=has_front_data,
            has_xray_data=has_xray_data,
            today_iso=date.today().isoformat(),
        )

    if analysis_type not in _CP_VALID_TYPES:
        flash("Unknown analysis type.", "error")
        return redirect(url_for("compare_progress", patient_id=patient_id, case_id=case_id))

    # Validate that baseline actually has this type
    if analysis_type == "side" and not has_side_data:
        flash("This baseline case does not contain side profile analysis data.", "error")
        return redirect(url_for("compare_progress", patient_id=patient_id, case_id=case_id))
    if analysis_type == "front" and not has_front_data:
        flash("This baseline case does not contain frontal analysis data.", "error")
        return redirect(url_for("compare_progress", patient_id=patient_id, case_id=case_id))
    if analysis_type == "xray" and not has_xray_data:
        flash("This baseline case does not contain X-ray analysis data.", "error")
        return redirect(url_for("compare_progress", patient_id=patient_id, case_id=case_id))

    # Step 2: Show baseline data + upload form for chosen type
    from utils.result_ui_data import (
        build_side_measurement_cards, build_frontal_measurement_cards,
        build_xray_measurement_cards,
    )
    from utils.frontal_measurements import calculate_frontal_measurements
    from utils.orthodontic_ai_inference import parse_xray_diagnosis_json

    b_side_cards = build_side_measurement_cards(_cp_parse_pts(b_side_result)) if b_side_result else []

    b_front_meas  = calculate_frontal_measurements(_cp_parse_pts(b_front_result)) \
                    if b_front_result and _cp_parse_pts(b_front_result) else None
    b_front_cards = build_frontal_measurement_cards(b_front_meas) if b_front_meas else []

    b_xray_cards  = build_xray_measurement_cards(b_ortho.landmarks_json if b_ortho else None)
    b_xray_diag   = parse_xray_diagnosis_json(b_ortho.diagnosis_json if b_ortho else None)

    b_side_diag  = SideDiagnosis.query.filter_by(
        case_id=baseline_case.id, doctor_id=doctor_id
    ).order_by(SideDiagnosis.created_at.desc()).first()
    b_front_diag = FrontalDiagnosis.query.filter_by(
        case_id=baseline_case.id, doctor_id=doctor_id
    ).order_by(FrontalDiagnosis.created_at.desc()).first()

    return render_template(
        "compare_progress.html",
        mode="upload",
        analysis_type=analysis_type,
        name=session.get("user_name"),
        active="patients",
        patient=patient,
        baseline_case=baseline_case,
        baseline_side_overlay=_cp_overlay_url(b_side_result),
        baseline_front_overlay=_cp_overlay_url(b_front_result),
        baseline_xray_overlay=_cp_ortho_overlay_url(b_ortho),
        baseline_side_cards=b_side_cards,
        baseline_front_cards=b_front_cards,
        baseline_xray_cards=b_xray_cards,
        baseline_xray_diag=b_xray_diag,
        baseline_side_diag=b_side_diag,
        baseline_front_diag=b_front_diag,
        today_iso=date.today().isoformat(),
    )


@app.route("/progress-comparison/<int:comparison_id>")
@login_required
def view_progress_comparison(comparison_id):
    """View a saved progress comparison. Never reruns any model."""
    doctor_id = session["user_id"]
    comp = ProgressComparison.query.get_or_404(comparison_id)

    if comp.doctor_id != doctor_id:
        flash("You are not allowed to view this comparison.", "error")
        return redirect(url_for("dashboard"))

    patient       = comp.patient
    baseline_case = comp.baseline_case
    new_case_obj  = comp.new_case
    analysis_type = comp.analysis_type or "side"  # default for old records

    # ── Parse stored JSON safely ─────────────────────────────────────────
    try:
        comparison_data = json.loads(comp.comparison_json) if comp.comparison_json else {}
    except Exception:
        comparison_data = {}
    try:
        summary_data = json.loads(comp.summary_json) if comp.summary_json else {}
    except Exception:
        summary_data = {}

    # Normalize rows — safe flat list of dicts, never strings
    comparison_rows_side    = normalize_comparison_rows(comparison_data, "side")
    comparison_rows_frontal = normalize_comparison_rows(comparison_data, "frontal")
    comparison_rows_xray    = normalize_comparison_rows(comparison_data, "xray")

    # For type-specific view we pick only the relevant rows
    if analysis_type == "side":
        main_rows = comparison_rows_side
    elif analysis_type == "front":
        main_rows = comparison_rows_frontal
    else:
        main_rows = comparison_rows_xray

    # ── Load display data from DB (no rerun) ─────────────────────────────
    from utils.result_ui_data import (
        build_side_measurement_cards, build_frontal_measurement_cards,
        build_xray_measurement_cards,
    )
    from utils.frontal_measurements import calculate_frontal_measurements
    from utils.orthodontic_ai_inference import parse_xray_diagnosis_json

    b_side  = Result.query.filter_by(case_id=baseline_case.id, view_type="SIDE").first()
    b_front = Result.query.filter_by(case_id=baseline_case.id, view_type="FRONT_NS").first()
    b_ortho = OrthoCase.query.filter_by(case_id=baseline_case.id).first()

    n_side  = Result.query.filter_by(case_id=new_case_obj.id, view_type="SIDE").first()   if new_case_obj else None
    n_front = Result.query.filter_by(case_id=new_case_obj.id, view_type="FRONT_NS").first() if new_case_obj else None
    n_ortho = OrthoCase.query.filter_by(case_id=new_case_obj.id).first()                  if new_case_obj else None

    b_side_cards  = build_side_measurement_cards(_cp_parse_pts(b_side)) if b_side else []
    b_front_meas  = calculate_frontal_measurements(_cp_parse_pts(b_front)) if b_front and _cp_parse_pts(b_front) else None
    b_front_cards = build_frontal_measurement_cards(b_front_meas) if b_front_meas else []
    b_xray_cards  = build_xray_measurement_cards(b_ortho.landmarks_json if b_ortho else None)
    b_xray_diag   = parse_xray_diagnosis_json(b_ortho.diagnosis_json if b_ortho else None)

    n_side_cards  = build_side_measurement_cards(_cp_parse_pts(n_side)) if n_side else []
    n_front_meas  = calculate_frontal_measurements(_cp_parse_pts(n_front)) if n_front and _cp_parse_pts(n_front) else None
    n_front_cards = build_frontal_measurement_cards(n_front_meas) if n_front_meas else []
    n_xray_cards  = build_xray_measurement_cards(n_ortho.landmarks_json if n_ortho else None)
    n_xray_diag   = parse_xray_diagnosis_json(n_ortho.diagnosis_json if n_ortho else None)

    b_side_diag  = SideDiagnosis.query.filter_by(case_id=baseline_case.id, doctor_id=doctor_id).order_by(SideDiagnosis.created_at.desc()).first()
    b_front_diag = FrontalDiagnosis.query.filter_by(case_id=baseline_case.id, doctor_id=doctor_id).order_by(FrontalDiagnosis.created_at.desc()).first()
    n_side_diag  = SideDiagnosis.query.filter_by(case_id=new_case_obj.id, doctor_id=doctor_id).order_by(SideDiagnosis.created_at.desc()).first() if new_case_obj else None
    n_front_diag = FrontalDiagnosis.query.filter_by(case_id=new_case_obj.id, doctor_id=doctor_id).order_by(FrontalDiagnosis.created_at.desc()).first() if new_case_obj else None

    return render_template(
        "compare_progress.html",
        mode="view",
        analysis_type=analysis_type,
        name=session.get("user_name"),
        active="patients",
        patient=patient,
        baseline_case=baseline_case,
        new_case=new_case_obj,
        comparison=comp,
        summary_data=summary_data,
        main_rows=main_rows,
        comparison_rows_side=comparison_rows_side,
        comparison_rows_frontal=comparison_rows_frontal,
        comparison_rows_xray=comparison_rows_xray,
        baseline_side_overlay=_cp_overlay_url(b_side),
        baseline_front_overlay=_cp_overlay_url(b_front),
        baseline_xray_overlay=_cp_ortho_overlay_url(b_ortho),
        new_side_overlay=_cp_overlay_url(n_side),
        new_front_overlay=_cp_overlay_url(n_front),
        new_xray_overlay=_cp_ortho_overlay_url(n_ortho),
        baseline_side_cards=b_side_cards,
        baseline_front_cards=b_front_cards,
        baseline_xray_cards=b_xray_cards,
        baseline_xray_diag=b_xray_diag,
        new_side_cards=n_side_cards,
        new_front_cards=n_front_cards,
        new_xray_cards=n_xray_cards,
        new_xray_diag=n_xray_diag,
        baseline_side_diag=b_side_diag,
        baseline_front_diag=b_front_diag,
        new_side_diag=n_side_diag,
        new_front_diag=n_front_diag,
    )


@app.route("/progress-comparison/<int:comparison_id>/notes", methods=["POST"])
@login_required
def save_comparison_notes(comparison_id):
    """Save/update doctor notes on a saved progress comparison."""
    import html as _html_module
    doctor_id = session["user_id"]
    comp = ProgressComparison.query.get_or_404(comparison_id)
    if comp.doctor_id != doctor_id:
        return jsonify({"success": False, "error": "Not authorised."}), 403

    notes = _html_module.escape((request.form.get("doctor_notes") or "").strip())[:5000]
    comp.doctor_notes = notes or None
    comp.updated_at   = datetime.utcnow()
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.error("save_comparison_notes DB error: %s", exc)
        flash("Could not save notes. Please try again.", "error")
        return redirect(url_for("view_progress_comparison", comparison_id=comparison_id))

    flash("Notes saved.", "success")
    return redirect(url_for("view_progress_comparison", comparison_id=comparison_id))


@app.route("/result/<int:case_id>")
@login_required
def view_result(case_id):
    case = Case.query.get_or_404(case_id)

    # Security: ensure doctor owns the case
    if case.doctor_id != session["user_id"]:
        flash("You are not allowed to view this case.", "error")
        return redirect(url_for("dashboard"))

    # front = Result.query.filter_by(case_id=case_id, view_type="FRONT").first()
    side = Result.query.filter_by(case_id=case_id, view_type="SIDE").first()
    front_ns = Result.query.filter_by(case_id=case_id, view_type="FRONT_NS").first()
    # 🔹 Get original uploaded images (for landmark editing)
    # front_img = Image.query.filter_by(case_id=case_id, view_type="FRONT").first()
    side_img  = Image.query.filter_by(case_id=case_id, view_type="SIDE").first()
    front_ns_img = Image.query.filter_by(case_id=case_id, view_type="FRONT_NS").first()

    # front_original = "/" + front_img.file_path if front_img else None
    side_original  = "/" + side_img.file_path if side_img else None
    front_ns_original = "/" + front_ns_img.file_path if front_ns_img else None

    # Parse landmarks JSON safely
    def parse_points(r):
        if not r:
            return []
        try:
            return json.loads(r.landmarks_json)
        except Exception:
            return []

    def parse_confidence(r):
        """Return list of 0-100 floats, or [] if not stored (older cases)."""
        if not r or not getattr(r, "confidence_json", None):
            return []
        try:
            return json.loads(r.confidence_json)
        except Exception:
            return []

    # front_points = parse_points(front)
    side_points = parse_points(side)
    front_ns_points = parse_points(front_ns)
    side_confidence = parse_confidence(side)
    front_ns_confidence = parse_confidence(front_ns)

    front_ns_measurements = None
    if front_ns_points and isinstance(front_ns_points, list) and len(front_ns_points) >= 34:
        try:
            front_ns_measurements = calculate_frontal_measurements(front_ns_points)
        except Exception:
            front_ns_measurements = None

    view_summary = build_case_view_summary(case)

    def _static_url_for_image(img_row):
        if not img_row or not img_row.file_path:
            return None
        fp = static_url_filename(img_row.file_path)
        if not fp:
            return None
        return url_for("static", filename=fp)

    side_upload_url = _static_url_for_image(side_img) if side_img and not side else None
    front_upload_url = _static_url_for_image(front_ns_img) if front_ns_img and not front_ns else None

    patient = case.patient
    portal_code = None
    if patient:
        code_row = (
            PatientUploadCode.query.filter_by(
                patient_id=patient.id,
                doctor_id=session["user_id"],
            )
            .order_by(PatientUploadCode.created_at.asc())
            .first()
        )
        if not code_row:
            code_row = get_or_create_patient_access_code(patient.id, session["user_id"])
        portal_code = code_row.code

    next_appt = None
    case_appt = None
    if patient:
        next_appt = _next_patient_appointment(session["user_id"], patient.id)
        case_appt = (
            Appointment.query.filter_by(
                doctor_id=session["user_id"],
                patient_id=patient.id,
                case_id=case.id,
            )
            .filter(Appointment.status.in_(ACTIVE_APPOINTMENT_STATUSES))
            .filter(Appointment.appointment_date >= date.today())
            .order_by(Appointment.appointment_date.asc(), Appointment.appointment_time.asc())
            .first()
        )
        if case_appt:
            next_appt = case_appt

    status_label, status_tone = case_status_display(case.status)
    age_line = ""
    if patient and patient.age is not None:
        age_line = f"{patient.age}y"
    gender_line = ""
    if patient and patient.gender:
        gender_line = patient.gender.title()
    type_line = case_type_label(case.case_type)

    side_measurement_cards = build_side_measurement_cards(side_points) if side_points else []
    frontal_measurement_cards = (
        build_frontal_measurement_cards(front_ns_measurements) if front_ns_measurements else []
    )
    ai_insights = build_ai_insights(
        side_points,
        front_ns_measurements,
        (getattr(case, "doctor_comment", None) or "").strip(),
    )
    measurement_treatment_advice = build_measurement_treatment_advice(
        side_points,
        front_ns_measurements,
    )

    # Ortho (X-ray AI) result for this case
    ortho_rec = OrthoCase.query.filter_by(case_id=case_id).first()
    ortho_diagnosis = parse_xray_diagnosis_json(ortho_rec.diagnosis_json if ortho_rec else None)
    xray_measurement_cards = []
    if ortho_rec and ortho_rec.landmarks_json:
        xray_measurement_cards = build_xray_measurement_cards(ortho_rec.landmarks_json)
    ortho_overlay = None
    if ortho_rec and ortho_rec.overlay_path:
        fp = ortho_rec.overlay_path.replace("\\", "/")
        for prefix in ("static/", "static\\"):
            if fp.startswith(prefix):
                fp = fp[len(prefix):]
                break
        ortho_overlay = fp

    return render_template(
        "result.html",
        name=session.get("user_name"),
        active="analysis",
        case=case,
        patient=patient,
        portal_code=portal_code,
        next_appt=next_appt,
        status_label=status_label,
        status_tone=status_tone,
        age_line=age_line,
        gender_line=gender_line,
        type_line=type_line,
        today_iso=date.today().isoformat(),
        case_appt=case_appt,
        side_overlay=side.overlay_path if side else None,
        front_ns_overlay=front_ns.overlay_path if front_ns else None,
        side_points=side_points,
        front_ns_points=front_ns_points,
        side_original=side_original,
        front_ns_original=front_ns_original,
        front_ns_measurements=front_ns_measurements,
        side_confidence=side_confidence,
        front_ns_confidence=front_ns_confidence,
        view_summary=view_summary,
        friendly_fail=FRIENDLY_FAIL,
        side_upload_url=side_upload_url,
        front_upload_url=front_upload_url,
        side_measurement_cards=side_measurement_cards,
        frontal_measurement_cards=frontal_measurement_cards,
        ai_insights=ai_insights,
        measurement_treatment_advice=measurement_treatment_advice,
        ortho_diagnosis=ortho_diagnosis,
        ortho_overlay=ortho_overlay,
        ortho_status=ortho_rec.status if ortho_rec else None,
        ortho_error=ortho_rec.error_message if ortho_rec else None,
        ortho_rec=ortho_rec,
        doctor_name=session.get("user_name", ""),
        xray_measurement_cards=xray_measurement_cards,
        # Most-recent side diagnosis (if the doctor has run it)
        side_diag=SideDiagnosis.query.filter_by(
            case_id=case_id, doctor_id=session["user_id"]
        ).order_by(SideDiagnosis.created_at.desc()).first(),
        # Most-recent frontal diagnosis (if the doctor has run it)
        frontal_diag=FrontalDiagnosis.query.filter_by(
            case_id=case_id, doctor_id=session["user_id"]
        ).order_by(FrontalDiagnosis.created_at.desc()).first(),
    )


# ── Side Diagnosis route ────────────────────────────────────────────────────

@app.route("/case/<int:case_id>/side-diagnosis", methods=["POST"])
@login_required
def run_side_diagnosis_route(case_id):
    """
    POST /case/<id>/side-diagnosis
    Body (JSON): {"growth_stage": "adult" | "growing"}
    Returns JSON with diagnosis result and stores it in the DB.
    """
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session["user_id"]:
        return jsonify({"success": False, "error": "Not authorised."}), 403

    # ── Read growth_stage from request ──────────────────────────────────────
    data = request.get_json(silent=True) or {}
    growth_stage = (data.get("growth_stage") or "").strip().lower()
    if growth_stage not in ("adult", "growing"):
        return jsonify({
            "success": False,
            "error": "growth_stage must be 'adult' or 'growing'.",
        }), 400

    # ── Load side landmarks ─────────────────────────────────────────────────
    side_result = Result.query.filter_by(case_id=case_id, view_type="SIDE").first()
    if not side_result or not side_result.landmarks_json:
        return jsonify({
            "success": False,
            "error": "No side-view landmarks found. Run side analysis first.",
        }), 400

    try:
        landmarks = json.loads(side_result.landmarks_json)
    except Exception:
        return jsonify({"success": False, "error": "Corrupt landmark data."}), 500

    # ── Run XGBoost pipeline (old overall ML — kept for DB persistence) ────────
    result = run_side_diagnosis(landmarks, growth_stage)
    if not result["success"]:
        return jsonify({"success": False, "error": result.get("error", "Diagnosis failed.")}), 500

    # ── Persist old overall ML result to DB (unchanged) ─────────────────────
    angles = result["angles"]
    diag   = result["diagnosis"]
    treat  = result.get("treatment")

    rec = SideDiagnosis(
        case_id     = case_id,
        doctor_id   = session["user_id"],
        growth_stage= growth_stage,
        nasiolabial_angle       = angles.get("nasiolabial"),
        profile_convexity_angle = angles.get("profile_convexity"),
        total_convexity_angle   = angles.get("total_convexity"),
        mentolabial_angle       = angles.get("mentolabial"),
        diagnosis_label         = diag["label"],
        diagnosis_confidence    = diag["confidence"],
        diagnosis_breakdown_json= json.dumps(diag["breakdown"]),
        treatment_label         = treat["label"]       if treat else None,
        treatment_confidence    = treat["confidence"]  if treat else None,
        treatment_breakdown_json= json.dumps(treat["breakdown"]) if treat else None,
    )
    db.session.add(rec)
    db.session.commit()

    # ── Run measurement-level ML (new per-angle models) ──────────────────────
    measurement_ml = predict_side_measurement_analysis(
        nasiolabial      = angles["nasiolabial"],
        profile_convexity= angles["profile_convexity"],
        total_convexity  = angles["total_convexity"],
        mentolabial      = angles["mentolabial"],
        growth_stage     = growth_stage,
    )

    # ── Persist / refresh per-measurement doctor-review rows ──────────────────
    if measurement_ml.get("success"):
        _upsert_measurement_reviews(case_id, session["user_id"],
                                    measurement_ml["measurements"])

    return jsonify({
        "success":        True,
        "diagnosis_id":   rec.id,
        "angles":         result["angles"],
        # Old overall ML fields kept for backward compatibility but not
        # displayed in the frontend.
        "diagnosis":      result["diagnosis"],
        "treatment":      result.get("treatment"),
        # New measurement-level ML result — displayed in the frontend.
        "measurement_ml": measurement_ml,
    })


# ── Doctor-review + profile-simulation helpers ──────────────────────────────

# Canonical order of the four measurement-level results.
MEAS_ORDER = ["nasiolabial", "profile_convexity", "total_convexity", "mentolabial"]


def _upsert_measurement_reviews(case_id, doctor_id, measurements):
    """Create/refresh MeasurementReview rows from a measurement_ml dict.

    Existing approve/decline decisions are preserved while the model output is
    unchanged; if the model diagnosis changed, the decision is reset to pending.
    """
    for key in MEAS_ORDER:
        m = measurements.get(key)
        if not m:
            continue
        diag_code = sim_rules.normalize_diagnosis(m.get("diagnosis", ""))
        row = MeasurementReview.query.filter_by(
            case_id=case_id, doctor_id=doctor_id, measurement_key=key
        ).first()
        if row is None:
            row = MeasurementReview(
                case_id=case_id, doctor_id=doctor_id, measurement_key=key,
                review_status="pending",
            )
            db.session.add(row)
        elif row.model_diagnosis != m.get("diagnosis"):
            # Model output changed → previous decision no longer valid.
            row.review_status = "pending"
            row.reviewed_at = None
        row.measurement_label    = m.get("display_name")
        row.angle                = m.get("angle")
        row.model_diagnosis      = m.get("diagnosis")
        row.diagnosis_code       = diag_code
        row.diagnosis_confidence = m.get("diagnosis_confidence")
        row.model_treatment      = m.get("treatment")
        row.treatment_confidence = m.get("treatment_confidence")
        expl = m.get("treatment_explanation")
        row.treatment_explanation_json = json.dumps(expl) if expl is not None else None
    db.session.commit()


def _merge_supported(pairs):
    """Group (label, support) pairs, preserving order, collecting supporters."""
    seen, order = {}, []
    for label, support in pairs:
        if label not in seen:
            seen[label] = []
            order.append(label)
        seen[label].append(support)
    return [{"label": lbl, "supported_by": seen[lbl]} for lbl in order]


def _review_signature(rows, selected_changes, strength):
    """Stable hash of approved review + selected changes + strength (for dedup)."""
    import hashlib
    approved = sorted(
        f"{r.measurement_key}:{r.diagnosis_code}"
        for r in rows if r.review_status == "approved"
    )
    payload = "|".join(approved) + "#" + ",".join(sorted(selected_changes)) + "#" + (strength or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _case_side_image_abs(case_id):
    """Absolute filesystem path to the case's ORIGINAL side image, or None."""
    img = Image.query.filter_by(case_id=case_id, view_type="SIDE").first()
    if not img or not img.file_path:
        return None
    abs_path = os.path.normpath(os.path.join(BASE_DIR, img.file_path.lstrip("/\\")))
    return abs_path if os.path.isfile(abs_path) else None


def _serialize_simulation(sim):
    if not sim:
        return None
    try:
        changes = json.loads(sim.selected_changes_json) if sim.selected_changes_json else []
    except Exception:
        changes = []
    return {
        "id": sim.id,
        "simulation_url": "/" + sim.image_path.lstrip("/") if sim.image_path else None,
        "selected_changes": changes,
        "selected_change_labels": [sim_rules.change_label(c) for c in changes],
        "strength": sim.strength,
        "strength_label": sim_rules.strength_label(sim.strength or ""),
        "gemini_model": sim.gemini_model,
        "created_at": sim.created_at.isoformat() if sim.created_at else None,
        "disclaimer": "Educational AI-generated visualization; not a guaranteed clinical outcome.",
    }


def _build_review_state(case_id, doctor_id):
    """Build the full doctor-review + simulation state for the frontend."""
    rows = MeasurementReview.query.filter_by(
        case_id=case_id, doctor_id=doctor_id
    ).all()
    by_key = {r.measurement_key: r for r in rows}

    measurements = []
    for key in MEAS_ORDER:
        r = by_key.get(key)
        if not r:
            continue
        measurements.append({
            "measurement_key": key,
            "measurement_label": r.measurement_label,
            "angle": r.angle,
            "model_diagnosis": r.model_diagnosis,
            "diagnosis_code": r.diagnosis_code,
            "is_abnormal": sim_rules.is_abnormal(r.diagnosis_code or "normal"),
            "diagnosis_confidence": r.diagnosis_confidence,
            "model_treatment": r.model_treatment,
            "treatment_confidence": r.treatment_confidence,
            "treatment_explanation": (
                json.loads(r.treatment_explanation_json)
                if r.treatment_explanation_json else None
            ),
            "review_status": r.review_status,
        })

    total = len(measurements)
    reviewed = sum(1 for m in measurements if m["review_status"] != "pending")
    approved = [m for m in measurements if m["review_status"] == "approved"]
    approved_abnormal = [m for m in approved if m["is_abnormal"]]

    diagnosis_summary = _merge_supported(
        [(m["model_diagnosis"], m["measurement_label"]) for m in approved]
    )
    treatment_summary = _merge_supported(
        [(m["model_treatment"], m["measurement_label"]) for m in approved]
    )

    approved_codes = [m["diagnosis_code"] for m in approved]
    options = sim_rules.available_changes(approved_codes)
    diag_conflicts = sim_rules.find_diagnosis_conflicts(approved_codes)

    # Determine whether simulation can run, with a user-facing blocker message.
    can_simulate = False
    blocker = None
    if total < 4 or reviewed < total or total == 0:
        blocker = f"{reviewed} of {total or 4} findings reviewed"
    elif not approved:
        blocker = "No doctor-approved findings are available for simulation."
    elif not approved_abnormal:
        blocker = "No abnormal doctor-approved finding is available for simulation."
    elif diag_conflicts:
        pairs = "; ".join(" vs ".join(p) for p in diag_conflicts)
        blocker = f"Approved findings conflict: {pairs}. Decline one to continue."
    elif not _case_side_image_abs(case_id):
        blocker = "The original patient side image is missing."
    elif not sim_gemini.is_available():
        blocker = "The image-generation service is not configured."
    else:
        can_simulate = True

    existing = ProfileSimulation.query.filter_by(
        case_id=case_id, doctor_id=doctor_id
    ).order_by(ProfileSimulation.created_at.desc()).first()

    return {
        "success": True,
        "measurements": measurements,
        "progress": {"reviewed": reviewed, "total": total or 4},
        "diagnosis_summary": diagnosis_summary,
        "treatment_summary": treatment_summary,
        "simulation_options": options,
        "strengths": [{"code": c, "label": l} for c, l in sim_rules.STRENGTHS.items()],
        "default_strength": sim_rules.DEFAULT_STRENGTH,
        "can_simulate": can_simulate,
        "blocker_message": blocker,
        "side_image_available": _case_side_image_abs(case_id) is not None,
        "gemini_available": sim_gemini.is_available(),
        "latest_simulation": _serialize_simulation(existing),
        "disclaimer": (
            "These outputs are clinical decision-support considerations and "
            "must be reviewed by a qualified orthodontist."
        ),
    }


def _authorize_case_doctor(case_id):
    """Return (case, None) when current doctor owns the case, else (None, response)."""
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session.get("user_id"):
        return None, (jsonify({"success": False, "error": "Not authorised."}), 403)
    return case, None


@app.route("/case/<int:case_id>/review-state", methods=["GET"])
@login_required
def measurement_review_state(case_id):
    """Return the saved doctor-review + simulation state for restore on load."""
    _, err = _authorize_case_doctor(case_id)
    if err:
        return err
    return jsonify(_build_review_state(case_id, session["user_id"]))


@app.route("/case/<int:case_id>/measurement-review", methods=["POST"])
@login_required
def measurement_review_update(case_id):
    """Record an approve/decline decision for one measurement card."""
    _, err = _authorize_case_doctor(case_id)
    if err:
        return err

    data = request.get_json(silent=True) or {}
    key = (data.get("measurement_key") or "").strip()
    status = (data.get("review_status") or "").strip().lower()

    if key not in MEAS_ORDER:
        return jsonify({"success": False, "error": "Unknown measurement."}), 400
    if status not in ("pending", "approved", "declined"):
        return jsonify({"success": False, "error": "Invalid review status."}), 400

    row = MeasurementReview.query.filter_by(
        case_id=case_id, doctor_id=session["user_id"], measurement_key=key
    ).first()
    if row is None:
        return jsonify({
            "success": False,
            "error": "Run the ML analysis before reviewing findings.",
        }), 400

    row.review_status = status
    row.reviewed_at = datetime.utcnow() if status != "pending" else None
    db.session.commit()

    return jsonify(_build_review_state(case_id, session["user_id"]))


@app.route("/api/cases/<int:case_id>/profile-simulation", methods=["POST"])
@login_required
def profile_simulation_generate(case_id):
    """Generate a doctor-reviewed Gemini profile simulation for a case."""
    _, err = _authorize_case_doctor(case_id)
    if err:
        return err

    doctor_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    selected_changes = data.get("selected_changes") or []
    strength = (data.get("strength") or "").strip().lower()
    force_regenerate = bool(data.get("regenerate"))

    if not isinstance(selected_changes, list):
        return jsonify({"success": False, "error": "Invalid selected changes."}), 400
    # Deduplicate + keep only known change codes.
    selected_changes = [c for c in dict.fromkeys(selected_changes)
                        if c in sim_rules.CHANGE_LABELS]

    # ── Load stored review decisions (never trust browser diagnosis text) ─────
    rows = MeasurementReview.query.filter_by(
        case_id=case_id, doctor_id=doctor_id
    ).all()
    by_key = {r.measurement_key: r for r in rows}

    if len([k for k in MEAS_ORDER if k in by_key]) < 4:
        return jsonify({"success": False,
                        "error": "Run the ML analysis before generating a simulation."}), 400
    if any(by_key[k].review_status == "pending" for k in MEAS_ORDER):
        return jsonify({"success": False,
                        "error": "All four findings must be reviewed first."}), 400

    approved_rows = [by_key[k] for k in MEAS_ORDER if by_key[k].review_status == "approved"]
    approved_findings = [
        ApprovedFinding(label=r.model_diagnosis, code=r.diagnosis_code or "unknown")
        for r in approved_rows
    ]

    side_abs = _case_side_image_abs(case_id)
    if not side_abs:
        return jsonify({"success": False,
                        "error": "The original patient side image is missing."}), 400

    if not sim_gemini.is_available():
        return jsonify({"success": False,
                        "error": "The image-generation service is not configured."}), 503

    # ── Dedup: reuse a saved simulation with the same signature ───────────────
    try:
        strength_check = sim_rules.normalize_strength(strength) or strength
        signature = _review_signature(approved_rows, selected_changes, strength_check)
    except Exception:
        signature = None

    if signature and not force_regenerate:
        existing = ProfileSimulation.query.filter_by(
            case_id=case_id, doctor_id=doctor_id, review_signature=signature
        ).order_by(ProfileSimulation.created_at.desc()).first()
        if existing and existing.image_path:
            payload = _serialize_simulation(existing)
            payload["success"] = True
            payload["reused"] = True
            return jsonify(payload)

    # ── Generate (validation + conflict checks happen in the service) ─────────
    out_dir = os.path.join(BASE_DIR, "generated_simulations", "profile")
    try:
        result = run_profile_simulation(
            source_image_path=side_abs,
            approved_findings=approved_findings,
            selected_change_codes=selected_changes,
            strength=strength,
            output_dir=out_dir,
            case_id=case_id,
        )
    except SimulationValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except GeminiUnavailableError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except GeminiGenerationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 502
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        app.logger.exception("profile_simulation failed for case %s", case_id)
        return jsonify({"success": False,
                        "error": "The simulation could not be generated."}), 500

    rel_path = "generated_simulations/profile/" + os.path.basename(result["image_path"])
    sim = ProfileSimulation(
        case_id=case_id,
        doctor_id=doctor_id,
        selected_changes_json=json.dumps(result["selected_changes"]),
        strength=result["strength"],
        review_signature=signature,
        image_path=rel_path,
        gemini_model=result["gemini_model"],
    )
    db.session.add(sim)
    db.session.commit()

    return jsonify({
        "success": True,
        "simulation_url": "/" + rel_path,
        "selected_changes": result["selected_changes"],
        "selected_change_labels": [sim_rules.change_label(c) for c in result["selected_changes"]],
        "strength": result["strength"],
        "strength_label": sim_rules.strength_label(result["strength"]),
        "gemini_model": result["gemini_model"],
        "created_at": sim.created_at.isoformat(),
        "disclaimer": result["disclaimer"],
    })


@app.route("/generated_simulations/<path:filename>")
@login_required
def serve_generated_simulation(filename):
    """Serve a generated simulation image (doctor-restricted, sanitized path)."""
    base = os.path.join(BASE_DIR, "generated_simulations")
    safe = os.path.normpath(os.path.join(base, filename))
    if not safe.startswith(os.path.normpath(base) + os.sep):
        return ("Not found", 404)
    if not os.path.isfile(safe):
        return ("Not found", 404)
    # Only the owning doctor may view a simulation file.
    rel = "generated_simulations/" + filename.replace("\\", "/")
    sim = ProfileSimulation.query.filter_by(image_path=rel).first()
    if sim and sim.doctor_id != session.get("user_id"):
        return ("Not authorised", 403)
    return send_file(safe)


# ── Smile Adjustment route ──────────────────────────────────────────────────

@app.route("/case/<int:case_id>/smile-adjustment", methods=["POST"])
@login_required
def smile_adjustment_generate(case_id):
    """Generate an esthetic smile-adjustment edit from an uploaded smiling photo.

    multipart/form-data:
        image        (file)
        tooth_style  (natural | hollywood | triangular)
        shade        (BL1 | BL2 | A1 | A2 | A3 | B1 | B2 | C1)
        gummy_mm     (0–5)
    """
    _, err = _authorize_case_doctor(case_id)
    if err:
        return err

    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"success": False,
                        "error": "Please upload a smiling patient photo first."}), 400

    try:
        image_bytes = upload.read()
    except Exception:
        return jsonify({"success": False,
                        "error": "The uploaded image could not be read."}), 400

    tooth_style = request.form.get("tooth_style", "")
    shade = request.form.get("shade", "")
    gummy_mm = request.form.get("gummy_mm", "")

    out_dir = os.path.join(BASE_DIR, "generated_simulations", "smile")
    try:
        result = run_smile_adjustment(
            image_bytes=image_bytes,
            filename=upload.filename,
            tooth_style=tooth_style,
            shade=shade,
            gummy_mm=gummy_mm,
            output_dir=out_dir,
            case_id=case_id,
        )
    except (SmileSelectionError, SmileImageError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except GeminiUnavailableError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except GeminiGenerationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 502
    except Exception:
        app.logger.exception("smile_adjustment failed for case %s", case_id)
        return jsonify({"success": False,
                        "error": "The smile adjustment could not be generated."}), 500

    return jsonify({
        "success": True,
        "original_url": "/" + result["original_url_path"],
        "result_url": "/" + result["result_url_path"],
        "tooth_style": result["tooth_style"],
        "tooth_style_label": result["tooth_style_label"],
        "shade": result["shade"],
        "shade_description": result["shade_description"],
        "gummy_mm": result["gummy_mm"],
        "gemini_model": result["gemini_model"],
        "disclaimer": (
            "Educational esthetic smile simulation; not a guaranteed clinical outcome."
        ),
    })


# ── Frontal Diagnosis route ─────────────────────────────────────────────────

@app.route("/case/<int:case_id>/frontal-diagnosis", methods=["POST"])
@login_required
def run_frontal_diagnosis_route(case_id):
    """
    POST /case/<id>/frontal-diagnosis
    No request body required.
    Loads FRONT_NS landmarks, runs six-model frontal pipeline, persists result.
    """
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session["user_id"]:
        return jsonify({"success": False, "error": "Not authorised."}), 403

    front_ns = Result.query.filter_by(case_id=case_id, view_type="FRONT_NS").first()
    if not front_ns or not front_ns.landmarks_json:
        return jsonify({
            "success": False,
            "error": "No frontal landmarks found. Run front analysis first.",
        }), 400

    try:
        landmarks = json.loads(front_ns.landmarks_json)
    except Exception:
        return jsonify({"success": False, "error": "Corrupt frontal landmark data."}), 500

    if not isinstance(landmarks, list) or len(landmarks) < 34:
        n = len(landmarks) if isinstance(landmarks, list) else 0
        return jsonify({
            "success": False,
            "error": f"Expected 34 frontal landmarks, got {n}.",
        }), 400

    outcome = predict_frontal_diagnosis(landmarks)
    if not outcome["success"]:
        return jsonify({"success": False, "error": outcome.get("error", "Diagnosis failed.")}), 500

    results_json = json.dumps(outcome["results"])

    # Update existing record for this (case, doctor) pair rather than appending
    try:
        rec = FrontalDiagnosis.query.filter_by(
            case_id=case_id, doctor_id=session["user_id"]
        ).first()
        if rec:
            rec.results_json = results_json
            rec.created_at   = datetime.utcnow()
        else:
            rec = FrontalDiagnosis(
                case_id      = case_id,
                doctor_id    = session["user_id"],
                results_json = results_json,
            )
            db.session.add(rec)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.error("frontal_diagnosis DB error: %s", exc)
        return jsonify({"success": False, "error": "Database error saving diagnosis."}), 500

    return jsonify({"success": True, "diagnosis_id": rec.id, "results": outcome["results"]})


# ── Standalone /predict route ──────────────────────────────────────────────
#
#  POST /predict
#  -------------
#  A lightweight, case-free endpoint: upload a side-view photo + growth_stage
#  and get a diagnosis JSON immediately.  No DB write, no login required.
#
#  multipart/form-data fields:
#    image        (file)   – the facial side-view photo
#    growth_stage (string) – "adult" | "growing"
#
#  Response 200:
#    {
#      "success":     true,
#      "diagnosis":   "Retruded chin",
#      "confidence":  91.4,
#      "angles": {
#        "nasiolabial":       107.3,
#        "profile_convexity": 163.1,
#        "total_convexity":   131.8,
#        "mentolabial":       119.2
#      },
#      "treatment": "No treatment / monitoring",   // or null
#      "treatment_confidence": 78.5               // or null
#    }
#
#  Response 4xx/5xx:
#    { "success": false, "error": "..." }

ALLOWED_PREDICT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@app.route("/predict", methods=["POST"])
def predict():
    """
    Standalone HRNet → XGBoost diagnosis endpoint (no case / no login needed).
    Accepts a fresh image upload and returns a diagnosis JSON.
    """
    # ── Validate image ────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in request."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"success": False, "error": "Empty file upload."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_PREDICT_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file type '{ext}'. Use JPG, PNG, WEBP, or BMP.",
        }), 400

    # ── Validate growth_stage ─────────────────────────────────────────────────
    growth_stage = (request.form.get("growth_stage") or "").strip().lower()
    if growth_stage not in ("adult", "growing"):
        return jsonify({
            "success": False,
            "error": "growth_stage must be 'adult' or 'growing'.",
        }), 400

    # ── Run full pipeline ─────────────────────────────────────────────────────
    image_bytes = file.read()
    result = full_pipeline_from_bytes(image_bytes, growth_stage, suffix=ext)

    if not result.get("success"):
        return jsonify({
            "success": False,
            "error": result.get("error", "Diagnosis pipeline failed."),
        }), 500

    # ── Flatten response ──────────────────────────────────────────────────────
    diag  = result["diagnosis"]
    treat = result.get("treatment")

    return jsonify({
        "success":              True,
        "diagnosis":            diag["label"],
        "confidence":           diag["confidence"],
        "confidence_level":     diag["confidence_level"],
        "diagnosis_breakdown":  diag.get("breakdown", []),
        "angles":               result["angles"],
        "growth_stage":         result["growth_stage"],
        "treatment":            treat["label"]      if treat else None,
        "treatment_confidence": treat["confidence"] if treat else None,
        "treatment_breakdown":  treat.get("breakdown", []) if treat else [],
    })


@app.route("/case/<int:case_id>/simulation/<int:sim_id>/delete", methods=["POST"])
@login_required
def simulation_delete(case_id, sim_id):
    """Delete a saved simulation record and its image file."""
    from flask import jsonify as _json

    sim = TreatmentSimulation.query.get_or_404(sim_id)
    if sim.doctor_id != session["user_id"] or sim.case_id != case_id:
        return _json({"success": False, "error": "Not allowed"}), 403

    # Remove file if it exists
    if sim.image_path:
        fp = os.path.normpath(os.path.join(BASE_DIR, sim.image_path.lstrip("/")))
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass

    db.session.delete(sim)
    db.session.commit()
    return _json({"success": True})


@app.route("/case/<int:case_id>/add-xray", methods=["POST"])
@login_required
def add_xray_to_case(case_id):
    """Add (or replace) X-ray analysis on an existing case."""
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session["user_id"]:
        flash("Not allowed.", "error")
        return redirect(url_for("dashboard"))

    xray_file = request.files.get("xray_file")
    if not xray_file or xray_file.filename == "":
        flash("Please select a cephalometric X-ray image to upload.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    # Save uploaded X-ray
    xray_name = f"{case_id}_xray_{uuid.uuid4().hex}.jpg"
    xray_upload_dir = os.path.join(BASE_DIR, "static", "uploads", "ortho")
    os.makedirs(xray_upload_dir, exist_ok=True)
    xray_abs = os.path.join(xray_upload_dir, xray_name)
    xray_file.save(xray_abs)

    xray_overlay_dir = os.path.join(BASE_DIR, "static", "results", "ortho")
    os.makedirs(xray_overlay_dir, exist_ok=True)
    xray_overlay_abs = os.path.join(xray_overlay_dir, f"overlay_{xray_name}")

    # Create or update the OrthoCase record for this case
    ortho_rec = OrthoCase.query.filter_by(case_id=case_id).first()
    if ortho_rec is None:
        ortho_rec = OrthoCase(
            doctor_id=session["user_id"],
            patient_id=case.patient_id,
            case_id=case_id,
        )
        db.session.add(ortho_rec)

    ortho_rec.image_path = os.path.join("static", "uploads", "ortho", xray_name)
    ortho_rec.status = "PENDING"
    ortho_rec.error_message = None
    ortho_rec.overlay_path = None
    ortho_rec.landmarks_json = None
    ortho_rec.diagnosis_json = None
    db.session.commit()

    # Run inference (reusing the same pipeline as new analysis)
    xray_result = run_ortho_analysis(xray_abs, xray_overlay_abs)
    if xray_result["success"]:
        ortho_rec.overlay_path = os.path.join("static", "results", "ortho", f"overlay_{xray_name}")
        ortho_rec.landmarks_json = json.dumps(xray_result["landmarks"])
        ortho_rec.diagnosis_json = json.dumps(xray_result["diagnosis"])
        ortho_rec.status = "DONE"
        flash("X-ray AI analysis complete.", "success")
    else:
        ortho_rec.status = "FAILED"
        ortho_rec.error_message = xray_result.get("error", "Unknown error")
        flash(f"X-ray analysis failed: {ortho_rec.error_message}", "error")
    db.session.commit()

    return redirect(url_for("view_result", case_id=case_id) + "#tab-panel-xray")


# ─────────────────────────────────────────────────────────────────────────────
# Explainable AI — SHAP endpoint for X-ray cephalometric diagnoses
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/case/<int:case_id>/xray-shap")
@login_required
def xray_shap_endpoint(case_id):
    """
    Return SHAP feature-importance explanations for all 11 X-ray diagnoses.

    This endpoint is called by JavaScript on the X-ray results tab.
    It computes SHAP on-demand from the stored image; no patient SHAP
    data is persisted to the database.

    Response JSON:
        {
            "success": true,
            "shap_data": {
                "<model_key>": {
                    "diagnosis", "status", "predicted_label",
                    "predicted_probability", "class_probabilities",
                    "supporting_features", "opposing_features",
                    "additivity_valid", "additivity_difference"
                },
                ...
            }
        }
    """
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session.get("user_id"):
        return jsonify({"success": False, "error": "Not authorized."}), 403

    ortho_rec = OrthoCase.query.filter_by(case_id=case_id).first()
    if not ortho_rec:
        return jsonify({"success": False, "error": "No X-ray record for this case."}), 404
    if ortho_rec.status != "DONE":
        return jsonify({"success": False, "error": "X-ray analysis is not complete."}), 400

    # Resolve the stored relative image path to an absolute path
    image_rel = (ortho_rec.image_path or "").replace("\\", "/")
    image_abs = os.path.join(BASE_DIR, image_rel)
    if not os.path.isfile(image_abs):
        return jsonify({
            "success": False,
            "error": "X-ray image file not found on disk. "
                     "The file may have been moved or deleted.",
        }), 404

    try:
        from utils.xray_shap import diagnose_xray_with_shap, shap_results_to_json_safe
        result = diagnose_xray_with_shap(image_abs, top_n=5)
        json_safe = shap_results_to_json_safe(
            result["shap_results"],
            reference_meta=result.get("reference_meta"),
        )
        return jsonify({"success": True, "shap_data": json_safe})

    except ImportError as imp_err:
        return jsonify({
            "success": False,
            "error": f"SHAP library not available: {imp_err}",
        }), 500

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"SHAP computation failed: {str(exc)}",
        }), 500


@app.route("/debug/landmark-preview/<int:case_id>")
@login_required
def landmark_preview(case_id):
    """
    Step 1 diagnostic tool — numbered side-landmark preview.

    Returns a PNG image with every predicted side landmark drawn as a
    large coloured dot labelled with its 0-based index (0-19).
    A colour legend strip is appended at the bottom.

    Access via:  /debug/landmark-preview/<case_id>
    Or via the 'Landmark Index Preview' link on the result page.
    """
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session["user_id"]:
        flash("Not allowed.", "error")
        return redirect(url_for("dashboard"))

    # Find the SIDE Result record for this case
    side_result = Result.query.filter_by(case_id=case_id, view_type="SIDE").first()
    if not side_result or not side_result.landmarks_json:
        flash("No side-view landmark data found for this case.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    # Find the original uploaded image
    side_image = Image.query.filter_by(case_id=case_id, view_type="SIDE").first()
    if not side_image:
        flash("Original side image not found for this case.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    # Resolve the absolute path the same way view_result does
    raw_path = (side_image.file_path or "").replace("\\", "/").lstrip("/")
    img_path = os.path.normpath(os.path.join(BASE_DIR, raw_path))
    if not os.path.isfile(img_path):
        flash(f"Side image file not found on disk: {img_path}", "error")
        return redirect(url_for("view_result", case_id=case_id))

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        flash("Could not read the side image file.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    png_bytes = preview_to_png_bytes(img_bgr, side_result.landmarks_json)

    buf = io.BytesIO(png_bytes)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=False,
        download_name=f"landmark_preview_case{case_id}.png",
    )


@app.route("/ortho-review/<int:case_id>", methods=["POST"])
@login_required
def ortho_review(case_id):
    """Save doctor review of X-ray AI result."""
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session["user_id"]:
        flash("Not allowed.", "error")
        return redirect(url_for("dashboard"))

    ortho_rec = OrthoCase.query.filter_by(case_id=case_id).first()
    if not ortho_rec:
        flash("No X-ray AI result found for this case.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    ortho_rec.reviewed = True
    ortho_rec.reviewed_at = datetime.utcnow()
    ortho_rec.doctor_final_diagnosis = request.form.get("doctor_final_diagnosis", "").strip() or None
    ortho_rec.doctor_review_notes = request.form.get("doctor_review_notes", "").strip() or None
    db.session.commit()

    flash("AI result marked as reviewed.", "success")
    return redirect(url_for("view_result", case_id=case_id) + "#xray-review")


@app.route("/report/<int:case_id>")
@login_required
def download_report(case_id):
    case = Case.query.get_or_404(case_id)

    # Security: ensure doctor owns the case
    if case.doctor_id != session["user_id"]:
        flash("Not allowed.", "error")
        return redirect(url_for("dashboard"))

    # ----------------------------
    # Doctor info (name)
    # ----------------------------
    doctor = User.query.get(case.doctor_id)
    doctor_name = None

    # Prefer DB name if exists, else fallback to session
    if doctor and hasattr(doctor, "name") and doctor.name:
        doctor_name = doctor.name
    else:
        doctor_name = session.get("user_name") or "—"

    # ----------------------------
    # Patient info (safe)
    # ----------------------------
    patient = None

    # Option A: case has patient_id (new schema)
    if hasattr(case, "patient_id") and getattr(case, "patient_id", None):
        patient = Patient.query.get(case.patient_id)

    # Option B: case still uses patient_code (old schema)
    if patient is None:
        code = getattr(case, "patient_code", None)
        if code is not None:
            patient = Patient.query.filter_by(
                doctor_id=case.doctor_id,
                patient_code=code
            ).first()

    # ----------------------------
    # Results
    # ----------------------------
    # front = Result.query.filter_by(case_id=case_id, view_type="FRONT").first()
    side = Result.query.filter_by(case_id=case_id, view_type="SIDE").first()
    front_ns = Result.query.filter_by(case_id=case_id, view_type="FRONT_NS").first()

    if case.status in ("NEEDS_REUPLOAD", "FAILED"):
        flash(
            case.failure_message or "Report is not available until all photos are analyzed successfully.",
            "error",
        )
        return redirect(url_for("view_result", case_id=case_id))

    if not side and not front_ns:
        flash("No analyzed results available for PDF export.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    def parse_points(r):
        if not r:
            return []
        try:
            return json.loads(r.landmarks_json)
        except Exception:
            return []

    # front_points = parse_points(front)
    side_points = parse_points(side)
    front_ns_points = parse_points(front_ns)

    # ----------------------------
    # Generate PDF
    # ----------------------------
    ortho_rec_for_pdf = OrthoCase.query.filter_by(case_id=case_id).first()
    pdf_path = generate_case_pdf(
        case,
        # front,
        side,
        # front_points,
        side_points,
        patient=patient,
        doctor_name=doctor_name,
        front_ns=front_ns,
        front_ns_points=front_ns_points,
        ortho_rec=ortho_rec_for_pdf,
    )

    # ----------------------------
    # Save/Update report record
    # ----------------------------
    pdf_stored = normalize_stored_path(pdf_path)
    existing = Report.query.filter_by(case_id=case_id).first()
    if existing:
        existing.file_path = pdf_stored
        existing.created_at = datetime.utcnow()
    else:
        db.session.add(Report(case_id=case_id, file_path=pdf_stored))

    db.session.commit()

    pdf_abs = resolve_project_path(pdf_stored)
    return send_file(
        pdf_abs,
        as_attachment=True,
        download_name=f"case_{case_id}_report.pdf"
    )




@app.route("/reports")
@login_required
def reports():
    doctor_id = session["user_id"]

    # show only cases belonging to this doctor
    cases = (
        Case.query.options(joinedload(Case.patient))
        .filter_by(doctor_id=doctor_id)
        .order_by(Case.created_at.desc())
        .all()
    )

    # map case_id -> report
    reps = Report.query.join(Case, Report.case_id == Case.id).filter(Case.doctor_id == doctor_id).all()
    rep_map = {r.case_id: r for r in reps}

    return render_template(
        "reports.html",
        name=session.get("user_name"),
        active="reports",
        cases=cases,
        rep_map=rep_map
    )
@app.route("/case/<int:case_id>/update-landmarks", methods=["POST"])
@login_required
def update_landmarks(case_id):
    case = Case.query.get_or_404(case_id)

    if case.doctor_id != session["user_id"]:
        flash("Not allowed.", "error")
        return redirect(url_for("dashboard"))

    view_type = request.form.get("view_type", "").strip().upper()  # SIDE or FRONT_NS
    points_json = request.form.get("points_json", "").strip()

    if view_type not in ("SIDE", "FRONT_NS"):
        flash("Invalid view type.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    # Validate JSON
    try:
        points = json.loads(points_json)
        if not isinstance(points, list) or len(points) == 0:
            raise ValueError("Empty points")
    except Exception:
        flash("Invalid landmarks data.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    # Update result row
    res = Result.query.filter_by(case_id=case_id, view_type=view_type).first()
    if not res:
        flash("Result not found for this view.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    res.landmarks_json = json.dumps(points)

    # Re-draw overlay using original uploaded image
    img_row = Image.query.filter_by(case_id=case_id, view_type=view_type).first()
    if not img_row:
        flash("Original image not found.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    ensure_static_subdirs("results")
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    new_overlay = results_abs(f"{case_id}_{view_type.lower()}_overlay.jpg", BASE_DIR)

    image_abs = resolve_project_path(img_row.file_path, BASE_DIR)
    image_bgr = cv2.imread(image_abs) if image_abs else None
    if image_bgr is None:
        flash("Could not read original image.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    pts = [(int(p["x"]), int(p["y"])) for p in points]

    overlay = draw_points(image_bgr, pts)
    save_overlay_image(overlay, new_overlay)

    res.overlay_path = overlay_rel(case_id, view_type)

    db.session.commit()

    flash(f"{view_type} landmarks updated.", "success")
    return redirect(url_for("view_result", case_id=case_id))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/model-health")
@login_required
def model_health():
    """Verify SIDE and FRONT_NS landmark models load and can infer."""
    report = run_model_health_check()
    return render_template(
        "model_health.html",
        report=report,
        name=session.get("user_name"),
    )


@app.route("/admin")
@admin_required
def admin_dashboard():
    # example stats
    total_users = User.query.count()
    total_cases = Case.query.count()
    completed_cases = Case.query.filter_by(status="COMPLETED").count()
    pending_cases = Case.query.filter_by(status="PENDING").count()

    return render_template(
        "admin_dashboard.html",
        active="admin_dashboard",
        name=session.get("user_name"),
        total_users=total_users,
        total_cases=total_cases,
        completed_cases=completed_cases,
        pending_cases=pending_cases,
    )


@app.route("/admin/datasets")
@admin_required
def admin_datasets():
    return render_template("admin_datasets.html", active="admin_datasets", name=session.get("user_name"))

# ------------------------
# ADMIN - USERS CRUD
# ------------------------

@app.route("/admin/users")
@admin_required
def admin_users():
    q = request.args.get("q", "").strip()
    role = request.args.get("role", "ALL").strip().upper()

    query = User.query

    if role != "ALL":
        query = query.filter(User.role == role)

    if q:
        query = query.filter(
            (User.name.ilike(f"%{q}%")) | (User.email.ilike(f"%{q}%"))
        )

    users = query.order_by(User.id.desc()).all()

    return render_template(
        "admin_users.html",
        active="admin_users",
        name=session.get("user_name"),
        users=users,
        q=q,
        role=role
    )


@app.route("/admin/users/create", methods=["POST"])
@admin_required
def admin_create_user():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    role = request.form.get("role", "DOCTOR").strip().upper()

    if not name or not email or not password:
        flash("Name, email, and password are required.", "error")
        return redirect(url_for("admin_users"))

    if role not in ["DOCTOR", "ADMIN"]:
        role = "DOCTOR"

    exists = User.query.filter_by(email=email).first()
    if exists:
        flash("Email already exists.", "error")
        return redirect(url_for("admin_users"))

    u = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password),
        role=role
    )
    db.session.add(u)
    db.session.commit()

    flash("User created successfully.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/update", methods=["POST"])
@admin_required
def admin_update_user(user_id):
    u = User.query.get_or_404(user_id)

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    role = request.form.get("role", u.role).strip().upper()
    new_password = request.form.get("new_password", "").strip()

    if not name or not email:
        flash("Name and email are required.", "error")
        return redirect(url_for("admin_users"))

    # avoid duplicate email
    email_owner = User.query.filter_by(email=email).first()
    if email_owner and email_owner.id != u.id:
        flash("Email already used by another account.", "error")
        return redirect(url_for("admin_users"))

    if role not in ["DOCTOR", "ADMIN"]:
        role = "DOCTOR"

    u.name = name
    u.email = email
    u.role = role

    if new_password:
        if len(new_password) < 6:
            flash("New password must be at least 6 characters.", "error")
            return redirect(url_for("admin_users"))
        u.password_hash = generate_password_hash(new_password)

    db.session.commit()
    flash("User updated.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    u = User.query.get_or_404(user_id)

    # Safety: prevent deleting yourself
    if u.id == session.get("user_id"):
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("admin_users"))

    db.session.delete(u)
    db.session.commit()

    flash("User deleted.", "success")
    return redirect(url_for("admin_users"))


# ------------------------
# ADMIN - PATIENTS CRUD
# ------------------------


@app.route("/patients", methods=["GET"])
@login_required
def patients():
    doctor_id = session["user_id"]

    q = (request.args.get("q") or "").strip()
    query = Patient.query.filter_by(doctor_id=doctor_id)

    if q:
        if q.isdigit():
            query = query.filter(Patient.patient_code == int(q))
        else:
            like = f"%{q}%"
            query = query.filter(
                or_(
                    cast(Patient.patient_code, String).like(like),
                    Patient.name.ilike(like)  # remove if you don't have name yet
                )
            )

    patients_list = query.order_by(Patient.created_at.desc()).all()

    access_code_by_patient = {}
    latest_case_by_patient = {}
    case_counts_by_patient = {}
    cases_by_patient = {}
    for p in patients_list:
        row = (
            PatientUploadCode.query.filter_by(
                patient_id=p.id,
                doctor_id=doctor_id,
            )
            .order_by(PatientUploadCode.created_at.asc())
            .first()
        )
        if row:
            access_code_by_patient[p.id] = row.code

        p_cases = (
            Case.query.filter_by(patient_id=p.id, doctor_id=doctor_id)
            .order_by(Case.created_at.desc())
            .all()
        )
        cases_by_patient[p.id] = p_cases
        latest_case_by_patient[p.id] = p_cases[0] if p_cases else None

        # Case status counts
        total = len(p_cases)
        needs_review = sum(1 for c in p_cases if c.status in ("PENDING_REVIEW", "PENDING", "PROCESSING"))
        needs_reupload = sum(1 for c in p_cases if c.status in ("NEEDS_REUPLOAD", "FAILED"))
        reviewed = sum(1 for c in p_cases if c.status in ("REVIEWED", "REVIWED", "COMPLETED"))
        case_counts_by_patient[p.id] = {
            "total": total,
            "needs_review": needs_review,
            "needs_reupload": needs_reupload,
            "reviewed": reviewed,
        }

    selected_patient_id = request.args.get("patient_id", type=int)

    return render_template(
        "patients.html",
        name=session.get("user_name"),
        active="patients",
        patients=patients_list,
        q=q,
        access_code_by_patient=access_code_by_patient,
        latest_case_by_patient=latest_case_by_patient,
        case_counts_by_patient=case_counts_by_patient,
        cases_by_patient=cases_by_patient,
        selected_patient_id=selected_patient_id,
        medical_condition_labels=MEDICAL_CONDITION_LABELS,
    )


@app.route("/patients/add", methods=["POST"])
@login_required
def doctor_add_patient():
    doctor_id = session["user_id"]
    name = clean_name(request.form.get("name", ""))
    patient_code = (request.form.get("patient_code") or "").strip().upper()
    age_raw = (request.form.get("age") or "").strip()
    gender = (request.form.get("gender") or "").strip().upper()

    if not name:
        flash("Patient name is required.", "error")
        return redirect(url_for("patients"))

    if not patient_code:
        flash("Patient code is required.", "error")
        return redirect(url_for("patients"))

    existing = Patient.query.filter_by(
        patient_code=patient_code,
        doctor_id=doctor_id,
    ).first()
    if existing:
        flash("A patient with this code already exists.", "error")
        return redirect(url_for("patients"))

    age_val = None
    if age_raw:
        try:
            age_val = int(age_raw)
            if age_val < 1 or age_val > 119:
                raise ValueError()
        except Exception:
            flash("Age must be between 1 and 119.", "error")
            return redirect(url_for("patients"))

    if gender and gender not in ["MALE", "FEMALE"]:
        flash("Gender must be Male or Female.", "error")
        return redirect(url_for("patients"))

    patient = Patient(
        doctor_id=doctor_id,
        patient_code=patient_code,
        name=name,
        age=age_val,
        gender=gender if gender else None,
    )

    # ── Medical history (Phase 1) ──
    med = _parse_medical_history_form(request.form)
    if "other" in med["conditions"] and not med["other_details"]:
        flash("Please specify the other medical condition.", "error")
        return redirect(url_for("patients"))
    _apply_medical_history(patient, med)

    db.session.add(patient)
    db.session.commit()
    flash(f"Patient {name} ({patient_code}) added.", "success")
    return redirect(url_for("patients", patient_id=patient.id))


@app.route("/admin/patients")
@admin_required
def admin_patients():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    doctors = User.query.order_by(User.id.desc()).all()   # if you need doctor dropdown
    return render_template("admin_patients.html", patients=patients, doctors=doctors)


@app.route("/admin/patients/create", methods=["POST"])
@admin_required
def admin_create_patient():
    doctor_id = request.form.get("doctor_id", "").strip()
    patient_code = request.form.get("patient_code", "").strip()
    name = request.form.get("name", "").strip()
    age = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip().upper()

    # ---------- Doctor ----------
    if not doctor_id.isdigit():
        flash("Doctor ID must be a number.", "error")
        return redirect(url_for("admin_patients"))

    doctor = User.query.get(int(doctor_id))
    if not doctor:
        flash("Doctor not found.", "error")
        return redirect(url_for("admin_patients"))

    # ---------- Patient Code ----------
    if not patient_code:
        flash("Patient code is required.", "error")
        return redirect(url_for("admin_patients"))

    # ---------- Patient Name ----------
    if not name:
        flash("Patient name is required.", "error")
        return redirect(url_for("admin_patients"))

    # ---------- Age ----------
    age_val = None
    if age:
        if not age.isdigit() or not (1 <= int(age) <= 119):
            flash("Age must be a number between 1 and 119.", "error")
            return redirect(url_for("admin_patients"))
        age_val = int(age)

    # ---------- Gender ----------
    if gender and gender not in ["MALE", "FEMALE"]:
        flash("Gender must be Male or Female.", "error")
        return redirect(url_for("admin_patients"))

    # ---------- Create ----------
    patient = Patient(
        doctor_id=int(doctor_id),
        patient_code=patient_code,
        name=name,
        age=age_val,
        gender=gender if gender else None
    )

    db.session.add(patient)
    db.session.commit()

    flash("Patient created successfully.", "success")
    return redirect(url_for("admin_patients"))



@app.route("/admin/patients/<int:patient_id>/update", methods=["POST"])
@admin_required
def admin_update_patient(patient_id):
    p = Patient.query.get_or_404(patient_id)

    patient_code = request.form.get("patient_code", "").strip()
    name = request.form.get("name", "").strip()
    age = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip().upper()

    # ---------- Code ----------
    if not patient_code:
        flash("Patient code is required.", "error")
        return redirect(url_for("admin_patients"))

    # ---------- Name ----------
    if not name:
        flash("Patient name is required.", "error")
        return redirect(url_for("admin_patients"))

    # ---------- Age ----------
    age_val = None
    if age:
        if not age.isdigit() or not (1 <= int(age) <= 119):
            flash("Age must be a number between 1 and 119.", "error")
            return redirect(url_for("admin_patients"))
        age_val = int(age)

    # ---------- Gender ----------
    if gender and gender not in ["MALE", "FEMALE"]:
        flash("Gender must be Male or Female.", "error")
        return redirect(url_for("admin_patients"))

    # ---------- Save ----------
    p.patient_code = patient_code
    p.name = name
    p.age = age_val
    p.gender = gender if gender else None

    db.session.commit()

    flash("Patient updated successfully.", "success")
    return redirect(url_for("admin_patients"))


@app.route("/admin/patients/<int:patient_id>/delete", methods=["POST"])
@admin_required
def admin_delete_patient(patient_id):
    p = Patient.query.get_or_404(patient_id)
    patient_name = p.name

    try:
        # Delete portal codes, cases (images/results/reports), messages, appointments, then patient.
        purge_patient_completely(p)
        db.session.commit()
    except Exception:
        db.session.rollback()
        flash("Could not delete patient. Please try again.", "error")
        return redirect(url_for("admin_patients"))

    flash(f'Patient "{patient_name}" deleted.', "success")
    return redirect(url_for("admin_patients"))


@app.route("/about")
@login_required
def about():
    return render_template("about.html", name=session.get("user_name"), active="about")


@app.route("/contact", methods=["GET", "POST"])
@login_required
def contact():
    if request.method == "POST":
        # simple validation (you can later store in DB or send email)
        name = request.form.get("name","").strip()
        email = request.form.get("email","").strip()
        subject = request.form.get("subject","").strip()
        message = request.form.get("message","").strip()

        if not name or not email or not subject or not message:
            flash("Please fill all fields.", "error")
            return redirect(url_for("contact"))

        flash("Message sent successfully. Thank you!", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html", name=session.get("user_name"), active="contact")

@app.route("/debug/fix-patient-doctors")
def fix_patient_doctors():
    Patient.query.filter(Patient.doctor_id == None).update(
        {"doctor_id": session.get("user_id")}
    )
    db.session.commit()
    return "Patients updated"


@app.route("/patient/register" , methods=["GET" , "POST"])
def patient_register():
    if request.method == "POST":
        name = clean_name(request.form.get("name"))
        email = clean_email(request.form.get("email"))
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not name:
            flash("Name is required." , "error")
            return redirect(url_for("patient_register"))
        
        if not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("patient_register"))
        
        if not is_valid_password(password, min_len=8):
            flash("Password must be at least 8 characters.", "error")
            return redirect(url_for("patient_register"))
        
        if password != confirm:
            flash("Passwords do not match. " , "error")
            return redirect(url_for("patient_register"))
        
        existing = PatientAuth.query.filter_by(email=email).first()
        if existing:
            flash("Email already registered. Please login.","error")
            return redirect(url_for("login"))
        
        patient_user = PatientAuth(
            name=name,
            email=email
        )
        patient_user.set_password(password)

        db.session.add(patient_user)
        db.session.commit()

        flash("Patient account created successfully. Please Login.", "success")
        return redirect(url_for("login"))
    
    return render_template("patient_register.html")


@app.route("/patient/dashboard")
@patient_login_required
def patient_dashboard():
    patient_auth_id = session["patient_auth_id"]

    linked_profiles = Patient.query.filter_by(patient_auth_id=patient_auth_id).all()

    total_doctors = len({p.doctor_id for p in linked_profiles})
    total_profiles = len(linked_profiles)

    return render_template(
        "patient_dashboard.html",
        patient_name=session.get("patient_name"),
        linked_profiles=linked_profiles,
        total_doctors=total_doctors,
        total_profiles=total_profiles,
        active="patient_dashboard"
    )

@app.route("/patients/<int:patient_id>/generate-code", methods=["post"])
@login_required
def generate_patient_code(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=session["user_id"]
    ).first_or_404()

    code_row = get_or_create_patient_access_code(patient.id, session["user_id"])
    flash(f"Patient access code for {patient.name}: {code_row.code}", "success")
    ret = request.form.get("redirect_to")
    cid = request.form.get("case_id", type=int)
    if ret == "result" and cid:
        return redirect(url_for("view_result", case_id=cid))
    return redirect(url_for("patients", patient_id=patient.id))


@app.route("/patients/<int:patient_id>/regenerate-code", methods=["POST"])
@login_required
def regenerate_patient_code_route(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=session["user_id"],
    ).first_or_404()

    code_row = regenerate_patient_access_code(patient.id, session["user_id"])
    flash(f"New access code for {patient.name}: {code_row.code} (previous code invalidated).", "success")
    ret = request.form.get("redirect_to")
    cid = request.form.get("case_id", type=int)
    if ret == "result" and cid:
        return redirect(url_for("view_result", case_id=cid))
    return redirect(url_for("patients", patient_id=patient.id))


@app.route("/patients/<int:patient_id>/notes", methods=["POST"])
@login_required
def save_patient_notes(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=session["user_id"],
    ).first_or_404()

    notes = (request.form.get("private_notes") or "").strip()
    patient.private_notes = notes if notes else None
    patient.private_notes_updated_at = datetime.utcnow()
    db.session.commit()
    flash(f"Private notes saved for {patient.name}.", "success")
    ret = request.form.get("redirect_to")
    cid = request.form.get("case_id", type=int)
    if ret == "result" and cid:
        return redirect(url_for("view_result", case_id=cid))
    return redirect(url_for("patients", patient_id=patient.id))


@app.route("/patients/<int:patient_id>/medical-history", methods=["POST"])
@login_required
def edit_medical_history(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=session["user_id"],
    ).first_or_404()

    med = _parse_medical_history_form(request.form)
    if "other" in med["conditions"] and not med["other_details"]:
        flash("Please specify the other medical condition.", "error")
        return redirect(url_for("patients", patient_id=patient.id))

    _apply_medical_history(patient, med)
    db.session.commit()
    flash(f"Medical history updated for {patient.name}.", "success")
    return redirect(url_for("patients", patient_id=patient.id))


@app.route("/patient/connect-doctor", methods=["GET", "POST"])
@patient_login_required
def patient_connect_doctor():
    if request.method == "POST":
        code_value = (request.form.get("code") or "").strip().upper()

        if not code_value:
            flash("Please enter a code.", "error")
            return redirect(url_for("patient_connect_doctor"))

        code_row = PatientUploadCode.query.filter_by(code=code_value).first()

        if not code_row:
            flash("Invalid code.", "error")
            return redirect(url_for("patient_connect_doctor"))

        if code_row.used_at is not None:
            flash("This code has already been used.", "error")
            return redirect(url_for("patient_connect_doctor"))

        patient = Patient.query.get(code_row.patient_id)
        if not patient:
            flash("Linked patient record was not found.", "error")
            return redirect(url_for("patient_connect_doctor"))

        # link logged-in patient account to this doctor-side patient record
        patient.patient_auth_id = session["patient_auth_id"]
        code_row.used_at = datetime.utcnow()

        db.session.commit()

        flash("Doctor connection completed successfully.", "success")
        return redirect(url_for("patient_dashboard"))

    return render_template("patient_connect_doctor.html")






@app.route("/patient-portal/<code>", methods=["GET"])
def patient_portal(code):
    code_row = PatientUploadCode.query.filter_by(code=code.upper()).first_or_404()
    patient = Patient.query.get_or_404(code_row.patient_id)
    doctor = User.query.get_or_404(code_row.doctor_id)

    patient_cases = (
        Case.query
        .filter_by(patient_id=patient.id, doctor_id=doctor.id)
        .order_by(Case.created_at.desc())
        .all()
    )

    latest_case = patient_cases[0] if len(patient_cases) > 0 else None

    next_appointment = _next_patient_appointment(doctor.id, patient.id)

    availability_rows = (
        DoctorAvailability.query.filter_by(doctor_id=doctor.id, is_active=True)
        .order_by(DoctorAvailability.weekday.asc(), DoctorAvailability.slot_time.asc())
        .all()
    )
    bookable_dates = get_bookable_dates(doctor.id, availability_rows)
    selected_book_date = request.args.get("book_date", "").strip()
    book_slots = []
    if selected_book_date:
        try:
            bd = datetime.strptime(selected_book_date, "%Y-%m-%d").date()
            if bd >= date.today():
                book_slots = get_available_slots_for_date(doctor.id, bd, availability_rows)
        except ValueError:
            selected_book_date = ""

    can_book_followup = bool(
        latest_case
        and latest_case.follow_up_requested
        and bookable_dates
        and not next_appointment
    )

    def _status_label(st):
        if st == "NEEDS_REUPLOAD":
            return "Needs re-upload"
        if st == "PENDING_REVIEW":
            return "Analyzed — awaiting doctor review"
        if st == "REVIEWED" or st == "REVIWED":
            return "Reviewed"
        if st == "FAILED":
            return "Could not analyze"
        if st == "PROCESSING":
            return "Processing"
        return st.replace("_", " ").title()

    portal_reports = _patient_portal_reports(patient.id, doctor.id)
    last_visit = latest_case.created_at if latest_case else patient.created_at

    return render_template(
        "patient_portal.html",
        patient=patient,
        doctor=doctor,
        code_row=code_row,
        cases=patient_cases,
        latest_case=latest_case,
        status_label=_status_label,
        friendly_fail=FRIENDLY_FAIL,
        next_appointment=next_appointment,
        portal_reports=portal_reports,
        reports_count=len(portal_reports),
        submissions_count=len(patient_cases),
        last_visit=last_visit,
        can_book_followup=can_book_followup,
        bookable_dates=bookable_dates,
        book_slots=book_slots,
        selected_book_date=selected_book_date,
        weekday_label=weekday_label,
    )


@app.route("/patient-portal/<code>/book-appointment", methods=["POST"])
def patient_book_appointment(code):
    code_row = PatientUploadCode.query.filter_by(code=code.upper()).first_or_404()
    patient = Patient.query.get_or_404(code_row.patient_id)
    doctor_id = code_row.doctor_id

    date_str = (request.form.get("appointment_date") or "").strip()
    time_str = (request.form.get("appointment_time") or "").strip()

    try:
        appt_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        appt_time = datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        flash("Please select a valid date and time.", "error")
        return redirect(url_for("patient_portal", code=code_row.code))

    if appt_date < date.today():
        flash("Please choose a future date.", "error")
        return redirect(url_for("patient_portal", code=code_row.code))

    availability_rows = DoctorAvailability.query.filter_by(
        doctor_id=doctor_id, is_active=True
    ).all()
    slots = get_available_slots_for_date(doctor_id, appt_date, availability_rows)
    matching = [s for s in slots if s["time"] == appt_time and s["available"]]
    if not matching:
        flash("That time is no longer available. Please choose another slot.", "error")
        return redirect(url_for("patient_portal", code=code_row.code, book_date=date_str))

    latest_case = (
        Case.query.filter_by(patient_id=patient.id, doctor_id=doctor_id)
        .order_by(Case.created_at.desc())
        .first()
    )

    appt = Appointment(
        doctor_id=doctor_id,
        patient_id=patient.id,
        case_id=latest_case.id if latest_case else None,
        reason="Follow-up checkup (patient booking)",
        appointment_date=appt_date,
        appointment_time=appt_time,
        status="BOOKED",
        source="patient",
        notes=None,
    )
    db.session.add(appt)
    db.session.commit()

    flash("Your appointment has been booked. We look forward to seeing you!", "success")
    return redirect(url_for("patient_portal", code=code_row.code))


@app.route("/patient-portal/<code>/message", methods=["POST"])
def patient_send_message(code):
    code_row = PatientUploadCode.query.filter_by(code=code.upper()).first_or_404()
    patient = Patient.query.get_or_404(code_row.patient_id)
    question = (request.form.get("question") or "").strip()

    if len(question) < 5:
        flash("Please write a short question (at least a few words).", "error")
        return redirect(url_for("patient_portal", code=code_row.code))
    if len(question) > 280:
        flash("Please keep your question under 280 characters.", "error")
        return redirect(url_for("patient_portal", code=code_row.code))

    msg = PatientMessage(
        patient_id=patient.id,
        doctor_id=code_row.doctor_id,
        question=question,
        read=False,
    )
    db.session.add(msg)
    db.session.commit()
    flash("Your question was sent to your doctor.", "success")
    return redirect(url_for("patient_portal", code=code_row.code))


@app.route("/patient-portal/<code>/report/<int:case_id>")
def patient_portal_report(code, case_id):
    code_row = PatientUploadCode.query.filter_by(code=code.upper()).first_or_404()
    case = Case.query.filter_by(
        id=case_id,
        patient_id=code_row.patient_id,
        doctor_id=code_row.doctor_id,
    ).first_or_404()
    report = Report.query.filter_by(case_id=case.id).first_or_404()
    path = resolve_project_path(report.file_path)
    if not path or not os.path.isfile(path):
        flash("Report file is not available yet.", "error")
        return redirect(url_for("patient_portal", code=code_row.code))
    return send_file(path, as_attachment=True, download_name=f"case_{case.id}_report.pdf")


@app.route("/patient-portal/<code>/upload", methods=["POST"])
def patient_upload_progress(code):
    code_row = PatientUploadCode.query.filter_by(code=code.upper()).first_or_404()
    patient = Patient.query.get_or_404(code_row.patient_id)

    side_file = request.files.get("side")
    front_ns_file = request.files.get("front_ns")

    def _upload_nonempty(f):
        return f and getattr(f, "filename", None) and str(f.filename).strip() != ""

    has_side = _upload_nonempty(side_file)
    has_front = _upload_nonempty(front_ns_file)

    if not has_side and not has_front:
        flash("Please upload at least one photo (side and/or frontal).", "error")
        return redirect(url_for("patient_portal", code=code_row.code))

    ensure_static_subdirs("uploads", "results")
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    new_case = Case(
        doctor_id=code_row.doctor_id,
        patient_id=patient.id,
        case_type="FOLLOW_UP",
        status="PROCESSING",
    )
    db.session.add(new_case)
    db.session.commit()

    outcomes = []
    try:
        if has_side:
            side_name = f"{new_case.id}_side_{uuid.uuid4().hex}.jpg"
            side_rel = upload_rel(side_name)
            side_abs = upload_abs(side_name, BASE_DIR)
            side_file.save(side_abs)
            db.session.add(Image(case_id=new_case.id, view_type="SIDE", file_path=side_rel))
            db.session.commit()

            side_out = run_view_analysis(new_case.id, side_abs, "SIDE")
            if side_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="SIDE",
                        landmarks_json=side_out["landmarks_json"],
                        overlay_path=normalize_stored_path(side_out["overlay_path"]),
                        confidence_json=side_out.get("confidence_json"),
                    )
                )
            outcomes.append(
                {
                    "uploaded": True,
                    "success": side_out["success"],
                    "message": side_out.get("message") or "",
                    "failed_stage": side_out.get("failed_stage"),
                }
            )

        if has_front:
            front_name = f"{new_case.id}_front_ns_{uuid.uuid4().hex}.jpg"
            front_rel = upload_rel(front_name)
            front_abs = upload_abs(front_name, BASE_DIR)
            front_ns_file.save(front_abs)
            db.session.add(Image(case_id=new_case.id, view_type="FRONT_NS", file_path=front_rel))
            db.session.commit()

            front_out = run_view_analysis(new_case.id, front_abs, "FRONT_NS")
            if front_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="FRONT_NS",
                        landmarks_json=front_out["landmarks_json"],
                        overlay_path=normalize_stored_path(front_out["overlay_path"]),
                        confidence_json=front_out.get("confidence_json"),
                    )
                )
            outcomes.append(
                {
                    "uploaded": True,
                    "success": front_out["success"],
                    "message": front_out.get("message") or "",
                    "failed_stage": front_out.get("failed_stage"),
                }
            )

        _finalize_case_after_analysis(new_case, outcomes)
        db.session.commit()

        if new_case.status in ("NEEDS_REUPLOAD", "FAILED"):
            flash(new_case.failure_message or MSG_ANALYSIS_FAILED, "error")
            return redirect(url_for("patient_portal", code=code_row.code))

        flash("Photos uploaded successfully. Your doctor can review them soon.", "success")
        return redirect(url_for("patient_portal", code=code_row.code))

    except Exception:
        new_case.status = "FAILED"
        new_case.failure_message = MSG_ANALYSIS_FAILED
        db.session.commit()
        flash(MSG_ANALYSIS_FAILED, "error")
        return redirect(url_for("patient_portal", code=code_row.code))

@app.route("/cases/<int:case_id>/review", methods=["GET", "POST"])
def review_case(case_id):
    if "user_id" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))
    
    case= Case.query.get_or_404(case_id)

    if case.doctor_id != session["user_id"]:
        flash("Unathorized access." ,"error")
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        doctor_comment = request.form.get("doctor_comment", "").strip()
        follow_up = request.form.get("follow_up_requested") == "1"

        case.doctor_comment = doctor_comment
        case.reviewed_at = datetime.utcnow()
        case.follow_up_requested = follow_up
        case.status = "REVIWED"

        db.session.commit()

        flash("Doctor review saved successfully.", "success")
        if follow_up:
            flash("Patient can request a follow-up from their portal.", "success")
        return redirect(url_for("view_result", case_id=case.id))

    return render_template(
        "review_case.html",
        case=case,
        name=session.get("user_name"),
        active="analysis",
    )



@app.route("/patient-access", methods=["GET", "POST"])
def patient_access():
    if request.method == "POST":
        code = (request.form.get("code") or "").strip().upper()

        if not code:
            flash("Please enter your access code.", "error")
            return redirect(url_for("patient_access"))

        code_row = PatientUploadCode.query.filter_by(code=code).first()

        if not code_row:
            flash("That code is not recognized. Please check with your doctor and try again.", "error")
            return redirect(url_for("patient_access"))

        return redirect(url_for("patient_portal", code=code_row.code))

    return render_template("patient_access.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()

    calibration_points = data.get("calibration_points", [])
    facial_points = data.get("facial_points", [])
    real_distance_mm = data.get("real_distance_mm", None)

    try:
        result = calculate_measurement(
            calibration_points=calibration_points,
            facial_points=facial_points,
            real_distance_mm=real_distance_mm
        )
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Something went wrong during calculation."}), 500



@app.route("/calibration", methods=["GET", "POST"])
def calibration():
    image_url = None
    error = None

    upload_folder = static_dir("uploads", base_dir=BASE_DIR)
    os.makedirs(upload_folder, exist_ok=True)

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file uploaded."
            return render_template("calibration.html", error=error, image_url=image_url)

        file = request.files["image"]

        if file.filename == "":
            error = "No file selected."
            return render_template("calibration.html", error=error, image_url=image_url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template("calibration.html", image_url=image_url, error=error)


@app.route("/case/<int:case_id>/measurement/<measurement_type>")
@login_required
def view_measurement(case_id, measurement_type):
    case = Case.query.get_or_404(case_id)

    if case.doctor_id != session["user_id"]:
        flash("You are not allowed to view this case.", "error")
        return redirect(url_for("dashboard"))

    side_result = Result.query.filter_by(case_id=case_id, view_type="SIDE").first()
    side_img = Image.query.filter_by(case_id=case_id, view_type="SIDE").first()

    if not side_result or not side_img:
        flash("Side image or landmarks not found for this case.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    pts_orig = json.loads(side_result.landmarks_json)

    image_path = resolve_project_path(side_img.file_path, BASE_DIR)

    measurement = analyze_measurement(image_path, pts_orig, measurement_type, case_id)

    return render_template(
        "measurement_detail.html",
        case=case,
        measurement=measurement
    )


@app.route("/case/<int:case_id>/measurements")
@login_required
def view_all_measurements(case_id):
    # Redirect to the Measurements tab on the main result page.
    return redirect(url_for("view_result", case_id=case_id) + "#tab-panel-measurements")

if __name__ == "__main__":
    with app.app_context():
        _ensure_sqlite_columns()
        _normalize_database_paths()
    app.run(debug=True)