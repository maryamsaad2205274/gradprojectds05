from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sqlalchemy import cast, String, or_
from datetime import datetime
from utils.calibration import calculate_measurement
from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from utils.calibration import calculate_measurement
from utils.inference import predict_landmarks, save_overlay_image, landmarks_to_json

from utils.measurements import analyze_measurement
from utils.frontal_measurements import calculate_frontal_measurements
from utils.case_pdf import render_case_pdf
from utils.analysis_pipeline import run_view_analysis
from utils.image_validation import FRIENDLY_FAIL, MSG_ANALYSIS_FAILED
from utils.model_health import run_model_health_check

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


def patient_login_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if "patient_auth_id" not in session:
            return redirect(url_for("patient_login"))
        return route_function(*args, **kwargs)
    return wrapper

app = Flask(__name__)
app.secret_key = "change-this-to-any-random-string"

# Database config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------------
# Database Model: User
# ------------------------
# ------------------------
# Database Models
# ------------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)
    role = db.Column(db.String(20), default="DOCTOR")  # DOCTOR or ADMIN

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class PatientAuth(db.Model):
    __tablename__ = "patient_auth"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Patient(db.Model):
    __tablename__ = "patient"

    id = db.Column(db.Integer, primary_key=True)

    patient_code = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(120), nullable=False)

    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)

    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_auth_id = db.Column(db.Integer, db.ForeignKey("patient_auth.id"), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    doctor = db.relationship("User", backref=db.backref("patients", lazy=True))
    patient_auth = db.relationship("PatientAuth", backref=db.backref("patient_profile", uselist=False))


class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True)

    case_type=db.Column(db.String(20), default="INITIAL")
    status = db.Column(db.String(20), default="PENDING")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref="cases")
    doctor = db.relationship("User", backref=db.backref("cases", lazy=True))

    doctor_comment=db.Column(db.Text, nullable=True)
    reviewed_at= db.Column(db.DateTime, nullable=True)
    failure_message = db.Column(db.Text, nullable=True)

   


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    view_type = db.Column(db.String(10), nullable=False)  # FRONT / SIDE
    file_path = db.Column(db.String(300), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    case = db.relationship("Case", backref=db.backref("images", lazy=True, cascade="all, delete-orphan"))


class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    view_type = db.Column(db.String(10), nullable=False)  # FRONT / SIDE
    landmarks_json = db.Column(db.Text, nullable=False)
    overlay_path = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    case = db.relationship("Case", backref=db.backref("results", lazy=True, cascade="all, delete-orphan"))


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    case = db.relationship("Case", backref=db.backref("report", uselist=False, cascade="all, delete-orphan"))


class PatientUploadCode(db.Model):
    __tablename__ = "patient_upload_code"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)
    used_at = db.Column(db.DateTime, nullable=True)

    doctor = db.relationship("User", backref=db.backref("upload_codes", lazy=True))
    patient = db.relationship("Patient", backref=db.backref("upload_codes", lazy=True))

    @property
    def is_used(self):
        return self.used_at is not None

    @property
    def is_expired(self):
        return self.expires_at is not None and datetime.utcnow() > self.expires_at
    


def _ensure_sqlite_columns():
    """Lightweight migration for columns added after first deploy."""
    from sqlalchemy import inspect, text

    try:
        insp = inspect(db.engine)
        if not insp.has_table("case"):
            return
        cols = {c["name"] for c in insp.get_columns("case")}
        if "failure_message" not in cols:
            with db.engine.begin() as conn:
                conn.execute(text("ALTER TABLE \"case\" ADD COLUMN failure_message TEXT"))
    except Exception:
        pass


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

    total_patients = Patient.query.filter_by(doctor_id=doctor_id).count()
    total_cases = Case.query.filter_by(doctor_id=doctor_id).count()
    completed_cases = Case.query.filter_by(doctor_id=doctor_id, status="COMPLETED").count()
    pending_cases = Case.query.filter_by(doctor_id=doctor_id, status="PENDING").count()

    recent_cases = (
        Case.query
        .filter_by(doctor_id=doctor_id)
        .order_by(Case.created_at.desc())
        .limit(5)
        .all()
    )

    return render_template(
        "dashboard.html",
        name=session.get("user_name"),
        active="dashboard",
        total_patients=total_patients,
        total_cases=total_cases,
        completed_cases=completed_cases,
        pending_cases=pending_cases,
        recent_cases=recent_cases
    )


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
            patients=patients
        )

    # =========================
    # POST → run analysis (one or both views)
    # =========================
    side_file = request.files.get("side")
    front_ns_file = request.files.get("front_ns")

    def _upload_nonempty(f):
        return f and getattr(f, "filename", None) and str(f.filename).strip() != ""

    has_side = _upload_nonempty(side_file)
    has_front_ns = _upload_nonempty(front_ns_file)

    if not has_side and not has_front_ns:
        flash("Please upload at least one image: side view and/or frontal non-smile.", "error")
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

    if new_code:
        new_code = new_code.upper()

        patient = Patient.query.filter_by(
            patient_code=new_code,
            doctor_id=session["user_id"]
        ).first()

        if not patient:
            if not new_name:
                flash("Please enter the patient name.", "error")
                return redirect(url_for("new_analysis"))

            age_val = None
            if new_age:
                try:
                    age_val = int(new_age)
                    if age_val < 1 or age_val > 119:
                        raise ValueError()
                except Exception:
                    flash("Age must be a valid number between 1 and 119.", "error")
                    return redirect(url_for("new_analysis"))

            if new_gender and new_gender not in ["MALE", "FEMALE"]:
                flash("Gender must be MALE or FEMALE (or leave it empty).", "error")
                return redirect(url_for("new_analysis"))

            patient = Patient(
                patient_code=new_code,
                name=new_name,
                age=age_val,
                gender=new_gender if new_gender else None,
                doctor_id=session["user_id"]
            )
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

    # =========================
    # Create case
    # =========================
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    new_case = Case(
        doctor_id=session["user_id"],
        patient_id=patient.id,
        case_type="INITIAL",
        status="PROCESSING"
    )
    db.session.add(new_case)
    db.session.commit()

    outcomes = []
    try:
        if has_side:
            side_name = f"{new_case.id}_side_{uuid.uuid4().hex}.jpg"
            side_path = os.path.join("static/uploads", side_name)
            side_file.save(side_path)
            db.session.add(Image(case_id=new_case.id, view_type="SIDE", file_path=side_path))
            db.session.commit()

            side_out = run_view_analysis(new_case.id, side_path, "SIDE")
            if side_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="SIDE",
                        landmarks_json=side_out["landmarks_json"],
                        overlay_path=side_out["overlay_path"],
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
            front_path = os.path.join("static/uploads", front_name)
            front_ns_file.save(front_path)
            db.session.add(Image(case_id=new_case.id, view_type="FRONT_NS", file_path=front_path))
            db.session.commit()

            front_out = run_view_analysis(new_case.id, front_path, "FRONT_NS")
            if front_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="FRONT_NS",
                        landmarks_json=front_out["landmarks_json"],
                        overlay_path=front_out["overlay_path"],
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
            return redirect(url_for("view_result", case_id=new_case.id))

        parts = []
        if has_side:
            parts.append("side")
        if has_front_ns:
            parts.append("frontal non-smile")
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

    # Security: only owner
    if case.doctor_id != session["user_id"]:
        flash("Not allowed.", "error")
        return redirect(url_for("history"))

    # Delete related records first
    Result.query.filter_by(case_id=case_id).delete()
    Image.query.filter_by(case_id=case_id).delete()

    db.session.delete(case)
    db.session.commit()

    flash("Case deleted.", "success")
    return redirect(url_for("history"))


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

    return render_template(
        "patient_cases.html",
        name=session.get("user_name"),
        active="patients",
        patient=patient,
        cases=cases,
        case_summaries=case_summaries,
        access_code=access_code_row.code if access_code_row else None,
    )





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

    # front_points = parse_points(front)
    side_points = parse_points(side)
    front_ns_points = parse_points(front_ns)

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
        fp = img_row.file_path.replace("\\", "/")
        if fp.startswith("static/"):
            fp = fp[len("static/") :]
        return url_for("static", filename=fp)

    side_upload_url = _static_url_for_image(side_img) if side_img and not side else None
    front_upload_url = _static_url_for_image(front_ns_img) if front_ns_img and not front_ns else None

    return render_template(
    "result.html",
    name=session.get("user_name"),
    active="cases",
    case=case,
    side_overlay=side.overlay_path if side else None,
    front_ns_overlay=front_ns.overlay_path if front_ns else None,
    side_points=side_points,
    front_ns_points=front_ns_points,
    side_original=side_original,
    front_ns_original=front_ns_original,
    front_ns_measurements=front_ns_measurements,
    view_summary=view_summary,
    friendly_fail=FRIENDLY_FAIL,
    side_upload_url=side_upload_url,
    front_upload_url=front_upload_url,
)

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
    )

    # ----------------------------
    # Save/Update report record
    # ----------------------------
    existing = Report.query.filter_by(case_id=case_id).first()
    if existing:
        existing.file_path = pdf_path
        existing.created_at = datetime.utcnow()
    else:
        db.session.add(Report(case_id=case_id, file_path=pdf_path))

    db.session.commit()

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"case_{case_id}_report.pdf"
    )




@app.route("/reports")
@login_required
def reports():
    doctor_id = session["user_id"]

    # show only cases belonging to this doctor
    cases = Case.query.filter_by(doctor_id=doctor_id).order_by(Case.created_at.desc()).all()

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

    os.makedirs("static/results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    new_overlay = os.path.join(
        "static/results",
        f"{case_id}_{view_type.lower()}_overlay.jpg"
    )

    image_bgr = cv2.imread(img_row.file_path)
    if image_bgr is None:
        flash("Could not read original image.", "error")
        return redirect(url_for("view_result", case_id=case_id))

    pts = [(int(p["x"]), int(p["y"])) for p in points]

    overlay = draw_points(image_bgr, pts)
    save_overlay_image(overlay, new_overlay)

    res.overlay_path = f"results/{case_id}_{view_type.lower()}_overlay.jpg"

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

    return render_template(
        "patients.html",
        name=session.get("user_name"),
        active="patients",
        patients=patients_list,
        q=q,
        access_code_by_patient=access_code_by_patient,
    )

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

    db.session.delete(p)
    db.session.commit()

    flash("Patient deleted.", "success")
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
    return redirect(url_for("patients"))


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

    return render_template(
        "patient_portal.html",
        patient=patient,
        doctor=doctor,
        code_row=code_row,
        cases=patient_cases,
        latest_case=latest_case,
        status_label=_status_label,
        friendly_fail=FRIENDLY_FAIL,
    )

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

    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

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
            side_path = os.path.join("static/uploads", side_name)
            side_file.save(side_path)
            db.session.add(Image(case_id=new_case.id, view_type="SIDE", file_path=side_path))
            db.session.commit()

            side_out = run_view_analysis(new_case.id, side_path, "SIDE")
            if side_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="SIDE",
                        landmarks_json=side_out["landmarks_json"],
                        overlay_path=side_out["overlay_path"],
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
            front_path = os.path.join("static/uploads", front_name)
            front_ns_file.save(front_path)
            db.session.add(Image(case_id=new_case.id, view_type="FRONT_NS", file_path=front_path))
            db.session.commit()

            front_out = run_view_analysis(new_case.id, front_path, "FRONT_NS")
            if front_out["success"]:
                db.session.add(
                    Result(
                        case_id=new_case.id,
                        view_type="FRONT_NS",
                        landmarks_json=front_out["landmarks_json"],
                        overlay_path=front_out["overlay_path"],
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

        case.doctor_comment = doctor_comment
        case.reviewed_at = datetime.utcnow()

    
        case.status="REVIWED"

        db.session.commit()

        flash("Doctor comment saved successfully." , "success")
        return redirect(url_for("view_result", case_id=case.id))
    
    return render_template("review_case.html", case=case)



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

    upload_folder = os.path.join("static", "uploads")
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

    base_dir = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.join(base_dir, side_img.file_path)

    measurement = analyze_measurement(image_path, pts_orig, measurement_type, case_id)

    return render_template(
        "measurement_detail.html",
        case=case,
        measurement=measurement
    )


@app.route("/case/<int:case_id>/measurements")
@login_required
def view_all_measurements(case_id):
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

    base_dir = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.join(base_dir, side_img.file_path)

    measurement_types = [
        "nasiolabial",
        "profile_convexity",
        "total_facial_convexity",
        "mentolabial"
    ]

    measurements = [
        analyze_measurement(image_path, pts_orig, mtype, case_id)
        for mtype in measurement_types
    ]

    return render_template(
        "measurements_all.html",
        case=case,
        measurements=measurements
    )

if __name__ == "__main__":
    app.run(debug=True)
