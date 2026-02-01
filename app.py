from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from functools import wraps
import uuid
import json
from utils.inference import predict_landmarks
from utils.draw_landmarks import draw_points
from datetime import datetime
from flask import send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import re
from PIL import Image

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
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)
    role = db.Column(db.String(20), default="DOCTOR")  # DOCTOR or ADMIN

class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # ✅ add these:
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True)
    patient = db.relationship("Patient", backref="cases")

    status = db.Column(db.String(20), default="PENDING")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    view_type = db.Column(db.String(10), nullable=False)  # FRONT/SIDE
    file_path = db.Column(db.String(300), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    view_type = db.Column(db.String(10), nullable=False)  # FRONT/SIDE
    landmarks_json = db.Column(db.Text, nullable=False)   # store points as JSON string
    overlay_path = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    patient_code = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(120), nullable=True)   

    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)  # e.g. static/reports/case_1_report.pdf
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def _as_xy_list(points):
    out = []
    for p in points or []:
        if isinstance(p, dict):
            out.append((float(p.get("x", 0)), float(p.get("y", 0))))
        else:
            out.append((float(p[0]), float(p[1])))
    return out

def _draw_points_table(c, title, points_xy, x0, y0, row_h=14):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y0, title)

    c.setFont("Helvetica", 10)
    y = y0 - 18

    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0, y, "ID")
    c.drawString(x0 + 50, y, "X")
    c.drawString(x0 + 120, y, "Y")
    c.setFont("Helvetica", 10)
    y -= row_h

    for i, (xv, yv) in enumerate(points_xy[:17], start=1):
        if y < 60:
            c.showPage()
            y = 780
            c.setFont("Helvetica", 10)
        c.drawString(x0, y, f"{i:02d}")
        c.drawString(x0 + 50, y, f"{xv:.1f}")
        c.drawString(x0 + 120, y, f"{yv:.1f}")
        y -= row_h


import os, json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

def _as_xy_list(points):
    """Supports dict {'x','y'} and legacy [x,y]. Returns list of (x,y)."""
    out = []
    for p in points or []:
        if isinstance(p, dict):
            out.append((float(p.get("x", 0)), float(p.get("y", 0))))
        else:
            out.append((float(p[0]), float(p[1])))
    return out

def _draw_points_table(c, title, points_xy, x0, y0, row_h=14):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y0, title)

    c.setFont("Helvetica", 10)
    y = y0 - 18

    # header
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0, y, "ID")
    c.drawString(x0 + 50, y, "X")
    c.drawString(x0 + 120, y, "Y")
    c.setFont("Helvetica", 10)
    y -= row_h

    for i, (xv, yv) in enumerate(points_xy[:17], start=1):
        if y < 60:          # page bottom margin
            c.showPage()
            y = 780
            c.setFont("Helvetica", 10)
        c.drawString(x0, y, f"{i:02d}")
        c.drawString(x0 + 50, y, f"{xv:.1f}")
        c.drawString(x0 + 120, y, f"{yv:.1f}")
        y -= row_h


def generate_case_pdf(case, front, side, front_points, side_points, patient=None, doctor_name=None):
    os.makedirs("static/reports", exist_ok=True)
    pdf_path = os.path.join("static/reports", f"case_{case.id}_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4

    # ------------------------
    # Safe values
    # ------------------------
    doctor_name = doctor_name or "—"

    # patient may be None if not linked
    patient_name = getattr(patient, "name", None) or "—"
    patient_age = getattr(patient, "age", None)
    patient_gender = getattr(patient, "gender", None)

    # code can be in patient or case (depending on your DB)
    patient_code = getattr(patient, "patient_code", None) or getattr(patient, "code", None) or (case.patient_code or "—")

    # display strings
    age_str = str(patient_age) if patient_age is not None else "—"
    gender_str = patient_gender if patient_gender else "—"

    # ======================
    # PAGE 1 (HEADER + FRONT)
    # ======================

    # ---- Title ----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, H - 50, f"Orthodontic Landmark Report (Case #{case.id})")

    # ---- Case Info Block ----
    y = H - 85
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Case Information")
    y -= 14

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Doctor: {doctor_name}")
    y -= 16
    c.drawString(40, y, f"Patient Name: {patient_name}")
    y -= 16
    c.drawString(40, y, f"Patient Code: {patient_code}")
    y -= 16
    c.drawString(40, y, f"Age: {age_str}    Gender: {gender_str}")
    y -= 16
    c.drawString(40, y, f"Status: {case.status}")
    y -= 16
    c.drawString(40, y, f"Created: {case.created_at.strftime('%Y-%m-%d %H:%M')}")

    # ---- FRONT PAGE: overlay + coords ----
    front_xy = _as_xy_list(front_points)

    y_cursor = y - 30  # start after info block

    if front and front.overlay_path and os.path.exists(front.overlay_path):
        img = ImageReader(front.overlay_path)
        img_w = 260
        img_h = 260
        c.drawImage(
            img,
            40,
            y_cursor - img_h,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask='auto'
        )
        _draw_points_table(c, "Front Landmarks (X,Y)", front_xy, x0=320, y0=y_cursor)
    else:
        _draw_points_table(c, "Front Landmarks (X,Y)", front_xy, x0=40, y0=y_cursor)

    c.showPage()

    # ======================
    # PAGE 2 (SIDE)
    # ======================

    # Optional: repeat small header on page 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, H - 50, f"Case #{case.id} — Side View Results")

    c.setFont("Helvetica", 11)
    c.drawString(40, H - 70, f"Patient: {patient_name}   |   Code: {patient_code}")

    side_xy = _as_xy_list(side_points)

    y_cursor = H - 100
    if side and side.overlay_path and os.path.exists(side.overlay_path):
        img = ImageReader(side.overlay_path)
        img_w = 260
        img_h = 260
        c.drawImage(
            img,
            40,
            y_cursor - img_h,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask='auto'
        )
        _draw_points_table(c, "Side Landmarks (X,Y)", side_xy, x0=320, y0=y_cursor)
    else:
        _draw_points_table(c, "Side Landmarks (X,Y)", side_xy, x0=40, y0=y_cursor)

    c.save()
    return pdf_path


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

with app.app_context():
    db.create_all()

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

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = clean_email(request.form.get("email"))
        password = request.form.get("password", "")

        # -------------------
        # VERIFICATION (format)
        # -------------------
        if not is_valid_email(email):
            flash("Please enter a valid email.", "error")
            return redirect(url_for("login"))

        if not password:
            flash("Password is required.", "error")
            return redirect(url_for("login"))

        # -------------------
        # VALIDATION (correct credentials)
        # -------------------
        user = User.query.filter_by(email=email).first()

        # important: generic error (don’t tell them which field is wrong)
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        # success session
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
    # POST → run analysis
    # =========================
    front_file = request.files.get("front")
    side_file  = request.files.get("side")

    # Quick-add fields (top form)
    new_name   = (request.form.get("new_name") or "").strip()
    new_code   = (request.form.get("new_code") or "").strip()
    new_age    = (request.form.get("new_age") or "").strip()
    new_gender = (request.form.get("new_gender") or "").strip().upper()

    # Existing patient selection
    patient_id = (request.form.get("patient_id") or "").strip()

    # Must have images
    if not front_file or not side_file:
        flash("Please upload both front and side images.", "error")
        return redirect(url_for("new_analysis"))

    # -------------------------
    # Decide which patient to use
    # -------------------------
    patient = None

    if new_code:
        # normalize code
        new_code = new_code.upper()

        # if patient already exists for this doctor → reuse it
        patient = Patient.query.filter_by(
            patient_code=new_code,
            doctor_id=session["user_id"]
        ).first()

        if not patient:
            # require name if creating new patient
            if not new_name:
                flash("Please enter the patient name.", "error")
                return redirect(url_for("new_analysis"))

            # validate age
            age_val = None
            if new_age:
                try:
                    age_val = int(new_age)
                    if age_val < 1 or age_val > 119:
                        raise ValueError()
                except Exception:
                    flash("Age must be a valid number between 1 and 119.", "error")
                    return redirect(url_for("new_analysis"))

            # validate gender (optional)
            if new_gender and new_gender not in ["MALE", "FEMALE"]:
                flash("Gender must be MALE or FEMALE (or leave it empty).", "error")
                return redirect(url_for("new_analysis"))

            # create patient
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
        # no quick-add → must choose an existing patient
        if not patient_id:
            flash("Please select a patient (or add a new one).", "error")
            return redirect(url_for("new_analysis"))

        patient = Patient.query.filter_by(
            id=patient_id,
            doctor_id=session["user_id"]
        ).first()

        if not patient:
            flash("Selected patient not found.",
