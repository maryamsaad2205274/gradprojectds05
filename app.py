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
    patient_code = db.Column(db.String(60), nullable=True)
    status = db.Column(db.String(20), default="PENDING")  # PENDING/COMPLETED/FAILED
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
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_code = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)  # e.g. static/reports/case_1_report.pdf
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
def generate_case_pdf(case, front, side, front_points, side_points):
    os.makedirs("static/reports", exist_ok=True)
    pdf_path = os.path.join("static", "reports", f"case_{case.id}_report.pdf")

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    def header(title):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, height - 2*cm, title)

        c.setFont("Helvetica", 10)
        y = height - 2.8*cm
        c.drawString(2*cm, y, f"Doctor: {session.get('user_name','Doctor')}")
        y -= 0.55*cm
        c.drawString(2*cm, y, f"Case ID: {case.id}")
        y -= 0.55*cm
        c.drawString(2*cm, y, f"Patient Code: {case.patient_code or '-'}")
        y -= 0.55*cm
        c.drawString(2*cm, y, f"Status: {case.status}")
        y -= 0.55*cm
        c.drawString(2*cm, y, f"Created At: {case.created_at.strftime('%Y-%m-%d %H:%M')}")
        return y - 0.8*cm

    def abs_path(rel_path):
        if not rel_path:
            return None
        return os.path.join(BASE_DIR, rel_path.replace("/", os.sep))

    def draw_image_block(img_path, title, y_start):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y_start, title)
        y = y_start - 0.6*cm

        if img_path and os.path.exists(img_path):
            max_w = width - 4*cm
            max_h = 9*cm
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            scale = min(max_w/iw, max_h/ih)
            w = iw * scale
            h = ih * scale
            c.drawImage(img, 2*cm, y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y = y - h - 0.8*cm
        else:
            c.setFont("Helvetica", 10)
            c.drawString(2*cm, y, "Image not available.")
            y -= 1.0*cm
        return y

    def draw_coords_table(points, title, y_start):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y_start, title)
        y = y_start - 0.7*cm

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2*cm, y, "ID")
        c.drawString(4*cm, y, "X")
        c.drawString(8*cm, y, "Y")
        y -= 0.5*cm

        c.setFont("Helvetica", 10)
        row_h = 0.45*cm
        for i, p in enumerate(points[:17], start=1):
            if y < 2*cm:
                c.showPage()
                y = height - 2*cm
                c.setFont("Helvetica-Bold", 12)
                c.drawString(2*cm, y, title + " (cont.)")
                y -= 0.9*cm
                c.setFont("Helvetica-Bold", 10)
                c.drawString(2*cm, y, "ID")
                c.drawString(4*cm, y, "X")
                c.drawString(8*cm, y, "Y")
                y -= 0.5*cm
                c.setFont("Helvetica", 10)

            x_val = p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else ""
            y_val = p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else ""

            c.drawString(2*cm, y, str(i))
            c.drawString(4*cm, y, str(x_val))
            c.drawString(8*cm, y, str(y_val))
            y -= row_h

        return y - 0.6*cm

    # Page 1
    y = header("DentalLandmark — Case Report")
    front_img = abs_path(front.overlay_path) if front else None
    side_img = abs_path(side.overlay_path) if side else None
    y = draw_image_block(front_img, "Front Overlay", y)
    y = draw_image_block(side_img, "Side Overlay", y)

    # Page 2
    c.showPage()
    y = header("Landmark Coordinates")
    y = draw_coords_table(front_points, "Front Landmarks (first 17)", y)
    y = draw_coords_table(side_points, "Side Landmarks (first 17)", y)

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


@app.route("/new-analysis")
@login_required
def new_analysis():
    return render_template("new_analysis.html", name=session.get("user_name"), active="new_analysis")

@app.route("/history")
@login_required
def history():
    doctor_id = session["user_id"]

    q = request.args.get("q", "").strip()
    status = request.args.get("status", "ALL").strip().upper()

    query = Case.query.filter_by(doctor_id=doctor_id)

    if status != "ALL":
        query = query.filter(Case.status == status)

    if q:
        # Search by patient_code OR by case id if q is numeric
        if q.isdigit():
            query = query.filter(Case.id == int(q))
        else:
            query = query.filter(Case.patient_code.ilike(f"%{q}%"))

    cases = query.order_by(Case.created_at.desc()).all()

    # counts for filter chips
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

@app.route("/patients", methods=["GET", "POST"])
@login_required
def patients():
    doctor_id = session["user_id"]

    # Add patient (simple form)
    if request.method == "POST":
        patient_code = request.form.get("patient_code", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()

        if not patient_code:
            flash("Patient code is required.", "error")
            return redirect(url_for("patients"))

        # Prevent duplicates per doctor
        existing = Patient.query.filter_by(doctor_id=doctor_id, patient_code=patient_code).first()
        if existing:
            flash("Patient code already exists.", "error")
            return redirect(url_for("patients"))

        p = Patient(
            doctor_id=doctor_id,
            patient_code=patient_code
        )
        # store optional fields if you add them later
        # for now we keep only patient_code in DB

        db.session.add(p)
        db.session.commit()

        flash("Patient added successfully.", "success")
        return redirect(url_for("patients"))

    # List patients
    q = request.args.get("q", "").strip()
    query = Patient.query.filter_by(doctor_id=doctor_id)
    if q:
        query = query.filter(Patient.patient_code.ilike(f"%{q}%"))
    patients_list = query.order_by(Patient.created_at.desc()).all()

    return render_template(
        "patients.html",
        name=session.get("user_name"),
        active="patients",
        patients=patients_list,
        q=q
    )


@app.route("/patients/<int:patient_id>/cases")
@login_required
def patient_cases(patient_id):
    doctor_id = session["user_id"]

    patient = Patient.query.filter_by(id=patient_id, doctor_id=doctor_id).first_or_404()
    cases = (
        Case.query
        .filter_by(doctor_id=doctor_id, patient_code=patient.patient_code)
        .order_by(Case.created_at.desc())
        .all()
    )

    return render_template(
        "patient_cases.html",
        name=session.get("user_name"),
        active="patients",
        patient=patient,
        cases=cases
    )


@app.route("/submit-analysis", methods=["POST"])
@login_required
def submit_analysis():
    front_file = request.files.get("front")
    side_file = request.files.get("side")
    patient_code = request.form.get("patient_code", "").strip()

    if not front_file or not side_file:
        flash("Please upload both front and side images.", "error")
        return redirect(url_for("new_analysis"))

    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

    # 1) Create case as PENDING immediately
    new_case = Case(
        doctor_id=session["user_id"],
        patient_code=patient_code,
        status="PENDING"
    )
    db.session.add(new_case)
    db.session.commit()

    try:
        # 2) Save images
        front_name = f"{new_case.id}_front_{uuid.uuid4().hex}.jpg"
        side_name  = f"{new_case.id}_side_{uuid.uuid4().hex}.jpg"

        front_path = os.path.join("static/uploads", front_name)
        side_path  = os.path.join("static/uploads", side_name)

        front_file.save(front_path)
        side_file.save(side_path)

        db.session.add(Image(case_id=new_case.id, view_type="FRONT", file_path=front_path))
        db.session.add(Image(case_id=new_case.id, view_type="SIDE", file_path=side_path))
        db.session.commit()

        # 3) Predict landmarks
        front_points = predict_landmarks(front_path)
        side_points  = predict_landmarks(side_path)

        # 4) Draw overlays
        front_overlay = os.path.join("static/results", f"{new_case.id}_front_overlay.jpg")
        side_overlay  = os.path.join("static/results", f"{new_case.id}_side_overlay.jpg")

        draw_points(front_path, front_points, front_overlay)
        draw_points(side_path, side_points, side_overlay)

        # 5) Save results
        db.session.add(Result(
            case_id=new_case.id,
            view_type="FRONT",
            landmarks_json=json.dumps(front_points),
            overlay_path=front_overlay
        ))
        db.session.add(Result(
            case_id=new_case.id,
            view_type="SIDE",
            landmarks_json=json.dumps(side_points),
            overlay_path=side_overlay
        ))

        new_case.status = "COMPLETED"
        db.session.commit()

        return redirect(url_for("view_result", case_id=new_case.id))

    except Exception as e:
        # If anything fails, mark FAILED
        new_case.status = "FAILED"
        db.session.commit()

        flash(f"Analysis failed: {str(e)}", "error")
        return redirect(url_for("history"))



@app.route("/result/<int:case_id>")
@login_required
def view_result(case_id):
    case = Case.query.get_or_404(case_id)

    # Security: ensure doctor owns the case
    if case.doctor_id != session["user_id"]:
        flash("You are not allowed to view this case.", "error")
        return redirect(url_for("dashboard"))

    front = Result.query.filter_by(case_id=case_id, view_type="FRONT").first()
    side = Result.query.filter_by(case_id=case_id, view_type="SIDE").first()
    # 🔹 Get original uploaded images (for landmark editing)
    front_img = Image.query.filter_by(case_id=case_id, view_type="FRONT").first()
    side_img  = Image.query.filter_by(case_id=case_id, view_type="SIDE").first()

    front_original = "/" + front_img.file_path if front_img else None
    side_original  = "/" + side_img.file_path if side_img else None

    # Parse landmarks JSON safely
    def parse_points(r):
        if not r:
            return []
        try:
            return json.loads(r.landmarks_json)
        except Exception:
            return []

    front_points = parse_points(front)
    side_points = parse_points(side)

    return render_template(
    "result.html",
    name=session.get("user_name"),
    active="cases",
    case=case,
    front_overlay="/" + front.overlay_path if front else None,
    side_overlay="/" + side.overlay_path if side else None,
    front_points=front_points,
    side_points=side_points,
    front_original=front_original,
    side_original=side_original
)


@app.route("/report/<int:case_id>")
@login_required
def download_report(case_id):
    case = Case.query.get_or_404(case_id)
    if case.doctor_id != session["user_id"]:
        flash("Not allowed.", "error")
        return redirect(url_for("dashboard"))

    front = Result.query.filter_by(case_id=case_id, view_type="FRONT").first()
    side = Result.query.filter_by(case_id=case_id, view_type="SIDE").first()

    def parse_points(r):
        if not r:
            return []
        try:
            return json.loads(r.landmarks_json)
        except Exception:
            return []

    front_points = parse_points(front)
    side_points = parse_points(side)

    # Generate pdf file
    pdf_path = generate_case_pdf(case, front, side, front_points, side_points)

    # Save/Update report record
    existing = Report.query.filter_by(case_id=case_id).first()
    if existing:
        existing.file_path = pdf_path
        existing.created_at = datetime.utcnow()
    else:
        db.session.add(Report(case_id=case_id, file_path=pdf_path))
    db.session.commit()

    return send_file(pdf_path, as_attachment=True, download_name=f"case_{case_id}_report.pdf")

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

    view_type = request.form.get("view_type", "").strip().upper()  # FRONT or SIDE
    points_json = request.form.get("points_json", "").strip()

    if view_type not in ["FRONT", "SIDE"]:
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
    new_overlay = os.path.join("static/results", f"{case_id}_{view_type.lower()}_overlay.jpg")

    draw_points(img_row.file_path, points, new_overlay)
    res.overlay_path = new_overlay

    db.session.commit()
    flash(f"{view_type} landmarks updated.", "success")
    return redirect(url_for("view_result", case_id=case_id))

@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


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



if __name__ == "__main__":
    app.run(debug=True)
