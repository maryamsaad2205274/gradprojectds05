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


def login_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return route_function(*args, **kwargs)
    return wrapper

# Create tables once
with app.app_context():
    db.create_all()

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
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        session["user_id"] = user.id
        session["user_name"] = user.name
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not name or not email or not password:
            flash("Please fill all fields.", "error")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("login"))

        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
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
    cases = Case.query.filter_by(doctor_id=session["user_id"]).order_by(Case.created_at.desc()).all()
    return render_template("history.html", name=session.get("user_name"), cases=cases, active="cases")

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", name=session.get("user_name"), active="settings")
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

    # Ensure folders exist
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

    # Create case
    new_case = Case(
        doctor_id=session["user_id"],
        patient_code=patient_code,
        status="PENDING"
    )
    db.session.add(new_case)
    db.session.commit()

    # Save images with unique names
    front_name = f"{new_case.id}_front_{uuid.uuid4().hex}.jpg"
    side_name = f"{new_case.id}_side_{uuid.uuid4().hex}.jpg"

    front_path = os.path.join("static/uploads", front_name)
    side_path = os.path.join("static/uploads", side_name)

    front_file.save(front_path)
    side_file.save(side_path)

    # Save image records
    db.session.add(Image(case_id=new_case.id, view_type="FRONT", file_path=front_path))
    db.session.add(Image(case_id=new_case.id, view_type="SIDE", file_path=side_path))
    db.session.commit()

    # Predict landmarks (dummy for now)
    front_points = predict_landmarks(front_path)
    side_points = predict_landmarks(side_path)

    # Draw overlays
    front_overlay = os.path.join("static/results", f"{new_case.id}_front_overlay.jpg")
    side_overlay = os.path.join("static/results", f"{new_case.id}_side_overlay.jpg")

    draw_points(front_path, front_points, front_overlay)
    draw_points(side_path, side_points, side_overlay)

    # Save results
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
        side_points=side_points
    )
@app.route("/report/<int:case_id>")
@login_required
def download_report(case_id):
    case = Case.query.get_or_404(case_id)

    # Security: ensure doctor owns the case
    if case.doctor_id != session["user_id"]:
        flash("You are not allowed to download this report.", "error")
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

    # Output folder
    os.makedirs("static/reports", exist_ok=True)
    pdf_path = os.path.join("static", "reports", f"case_{case_id}_report.pdf")

    # Build absolute paths for images
    def abs_path(rel_path):
        if not rel_path:
            return None
        # rel_path stored like "static/results/xxx.jpg"
        return os.path.join(BASE_DIR, rel_path.replace("/", os.sep))

    front_img = abs_path(front.overlay_path) if front else None
    side_img = abs_path(side.overlay_path) if side else None

    # ---------- PDF generation ----------
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

        # table header
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2*cm, y, "ID")
        c.drawString(4*cm, y, "X")
        c.drawString(8*cm, y, "Y")
        y -= 0.5*cm

        c.setFont("Helvetica", 10)
        row_h = 0.45*cm
        for i, p in enumerate(points[:17], start=1):
            # new page if needed
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

    # Page 1: Summary + images
    y = header("DentalLandmark — Case Report")
    y = draw_image_block(front_img, "Front Overlay", y)
    y = draw_image_block(side_img, "Side Overlay", y)

    # Page 2+: tables
    c.showPage()
    y = header("Landmark Coordinates")
    y = draw_coords_table(front_points, "Front Landmarks (first 17)", y)
    y = draw_coords_table(side_points, "Side Landmarks (first 17)", y)

    c.save()
    # ---------- end PDF generation ----------

    return send_file(pdf_path, as_attachment=True, download_name=f"case_{case_id}_report.pdf")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
