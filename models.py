"""SQLAlchemy models — single shared `db` from extensions."""

from datetime import date, datetime, time

from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db


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
    private_notes = db.Column(db.Text, nullable=True)
    private_notes_updated_at = db.Column(db.DateTime, nullable=True)

    doctor = db.relationship("User", backref=db.backref("patients", lazy=True))
    patient_auth = db.relationship("PatientAuth", backref=db.backref("patient_profile", uselist=False))

    upload_codes = db.relationship(
        "PatientUploadCode",
        back_populates="patient",
        cascade="all, delete-orphan",
        lazy=True,
    )
    cases = db.relationship(
        "Case",
        back_populates="patient",
        cascade="all, delete-orphan",
        lazy=True,
    )
    messages = db.relationship(
        "PatientMessage",
        back_populates="patient",
        cascade="all, delete-orphan",
        lazy=True,
    )
    appointments = db.relationship(
        "Appointment",
        back_populates="patient",
        cascade="all, delete-orphan",
        lazy=True,
    )


class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True)

    case_type = db.Column(db.String(20), default="INITIAL")
    status = db.Column(db.String(20), default="PENDING")
    case_date = db.Column(db.Date, nullable=True)
    follow_up_requested = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", back_populates="cases")
    doctor = db.relationship("User", backref=db.backref("cases", lazy=True))

    doctor_comment = db.Column(db.Text, nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    failure_message = db.Column(db.Text, nullable=True)

    @property
    def display_case_date(self):
        if self.case_date:
            return self.case_date
        if self.created_at:
            return self.created_at.date()
        return date.today()


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
    # JSON list of 0–100 floats — one sigmoid-scaled confidence per landmark.
    # Null for rows created before this feature was added.
    confidence_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    case = db.relationship("Case", backref=db.backref("results", lazy=True, cascade="all, delete-orphan"))


class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    case = db.relationship("Case", backref=db.backref("report", uselist=False, cascade="all, delete-orphan"))


class TreatmentSimulation(db.Model):
    """Saved treatment simulation result for a side-analysis case."""
    __tablename__ = "treatment_simulation"

    id          = db.Column(db.Integer, primary_key=True)
    case_id     = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    doctor_id   = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    # Path to the saved side-by-side comparison image
    image_path  = db.Column(db.String(300), nullable=True)
    # JSON: {"upper_lip": 5.0, "lower_lip": -3.0, ...}
    sliders_json = db.Column(db.Text, nullable=True)
    # JSON: [[x,y], ...] — 20 simulated landmark positions
    simulated_landmarks_json = db.Column(db.Text, nullable=True)
    notes       = db.Column(db.Text, nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    case = db.relationship(
        "Case",
        backref=db.backref("simulations", lazy=True, cascade="all, delete-orphan"),
    )


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
    patient = db.relationship("Patient", back_populates="upload_codes")

    @property
    def is_used(self):
        return self.used_at is not None

    @property
    def is_expired(self):
        return self.expires_at is not None and datetime.utcnow() > self.expires_at


class Appointment(db.Model):
    __tablename__ = "appointment"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=True)

    reason = db.Column(db.String(200), nullable=False)
    appointment_date = db.Column(db.Date, nullable=False)
    appointment_time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(20), default="SCHEDULED")
    source = db.Column(db.String(20), default="doctor")  # doctor | patient
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    doctor = db.relationship("User", backref=db.backref("appointments", lazy=True))
    patient = db.relationship("Patient", back_populates="appointments")
    case = db.relationship("Case", backref=db.backref("appointments", lazy=True))

    @property
    def starts_at(self):
        return datetime.combine(self.appointment_date, self.appointment_time)


class DoctorAvailability(db.Model):
    __tablename__ = "doctor_availability"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    weekday = db.Column(db.Integer, nullable=False)  # 0=Monday … 6=Sunday (Python weekday)
    slot_time = db.Column(db.Time, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    doctor = db.relationship("User", backref=db.backref("availability_slots", lazy=True))


class PatientMessage(db.Model):
    __tablename__ = "patient_message"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    question = db.Column(db.Text, nullable=False)
    read = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", back_populates="messages")
    doctor = db.relationship("User", backref=db.backref("patient_messages", lazy=True))


class OrthoCase(db.Model):
    __tablename__ = "ortho_case"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=True)
    case_id = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=True)
    image_path = db.Column(db.String(300), nullable=False)
    overlay_path = db.Column(db.String(300), nullable=True)
    landmarks_json = db.Column(db.Text, nullable=True)
    diagnosis_json = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default="PENDING")  # PENDING | DONE | FAILED
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Doctor review fields
    reviewed = db.Column(db.Boolean, default=False, nullable=False)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    doctor_final_diagnosis = db.Column(db.Text, nullable=True)
    doctor_review_notes = db.Column(db.Text, nullable=True)

    doctor = db.relationship("User", backref=db.backref("ortho_cases", lazy=True))
    patient = db.relationship("Patient", backref=db.backref("ortho_cases", lazy=True))
    case = db.relationship("Case", backref=db.backref("ortho_case", uselist=False))


class SideDiagnosis(db.Model):
    """
    XGBoost-based diagnosis from side-view angles + growth stage.

    Pipeline:
      stored side landmarks → 4 clinical angles + growth_stage
      → StandardScaler → XGBoost → label-decoded diagnosis (+ optional treatment)
    """
    __tablename__ = "side_diagnosis"

    id          = db.Column(db.Integer, primary_key=True)
    case_id     = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    doctor_id   = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # Input recorded for audit / reproducibility
    growth_stage = db.Column(db.String(20), nullable=False)  # 'adult' | 'growing'

    # The 4 computed angles (degrees, rounded to 2 dp)
    nasiolabial_angle       = db.Column(db.Float, nullable=True)
    profile_convexity_angle = db.Column(db.Float, nullable=True)
    total_convexity_angle   = db.Column(db.Float, nullable=True)
    mentolabial_angle       = db.Column(db.Float, nullable=True)

    # Diagnosis output
    diagnosis_label      = db.Column(db.String(120), nullable=True)
    diagnosis_confidence = db.Column(db.Float, nullable=True)   # 0-100 %
    # Full per-class breakdown JSON: [{"label": "...", "probability": 12.3}, ...]
    diagnosis_breakdown_json = db.Column(db.Text, nullable=True)

    # Treatment output (present only when treatment model is installed)
    treatment_label      = db.Column(db.String(120), nullable=True)
    treatment_confidence = db.Column(db.Float, nullable=True)   # 0-100 %
    treatment_breakdown_json = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    case   = db.relationship("Case", backref=db.backref(
        "side_diagnoses", lazy=True, cascade="all, delete-orphan"))
    doctor = db.relationship("User", backref=db.backref("side_diagnoses", lazy=True))


class MeasurementReview(db.Model):
    """Doctor-review state for one measurement-level ML result.

    One row per (case, doctor, measurement_key).  Stores the model outputs
    (for backend re-validation — never trust the browser) plus the doctor's
    approve/decline decision so it can be restored on refresh.
    """
    __tablename__ = "measurement_review"

    id        = db.Column(db.Integer, primary_key=True)
    case_id   = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    measurement_key   = db.Column(db.String(40), nullable=False)   # nasiolabial, ...
    measurement_label = db.Column(db.String(80), nullable=True)
    angle             = db.Column(db.Float, nullable=True)

    model_diagnosis      = db.Column(db.String(160), nullable=True)
    diagnosis_code       = db.Column(db.String(80), nullable=True)
    diagnosis_confidence = db.Column(db.Float, nullable=True)   # 0-1
    model_treatment      = db.Column(db.String(160), nullable=True)
    treatment_confidence = db.Column(db.Float, nullable=True)   # 0-1

    # pending | approved | declined
    review_status = db.Column(db.String(20), default="pending", nullable=False)
    reviewed_at   = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("case_id", "doctor_id", "measurement_key",
                            name="uq_measurement_review_case_doctor_key"),
    )

    case   = db.relationship("Case", backref=db.backref(
        "measurement_reviews", lazy=True, cascade="all, delete-orphan"))
    doctor = db.relationship("User", backref=db.backref("measurement_reviews", lazy=True))


class ProfileSimulation(db.Model):
    """A saved Gemini profile-simulation result for a reviewed case."""
    __tablename__ = "profile_simulation"

    id        = db.Column(db.Integer, primary_key=True)
    case_id   = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # JSON list of selected internal change codes
    selected_changes_json = db.Column(db.Text, nullable=True)
    strength              = db.Column(db.String(20), nullable=True)

    # Stable signature of (approved review + selected changes + strength) used
    # to avoid charging for a duplicate generation.
    review_signature = db.Column(db.String(64), nullable=True, index=True)

    # Application-relative path to the saved generated image
    image_path   = db.Column(db.String(300), nullable=True)
    gemini_model = db.Column(db.String(60), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    case   = db.relationship("Case", backref=db.backref(
        "profile_simulations", lazy=True, cascade="all, delete-orphan"))
    doctor = db.relationship("User", backref=db.backref("profile_simulations", lazy=True))


class FrontalDiagnosis(db.Model):
    """
    Six-model ML diagnosis from frontal-view landmarks.

    Pipeline:
      stored FRONT_NS landmarks (34 points) → 6 computed features
      → 6 independent models → 6 diagnosis + treatment results stored as JSON
    """
    __tablename__ = "frontal_diagnosis"

    id        = db.Column(db.Integer, primary_key=True)
    case_id   = db.Column(db.Integer, db.ForeignKey("case.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # Full 6-result dict serialised as JSON text
    results_json = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    case   = db.relationship("Case", backref=db.backref(
        "frontal_diagnoses", lazy=True, cascade="all, delete-orphan"))
    doctor = db.relationship("User", backref=db.backref("frontal_diagnoses", lazy=True))
