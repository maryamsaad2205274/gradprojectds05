(function () {
  function distance(a, b) {
    return Math.hypot(b[0] - a[0], b[1] - a[1]);
  }

  function fmtPx(value) {
    return Number.isFinite(value) ? `${value.toFixed(1)} px` : "-";
  }

  function fmtMm(value) {
    return Number.isFinite(value) ? `${value.toFixed(2)} mm` : "-";
  }

  function fmtDeg(value) {
    return Number.isFinite(value) ? `${value.toFixed(2)}°` : "-";
  }

  const DEFAULT_TREATMENT_REVIEW =
    "This measurement should be reviewed by the orthodontist together with the patient's clinical records.";

  const TREATMENT_MEASUREMENT_KEYS = new Set([
    "nasiolabial",
    "profile_convexity",
    "total_facial_convexity",
    "mentolabial",
    "mandibular_width",
    "interpupillary_line",
    "rule_of_fifths",
    "facial_midline",
  ]);

  const DRAW = {
    landmarkCyan: "#22d3ee",
    landmarkYellow: "#facc15",
    landmarkSelected: "#ea580c",
    lineMeasure: "#06b6d4",
    lineHighlight: "#0284c7",
    arcAngle: "#f97316",
    calibrate: "#dc2626",
    manualMeasure: "#2563eb",
    landmarkRadius: 5,
    selectedRadius: 8,
    lineWidth: 3.5,
    lineWidthBold: 5,
    arcRadius: 36,
  };

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtRatio(value) {
    return Number.isFinite(value) ? value.toFixed(3) : "-";
  }

  function toPoints(rawPoints) {
    if (!Array.isArray(rawPoints)) return [];
    return rawPoints.map((p) => {
      if (Array.isArray(p)) return [Number(p[0]), Number(p[1])];
      return [Number(p.x), Number(p.y)];
    });
  }

  /** Angle at vertex B between segments BA and BC, in degrees */
  function angleAt(vertex, a, c) {
    const v1 = [a[0] - vertex[0], a[1] - vertex[1]];
    const v2 = [c[0] - vertex[0], c[1] - vertex[1]];
    const m1 = Math.hypot(v1[0], v1[1]);
    const m2 = Math.hypot(v2[0], v2[1]);
    if (!m1 || !m2) return NaN;
    let cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2);
    cos = Math.max(-1, Math.min(1, cos));
    return (Math.acos(cos) * 180) / Math.PI;
  }

  class MeasurementViewer {
    constructor(config) {
      this.root = document.getElementById(config.rootId);
      if (!this.root) return;

      this.image = this.root.querySelector("[data-measure-image]");
      this.canvas = this.root.querySelector("[data-measure-canvas]");
      this.card = this.root.querySelector("[data-measure-card]");
      this.realDistanceInput = this.root.querySelector("[data-real-distance]");
      this.scaleText = this.root.querySelector("[data-scale-text]");
      this.statusText = this.root.querySelector("[data-calibration-status]");
      this.startCalibrationBtn = this.root.querySelector("[data-calibrate-button]");
      this.saveCalibrationBtn = this.root.querySelector("[data-save-calibration]");
      this.measureBtn = this.root.querySelector("[data-measure-button]");
      this.resetCalibrationBtn = this.root.querySelector("[data-reset-calibration]");
      this.resetMeasurementBtn = this.root.querySelector("[data-reset-measurement]");
      this.selectEl = this.root.querySelector("[data-measure-select]");
      this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
      this.points = toPoints(config.points);
      this.measurements = Array.isArray(config.measurements) ? config.measurements : [];
      this.measurementResults = config.measurementResults || null;
      this.treatmentAdvice =
        config.treatmentAdvice && typeof config.treatmentAdvice === "object"
          ? config.treatmentAdvice
          : {};

      this.calibrationPoints = [];
      this.measurementPoints = [];
      this.mmPerPixel = null;

      // idle | calibration | measurement
      this.mode = "idle";

      if (!this.image || !this.canvas || !this.card || !this.ctx) return;
      this.bindEvents();
      this.initMeasurementSelector();
      this.initCanvas();
      this.setStatus('Click "Start Calibration"');
      this.updateScaleText();
      this.updateResultCard();

      if (config.viewerKey && window.measurementViewers) {
        window.measurementViewers[config.viewerKey] = this;
      }
    }

    selectMeasurement(key) {
      if (!this.selectEl) return;
      const v = key == null || key === "" ? "" : String(key);
      if (v && !this.measurements.some((m) => m.key === v)) return;
      this.selectEl.value = v;
      this.mode = "idle";
      this.updateResultCard();
      this.render();
    }

    bindEvents() {
      if (this.startCalibrationBtn) {
        this.startCalibrationBtn.addEventListener("click", () => {
          this.mode = "calibration";
          this.calibrationPoints = [];
          this.setStatus("Click first reference point");
          this.render();
        });
      }

      if (this.saveCalibrationBtn) {
        this.saveCalibrationBtn.addEventListener("click", () => this.saveCalibration());
      }

      if (this.measureBtn) {
        this.measureBtn.addEventListener("click", () => {
          if (!Number.isFinite(this.mmPerPixel)) {
            this.setStatus("Save calibration first");
            return;
          }
          this.mode = "measurement";
          this.measurementPoints = [];
          this.setStatus("Click first measurement point");
          this.render();
          this.updateResultCard();
        });
      }

      if (this.resetCalibrationBtn) {
        this.resetCalibrationBtn.addEventListener("click", () => {
          this.calibrationPoints = [];
          this.mmPerPixel = null;
          this.mode = "idle";
          this.updateScaleText();
          this.setStatus('Click "Start Calibration"');
          this.render();
          this.updateResultCard();
        });
      }

      if (this.resetMeasurementBtn) {
        this.resetMeasurementBtn.addEventListener("click", () => {
          this.measurementPoints = [];
          this.mode = "idle";
          this.setStatus(Number.isFinite(this.mmPerPixel) ? 'Calibration saved. Click "Measure Distance"' : 'Click "Start Calibration"');
          this.render();
          this.updateResultCard();
        });
      }

      this.canvas.addEventListener("click", (event) => this.handleCanvasClick(event));
      window.addEventListener("resize", () => this.render());
      if (this.selectEl) {
        this.selectEl.addEventListener("change", () => {
          this.updateResultCard();
          this.render();
        });
      }
    }

    initMeasurementSelector() {
      if (!this.selectEl || this.measurements.length === 0) return;
      this.selectEl.innerHTML = "";
      const noneOpt = document.createElement("option");
      noneOpt.value = "";
      noneOpt.textContent = "None";
      noneOpt.selected = true;
      this.selectEl.appendChild(noneOpt);
      this.measurements.forEach((measurement) => {
        const option = document.createElement("option");
        option.value = measurement.key;
        option.textContent = measurement.name || measurement.key;
        this.selectEl.appendChild(option);
      });
    }

    initCanvas() {
      if (this.image.complete) {
        this.render();
      } else {
        this.image.addEventListener("load", () => this.render());
      }
    }

    handleCanvasClick(event) {
      if (this.mode !== "calibration" && this.mode !== "measurement") return;

      const point = this.eventToImagePoint(event);
      if (this.mode === "calibration") {
        if (this.calibrationPoints.length === 0) {
          this.calibrationPoints = [point];
          this.setStatus("Click second reference point");
        } else if (this.calibrationPoints.length === 1) {
          this.calibrationPoints = [this.calibrationPoints[0], point];
          this.setStatus("Enter real distance in mm");
        } else {
          this.calibrationPoints = [point];
          this.setStatus("Click second reference point");
        }
      } else if (this.mode === "measurement") {
        if (!Number.isFinite(this.mmPerPixel)) {
          this.setStatus("Save calibration first");
          return;
        }
        if (this.measurementPoints.length === 0) {
          this.measurementPoints = [point];
          this.setStatus("Click second measurement point");
        } else if (this.measurementPoints.length === 1) {
          this.measurementPoints = [this.measurementPoints[0], point];
          this.setStatus("Measurement ready");
        } else {
          this.measurementPoints = [point];
          this.setStatus("Click second measurement point");
        }
      }

      this.render();
      this.updateResultCard();
    }

    saveCalibration() {
      if (this.calibrationPoints.length !== 2) {
        this.setStatus("Click first reference point");
        return;
      }

      const realDistanceMm = Number(this.realDistanceInput ? this.realDistanceInput.value : NaN);
      if (!Number.isFinite(realDistanceMm) || realDistanceMm <= 0) {
        this.setStatus("Enter real distance in mm");
        return;
      }

      const referencePixelDistance = distance(this.calibrationPoints[0], this.calibrationPoints[1]);
      if (!Number.isFinite(referencePixelDistance) || referencePixelDistance <= 0) {
        this.setStatus("Reference points must be different");
        return;
      }

      this.mmPerPixel = realDistanceMm / referencePixelDistance;
      this.mode = "idle";
      this.updateScaleText();
      this.setStatus("Calibration saved");
      this.updateResultCard();
      this.render();
    }

    eventToImagePoint(event) {
      const rect = this.image.getBoundingClientRect();
      const displayX = event.clientX - rect.left;
      const displayY = event.clientY - rect.top;
      return [
        displayX * this.image.naturalWidth / rect.width,
        displayY * this.image.naturalHeight / rect.height,
      ];
    }

    imageToCanvas(point) {
      return [
        point[0] * this.canvas.width / this.image.naturalWidth,
        point[1] * this.canvas.height / this.image.naturalHeight,
      ];
    }

    resizeCanvas() {
      const rect = this.image.getBoundingClientRect();
      this.canvas.width = Math.round(rect.width);
      this.canvas.height = Math.round(rect.height);
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
    }

    drawPoint(point, color, label, radius) {
      const [x, y] = this.imageToCanvas(point);
      const r = radius || DRAW.landmarkRadius;
      this.ctx.beginPath();
      this.ctx.arc(x, y, r, 0, Math.PI * 2);
      this.ctx.fillStyle = color;
      this.ctx.fill();
      this.ctx.strokeStyle = "rgba(15, 23, 42, 0.35)";
      this.ctx.lineWidth = 1;
      this.ctx.stroke();

      if (label) {
        this.drawCanvasLabel(label, x + 6, y - 6, { fontSize: 10 });
      }
    }

    drawLine(a, b, color, lineWidth) {
      const [ax, ay] = this.imageToCanvas(a);
      const [bx, by] = this.imageToCanvas(b);
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = lineWidth || DRAW.lineWidth;
      this.ctx.lineCap = "round";
      this.ctx.beginPath();
      this.ctx.moveTo(ax, ay);
      this.ctx.lineTo(bx, by);
      this.ctx.stroke();
    }

    drawPolyline(points, color, lineWidth) {
      if (!Array.isArray(points) || points.length < 2) return;
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = lineWidth || DRAW.lineWidth;
      this.ctx.lineCap = "round";
      this.ctx.beginPath();
      points.forEach((point, idx) => {
        const [x, y] = this.imageToCanvas(point);
        if (idx === 0) this.ctx.moveTo(x, y);
        else this.ctx.lineTo(x, y);
      });
      this.ctx.stroke();
    }

    drawCanvasLabel(text, x, y, opts) {
      const fontSize = (opts && opts.fontSize) || 12;
      const font = `600 ${fontSize}px system-ui, Segoe UI, Arial, sans-serif`;
      this.ctx.font = font;
      const metrics = this.ctx.measureText(text);
      const padX = 6;
      const padY = 4;
      const boxW = metrics.width + padX * 2;
      const boxH = fontSize + padY * 2;
      const bx = x;
      const by = y - boxH;

      this.ctx.fillStyle = "rgba(15, 23, 42, 0.78)";
      this.ctx.beginPath();
      const r = 4;
      this.ctx.moveTo(bx + r, by);
      this.ctx.lineTo(bx + boxW - r, by);
      this.ctx.quadraticCurveTo(bx + boxW, by, bx + boxW, by + r);
      this.ctx.lineTo(bx + boxW, by + boxH - r);
      this.ctx.quadraticCurveTo(bx + boxW, by + boxH, bx + boxW - r, by + boxH);
      this.ctx.lineTo(bx + r, by + boxH);
      this.ctx.quadraticCurveTo(bx, by + boxH, bx, by + boxH - r);
      this.ctx.lineTo(bx, by + r);
      this.ctx.quadraticCurveTo(bx, by, bx + r, by);
      this.ctx.closePath();
      this.ctx.fill();

      this.ctx.fillStyle = "#ffffff";
      this.ctx.shadowColor = "rgba(0,0,0,0.45)";
      this.ctx.shadowBlur = 3;
      this.ctx.fillText(text, bx + padX, by + fontSize + padY - 2);
      this.ctx.shadowBlur = 0;
    }

    drawAngleArc(vertex, a, c, color) {
      const [vx, vy] = this.imageToCanvas(vertex);
      const [ax, ay] = this.imageToCanvas(a);
      const [cx, cy] = this.imageToCanvas(c);
      const angA = Math.atan2(ay - vy, ax - vx);
      const angC = Math.atan2(cy - vy, cx - vx);
      let start = angA;
      let end = angC;
      let sweep = end - start;
      while (sweep <= -Math.PI) sweep += Math.PI * 2;
      while (sweep > Math.PI) sweep -= Math.PI * 2;
      if (sweep < 0) {
        const tmp = start;
        start = end;
        end = tmp;
        sweep = -sweep;
      }

      const radius = DRAW.arcRadius;
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = 3;
      this.ctx.beginPath();
      this.ctx.arc(vx, vy, radius, start, end, false);
      this.ctx.stroke();

      const mid = start + sweep / 2;
      const labelX = vx + Math.cos(mid) * (radius + 14);
      const labelY = vy + Math.sin(mid) * (radius + 14);
      const deg = angleAt(vertex, a, c);
      if (Number.isFinite(deg)) {
        this.drawCanvasLabel(fmtDeg(deg), labelX, labelY);
      }
    }

    getPointById(id1based) {
      return this.points[id1based - 1];
    }

    drawLandmarkDot(point, index1based, isSelected) {
      const color = isSelected
        ? DRAW.landmarkSelected
        : index1based % 2 === 0
          ? DRAW.landmarkYellow
          : DRAW.landmarkCyan;
      const radius = isSelected ? DRAW.selectedRadius : DRAW.landmarkRadius;
      this.drawPoint(point, color, null, radius);
    }

    getSelectedPointIds(selected) {
      const ids = new Set();
      if (!selected) return ids;
      if (selected.kind === "polyline" && Array.isArray(selected.polyline)) {
        selected.polyline.forEach((id) => ids.add(id));
      }
      if (selected.kind === "angle" && Array.isArray(selected.anglePoints)) {
        selected.anglePoints.forEach((id) => ids.add(id));
      }
      if (selected.kind === "guides" && selected.key === "rule_of_fifths") {
        const h = (this.measurementResults && this.measurementResults.horizontal) || {};
        const fw = h.facial_width_bizygomatic_px || {};
        const pair = Array.isArray(fw.chosen_pair) ? fw.chosen_pair : [1, 28];
        pair.forEach((id) => ids.add(id));
      }
      const segments = Array.isArray(selected.segments) ? selected.segments : [];
      segments.forEach((segment) => {
        if (Array.isArray(segment)) {
          segment.forEach((id) => ids.add(id));
        }
      });
      return ids;
    }

    drawAllLandmarks(selectedIds) {
      this.points.forEach((point, idx) => {
        const id = idx + 1;
        this.drawLandmarkDot(point, id, selectedIds.has(id));
      });
    }

    getSelectedMeasurement() {
      if (!this.selectEl || this.measurements.length === 0) return null;
      const v = (this.selectEl.value || "").trim();
      if (!v) return null;
      return this.measurements.find((m) => m.key === v) || null;
    }

    drawSelectedMeasurement() {
      const selected = this.getSelectedMeasurement();
      if (!selected) return;

      if (selected.kind === "guides" && selected.key === "rule_of_fifths") {
        this.drawRuleOfFifthsGuides();
        return;
      }

      const selectedIds = this.getSelectedPointIds(selected);
      this.drawAllLandmarks(selectedIds);

      const lineColor = DRAW.lineMeasure;
      const bold = DRAW.lineWidthBold;

      if (selected.kind === "polyline" && Array.isArray(selected.polyline)) {
        const points = selected.polyline.map((id) => this.getPointById(id)).filter(Boolean);
        this.drawPolyline(points, lineColor, bold);
        selected.polyline.forEach((id) => {
          const p = this.getPointById(id);
          if (p) this.drawLandmarkDot(p, id, true);
        });
        return;
      }

      if (selected.kind === "angle" && Array.isArray(selected.anglePoints) && selected.anglePoints.length === 3) {
        const [idA, idV, idC] = selected.anglePoints;
        const pa = this.getPointById(idA);
        const pv = this.getPointById(idV);
        const pc = this.getPointById(idC);
        if (pa && pv && pc) {
          this.drawLine(pa, pv, lineColor, bold);
          this.drawLine(pv, pc, lineColor, bold);
          this.drawAngleArc(pv, pa, pc, DRAW.arcAngle);
          [idA, idV, idC].forEach((id) => {
            const p = this.getPointById(id);
            if (p) this.drawLandmarkDot(p, id, true);
          });
          const labels = selected.shortLabels || ["A", "B", "C"];
          const pts = [pa, pv, pc];
          [idA, idV, idC].forEach((id, i) => {
            const p = pts[i];
            const [x, y] = this.imageToCanvas(p);
            this.drawCanvasLabel(labels[i] || String(id), x + 8, y - 8, { fontSize: 11 });
          });
        }
        return;
      }

      const segments = Array.isArray(selected.segments) ? selected.segments : [];
      const shortLabels = selected.shortLabels || [];
      segments.forEach((segment, segIdx) => {
        const a = this.getPointById(segment[0]);
        const b = this.getPointById(segment[1]);
        if (!a || !b) return;
        this.drawLine(a, b, lineColor, bold);
        this.drawLandmarkDot(a, segment[0], true);
        this.drawLandmarkDot(b, segment[1], true);
        const label = shortLabels[segIdx];
        if (label) {
          const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
          const [x, y] = this.imageToCanvas(mid);
          this.drawCanvasLabel(label, x, y - 10);
        }
      });
    }

    drawRuleOfFifthsGuides() {
      const h = (this.measurementResults && this.measurementResults.horizontal) || {};
      const fw = h.facial_width_bizygomatic_px || {};
      const pair = Array.isArray(fw.chosen_pair) ? fw.chosen_pair : [1, 28];
      const leftPt = this.getPointById(pair[0]);
      const rightPt = this.getPointById(pair[1]);
      if (!leftPt || !rightPt) return;

      const idsForY = [6, 10, 11, 16, 18, 21, 24, 30, 34];
      const yVals = idsForY
        .map((id) => this.getPointById(id))
        .filter(Boolean)
        .map((p) => p[1]);
      if (yVals.length === 0) return;

      const yTop = Math.min(...yVals);
      const yBottom = Math.max(...yVals);
      const xMin = Math.min(leftPt[0], rightPt[0]);
      const xMax = Math.max(leftPt[0], rightPt[0]);
      const width = xMax - xMin;

      const selectedIds = this.getSelectedPointIds({ key: "rule_of_fifths", kind: "guides" });
      this.drawAllLandmarks(selectedIds);

      for (let i = 0; i <= 5; i += 1) {
        const x = xMin + (width * i / 5.0);
        const a = [x, yTop];
        const b = [x, yBottom];
        const strong = i === 0 || i === 5;
        this.drawLine(
          a,
          b,
          strong ? "rgba(6, 182, 212, 0.9)" : "rgba(6, 182, 212, 0.45)",
          strong ? DRAW.lineWidthBold : DRAW.lineWidth
        );
      }
      pair.forEach((id) => {
        const p = this.getPointById(id);
        if (p) this.drawLandmarkDot(p, id, true);
      });
      const [lx, ly] = this.imageToCanvas(leftPt);
      this.drawCanvasLabel("Rule of fifths", lx, ly - 12);
    }

    drawCalibration() {
      if (this.calibrationPoints.length === 0) return;
      if (this.calibrationPoints.length === 1) {
        this.drawPoint(this.calibrationPoints[0], "#dc2626", "A");
        return;
      }

      this.drawLine(this.calibrationPoints[0], this.calibrationPoints[1], "#dc2626");
      this.drawPoint(this.calibrationPoints[0], "#dc2626", "A");
      this.drawPoint(this.calibrationPoints[1], "#dc2626", "B");
    }

    drawMeasurement() {
      if (this.measurementPoints.length === 0) return;
      if (this.measurementPoints.length === 1) {
        this.drawPoint(this.measurementPoints[0], "#2563eb", "A");
        return;
      }

      this.drawLine(this.measurementPoints[0], this.measurementPoints[1], "#2563eb");
      this.drawPoint(this.measurementPoints[0], "#2563eb", "A");
      this.drawPoint(this.measurementPoints[1], "#2563eb", "B");
    }

    updateScaleText() {
      if (!this.scaleText) return;
      if (!Number.isFinite(this.mmPerPixel)) {
        this.scaleText.textContent = "No calibration yet";
        return;
      }
      this.scaleText.textContent = "Calibrated";
    }

    getTreatmentRecommendation(key) {
      if (!key || !TREATMENT_MEASUREMENT_KEYS.has(key)) return null;
      const text = this.treatmentAdvice && this.treatmentAdvice[key];
      if (text && String(text).trim()) return String(text).trim();
      return DEFAULT_TREATMENT_REVIEW;
    }

    updateResultCard() {
      const selected = this.getSelectedMeasurement();
      const selectedDetails = selected ? this.getSelectedMeasurementDetails(selected) : null;
      const measurementPx = this.measurementPoints.length === 2
        ? distance(this.measurementPoints[0], this.measurementPoints[1])
        : NaN;
      const measurementMm = Number.isFinite(measurementPx) && Number.isFinite(this.mmPerPixel)
        ? measurementPx * this.mmPerPixel
        : NaN;

      const selectedBlock = selectedDetails
        ? `
          <div class="measurement-card__eyebrow">Selected measurement</div>
          <div class="measurement-card__title">${escapeHtml(selectedDetails.title)}</div>
          <div class="measurement-card__value">${escapeHtml(selectedDetails.body)}</div>
        `
        : "";

      const treatmentText = selected ? this.getTreatmentRecommendation(selected.key) : null;
      const treatmentBlock = treatmentText
        ? `
          <div class="measurement-card__treatment">
            <div class="measurement-card__treatment-title">Treatment recommendation</div>
            <p class="measurement-card__treatment-text">${escapeHtml(treatmentText)}</p>
          </div>
        `
        : "";

      this.card.innerHTML = `
        ${selectedBlock}
        ${treatmentBlock}
        <div class="measurement-card__value measurement-card__value--cal-only">Millimeter distance: ${fmtMm(measurementMm)}</div>
      `;
    }

    setStatus(message) {
      if (this.statusText) this.statusText.textContent = message;
    }

    getSelectedMeasurementDetails(selected) {
      if (!selected) return null;
      const v = (this.measurementResults && this.measurementResults.vertical) || {};
      const h = (this.measurementResults && this.measurementResults.horizontal) || {};

      if (selected.kind === "angle" && Array.isArray(selected.anglePoints) && selected.anglePoints.length === 3) {
        const [idA, idV, idC] = selected.anglePoints;
        const pa = this.getPointById(idA);
        const pv = this.getPointById(idV);
        const pc = this.getPointById(idC);
        if (pa && pv && pc) {
          const deg = angleAt(pv, pa, pc);
          return { title: selected.name || "Angle", body: `${fmtDeg(deg)}` };
        }
      }

      if (selected.key === "middle_third") {
        return { title: "Middle Third", body: `${fmtPx(v.middle_third_px)}` };
      }
      if (selected.key === "lower_third") {
        return { title: "Lower Third", body: `${fmtPx(v.lower_third_px)}` };
      }
      if (selected.key === "upper_to_lower_lip_ratio") {
        return {
          title: "Upper lip / Lower lip ratio",
          body: `${fmtRatio(v.upper_to_lower_lip_ratio)} (upper ${fmtPx(v.upper_lip_height_px)}, lower ${fmtPx(v.lower_lip_height_px)})`,
        };
      }
      if (selected.key === "lip_length_at_rest") {
        return { title: "Lip length at rest", body: `${fmtPx(v.lip_length_at_rest_px)}` };
      }
      if (selected.key === "interlabial_gap") {
        const gap = v.interlabial_gap_px && v.interlabial_gap_px.value;
        return { title: "Interlabial gap", body: `${fmtPx(gap)} (approximate)` };
      }
      if (selected.key === "interpupillary_line") {
        const a = v.interpupillary_line_alignment || {};
        return { title: "Interpupillary line", body: `Angle ${fmtDeg(a.angle_deg)}, length ${fmtPx(a.length_px)}` };
      }
      if (selected.key === "commissure_line") {
        const a = v.commissure_line_alignment || {};
        return { title: "Commissure line", body: `Angle ${fmtDeg(a.angle_deg)}, length ${fmtPx(a.length_px)}` };
      }
      if (selected.key === "facial_midline") {
        const m = h.facial_midline || {};
        return { title: "Facial midline", body: `RMS deviation ${fmtPx(m.rms_deviation_px)}` };
      }
      if (selected.key === "rule_of_fifths") {
        const r = h.rule_of_fifths || {};
        return { title: "Rule of fifths", body: `Ideal fifth width ${fmtPx(r.fifth_px)}` };
      }
      if (selected.key === "facial_width") {
        const fw = h.facial_width_bizygomatic_px || {};
        return { title: "Facial width", body: `${fmtPx(fw.value)}` };
      }
      if (selected.key === "mandibular_width") {
        return { title: "Mandibular width", body: `${fmtPx(h.mandibular_width_bigonial_px)}` };
      }
      if (selected.key === "bizygomatic_to_bigonial_ratio") {
        return { title: "Bizygomatic / Bigonial ratio", body: `${fmtRatio(h.bizygomatic_to_bigonial_ratio)}` };
      }
      if (selected.key === "facial_index") {
        return { title: "Facial index", body: `${fmtRatio(h.facial_index_height_to_width)} (height ${fmtPx(h.facial_height_px)}, width ${fmtPx((h.facial_width_bizygomatic_px || {}).value)})` };
      }
      if (selected.key === "middle_lower_ratio") {
        return {
          title: "Middle / lower third ratio",
          body: `${fmtRatio(v.middle_to_lower_third_ratio)} (middle ${fmtPx(v.middle_third_px)}, lower ${fmtPx(v.lower_third_px)})`,
        };
      }
      if (selected.key === "interpupillary_parallel") {
        const par = v.interpupillary_vs_commissure_parallel || {};
        return {
          title: "Interpupillary vs commissure",
          body: `Difference ${fmtDeg(par.delta_deg)} (threshold ${fmtDeg(par.threshold_deg)})`,
        };
      }

      return { title: selected.name || "Measurement", body: "Select a measurement card or dropdown option to inspect this metric." };
    }

    render() {
      if (!this.image.naturalWidth || !this.image.naturalHeight) return;
      this.resizeCanvas();
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      const selected = this.getSelectedMeasurement();
      if (selected) {
        this.drawSelectedMeasurement();
      }
      this.drawCalibration();
      this.drawMeasurement();
    }
  }

  window.MeasurementViewer = MeasurementViewer;
}());
