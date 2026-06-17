/**
 * predict_widget.js
 * -----------------
 * Self-contained widget for the standalone POST /predict endpoint.
 *
 * Drop the HTML snippet below into any page and include this script.
 * No dependencies.  Works in all modern browsers.
 *
 * HTML template:
 * ──────────────────────────────────────────────────────────────────────────
 *
 *  <!-- Growth stage -->
 *  <div class="predict-stage-row">
 *    <button type="button" class="predict-stage-btn active" data-stage="adult">
 *      👤 Adult
 *    </button>
 *    <button type="button" class="predict-stage-btn" data-stage="growing">
 *      🌱 Growing
 *    </button>
 *  </div>
 *
 *  <!-- Image drop zone -->
 *  <label id="predictDropZone" class="predict-drop-zone">
 *    <span id="predictDropLabel">Drop a side-view photo or click to browse</span>
 *    <input id="predictFileInput" type="file"
 *           accept="image/jpeg,image/png,image/webp,image/bmp"
 *           style="display:none">
 *  </label>
 *
 *  <!-- Run button -->
 *  <button type="button" id="predictRunBtn" disabled>Run Diagnosis</button>
 *
 *  <!-- Status line -->
 *  <p id="predictStatus"></p>
 *
 *  <!-- Result area (populated by JS) -->
 *  <div id="predictResultArea"></div>
 *
 * ──────────────────────────────────────────────────────────────────────────
 */

(function () {
  "use strict";

  // ── Config ──────────────────────────────────────────────────────────────
  var PREDICT_URL = "/predict";

  // ── State ───────────────────────────────────────────────────────────────
  var selectedFile = null;
  var selectedStage = "adult";

  // ── DOM refs (resolved lazily after DOMContentLoaded) ───────────────────
  var stageButtons, dropZone, fileInput, runBtn, statusEl, resultEl;

  function init() {
    stageButtons = document.querySelectorAll(".predict-stage-btn");
    dropZone     = document.getElementById("predictDropZone");
    fileInput    = document.getElementById("predictFileInput");
    runBtn       = document.getElementById("predictRunBtn");
    statusEl     = document.getElementById("predictStatus");
    resultEl     = document.getElementById("predictResultArea");

    if (!dropZone || !fileInput || !runBtn || !statusEl || !resultEl) return;

    // Growth-stage toggle
    stageButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        selectedStage = this.dataset.stage;
        stageButtons.forEach(function (b) { b.classList.remove("active"); });
        this.classList.add("active");
      }.bind(btn));
    });

    // File input change
    fileInput.addEventListener("change", function () {
      if (this.files && this.files[0]) {
        selectedFile = this.files[0];
        document.getElementById("predictDropLabel").textContent =
          "✓ " + selectedFile.name;
        runBtn.disabled = false;
        setStatus("", "");
      }
    });

    // Click on drop zone → open file picker
    dropZone.addEventListener("click", function () {
      fileInput.click();
    });

    // Drag-and-drop support
    dropZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", function () {
      dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", function (e) {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      var f = e.dataTransfer.files[0];
      if (f) {
        selectedFile = f;
        document.getElementById("predictDropLabel").textContent = "✓ " + f.name;
        runBtn.disabled = false;
        setStatus("", "");
      }
    });

    // Run button
    runBtn.addEventListener("click", runPredict);
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  function setStatus(type, msg) {
    statusEl.className = "predict-status" + (type ? " predict-status--" + type : "");
    statusEl.textContent = msg;
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ── Main request ─────────────────────────────────────────────────────────
  function runPredict() {
    if (!selectedFile) {
      setStatus("error", "Please select an image first.");
      return;
    }

    runBtn.disabled = true;
    setStatus("running", "⏳ Running HRNet + diagnosis…");
    resultEl.innerHTML = "";

    var fd = new FormData();
    fd.append("image", selectedFile);
    fd.append("growth_stage", selectedStage);

    fetch(PREDICT_URL, { method: "POST", body: fd })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        runBtn.disabled = false;
        if (data.success) {
          setStatus("done", "✓ Diagnosis complete");
          renderResult(data);
        } else {
          setStatus("error", "✗ " + (data.error || "Diagnosis failed"));
        }
      })
      .catch(function (err) {
        runBtn.disabled = false;
        setStatus("error", "✗ Network error: " + err.message);
      });
  }

  // ── Render result ─────────────────────────────────────────────────────────
  function renderResult(data) {
    var confBar = bar(data.confidence, "#6366f1");

    var diagHtml =
      "<div class='predict-card predict-card--highlight'>" +
        "<p class='predict-card__label'>🔍 Diagnosis</p>" +
        "<p class='predict-card__value'>" + escHtml(data.diagnosis) + "</p>" +
        "<div class='predict-conf-row'>" + confBar +
          "<span class='predict-conf-pct'>" + data.confidence.toFixed(1) + "%</span>" +
        "</div>" +
        breakdownHtml(data.diagnosis_breakdown) +
      "</div>";

    var treatHtml = "";
    if (data.treatment) {
      treatHtml =
        "<div class='predict-card'>" +
          "<p class='predict-card__label'>💊 Treatment</p>" +
          "<p class='predict-card__value'>" + escHtml(data.treatment) + "</p>" +
          "<div class='predict-conf-row'>" + bar(data.treatment_confidence, "#10b981") +
            "<span class='predict-conf-pct'>" +
              data.treatment_confidence.toFixed(1) + "%" +
            "</span>" +
          "</div>" +
          breakdownHtml(data.treatment_breakdown) +
        "</div>";
    }

    var anglesHtml =
      "<div class='predict-angles'>" +
        "<p class='predict-angles__title'>📐 Clinical angles</p>" +
        "<div class='predict-angles__grid'>" +
          angleChip("Nasiolabial",       data.angles.nasiolabial) +
          angleChip("Profile convexity", data.angles.profile_convexity) +
          angleChip("Total convexity",   data.angles.total_convexity) +
          angleChip("Mentolabial",       data.angles.mentolabial) +
        "</div>" +
      "</div>";

    resultEl.innerHTML =
      "<div class='predict-result-grid'>" + diagHtml + treatHtml + "</div>" +
      anglesHtml;
  }

  function bar(pct, color) {
    return (
      "<div class='predict-bar-track'>" +
        "<div class='predict-bar-fill' style='width:" + pct + "%;background:" + color + ";'></div>" +
      "</div>"
    );
  }

  function angleChip(name, val) {
    return (
      "<div class='predict-angle-chip'>" +
        "<span class='predict-angle-chip__name'>" + name + "</span>" +
        "<span class='predict-angle-chip__val'>" +
          (val != null ? val.toFixed(1) + "°" : "—") +
        "</span>" +
      "</div>"
    );
  }

  function breakdownHtml(items) {
    if (!items || !items.length) return "";
    return (
      "<ul class='predict-breakdown'>" +
      items.map(function (item, i) {
        var top = i === 0 ? " predict-breakdown__item--top" : "";
        return (
          "<li class='predict-breakdown__item" + top + "'>" +
            "<span class='predict-breakdown__label'>" + escHtml(item.label) + "</span>" +
            "<div class='predict-breakdown__bar-track'>" +
              "<div class='predict-breakdown__bar-fill' style='width:" +
                item.probability + "%;'></div>" +
            "</div>" +
            "<span class='predict-breakdown__pct'>" +
              Math.round(item.probability) + "%" +
            "</span>" +
          "</li>"
        );
      }).join("") +
      "</ul>"
    );
  }

  // ── Boot ─────────────────────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
