/**
 * Tab switching, measurement card clicks, and canvas viewer integration.
 */
(function () {
  function switchToTab(tabName) {
    document.querySelectorAll(".lov-tab").forEach(function (t) {
      t.classList.toggle("is-active", t.dataset.tab === tabName);
    });
    document.querySelectorAll(".tab-panel").forEach(function (p) {
      p.classList.remove("is-active");
    });
    const panel = document.getElementById("tab-panel-" + tabName);
    if (panel) panel.classList.add("is-active");
    window.dispatchEvent(new Event("resize"));
  }

  window.switchAnalysisTab = switchToTab;

  document.querySelectorAll(".lov-tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      const name = tab.dataset.tab;
      if (name) switchToTab(name);
    });
  });

  function scrollToViewer(viewKey) {
    const id = viewKey === "side" ? "sideMeasurementViewer" : "frontNsMeasurementViewer";
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function applyMeasurement(view, key) {
    switchToTab(view === "side" ? "side" : "front");
    const viewers = window.measurementViewers || {};
    const viewer = viewers[view];
    if (viewer && typeof viewer.selectMeasurement === "function") {
      viewer.selectMeasurement(key || "");
    }
    scrollToViewer(view);
    // Brief flash so the doctor sees which panel was activated
    const id = view === "side" ? "sideMeasurementViewer" : "frontNsMeasurementViewer";
    const el = document.getElementById(id);
    if (el) {
      el.classList.remove("meas-flash-target");
      void el.offsetWidth; // force reflow to restart animation
      el.classList.add("meas-flash-target");
    }
  }

  document.querySelectorAll(".meas-metric-card[data-vis-key]").forEach(function (card) {
    card.addEventListener("click", function () {
      const view = card.dataset.view;
      const key = card.dataset.visKey;
      if (!view || key === undefined) return;

      document.querySelectorAll(".meas-metric-card.is-selected").forEach(function (c) {
        c.classList.remove("is-selected");
      });
      card.classList.add("is-selected");

      applyMeasurement(view, key);
    });
  });

})();

/* ── Landmark confidence accordion ──────────────────────────────────────────
   toggleConf('side') / toggleConf('front')
   Collapses / expands the confidence grid under each analysis panel.
   Called inline from the panel title div.
   ─────────────────────────────────────────────────────────────────────────── */
function toggleConf(view) {
  var prefix = (view === "side") ? "side" : "front";
  var section = document.getElementById(prefix + "ConfSection");
  if (!section) return;
  section.classList.toggle("is-open");
  var btn = document.getElementById(prefix + "ConfToggle");
  if (btn) btn.setAttribute("aria-expanded", section.classList.contains("is-open").toString());
}
