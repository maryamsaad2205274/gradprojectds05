(function () {
  "use strict";

  var sidebar = document.querySelector(".sidebar");
  var toggle = document.getElementById("daSidebarToggle");
  if (toggle && sidebar) {
    toggle.addEventListener("click", function () {
      sidebar.classList.toggle("is-open");
    });
  }

  document.querySelectorAll(".photo-drop").forEach(function (zone) {
    var input = zone.querySelector('input[type="file"]');
    var preview = zone.querySelector(".photo-drop__preview");
    if (!input) return;

    function showPreview(file) {
      if (!file || !file.type.startsWith("image/")) return;
      var url = URL.createObjectURL(file);
      var img = preview || document.createElement("img");
      img.className = "photo-drop__preview";
      img.src = url;
      if (!preview) zone.appendChild(img);
      zone.classList.add("has-image");
    }

    input.addEventListener("change", function () {
      if (input.files && input.files[0]) showPreview(input.files[0]);
    });

    zone.addEventListener("dragover", function (e) {
      e.preventDefault();
      zone.classList.add("is-dragover");
    });
    zone.addEventListener("dragleave", function () {
      zone.classList.remove("is-dragover");
    });
    zone.addEventListener("drop", function (e) {
      e.preventDefault();
      zone.classList.remove("is-dragover");
      var file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file && input) {
        var dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        showPreview(file);
      }
    });
  });

  var addBtn = document.getElementById("openAddPatientModal");
  var modal = document.getElementById("addPatientModal");
  var closeBtn = document.getElementById("closeAddPatientModal");
  if (addBtn && modal) {
    addBtn.addEventListener("click", function () {
      modal.classList.add("is-open");
    });
  }
  if (closeBtn && modal) {
    closeBtn.addEventListener("click", function () {
      modal.classList.remove("is-open");
    });
  }
  if (modal) {
    modal.addEventListener("click", function (e) {
      if (e.target === modal) modal.classList.remove("is-open");
    });
  }
})();
