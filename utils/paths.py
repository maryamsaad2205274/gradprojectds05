"""
Cross-platform path helpers.

Database paths are stored with forward slashes (e.g. static/uploads/file.jpg).
Legacy Windows backslashes are normalized on read via normalize_stored_path().
"""
from __future__ import annotations

import os
from typing import Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def normalize_stored_path(path: Optional[str]) -> str:
    """Normalize a path string from DB or legacy Windows storage."""
    if not path:
        return ""
    p = str(path).strip().replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    return p


def join_stored(*parts: str) -> str:
    """Build a forward-slash path suitable for database storage."""
    segments = []
    for part in parts:
        if not part:
            continue
        norm = normalize_stored_path(part)
        for seg in norm.split("/"):
            if seg and seg != ".":
                segments.append(seg)
    return "/".join(segments)


def resolve_project_path(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    """
    Resolve a stored or relative path to an absolute filesystem path.
    Handles legacy values like static/uploads\\file.jpg.
    """
    if not path:
        return None
    base = os.path.abspath(base_dir or BASE_DIR)
    norm = normalize_stored_path(path)
    if os.path.isabs(norm):
        return os.path.normpath(norm)
    parts = [p for p in norm.split("/") if p and p != "."]
    if not parts:
        return None
    return os.path.normpath(os.path.join(base, *parts))


def static_url_filename(path: Optional[str]) -> Optional[str]:
    """Path relative to static/ for url_for('static', filename=...)."""
    if not path:
        return None
    norm = normalize_stored_path(path)
    if norm.startswith("static/"):
        return norm[len("static/") :]
    return norm


def static_dir(*parts: str, base_dir: Optional[str] = None) -> str:
    """Absolute path under project static/."""
    base = os.path.abspath(base_dir or BASE_DIR)
    if parts:
        return os.path.join(base, "static", *parts)
    return os.path.join(base, "static")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_static_subdirs(*names: str, base_dir: Optional[str] = None) -> None:
    """Create static/uploads, static/results, etc."""
    for name in names:
        ensure_dir(static_dir(name, base_dir=base_dir))


def upload_abs(filename: str, base_dir: Optional[str] = None) -> str:
    return static_dir("uploads", filename, base_dir=base_dir)


def upload_rel(filename: str) -> str:
    return join_stored("static", "uploads", filename)


def results_abs(filename: Optional[str] = None, base_dir: Optional[str] = None) -> str:
    d = static_dir("results", base_dir=base_dir)
    if filename:
        return os.path.join(d, filename)
    return d


def overlay_rel(case_id: int, view_type: str) -> str:
    """Relative overlay path stored in DB (under static/ when resolved)."""
    vt = str(view_type).lower()
    return join_stored("results", f"{case_id}_{vt}_overlay.jpg")


def resolve_overlay_path(overlay_path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    """Resolve Result.overlay_path (e.g. results/case_1_side_overlay.jpg)."""
    if not overlay_path:
        return None
    norm = normalize_stored_path(overlay_path)
    if norm.startswith("static/"):
        return resolve_project_path(norm, base_dir)
    return resolve_project_path(join_stored("static", norm), base_dir)


def resolve_for_imread(path: Optional[str], base_dir: Optional[str] = None) -> Optional[str]:
    """Resolve path and verify file exists (for cv2.imread)."""
    resolved = resolve_project_path(path, base_dir)
    if resolved and os.path.isfile(resolved):
        return resolved
    return resolved
