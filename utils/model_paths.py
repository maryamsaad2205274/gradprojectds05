"""
Resolve landmark model weight files under project/model/.
"""

from __future__ import annotations

import os
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Primary side-view weights (Colab SimpleHRNet, 20 landmarks @ 384/96)
SIDE_WEIGHT_FILE = "best_hrnet_landmarks.pth"
FRONT_WEIGHT_FILE = "best_hrnet_front34.pth"


def resolve_model_file(candidates: List[str]) -> Optional[str]:
    """Return first existing path under model/ for the given filenames."""
    for name in candidates:
        path = os.path.join(MODEL_DIR, name)
        if os.path.isfile(path):
            return path
    return None


def resolve_side_model_path() -> str:
    """Colab side weights: model/best_hrnet_landmarks.pth"""
    path = os.path.join(MODEL_DIR, SIDE_WEIGHT_FILE)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Side model not found: {path}")


def resolve_front_model_path() -> str:
    path = os.path.join(MODEL_DIR, FRONT_WEIGHT_FILE)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Front model not found: {path}")


def list_model_dir() -> List[str]:
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(
        f for f in os.listdir(MODEL_DIR) if f.lower().endswith((".pth", ".pt"))
    )
