import os
import json
import cv2
import numpy as np

from utils.paths import (
    BASE_DIR,
    join_stored,
    resolve_project_path,
    static_dir,
    ensure_dir,
)

MEASUREMENTS_DIR = static_dir("measurements")
ensure_dir(MEASUREMENTS_DIR)


def normalize_points(points):
    """
    Converts landmarks into Nx2 float array.
    Supports:
    - [[x, y], [x, y], ...]
    - [{"x": x, "y": y}, {"x": x, "y": y}, ...]
    """
    normalized = []

    for p in points:
        if isinstance(p, dict):
            normalized.append([float(p["x"]), float(p["y"])])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            normalized.append([float(p[0]), float(p[1])])
        else:
            raise ValueError(f"Unsupported landmark format: {p}")

    return np.array(normalized, dtype=np.float32)

def angle_ABC(A, B, C, eps=1e-9):
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    C = np.array(C, dtype=np.float32)

    BA = A - B
    BC = C - B

    cosang = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def draw_angle_image(image_path, pts_orig, idxs_1based, title, output_filename):
    abs_image_path = resolve_project_path(image_path)
    if not abs_image_path or not os.path.isfile(abs_image_path):
        raise FileNotFoundError(f"Image not found: {image_path} (resolved: {abs_image_path})")
    img = cv2.imread(abs_image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {abs_image_path}")

    A_i, B_i, C_i = idxs_1based
    A = pts_orig[A_i - 1]
    B = pts_orig[B_i - 1]
    C = pts_orig[C_i - 1]

    ang = angle_ABC(A, B, C)

    # draw all landmarks
    for i, (x, y) in enumerate(pts_orig, start=1):
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(img, str(i), (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # highlight A, B, C
    for p in [A, B, C]:
        cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 140, 255), -1)

    # angle rays
    cv2.line(img, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), (255, 0, 0), 2)
    cv2.line(img, (int(C[0]), int(C[1])), (int(B[0]), int(B[1])), (255, 0, 0), 2)

    h, w = img.shape[:2]

    # smart label position
    label_x = int(w * 0.05) if B[0] > w / 2 else int(w * 0.75)
    label_y = int(h * 0.08) if B[1] > h / 2 else int(h * 0.92)

    label = f"{ang:.1f}°"

    cv2.rectangle(img, (label_x, label_y - 35), (label_x + 130, label_y + 10), (0, 0, 0), -1)
    cv2.putText(img, label, (label_x + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    cv2.putText(img, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (20, 20, 20), 2)

    output_path = os.path.join(MEASUREMENTS_DIR, output_filename)
    cv2.imwrite(output_path, img)

    rel_path = join_stored("static", "measurements", output_filename)
    return ang, "/" + rel_path


def interpret_nasiolabial(ang):
    if ang > 110:
        return {
            "status": "Increased (>110°)",
            "meaning": "Possible high columella or retruded maxilla.",
            "treatment": "Class III cases may require surgical evaluation depending on full diagnosis."
        }
    elif ang < 90:
        return {
            "status": "Decreased (<90°)",
            "meaning": "Possible protruded maxilla.",
            "treatment": "Class II pattern may need extraction planning or surgery depending on full diagnosis."
        }
    return {
        "status": "Normal (90°–110°)",
        "meaning": "Within normal range.",
        "treatment": "No treatment implication from this angle alone."
    }


def interpret_profile_convexity(ang):
    if ang > 171:
        return {
            "status": "Increased (>171°)",
            "meaning": "Possible protruded chin.",
            "treatment": "May indicate Class III tendency; consider growth modification or surgery depending on case."
        }
    elif ang < 151:
        return {
            "status": "Decreased (<151°)",
            "meaning": "Possible retruded chin.",
            "treatment": "May indicate Class II tendency; consider advancement options depending on diagnosis."
        }
    return {
        "status": "Normal (151°–171°)",
        "meaning": "Within normal range.",
        "treatment": "No treatment implication from this angle alone."
    }


def interpret_total_facial_convexity(ang):
    if ang > 137:
        return {
            "status": "Increased (>137°)",
            "meaning": "Possible protruded chin.",
            "treatment": "May suggest Class III tendency; needs correlation with full clinical findings."
        }
    elif ang < 127:
        return {
            "status": "Decreased (<127°)",
            "meaning": "Possible retruded chin or prominent nasal effect.",
            "treatment": "May suggest Class II tendency; requires full orthodontic evaluation."
        }
    return {
        "status": "Normal (127°–137°)",
        "meaning": "Within normal range.",
        "treatment": "No treatment implication from this angle alone."
    }


def interpret_mentolabial(ang):
    if ang > 130:
        return {
            "status": "Increased (>130°)",
            "meaning": "Possible retruded chin, proclined lower teeth, or advanced mandible pattern.",
            "treatment": "May require orthodontic correction, genioplasty, or surgery depending on full diagnosis."
        }
    elif ang < 110:
        return {
            "status": "Decreased (<110°)",
            "meaning": "Possible prominent chin, retroclined lower teeth, or mandibular retrusion.",
            "treatment": "May require incisor correction, chin correction, or skeletal treatment depending on case."
        }
    return {
        "status": "Normal (110°–130°)",
        "meaning": "Within normal range.",
        "treatment": "No treatment implication from this angle alone."
    }


def analyze_measurement(image_path, pts_orig, measurement_type, case_id):
    image_path = resolve_project_path(image_path) or image_path
    pts_orig = normalize_points(pts_orig)

    configs = {
        "nasiolabial": {
            "title": "Nasiolabial Angle (7,8,10)",
            "points": (7, 8, 10),
            "interpreter": interpret_nasiolabial
        },
        "profile_convexity": {
            "title": "Profile Convexity Angle (3,8,17)",
            "points": (3, 8, 17),
            "interpreter": interpret_profile_convexity
        },
        "total_facial_convexity": {
            "title": "Total Facial Convexity (3,5,17)",
            "points": (3, 5, 17),
            "interpreter": interpret_total_facial_convexity
        },
        "mentolabial": {
            "title": "Mentolabial Angle (15,16,17)",
            "points": (15, 16, 17),
            "interpreter": interpret_mentolabial
        }
    }

    if measurement_type not in configs:
        raise ValueError("Invalid measurement type")

    config = configs[measurement_type]
    filename = f"case_{case_id}_{measurement_type}.jpg"

    angle_value, image_url = draw_angle_image(
        image_path=image_path,
        pts_orig=pts_orig,
        idxs_1based=config["points"],
        title=config["title"],
        output_filename=filename
    )

    interpretation = config["interpreter"](angle_value)

    return {
        "type": measurement_type,
        "title": config["title"],
        "angle": round(angle_value, 2),
        "image_url": image_url,
        "status": interpretation["status"],
        "meaning": interpretation["meaning"],
        "treatment": interpretation["treatment"]
    }