import os
import json
import cv2
import csv

BASE = r"C:\Users\marya\Desktop\GradLandmarksDataset"  

LAB_IMG_DIR = os.path.join(BASE, "labeled", "images")
LAB_LBL_DIR = os.path.join(BASE, "labeled", "labels")
UNLAB_IMG_DIR = os.path.join(BASE, "unlabeled", "images")

EXPECTED_LANDMARKS = 17

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def main():
    issues = []
    labeled_images = list_images(LAB_IMG_DIR)
    unlabeled_images = list_images(UNLAB_IMG_DIR)
    label_files = sorted([f for f in os.listdir(LAB_LBL_DIR) if f.lower().endswith(".json")])

    print("=== DATASET SUMMARY ===")
    print("Labeled images:", len(labeled_images))
    print("Label files   :", len(label_files))
    print("Unlabeled imgs:", len(unlabeled_images))
    print()

    print("=== CHECK 1: Image <-> Label pairing ===")
    img_to_json_missing = 0
    json_to_img_missing = 0

    labeled_set = set(labeled_images)
    label_set = set(label_files)

    for img in labeled_images:
        stem = os.path.splitext(img)[0]
        expected_json = stem + ".json"
        if expected_json not in label_set:
            issues.append(("MISSING_LABEL", img, expected_json))
            img_to_json_missing += 1

    for jf in label_files:
        stem = os.path.splitext(jf)[0]
        # allow any image extension
        found = any((stem + ext) in labeled_set for ext in [".jpg", ".jpeg", ".png", ".bmp"])
        if not found:
            issues.append(("MISSING_IMAGE", jf, stem))
            json_to_img_missing += 1

    print("Missing label files:", img_to_json_missing)
    print("Missing image files:", json_to_img_missing)
    print()

    print("=== CHECK 2: Label file validity ===")
    bad_json = 0
    bad_landmark_count = 0
    out_of_bounds = 0
    size_mismatch = 0
    unreadable_img = 0

    for jf in label_files:
        jp = os.path.join(LAB_LBL_DIR, jf)
        try:
            data = json.load(open(jp, "r", encoding="utf-8"))
        except Exception as e:
            issues.append(("BAD_JSON", jf, str(e)))
            bad_json += 1
            continue

        img_name = data.get("image")
        w = data.get("width")
        h = data.get("height")
        lms = data.get("landmarks", [])

        if img_name is None:
            issues.append(("JSON_MISSING_IMAGE_FIELD", jf, "image"))
        if w is None or h is None:
            issues.append(("JSON_MISSING_SIZE_FIELD", jf, f"width={w}, height={h}"))

        img_path = os.path.join(LAB_IMG_DIR, img_name) if img_name else None
        img = cv2.imread(img_path) if img_path and os.path.exists(img_path) else None
        if img is None:
            issues.append(("UNREADABLE_OR_MISSING_IMAGE", jf, img_path))
            unreadable_img += 1
            continue

        ih, iw = img.shape[:2]

        if (w is not None and h is not None) and (int(w) != int(iw) or int(h) != int(ih)):
            issues.append(("SIZE_MISMATCH", img_name, f"json=({w},{h}) vs actual=({iw},{ih})"))
            size_mismatch += 1

        if len(lms) != EXPECTED_LANDMARKS:
            issues.append(("BAD_LANDMARK_COUNT", img_name, f"{len(lms)} (expected {EXPECTED_LANDMARKS})"))
            bad_landmark_count += 1

        for lm in lms:
            x = lm.get("x")
            y = lm.get("y")
            lid = lm.get("id")
            if x is None or y is None:
                issues.append(("MISSING_COORD", img_name, f"id={lid}"))
                continue
            if not (0 <= int(x) < iw and 0 <= int(y) < ih):
                issues.append(("OUT_OF_BOUNDS", img_name, f"id={lid} x={x} y={y} img=({iw},{ih})"))
                out_of_bounds += 1

    print("Bad JSON files:", bad_json)
    print("Unreadable/missing images:", unreadable_img)
    print("Size mismatches:", size_mismatch)
    print("Bad landmark counts:", bad_landmark_count)
    print("Out-of-bounds landmarks:", out_of_bounds)
    print()

    report_path = os.path.join(BASE, "data_cleaning_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["issue_type", "file_or_image", "details"])
        for row in issues:
            writer.writerow(row)

    print("✅ Report saved:", report_path)
    print("Total issues found:", len(issues))

if __name__ == "__main__":
    main()
