"""
generate_xray_reference_stats.py
=================================
Compute per-feature reference statistics (median, Q1, Q3) for the 217
cephalometric features used by the DentAlign XGBoost diagnosis models.

These statistics power the SHAP explanation layer: for each top feature the
UI shows whether the patient's value is higher than, within, or lower than
the "available system-case reference range".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT LIMITATIONS — READ BEFORE USING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - The default source is the OrthoCase records stored in app.db.
  - These are NOT the original XGBoost model-training dataset
    (that data lives in Google Colab and is not available locally).
  - Reference comparisons shown in the UI are clearly labelled as
    "available system-case reference range" — never as "model reference
    range" or "training-data reference range".
  - The website code does not need to change when a better source is
    supplied — just regenerate this JSON file.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPPORTED INPUT SOURCES (use --source to select):

  system_db  (default)
      Loads 19-landmark JSON from completed OrthoCase rows in app.db,
      runs the same feature-engineering pipeline as the live app.

  landmarks_csv
      Reads a CSV where each row is one case and columns are
      landmark coordinates named x1,y1,x2,y2,...,x19,y19
      (1-based, pixel coordinates in original image space).
      Pass the file path with --input.

  features_csv
      Reads a CSV or NPY file that already contains 217 features
      in training-column order (column headers must match
      feature_columns.pkl).  No HRNet re-run needed.
      Pass the file path with --input.

USAGE:

  # Default — use system DB
  python scripts/generate_xray_reference_stats.py

  # Future: use original 400-image landmarks CSV
  python scripts/generate_xray_reference_stats.py \\
      --source landmarks_csv --input path/to/landmarks.csv

  # Future: use pre-computed 217-feature matrix
  python scripts/generate_xray_reference_stats.py \\
      --source features_csv --input path/to/features.csv

Output: model/xray/feature_reference_statistics.json
  {
    "metadata": {
        "reference_source": "available_system_cases",
        "sample_count": 30,
        ...
    },
    "statistics": {
        "p1_x": {"median": -0.2125, "q1": -0.2152, "q3": -0.2122},
        ...
    }
  }
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# ── Add project root to sys.path ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

OUTPUT_PATH = ROOT / "model" / "xray" / "feature_reference_statistics.json"
DB_PATH     = ROOT / "app.db"

# Canonical disclaimer text — used in the JSON and displayed in the UI.
# Do NOT change this string without also updating the UI disclaimer note.
_DISCLAIMER = (
    "Reference comparisons are based on {n} available system cases. "
    "They are not validated clinical normal ranges and do not represent "
    "the original model-training dataset."
)


# ─────────────────────────────────────────────────────────────────────────────
# Source loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_rows_from_db() -> tuple:
    """
    Load completed OrthoCase landmark records from app.db, run feature
    engineering, and return (feature_rows, source_label, n_cases_loaded).
    """
    print(f"Loading landmark records from: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    cur.execute(
        "SELECT landmarks_json FROM ortho_case "
        "WHERE landmarks_json IS NOT NULL AND status = 'DONE'"
    )
    rows = cur.fetchall()
    conn.close()

    all_landmarks = [json.loads(row[0]) for row in rows]
    n_cases = len(all_landmarks)
    print(f"Found {n_cases} completed OrthoCase records.")

    if n_cases == 0:
        print("ERROR: No completed OrthoCase records found in app.db.")
        print("  Please upload and process at least one X-ray before running this script.")
        sys.exit(1)

    from utils.orthodontic_ai_inference import (
        NUM_FEATURES,
        _build_feature_matrix,
        _generate_features,
    )

    print("Building 217-feature vectors from stored landmarks...")
    feature_rows = []
    failed_indices = []

    for i, lm_list in enumerate(all_landmarks):
        try:
            pts = np.array(lm_list, dtype=np.float32)
            if pts.shape != (19, 2):
                raise ValueError(f"Unexpected landmark shape {pts.shape}")
            feat_dict = _generate_features(pts)
            X = _build_feature_matrix(feat_dict)
            assert X.shape == (1, NUM_FEATURES)
            feature_rows.append(feat_dict)
        except Exception as e:
            failed_indices.append((i, str(e)))

    if failed_indices:
        print(f"WARNING: {len(failed_indices)} cases failed feature engineering:")
        for idx, err in failed_indices:
            print(f"  Case index {idx}: {err}")

    return feature_rows, "available_system_cases", n_cases


def load_feature_rows_from_landmarks_csv(csv_path: str) -> tuple:
    """
    Load a landmarks CSV with columns x1,y1,...,x19,y19 (pixel coordinates),
    run feature engineering, return (feature_rows, source_label, n_cases).
    """
    import pandas as pd
    from utils.orthodontic_ai_inference import (
        NUM_FEATURES,
        _build_feature_matrix,
        _generate_features,
    )

    print(f"Loading landmarks CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    n_cases = len(df)
    print(f"Found {n_cases} rows.")

    feature_rows = []
    failed_indices = []

    for i, row in df.iterrows():
        try:
            pts = np.array(
                [[row[f"x{j}"], row[f"y{j}"]] for j in range(1, 20)],
                dtype=np.float32,
            )
            if pts.shape != (19, 2):
                raise ValueError(f"Row {i}: unexpected shape {pts.shape}")
            feat_dict = _generate_features(pts)
            X = _build_feature_matrix(feat_dict)
            assert X.shape == (1, NUM_FEATURES)
            feature_rows.append(feat_dict)
        except Exception as e:
            failed_indices.append((i, str(e)))

    if failed_indices:
        print(f"WARNING: {len(failed_indices)} rows failed feature engineering:")
        for idx, err in failed_indices:
            print(f"  Row {idx}: {err}")

    return feature_rows, "landmarks_csv", n_cases


def load_feature_rows_from_features_csv(file_path: str) -> tuple:
    """
    Load a pre-computed 217-feature matrix (CSV or .npy).
    Column headers must match feature_columns.pkl.
    Returns (feature_rows, source_label, n_cases).
    """
    import pandas as pd
    from utils.orthodontic_ai_inference import NUM_FEATURES, _load_diagnosis_models

    print(f"Loading pre-computed feature matrix: {file_path}")

    if file_path.endswith(".npy"):
        arr = np.load(file_path)
        _, _, feature_cols = _load_diagnosis_models()
        if arr.shape[1] != NUM_FEATURES:
            print(f"ERROR: .npy has {arr.shape[1]} columns, expected {NUM_FEATURES}.")
            sys.exit(1)
        df = pd.DataFrame(arr, columns=feature_cols)
    else:
        df = pd.read_csv(file_path)
        if df.shape[1] != NUM_FEATURES:
            print(f"ERROR: CSV has {df.shape[1]} columns, expected {NUM_FEATURES}.")
            sys.exit(1)

    n_cases = len(df)
    print(f"Found {n_cases} rows with {df.shape[1]} features.")
    feature_rows = [row.to_dict() for _, row in df.iterrows()]
    return feature_rows, "features_csv", n_cases


# ─────────────────────────────────────────────────────────────────────────────
# Statistics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(feature_rows: list) -> dict:
    """Compute median, Q1, Q3 per feature column from a list of feature dicts."""
    import pandas as pd

    df = pd.DataFrame(feature_rows)
    print(f"Feature matrix shape: {df.shape}")

    stats: dict = {}
    for col in df.columns:
        series = df[col].dropna()
        stats[col] = {
            "median": round(float(series.median()), 6),
            "q1":     round(float(series.quantile(0.25)), 6),
            "q3":     round(float(series.quantile(0.75)), 6),
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate feature_reference_statistics.json for DentAlign SHAP."
    )
    parser.add_argument(
        "--source",
        choices=["system_db", "landmarks_csv", "features_csv"],
        default="system_db",
        help=(
            "Data source to use.  "
            "'system_db' (default): completed OrthoCase records from app.db.  "
            "'landmarks_csv': CSV with columns x1,y1,...,x19,y19.  "
            "'features_csv': pre-computed 217-feature CSV or .npy matrix."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        metavar="FILE",
        help="Path to input file (required for landmarks_csv and features_csv sources).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        metavar="FILE",
        help=f"Output JSON path (default: {OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DentAlign — X-Ray Feature Reference Statistics Generator")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.source == "system_db":
        feature_rows, source_label, n_raw = load_feature_rows_from_db()
    elif args.source == "landmarks_csv":
        if not args.input:
            print("ERROR: --input is required for --source landmarks_csv")
            sys.exit(1)
        feature_rows, source_label, n_raw = load_feature_rows_from_landmarks_csv(
            args.input
        )
    elif args.source == "features_csv":
        if not args.input:
            print("ERROR: --input is required for --source features_csv")
            sys.exit(1)
        feature_rows, source_label, n_raw = load_feature_rows_from_features_csv(
            args.input
        )

    n_valid = len(feature_rows)
    if n_valid == 0:
        print("ERROR: No cases produced valid features. Cannot generate statistics.")
        sys.exit(1)

    print(f"Successfully built features for {n_valid} / {n_raw} cases.")

    # ── Compute statistics ─────────────────────────────────────────────────────
    stats = compute_statistics(feature_rows)
    print(f"Computed statistics for {len(stats)} features.")

    # Spot check — show a few angle features
    print("\nSpot check (angle features):")
    shown = 0
    for name, s in stats.items():
        if name.startswith("a_"):
            print(f"  {name}: median={s['median']:.3f}  Q1={s['q1']:.3f}  Q3={s['q3']:.3f}")
            shown += 1
            if shown >= 3:
                break

    # ── Write output ──────────────────────────────────────────────────────────
    disclaimer = _DISCLAIMER.format(n=n_valid)

    output = {
        "metadata": {
            # Canonical fields read by xray_shap.load_reference_meta()
            "reference_source": source_label,
            "sample_count":     n_valid,
            # Legacy field kept for backwards compatibility
            "n_cases":          n_valid,
            "source": (
                f"Generated from source '{source_label}' "
                f"({n_valid} valid cases out of {n_raw} loaded). "
                "These are NOT the original XGBoost model training data."
            ),
            "disclaimer": disclaimer,
            "feature_count": len(stats),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "statistics": stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOutput written to: {out_path}")
    print(f"File size: {out_path.stat().st_size:,} bytes")

    if n_valid < 20:
        print(
            f"\nWARNING: Only {n_valid} valid cases — fewer than the minimum of 20 "
            "required for higher/lower/within comparison wording. "
            "The UI will show 'Insufficient reference cases for comparison.' "
            "for all features."
        )

    print("\nDone.")
    print(
        "\nTo regenerate from the original 400-image training data when available:\n"
        "  python scripts/generate_xray_reference_stats.py \\\n"
        "      --source features_csv --input path/to/training_features.csv\n"
        "No website code changes needed — the JSON format is identical."
    )


if __name__ == "__main__":
    main()
