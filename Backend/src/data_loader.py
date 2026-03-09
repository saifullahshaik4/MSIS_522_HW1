"""Load credit_score.csv, split into train/test, and save dataset summaries."""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .config import (
    DATA_CSV, TARGET_COL, ID_COLS, EXCLUDE_COLS,
    DATA_SUMMARY_DIR, MODELS_DIR,
    RANDOM_STATE, TEST_SIZE, CAT_COLS_OVERRIDE,
)
from .utils import save_json, ensure_dirs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data() -> tuple:
    """Load, validate, split, and summarise the dataset.

    Returns
    -------
    df, X_train, X_test, y_train, y_test,
    numeric_features, categorical_features, feature_cols
    """
    ensure_dirs(DATA_SUMMARY_DIR, MODELS_DIR)

    print(f"[data_loader] Loading {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV)
    print(f"[data_loader] Shape: {df.shape}")

    _validate(df)

    # ---- identify feature columns ----------------------------------------
    drop_cols = ID_COLS + EXCLUDE_COLS + [TARGET_COL]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # ---- detect column types ---------------------------------------------
    if CAT_COLS_OVERRIDE:
        categorical_features = [c for c in CAT_COLS_OVERRIDE if c in feature_cols]
    else:
        categorical_features = (
            df[feature_cols]
            .select_dtypes(include=["object", "category", "bool"])
            .columns.tolist()
        )
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    print(f"[data_loader] Features: {len(feature_cols)} "
          f"({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    print(f"[data_loader] Categorical: {categorical_features}")

    # ---- split -----------------------------------------------------------
    X = df[feature_cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[data_loader] Train: {X_train.shape}, Test: {X_test.shape}")

    # ---- save summaries --------------------------------------------------
    _save_summary(df, feature_cols, numeric_features, categorical_features)
    _save_metadata(feature_cols, numeric_features, categorical_features)

    return df, X_train, X_test, y_train, y_test, numeric_features, categorical_features, feature_cols


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")
    print("[data_loader] Schema validation passed.")


# ---------------------------------------------------------------------------
# Summary artefacts
# ---------------------------------------------------------------------------

def _save_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    y = df[TARGET_COL]
    n_missing = int(df[feature_cols].isnull().sum().sum())

    overview = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "n_features": len(feature_cols),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": TARGET_COL,
        "n_missing_total": n_missing,
        "n_duplicates": int(df.duplicated().sum()),
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "target_std": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }
    save_json(overview, DATA_SUMMARY_DIR / "dataset_overview.json")

    # Missing values per column
    mv = df[feature_cols + [TARGET_COL]].isnull().sum()
    mv_pct = (mv / len(df) * 100).round(4)
    mv_df = pd.DataFrame({"missing_count": mv, "missing_pct": mv_pct})
    mv_df = mv_df[mv_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
    mv_df.to_csv(DATA_SUMMARY_DIR / "missing_values.csv")
    print(f"  Saved → {DATA_SUMMARY_DIR / 'missing_values.csv'}")

    # Descriptive stats
    desc = df[numeric_features + [TARGET_COL]].describe().T
    desc.to_csv(DATA_SUMMARY_DIR / "describe.csv")
    print(f"  Saved → {DATA_SUMMARY_DIR / 'describe.csv'}")


def _save_metadata(
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    save_json(feature_cols, MODELS_DIR / "feature_columns.json")
    save_json({"target": TARGET_COL}, MODELS_DIR / "target_name.json")
    save_json(
        {"numeric": numeric_features, "categorical": categorical_features},
        MODELS_DIR / "feature_types.json",
    )
    # Save unique category values for the UI
    from pandas import read_csv
    df = read_csv(DATA_CSV)
    cat_values: dict[str, list] = {}
    for c in categorical_features:
        if c in df.columns:
            cat_values[c] = sorted(df[c].dropna().unique().tolist())
    save_json(cat_values, MODELS_DIR / "categorical_values.json")
