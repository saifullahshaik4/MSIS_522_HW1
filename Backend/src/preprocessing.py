"""Build, fit, and save ColumnTransformer preprocessing pipelines."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import MODELS_DIR, DATA_SUMMARY_DIR
from .utils import save_joblib, ensure_dirs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_and_fit_preprocessors(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple:
    """Fit two preprocessing pipelines on X_train and transform both sets.

    Returns
    -------
    preprocessor_scaled, preprocessor_tree,
    X_train_scaled, X_test_scaled,
    X_train_tree,   X_test_tree
    """
    ensure_dirs(MODELS_DIR, DATA_SUMMARY_DIR)

    print("[preprocessing] Fitting preprocessors …")

    # ------------------------------------------------------------------
    # Scaled pipeline (for Linear Regression & MLP)
    # numeric: impute + scale | categorical: impute + OHE
    # ------------------------------------------------------------------
    num_pipe_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("num", num_pipe_scaled, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )

    # ------------------------------------------------------------------
    # Tree pipeline (for DT / RF / XGBoost)
    # numeric: impute only | categorical: impute + OHE (trees handle scale)
    # ------------------------------------------------------------------
    num_pipe_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", num_pipe_tree, numeric_features),
            ("cat", cat_pipe_tree, categorical_features),
        ],
        remainder="drop",
    )

    # Fit on training data only
    X_train_scaled = preprocessor_scaled.fit_transform(X_train)
    X_test_scaled  = preprocessor_scaled.transform(X_test)

    X_train_tree = preprocessor_tree.fit_transform(X_train)
    X_test_tree  = preprocessor_tree.transform(X_test)

    # Derive the full output feature names
    ohe_cats_scaled = (
        preprocessor_scaled.named_transformers_["cat"]["ohe"]
        .get_feature_names_out(categorical_features).tolist()
        if categorical_features else []
    )
    output_feature_names_scaled = numeric_features + ohe_cats_scaled

    ohe_cats_tree = (
        preprocessor_tree.named_transformers_["cat"]["ohe"]
        .get_feature_names_out(categorical_features).tolist()
        if categorical_features else []
    )
    output_feature_names_tree = numeric_features + ohe_cats_tree

    save_joblib(preprocessor_scaled, MODELS_DIR / "preprocessor_scaled.pkl")
    save_joblib(preprocessor_tree,   MODELS_DIR / "preprocessor_tree.pkl")

    from .utils import save_json
    save_json(output_feature_names_scaled, MODELS_DIR / "preprocessed_feature_names_scaled.json")
    save_json(output_feature_names_tree,   MODELS_DIR / "preprocessed_feature_names_tree.json")

    _save_feature_metadata(X_train, numeric_features, preprocessor_tree)

    print(f"[preprocessing] Scaled output shape: {X_train_scaled.shape}")
    print(f"[preprocessing] Tree output shape:   {X_train_tree.shape}")
    print("[preprocessing] Done.")

    return (
        preprocessor_scaled, preprocessor_tree,
        X_train_scaled, X_test_scaled,
        X_train_tree, X_test_tree,
        output_feature_names_scaled, output_feature_names_tree,
    )


# ---------------------------------------------------------------------------
# Feature metadata for Streamlit UI
# ---------------------------------------------------------------------------

def _save_feature_metadata(
    X_train: pd.DataFrame,
    numeric_features: list[str],
    preprocessor_tree,
) -> None:
    """Compute per-numeric-feature statistics and persist for the frontend."""
    # Use the imputed values from the tree preprocessor (no scaling)
    X_imp_arr = preprocessor_tree.named_transformers_["num"]["imputer"].transform(
        X_train[numeric_features]
    )
    X_imp = pd.DataFrame(X_imp_arr, columns=numeric_features)

    feature_ranges = pd.DataFrame({
        "feature": numeric_features,
        "min":    X_imp.min().values,
        "max":    X_imp.max().values,
        "median": X_imp.median().values,
        "mean":   X_imp.mean().values,
        "std":    X_imp.std().values,
    }).set_index("feature")

    feature_ranges.to_csv(DATA_SUMMARY_DIR / "feature_ranges.csv")
    print(f"  Saved → {DATA_SUMMARY_DIR / 'feature_ranges.csv'}")

    feature_defaults = feature_ranges[["median"]].rename(columns={"median": "default"})
    feature_defaults.to_csv(DATA_SUMMARY_DIR / "feature_defaults.csv")
    print(f"  Saved → {DATA_SUMMARY_DIR / 'feature_defaults.csv'}")
