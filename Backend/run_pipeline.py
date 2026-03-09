"""
Full Backend Pipeline — Credit Score Dataset
============================================
Run from the project root:
    python Backend/run_pipeline.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT   = BACKEND_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from Backend.src.config import (
    DATA_SUMMARY_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR, SHAP_DIR,
)
from Backend.src.utils import ensure_dirs, load_joblib


def _section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # 0. Directories
    # ------------------------------------------------------------------
    _section("0. Creating output directories")
    ensure_dirs(DATA_SUMMARY_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR, SHAP_DIR)

    # ------------------------------------------------------------------
    # 1. Load data & split
    # ------------------------------------------------------------------
    _section("1. Loading and splitting data")
    from Backend.src.data_loader import load_data
    df, X_train, X_test, y_train, y_test, \
        numeric_features, categorical_features, feature_cols = load_data()

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    _section("2. Fitting preprocessors")
    from Backend.src.preprocessing import build_and_fit_preprocessors
    (
        preprocessor_scaled, preprocessor_tree,
        X_train_scaled, X_test_scaled,
        X_train_tree, X_test_tree,
        feat_names_scaled, feat_names_tree,
    ) = build_and_fit_preprocessors(
        X_train, X_test, numeric_features, categorical_features,
    )

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    _section("3. Exploratory data analysis")
    from Backend.src.eda import run_eda
    run_eda(df, numeric_features, categorical_features)

    # ------------------------------------------------------------------
    # 4. Linear Regression
    # ------------------------------------------------------------------
    _section("4. Training Linear Regression (baseline)")
    from Backend.src.train_baseline import train_linear_regression
    lr_model = train_linear_regression(X_train, y_train, preprocessor_scaled)

    # ------------------------------------------------------------------
    # 5. Decision Tree
    # ------------------------------------------------------------------
    _section("5. Training Decision Tree (GridSearchCV)")
    from Backend.src.train_tree import train_decision_tree
    dt_model = train_decision_tree(X_train, y_train, preprocessor_tree)

    # ------------------------------------------------------------------
    # 6. Random Forest
    # ------------------------------------------------------------------
    _section("6. Training Random Forest (GridSearchCV)")
    from Backend.src.train_forest import train_random_forest
    rf_model = train_random_forest(X_train, y_train, preprocessor_tree)

    # ------------------------------------------------------------------
    # 7. XGBoost
    # ------------------------------------------------------------------
    _section("7. Training XGBoost (GridSearchCV)")
    from Backend.src.train_boosting import train_xgboost
    xgb_model = train_xgboost(X_train, y_train, preprocessor_tree)

    # ------------------------------------------------------------------
    # 8. MLP
    # ------------------------------------------------------------------
    _section("8. Training MLP (Keras)")
    from Backend.src.train_mlp import train_mlp
    mlp_model = train_mlp(X_train, y_train, preprocessor_scaled)

    # ------------------------------------------------------------------
    # 9. Evaluation
    # ------------------------------------------------------------------
    _section("9. Evaluating all models on held-out test set")
    from Backend.src.evaluate import evaluate_all

    models = {
        "linear_regression": lr_model,
        "decision_tree":     dt_model,
        "random_forest":     rf_model,
        "xgboost":           xgb_model,
        "mlp":               mlp_model,
    }
    best_key, best_tree_key = evaluate_all(
        models, X_test, y_test, preprocessor_scaled,
    )

    # ------------------------------------------------------------------
    # 10. SHAP
    # ------------------------------------------------------------------
    _section(f"10. SHAP explainability ({best_tree_key})")
    from Backend.src.shap_analysis import run_shap
    run_shap(models[best_tree_key], X_test, feat_names_tree, best_tree_key)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    _section(f"Pipeline complete — {elapsed:.1f}s")
    print(f"  Best model     : {best_key}")
    print(f"  Best tree model: {best_tree_key}")
    print(f"  Outputs        : {BACKEND_DIR / 'outputs'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
