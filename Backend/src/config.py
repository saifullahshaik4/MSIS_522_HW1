"""Central configuration: paths, constants, and hyperparameter grids."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths (resolved relative to this file — works from any CWD)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SRC_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

DATA_DIR = REPO_ROOT / "Data"
OUTPUTS_DIR = BACKEND_DIR / "outputs"

DATA_SUMMARY_DIR = OUTPUTS_DIR / "data_summary"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
SHAP_DIR = OUTPUTS_DIR / "shap"

# ---------------------------------------------------------------------------
# Data file
# ---------------------------------------------------------------------------
DATA_CSV = DATA_DIR / "credit_score.csv"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
TARGET_COL = "CREDIT_SCORE"
ID_COLS = ["CUST_ID"]           # identifier columns — excluded from features
EXCLUDE_COLS = ["DEFAULT"]      # leakage-risk columns — excluded from features
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Categorical column(s) that need OneHotEncoding
CAT_COLS_OVERRIDE: list[str] = []   # leave empty = auto-detect from dtype

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
DT_PARAM_GRID = {
    "regressor__max_depth": [3, 5, 8, 12, 20, None],
    "regressor__min_samples_split": [2, 5, 10, 20],
    "regressor__min_samples_leaf": [1, 2, 5, 10],
}

RF_PARAM_GRID = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 5],
}

XGB_PARAM_GRID = {
    "regressor__n_estimators": [200, 500],
    "regressor__max_depth": [3, 5, 8],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0],
}

CV_FOLDS = 5
CV_SCORING = "neg_root_mean_squared_error"

# ---------------------------------------------------------------------------
# MLP settings
# ---------------------------------------------------------------------------
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 32
MLP_VALIDATION_SPLIT = 0.2
MLP_EARLY_STOPPING_PATIENCE = 10
