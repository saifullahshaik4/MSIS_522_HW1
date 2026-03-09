"""Load all pre-trained artefacts from Backend/outputs."""
from __future__ import annotations

import json
import joblib
import pandas as pd
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT    = FRONTEND_DIR.parent
OUTPUTS_DIR  = REPO_ROOT / "Backend" / "outputs"

DATA_SUMMARY = OUTPUTS_DIR / "data_summary"
PLOTS_DIR    = OUTPUTS_DIR / "plots"
METRICS_DIR  = OUTPUTS_DIR / "metrics"
MODELS_DIR   = OUTPUTS_DIR / "models"
SHAP_DIR     = OUTPUTS_DIR / "shap"


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


# ---- Dataset meta --------------------------------------------------------

def load_dataset_overview() -> dict:
    return _load_json(DATA_SUMMARY / "dataset_overview.json")


def load_feature_columns() -> list:
    return _load_json(MODELS_DIR / "feature_columns.json")


def load_feature_types() -> dict:
    return _load_json(MODELS_DIR / "feature_types.json")


def load_categorical_values() -> dict:
    p = MODELS_DIR / "categorical_values.json"
    return _load_json(p) if p.exists() else {}


def load_target_name() -> str:
    return _load_json(MODELS_DIR / "target_name.json")["target"]


def load_plot_interpretations() -> dict:
    return _load_json(DATA_SUMMARY / "plot_interpretations.json")


def load_feature_ranges() -> pd.DataFrame:
    return pd.read_csv(DATA_SUMMARY / "feature_ranges.csv", index_col="feature")


def load_feature_defaults() -> pd.DataFrame:
    return pd.read_csv(DATA_SUMMARY / "feature_defaults.csv", index_col="feature")


# ---- Metrics -------------------------------------------------------------

def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "model_comparison.csv", index_col="model")


def load_model_summary_text() -> str:
    return _read_text(METRICS_DIR / "model_summary.txt")


def load_mlp_history() -> dict:
    p = METRICS_DIR / "mlp_history.json"
    return _load_json(p) if p.exists() else {}


# ---- Best model names ----------------------------------------------------

def load_best_model_name() -> str:
    p = MODELS_DIR / "best_model_name.txt"
    return p.read_text().strip() if p.exists() else "unknown"


def load_best_tree_model_name() -> str:
    p = MODELS_DIR / "best_tree_model_name.txt"
    return p.read_text().strip() if p.exists() else "xgboost"


# ---- Models --------------------------------------------------------------

def load_sklearn_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}.pkl")


def load_keras_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(
            str(MODELS_DIR / "mlp_model.keras"), compile=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load MLP model. Ensure tensorflow-cpu==2.15.0 is installed. Error: {e}"
        )


def load_preprocessor(kind: str = "scaled"):
    return joblib.load(MODELS_DIR / f"preprocessor_{kind}.pkl")


def load_model_params(model_key: str) -> dict:
    p = MODELS_DIR / f"{model_key}_best_params.json"
    return _load_json(p) if p.exists() else {}


def load_preprocessed_feature_names(kind: str = "tree") -> list:
    p = MODELS_DIR / f"preprocessed_feature_names_{kind}.json"
    return _load_json(p) if p.exists() else []


# ---- Plot paths ----------------------------------------------------------

def get_plot_path(filename: str) -> Path:
    return PLOTS_DIR / filename


def get_shap_path(filename: str) -> Path:
    return SHAP_DIR / filename


# ---- SHAP artefacts -------------------------------------------------------

def load_shap_top_features() -> pd.DataFrame:
    p = SHAP_DIR / "shap_top_features.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_shap_metadata() -> dict:
    p = SHAP_DIR / "shap_metadata.json"
    return _load_json(p) if p.exists() else {}


def load_shap_interpretation() -> str:
    return _read_text(SHAP_DIR / "shap_interpretation.txt")
