"""Train a Linear Regression baseline (preprocessor_scaled baked in)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, RANDOM_STATE
from .utils import save_joblib, ensure_dirs


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train,
    preprocessor_scaled,
) -> Pipeline:
    ensure_dirs(MODELS_DIR)
    print("[train_baseline] Training Linear Regression …")
    pipe = Pipeline([
        ("preprocessor", preprocessor_scaled),
        ("regressor", LinearRegression(n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    save_joblib(pipe, MODELS_DIR / "linear_regression.pkl")
    print("[train_baseline] Done.")
    return pipe
