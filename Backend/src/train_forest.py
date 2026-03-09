"""Train a Random Forest regressor with GridSearchCV."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from .config import MODELS_DIR, RF_PARAM_GRID, CV_FOLDS, CV_SCORING, RANDOM_STATE
from .utils import save_joblib, save_json, ensure_dirs


def train_random_forest(
    X_train: pd.DataFrame,
    y_train,
    preprocessor_tree,
) -> Pipeline:
    ensure_dirs(MODELS_DIR)
    print("[train_forest] Running GridSearchCV for Random Forest …")

    pipe = Pipeline([
        ("preprocessor", preprocessor_tree),
        ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=RF_PARAM_GRID,
        scoring=CV_SCORING,
        cv=CV_FOLDS,
        n_jobs=1,   # inner RF already uses n_jobs=-1
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)

    best_params = {k.replace("regressor__", ""): v for k, v in gs.best_params_.items()}
    best_score  = float(-gs.best_score_)
    print(f"[train_forest] Best params: {best_params}")
    print(f"[train_forest] CV RMSE: {best_score:.4f}")

    save_joblib(gs.best_estimator_, MODELS_DIR / "random_forest.pkl")
    save_json({"best_params": best_params, "cv_rmse": best_score},
              MODELS_DIR / "random_forest_best_params.json")
    print("[train_forest] Done.")
    return gs.best_estimator_
