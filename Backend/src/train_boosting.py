"""Train an XGBoost regressor with GridSearchCV."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from .config import MODELS_DIR, XGB_PARAM_GRID, CV_FOLDS, CV_SCORING, RANDOM_STATE
from .utils import save_joblib, save_json, ensure_dirs


def train_xgboost(
    X_train: pd.DataFrame,
    y_train,
    preprocessor_tree,
) -> Pipeline:
    ensure_dirs(MODELS_DIR)
    print("[train_boosting] Running GridSearchCV for XGBoost …")

    pipe = Pipeline([
        ("preprocessor", preprocessor_tree),
        ("regressor", XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )),
    ])
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=XGB_PARAM_GRID,
        scoring=CV_SCORING,
        cv=CV_FOLDS,
        n_jobs=1,   # inner XGB uses n_jobs=-1
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)

    best_params = {k.replace("regressor__", ""): v for k, v in gs.best_params_.items()}
    best_score  = float(-gs.best_score_)
    print(f"[train_boosting] Best params: {best_params}")
    print(f"[train_boosting] CV RMSE: {best_score:.4f}")

    save_joblib(gs.best_estimator_, MODELS_DIR / "xgboost.pkl")
    save_json({"best_params": best_params, "cv_rmse": best_score},
              MODELS_DIR / "xgboost_best_params.json")
    print("[train_boosting] Done.")
    return gs.best_estimator_
