"""Build prediction input rows and route through the correct model."""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_DISPLAY_LIST = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "MLP (Neural Net)",
]

MODEL_FILE_MAP = {
    "Linear Regression": "linear_regression",
    "Decision Tree":     "decision_tree",
    "Random Forest":     "random_forest",
    "XGBoost":           "xgboost",
    "MLP (Neural Net)":  "mlp",
}

MODEL_USES_SCALED = {"Linear Regression", "MLP (Neural Net)"}


def build_input_row(
    user_inputs: dict,
    feature_cols: list[str],
    feature_defaults: pd.DataFrame,
    categorical_values: dict,
) -> pd.DataFrame:
    """Construct a single-row DataFrame with ALL feature columns.

    user_inputs keys match feature names; missing ones use median/mode defaults.
    """
    row: dict = {}
    for feat in feature_cols:
        if feat in user_inputs:
            row[feat] = user_inputs[feat]
        elif feat in feature_defaults.index:
            row[feat] = float(feature_defaults.loc[feat, "default"])
        else:
            # For categorical features: use first category value as default
            for cat_col, cats in categorical_values.items():
                if feat == cat_col:
                    row[feat] = cats[0] if cats else ""
                    break
            else:
                row[feat] = 0.0
    return pd.DataFrame([row], columns=feature_cols)


def predict(
    model_display_name: str,
    input_row: pd.DataFrame,
    preprocessor_scaled,
    preprocessor_tree,
) -> float:
    """Apply preprocessor and return the scalar prediction."""
    from .load_artifacts import load_sklearn_model, load_keras_model

    model_key = MODEL_FILE_MAP[model_display_name]

    if model_key == "mlp":
        model = load_keras_model()
        if model is None:
            raise RuntimeError(
                "MLP (Neural Net) requires TensorFlow, which is not installed in "
                "this deployment. Please select a different model."
            )
        X_arr = preprocessor_scaled.transform(input_row)
        pred  = model.predict(X_arr, verbose=0).flatten()[0]
    else:
        model = load_sklearn_model(model_key)
        pred  = model.predict(input_row)[0]

    return float(pred)


TREE_MODEL_KEYS = {"decision_tree", "random_forest", "xgboost"}


def compute_local_shap(
    model_display_name: str,
    input_row: pd.DataFrame,
    feature_names_tree: list[str],
) -> tuple | None:
    """Compute SHAP values for a single row using the tree-based model.

    Returns (shap_explanation, base_value) or None if model is not tree-based.
    """
    model_key = MODEL_FILE_MAP.get(model_display_name, "")
    if model_key not in TREE_MODEL_KEYS:
        return None

    try:
        import shap
        from .load_artifacts import load_sklearn_model

        model = load_sklearn_model(model_key)
        preprocessor = model.named_steps["preprocessor"]
        regressor    = model.named_steps["regressor"]

        X_arr = preprocessor.transform(input_row)

        explainer   = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_arr)   # shape (1, n_features)
        base_val    = float(explainer.expected_value)

        exp = shap.Explanation(
            values=shap_values[0],
            base_values=base_val,
            data=X_arr[0],
            feature_names=feature_names_tree,
        )
        return exp, base_val
    except Exception:
        return None
