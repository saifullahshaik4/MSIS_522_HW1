"""Evaluate all models on the held-out test set and save comparison artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
)

from .config import METRICS_DIR, PLOTS_DIR, MODELS_DIR
from .utils import save_json, save_text, ensure_dirs

sns.set_theme(style="whitegrid")

MODEL_DISPLAY = {
    "linear_regression": "Linear Regression",
    "decision_tree":     "Decision Tree",
    "random_forest":     "Random Forest",
    "xgboost":           "XGBoost",
    "mlp":               "MLP (Neural Net)",
}
TREE_MODELS = {"decision_tree", "random_forest", "xgboost"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all(
    models: dict,
    X_test: pd.DataFrame,
    y_test,
    preprocessor_scaled,
) -> tuple[str, str]:
    """Evaluate each model on the test set.

    Models are sklearn Pipelines (contain their own preprocessor) except MLP
    which expects X already preprocessed by preprocessor_scaled.

    Returns best_model_key, best_tree_model_key.
    """
    ensure_dirs(METRICS_DIR, PLOTS_DIR)
    y_true = np.array(y_test)
    rows = []

    for key, model in models.items():
        print(f"  Evaluating {key} …")
        y_pred = _predict(model, X_test, key, preprocessor_scaled)
        metrics = _compute_metrics(y_true, y_pred)
        rows.append({
            "model": key,
            "display_name": MODEL_DISPLAY.get(key, key),
            **metrics,
        })
        _plot_pred_vs_actual(y_true, y_pred, key)
        _plot_residuals(y_true, y_pred, key)

    cmp_df = pd.DataFrame(rows).set_index("model")
    cmp_df.to_csv(METRICS_DIR / "model_comparison.csv")
    cmp_df.reset_index().to_json(
        METRICS_DIR / "model_comparison.json", orient="records", indent=2
    )
    print(f"  Saved → {METRICS_DIR / 'model_comparison.csv'}")

    _plot_bar(cmp_df, "rmse", "RMSE",   "model_rmse_comparison.png", ascending=True)
    _plot_bar(cmp_df, "r2",   "R²",     "model_r2_comparison.png",   ascending=False)

    best_key      = str(cmp_df["rmse"].idxmin())
    tree_sub      = cmp_df[cmp_df.index.isin(TREE_MODELS)]
    best_tree_key = str(tree_sub["rmse"].idxmin()) if not tree_sub.empty else best_key

    _write_summary(cmp_df, best_key, best_tree_key)
    (MODELS_DIR / "best_model_name.txt").write_text(best_key)
    (MODELS_DIR / "best_tree_model_name.txt").write_text(best_tree_key)

    print(f"[evaluate] Best overall: {best_key}  |  Best tree: {best_tree_key}")
    return best_key, best_tree_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict(model, X_test: pd.DataFrame, key: str, preprocessor_scaled) -> np.ndarray:
    """Route prediction through the correct interface."""
    if key == "mlp":
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                X_arr = preprocessor_scaled.transform(X_test)
                return model.predict(X_arr, verbose=0).flatten()
        except Exception:
            pass
    # sklearn Pipeline (preprocessor + regressor)
    return model.predict(X_test)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    ev   = float(explained_variance_score(y_true, y_pred))
    mape = (
        float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        if np.all(y_true != 0) else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "explained_variance": ev, "mape": mape}


def _plot_pred_vs_actual(y_true, y_pred, key: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {MODEL_DISPLAY.get(key, key)}")
    fig.tight_layout()
    path = PLOTS_DIR / f"pred_vs_actual_{key}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _plot_residuals(y_true, y_pred, key: str) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=15, color="darkorange")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Residuals — {MODEL_DISPLAY.get(key, key)}", fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / f"residuals_{key}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _plot_bar(
    df: pd.DataFrame, metric: str, label: str,
    filename: str, ascending: bool = True,
) -> None:
    data = df[metric].sort_values(ascending=ascending)
    fig, ax = plt.subplots(figsize=(9, 5))
    best_val = data.min() if ascending else data.max()
    colors = ["#1976D2" if v == best_val else "#90CAF9" for v in data.values]
    bars = ax.bar(
        [MODEL_DISPLAY.get(k, k) for k in data.index],
        data.values, color=colors, edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set_title(f"Model Comparison — {label}", fontsize=14, fontweight="bold")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _write_summary(df: pd.DataFrame, best_key: str, best_tree_key: str) -> None:
    lines = ["=" * 60, "Model Performance Summary — Credit Score Dataset", "=" * 60, ""]
    for key in df.index:
        row = df.loc[key]
        lines += [
            f"  {MODEL_DISPLAY.get(key, key)}",
            f"    RMSE              : {row['rmse']:.4f}",
            f"    MAE               : {row['mae']:.4f}",
            f"    R²                : {row['r2']:.4f}",
            f"    Explained Variance: {row['explained_variance']:.4f}",
            f"    MAPE (%)          : {row['mape']:.2f}" if not np.isnan(row["mape"]) else "    MAPE (%)          : N/A",
            "",
        ]
    lines += [
        "=" * 60,
        f"  Best overall model  : {MODEL_DISPLAY.get(best_key, best_key)}",
        f"  Best tree model     : {MODEL_DISPLAY.get(best_tree_key, best_tree_key)}",
        "=" * 60,
    ]
    save_text("\n".join(lines), METRICS_DIR / "model_summary.txt")
