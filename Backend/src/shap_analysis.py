"""SHAP explainability for the best tree-based model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from .config import SHAP_DIR, MODELS_DIR
from .utils import save_json, save_text, ensure_dirs

SHAP_SAMPLE_SIZE = 500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_shap(
    model,           # sklearn Pipeline(preprocessor_tree + regressor)
    X_test: pd.DataFrame,
    preprocessed_feature_names: list[str],
    best_tree_key: str,
) -> None:
    """Compute SHAP values and save all required artefacts.

    The model is a full Pipeline. We extract the fitted regressor and
    apply it to the already-transformed test data so that SHAP sees
    a plain array (required by TreeExplainer).
    """
    ensure_dirs(SHAP_DIR)
    print(f"[shap_analysis] Running SHAP for '{best_tree_key}' …")

    # Extract preprocessor and regressor from the Pipeline
    preprocessor = model.named_steps["preprocessor"]
    regressor    = model.named_steps["regressor"]

    # Transform test data
    X_arr = preprocessor.transform(X_test)

    # Sub-sample for speed / memory
    rng = np.random.default_rng(42)
    n   = min(SHAP_SAMPLE_SIZE, X_arr.shape[0])
    idx = rng.choice(X_arr.shape[0], size=n, replace=False)
    X_sample = X_arr[idx]

    explainer   = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_sample)

    shap_df = pd.DataFrame(shap_values, columns=preprocessed_feature_names)

    # ------------------------------------------------------------------
    # 1. SHAP beeswarm summary
    # ------------------------------------------------------------------
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=preprocessed_feature_names,
        max_display=20, show=False,
    )
    fig = plt.gcf()
    fig.suptitle("SHAP Summary Plot (Beeswarm)", fontsize=13, fontweight="bold", y=1.01)
    _save_gcf(SHAP_DIR / "shap_summary.png")

    # ------------------------------------------------------------------
    # 2. SHAP bar plot
    # ------------------------------------------------------------------
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=preprocessed_feature_names,
        plot_type="bar", max_display=20, show=False,
    )
    fig = plt.gcf()
    fig.suptitle("SHAP Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold", y=1.01)
    _save_gcf(SHAP_DIR / "shap_bar.png")

    # ------------------------------------------------------------------
    # 3. SHAP waterfall for one observation
    # ------------------------------------------------------------------
    base_val = float(explainer.expected_value) \
        if not hasattr(explainer.expected_value, "__len__") \
        else float(explainer.expected_value[0])

    exp = shap.Explanation(
        values=shap_values[0],
        base_values=base_val,
        data=X_sample[0],
        feature_names=preprocessed_feature_names,
    )
    shap.plots.waterfall(exp, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Waterfall (Single Customer)", fontsize=13, fontweight="bold", y=1.01)
    _save_gcf(SHAP_DIR / "shap_waterfall.png")

    # ------------------------------------------------------------------
    # 4. Top 15 features CSV
    # ------------------------------------------------------------------
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top15 = mean_abs.head(15).reset_index()
    top15.columns = ["feature", "mean_abs_shap"]
    top15.to_csv(SHAP_DIR / "shap_top_features.csv", index=False)
    print(f"  Saved → {SHAP_DIR / 'shap_top_features.csv'}")

    top_feats = top15["feature"].tolist()
    save_json({"top_shap_features": top_feats}, SHAP_DIR / "shap_metadata.json")

    # ------------------------------------------------------------------
    # 5. Interpretation text
    # ------------------------------------------------------------------
    top3 = top_feats[:3]
    text = (
        f"SHAP (SHapley Additive exPlanations) analysis was applied to the best "
        f"tree-based model ({best_tree_key}) using {n:,} held-out test observations.\n\n"
        f"The three features with the greatest average impact on credit score predictions "
        f"are: {', '.join(top3)}.\n\n"
        "In the beeswarm plot, each point represents one customer. Red dots indicate "
        "high feature values; blue dots indicate low values. Points to the right of "
        "centre push the predicted score higher; points to the left push it lower.\n\n"
        "The bar chart ranks features by mean |SHAP value| — the average magnitude of "
        "impact regardless of direction. The waterfall plot explains one specific "
        "customer's prediction by decomposing it into individual feature contributions, "
        "showing the path from the baseline score to the final prediction."
    )
    save_text(text, SHAP_DIR / "shap_interpretation.txt")

    print("[shap_analysis] Done.")


def _save_gcf(path) -> None:
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {path}")
