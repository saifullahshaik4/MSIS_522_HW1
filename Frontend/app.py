"""
Credit Score Predictor — Streamlit App
=======================================
MSIS 522 HW1 · End-to-end data science workflow on credit_score.csv
All models are loaded from Backend/outputs — no retraining occurs.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT    = FRONTEND_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from Frontend.utils import load_artifacts as la
from Frontend.utils.prediction import (
    MODEL_DISPLAY_LIST, MODEL_FILE_MAP, TREE_MODEL_KEYS,
    build_input_row, predict, compute_local_shap,
)

# ---------------------------------------------------------------------------
# Page config  — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Score Predictor | MSIS 522",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — polished card look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* metric card borders */
div[data-testid="metric-container"] {
    background: #f0f4fb;
    border: 1px solid #d0daf0;
    border-radius: 10px;
    padding: 14px 16px;
}
/* tab font */
button[data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
/* sidebar accent */
section[data-testid="stSidebar"] { background: #f7f9fc; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load all artefacts (cached so they're only read once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_all():
    overview        = la.load_dataset_overview()
    feature_cols    = la.load_feature_columns()
    feature_types   = la.load_feature_types()
    cat_values      = la.load_categorical_values()
    interpretations = la.load_plot_interpretations()
    feat_ranges     = la.load_feature_ranges()
    feat_defaults   = la.load_feature_defaults()
    comparison_df   = la.load_model_comparison()
    shap_top        = la.load_shap_top_features()
    shap_meta       = la.load_shap_metadata()
    shap_interp     = la.load_shap_interpretation()
    mlp_history     = la.load_mlp_history()
    best_model      = la.load_best_model_name()
    best_tree       = la.load_best_tree_model_name()
    return (
        overview, feature_cols, feature_types, cat_values,
        interpretations, feat_ranges, feat_defaults,
        comparison_df, shap_top, shap_meta, shap_interp,
        mlp_history, best_model, best_tree,
    )


@st.cache_resource(show_spinner=False)
def _load_preprocessors():
    return la.load_preprocessor("scaled"), la.load_preprocessor("tree")


with st.spinner("Loading artefacts …"):
    (
        overview, feature_cols, feature_types, cat_values,
        interpretations, feat_ranges, feat_defaults,
        comparison_df, shap_top, shap_meta, shap_interp,
        mlp_history, best_model_key, best_tree_key,
    ) = _load_all()
    preprocessor_scaled, preprocessor_tree = _load_preprocessors()

numeric_features     = feature_types.get("numeric", [])
categorical_features = feature_types.get("categorical", [])

MODEL_DISPLAY_MAP = {
    "linear_regression": "Linear Regression",
    "decision_tree":     "Decision Tree",
    "random_forest":     "Random Forest",
    "xgboost":           "XGBoost",
    "mlp":               "MLP (Neural Net)",
}
REVERSE_MODEL_MAP = {v: k for k, v in MODEL_DISPLAY_MAP.items()}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/000000/money.png", width=60)
st.sidebar.title("Credit Score Predictor")
st.sidebar.markdown("*MSIS 522 HW1 · End-to-end ML pipeline*")
st.sidebar.markdown("---")

SECTIONS = [
    "🏠 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction",
]
section = st.sidebar.radio("Navigate to", SECTIONS, label_visibility="collapsed")
st.sidebar.markdown("---")

# Quick stats in sidebar
best_row     = comparison_df.loc[best_model_key]
best_display = MODEL_DISPLAY_MAP.get(best_model_key, best_model_key)
st.sidebar.markdown(f"""
**Best model:** {best_display}  
**RMSE:** `{best_row['rmse']:.2f}`  
**R²:** `{best_row['r2']:.4f}`
""")
st.sidebar.markdown("---")
st.sidebar.caption("All models trained **offline**. App loads saved artefacts — no retraining.")

# ---------------------------------------------------------------------------
# Global page header
# ---------------------------------------------------------------------------
st.title("💳 Credit Score Predictor")
st.markdown(
    "> **MSIS 522 HW1** — Complete end-to-end machine learning pipeline: "
    "exploratory analytics · 5 regression models · hyperparameter tuning · "
    "SHAP explainability · interactive prediction tool."
)
st.markdown("---")


# ===========================================================================
# HELPERS
# ===========================================================================

def _metric(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


def _metric_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, (lbl, val, hlp) in zip(cols, items):
        with col:
            _metric(lbl, val, hlp)


def _image(title: str, path: Path, interp: str = "", expander: bool = False) -> None:
    if not path.exists():
        st.warning(f"Plot not found: `{path.name}`")
        return
    if expander:
        with st.expander(title, expanded=True):
            st.image(str(path), use_container_width=True)
            if interp:
                st.caption(interp)
    else:
        st.subheader(title)
        st.image(str(path), use_container_width=True)
        if interp:
            st.info(interp)


def _fmt_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready copy with renamed columns and display names."""
    out = df.copy()
    if "display_name" in out.columns:
        out = out.rename(columns={"display_name": "Model"})
        out = out.set_index("Model") if "Model" in out.columns else out
    out = out.rename(columns={
        "rmse": "RMSE", "mae": "MAE", "r2": "R²",
        "explained_variance": "Expl. Var.", "mape": "MAPE (%)"
    })
    return out


# ===========================================================================
# SECTION 1 — Executive Summary
# ===========================================================================
if section == SECTIONS[0]:
    st.header("Executive Summary")
    st.markdown("*Project overview and best results at a glance*")
    st.markdown("---")

    _metric_row([
        ("Dataset Rows",  f"{overview['total_rows']:,}",   "Total customers in credit_score.csv"),
        ("Total Features", str(overview["n_features"]),    f"{overview['n_numeric_features']} numeric + {overview['n_categorical_features']} categorical"),
        ("Target: CREDIT_SCORE", "Regression",             "Predict continuous CREDIT_SCORE (300–800). Higher = better creditworthiness."),
        ("Best Model",    best_display,                    "Lowest RMSE on held-out 20% test set"),
        ("Best RMSE",     f"{best_row['rmse']:.2f}",       "Root Mean Squared Error on test set"),
        ("Best R²",       f"{best_row['r2']:.4f}",         "Coefficient of Determination (1 = perfect)"),
    ])

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("What Was Built")
        st.markdown(f"""
| Stage | Description |
|-------|-------------|
| **1. Data** | `credit_score.csv` — 1,000 customers, 84 features, 80 / 20 split |
| **2. EDA** | Distribution, correlation, heatmap, scatter plots, categorical analysis |
| **3. Preprocessing** | Median imputation · StandardScaler (LR/MLP) · OneHotEncoding (CAT_GAMBLING) |
| **4. Models** | Linear Regression · Decision Tree · Random Forest · XGBoost · Keras MLP |
| **5. Tuning** | GridSearchCV, 5-fold CV, RMSE scoring (DT · RF · XGBoost) |
| **6. Evaluation** | RMSE · MAE · R² · Explained Variance · MAPE on held-out test set |
| **7. Explainability** | SHAP TreeExplainer on `{best_tree_key}` |
| **8. App** | Streamlit — loads saved artefacts, interactive prediction |

**Target:** `CREDIT_SCORE` — a continuous credit-risk score ranging **300–800** (higher = better creditworthiness). Accurate prediction enables lenders to automate credit decisions, price loans appropriately, and reduce default risk while expanding access to qualified borrowers.  
The best-performing model is **{best_display}** with RMSE `{best_row['rmse']:.2f}` and R² `{best_row['r2']:.4f}`.
""")

    with col_r:
        st.subheader("Target Variable Stats")
        st.markdown(f"""
| Statistic | Value |
|-----------|-------|
| Mean | **{overview['target_mean']:.1f}** |
| Median | **{overview['target_median']:.1f}** |
| Std Dev | **{overview['target_std']:.1f}** |
| Min | **{overview['target_min']:.0f}** |
| Max | **{overview['target_max']:.0f}** |
| Missing | **{overview['n_missing_total']:,}** |
""")
        st.markdown("&nbsp;")
        st.image(str(la.get_plot_path("target_distribution.png")), use_container_width=True)

    st.markdown("---")
    st.subheader("All Models — Performance at a Glance")

    disp_df = _fmt_metrics_df(comparison_df.reset_index())
    if "display_name" in disp_df.columns:
        disp_df = disp_df.drop(columns=["display_name"])

    # Rebuild with display names as index
    disp_df2 = comparison_df.copy()
    if "display_name" in disp_df2.columns:
        disp_df2.index = disp_df2["display_name"]
        disp_df2 = disp_df2.drop(columns=["display_name"])
    disp_df2.index.name = "Model"
    disp_df2 = disp_df2.rename(columns={
        "rmse": "RMSE", "mae": "MAE", "r2": "R²",
        "explained_variance": "Expl. Var.", "mape": "MAPE (%)"
    })

    def _hl(s: pd.Series) -> list[str]:
        best = s.min() if s.name in ("RMSE", "MAE", "MAPE (%)") else s.max()
        return ["background-color: #c8e6c9; font-weight: bold"
                if v == best else "" for v in s]

    st.dataframe(
        disp_df2.style.apply(_hl, axis=0),
        use_container_width=True,
        height=220,
    )

    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        st.image(str(la.get_plot_path("model_rmse_comparison.png")), use_container_width=True)
    with col_d:
        st.image(str(la.get_plot_path("model_r2_comparison.png")), use_container_width=True)


# ===========================================================================
# SECTION 2 — Descriptive Analytics
# ===========================================================================
elif section == SECTIONS[1]:
    st.header("Descriptive Analytics")
    st.markdown("*All plots generated from `credit_score.csv` before any modelling*")
    st.markdown("---")

    _image(
        "Dataset Summary Dashboard",
        la.get_plot_path("summary_dashboard.png"),
        interpretations.get("summary_dashboard", ""),
    )
    st.markdown("---")

    _image(
        "Credit Score Distribution",
        la.get_plot_path("target_distribution.png"),
        interpretations.get("target_distribution", ""),
    )
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        _image(
            "Missing Values per Column",
            la.get_plot_path("missing_values_top20.png"),
            interpretations.get("missing_values_top20", ""),
        )
    with c2:
        _image(
            "Top 20 Features — Correlation with Target",
            la.get_plot_path("correlation_to_target_top20.png"),
            interpretations.get("correlation_to_target_top20", ""),
        )
    st.markdown("---")

    _image(
        "Correlation Heatmap — Top 15 Features + Target",
        la.get_plot_path("top_feature_heatmap.png"),
        interpretations.get("top_feature_heatmap", ""),
    )
    st.markdown("---")

    # Categorical breakdown
    for cat_col in categorical_features:
        _image(
            f"Credit Score by {cat_col}",
            la.get_plot_path(f"credit_score_by_category_{cat_col}.png"),
            interpretations.get(f"credit_score_by_category_{cat_col}", ""),
        )
        st.markdown("---")

    # Scatter plots — 2 per row
    st.subheader("Scatter Plots — Top 4 Correlated Features vs Credit Score")
    srow1 = st.columns(2)
    srow2 = st.columns(2)
    for idx, (col, i) in enumerate([(srow1[0], 1), (srow1[1], 2), (srow2[0], 3), (srow2[1], 4)]):
        with col:
            _image(
                f"Feature #{i}",
                la.get_plot_path(f"scatter_top_feature_{i}.png"),
                interpretations.get(f"scatter_top_feature_{i}", ""),
                expander=True,
            )


# ===========================================================================
# SECTION 3 — Model Performance
# ===========================================================================
elif section == SECTIONS[2]:
    st.header("Model Performance")
    st.markdown("*All metrics computed on the held-out 20% test split*")
    st.markdown("---")

    # ---- Comparison charts ------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        _image("RMSE Comparison (lower is better)", la.get_plot_path("model_rmse_comparison.png"))
    with c2:
        _image("R² Comparison (higher is better)",  la.get_plot_path("model_r2_comparison.png"))
    st.markdown("---")

    # ---- MLP training history ---------------------------------------------
    _image("MLP Training History", la.get_plot_path("mlp_training_history.png"),
           "MSE (left) and MAE (right) across epochs for the Keras neural network. "
           "Early stopping prevented overfitting; the validation curves guided the optimal stopping epoch.")
    st.markdown("---")

    # ---- Metrics table ----------------------------------------------------
    st.subheader("Full Metrics Table")
    disp_df = comparison_df.copy()
    if "display_name" in disp_df.columns:
        disp_df.index = disp_df["display_name"]
        disp_df = disp_df.drop(columns=["display_name"])
    disp_df.index.name = "Model"
    disp_df = disp_df.rename(columns={
        "rmse": "RMSE", "mae": "MAE", "r2": "R²",
        "explained_variance": "Expl. Var.", "mape": "MAPE (%)"
    })

    def _hl2(s: pd.Series) -> list[str]:
        if s.name in ("RMSE", "MAE", "MAPE (%)"):
            best = s.min()
        elif s.name in ("R²", "Expl. Var."):
            best = s.max()
        else:
            return [""] * len(s)
        return ["background-color: #c8e6c9; font-weight: bold"
                if v == best else "" for v in s]

    st.dataframe(
        disp_df.style.apply(_hl2, axis=0).format("{:.4f}", subset=disp_df.columns),
        use_container_width=True,
    )
    st.markdown("---")

    # ---- Per-model diagnostics -------------------------------------------
    st.subheader("Per-Model Diagnostics")
    chosen_disp = st.selectbox(
        "Select a model to inspect",
        list(MODEL_DISPLAY_MAP.values()),
        index=list(MODEL_DISPLAY_MAP.keys()).index(best_model_key),
    )
    chosen_key = REVERSE_MODEL_MAP[chosen_disp]

    if chosen_key in comparison_df.index:
        row = comparison_df.loc[chosen_key]
        _metric_row([
            ("RMSE",            f"{row['rmse']:.4f}",              "Root Mean Squared Error"),
            ("MAE",             f"{row['mae']:.4f}",               "Mean Absolute Error"),
            ("R²",              f"{row['r2']:.4f}",                "Coefficient of Determination"),
            ("Explained Var.",  f"{row['explained_variance']:.4f}","Explained Variance Score"),
            ("MAPE (%)",        f"{row['mape']:.2f}%",             "Mean Absolute Percentage Error"),
        ])
        st.markdown("&nbsp;")

    ca, cb = st.columns(2)
    with ca:
        _image(
            f"Predicted vs Actual — {chosen_disp}",
            la.get_plot_path(f"pred_vs_actual_{chosen_key}.png"),
        )
    with cb:
        _image(
            f"Residuals — {chosen_disp}",
            la.get_plot_path(f"residuals_{chosen_key}.png"),
        )

    params = la.load_model_params(chosen_key)
    if params:
        with st.expander(f"Best Hyperparameters — {chosen_disp} (from GridSearchCV)"):
            col_p1, col_p2 = st.columns([1, 2])
            with col_p1:
                if "best_params" in params:
                    p_df = pd.DataFrame(
                        list(params["best_params"].items()), columns=["Hyperparameter", "Value"]
                    )
                    st.dataframe(p_df, hide_index=True, use_container_width=True)
                    if "cv_rmse" in params:
                        st.caption(f"CV RMSE (5-fold): **{params['cv_rmse']:.4f}**")
            with col_p2:
                st.json(params)

    st.markdown("---")
    with st.expander("Written Performance Summary"):
        st.text(la.load_model_summary_text())


# ===========================================================================
# SECTION 4 — Explainability & Interactive Prediction
# ===========================================================================
elif section == SECTIONS[3]:
    st.header("Explainability & Interactive Prediction")
    st.markdown("---")

    tab_shap, tab_pred = st.tabs(["🔍 SHAP Explainability", "🎯 Interactive Prediction"])

    # -----------------------------------------------------------------------
    # Tab A — SHAP
    # -----------------------------------------------------------------------
    with tab_shap:
        st.markdown(
            f"SHAP analysis was applied to **{MODEL_DISPLAY_MAP.get(best_tree_key, best_tree_key)}** "
            f"— the best tree-based model (lowest RMSE among Decision Tree, Random Forest, XGBoost)."
        )

        if shap_interp:
            with st.expander("Interpretation", expanded=True):
                st.markdown(shap_interp)
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            _image(
                "SHAP Summary — Beeswarm",
                la.get_shap_path("shap_summary.png"),
                "Each dot is one customer. Red = high feature value, blue = low. "
                "Position on x-axis shows direction and magnitude of impact on predicted score.",
            )
        with c2:
            _image(
                "SHAP Feature Importance — Bar",
                la.get_shap_path("shap_bar.png"),
                "Mean absolute SHAP value per feature across all sampled test observations. "
                "The longer the bar, the more influential the feature on average.",
            )
        st.markdown("---")

        _image(
            "SHAP Waterfall — Single Customer",
            la.get_shap_path("shap_waterfall.png"),
            "Step-by-step decomposition of one customer's predicted credit score. "
            "Starting from the baseline (average prediction), each feature pushes the "
            "final score higher (red) or lower (blue).",
        )
        st.markdown("---")

        st.subheader("Top 15 Features by Mean |SHAP Value|")
        if not shap_top.empty:
            shap_disp = shap_top.copy()
            shap_disp.columns = ["Feature", "Mean |SHAP|"]
            shap_disp["Rank"] = range(1, len(shap_disp) + 1)
            shap_disp = shap_disp[["Rank", "Feature", "Mean |SHAP|"]]
            st.dataframe(
                shap_disp.style.background_gradient(subset=["Mean |SHAP|"], cmap="Blues"),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("SHAP artefacts not found — run the backend pipeline first.")

    # -----------------------------------------------------------------------
    # Tab B — Interactive Prediction
    # -----------------------------------------------------------------------
    with tab_pred:

        # ------------------------------------------------------------------
        # Helper: readable feature descriptions
        # ------------------------------------------------------------------
        FEAT_DESC = {
            "INCOME":             "Annual income ($)",
            "SAVINGS":            "Total savings ($)",
            "DEBT":               "Total outstanding debt ($)",
            "R_SAVINGS_INCOME":   "Savings ÷ Income ratio",
            "R_DEBT_INCOME":      "Debt ÷ Income ratio (key risk factor)",
            "R_DEBT_SAVINGS":     "Debt ÷ Savings ratio",
            "R_EXPENDITURE":      "Total expenditure ratio",
            "R_GAMBLING_INCOME":  "Gambling spend ÷ Income",
            "R_TAX_DEBT":         "Tax payments ÷ Debt",
            "R_ENTERTAINMENT":    "Entertainment spend ratio",
            "T_GAMBLING_12":      "Gambling spend — last 12 months ($)",
            "R_DEBT_SAVINGS":     "Debt ÷ Savings",
            "R_TRAVEL_DEBT":      "Travel spend ÷ Debt",
            "R_GROCERIES":        "Groceries spend ratio",
            "T_EXPENDITURE_6":    "Total expenditure — last 6 months ($)",
            "T_TAX_12":           "Tax payments — last 12 months ($)",
            "R_CLOTHING_INCOME":  "Clothing spend ÷ Income",
            "T_HOUSING_12":       "Housing costs — last 12 months ($)",
            "R_UTILITIES_DEBT":   "Utilities spend ÷ Debt",
            "CAT_GAMBLING":       "Gambling activity level",
            "CAT_DEBT":           "Has significant debt (0/1)",
            "CAT_CREDIT_CARD":    "Has credit card (0/1)",
            "CAT_MORTGAGE":       "Has mortgage (0/1)",
            "CAT_SAVINGS_ACCOUNT":"Has savings account (0/1)",
            "CAT_DEPENDENTS":     "Has dependents (0/1)",
        }

        # Categorise all numeric features
        def _feature_group(f: str) -> str:
            if f in ("INCOME", "SAVINGS", "DEBT",
                     "R_SAVINGS_INCOME", "R_DEBT_INCOME", "R_DEBT_SAVINGS"):
                return "Core Financials"
            if f.startswith("T_") and f.endswith("_12"):  return "Annual Spend ($)"
            if f.startswith("T_") and f.endswith("_6"):   return "Semi-Annual Spend ($)"
            if f.endswith("_INCOME"):                      return "Ratio to Income"
            if f.endswith("_DEBT"):                        return "Ratio to Debt"
            if f.endswith("_SAVINGS"):                     return "Ratio to Savings"
            if f.startswith("CAT_"):                       return "Binary Flags"
            return "Other Ratios"

        # Dataset mean score for comparison
        DATASET_MEAN = overview.get("target_mean", 587.0)

        st.markdown("""
**How it works:** Adjust values below for the features that matter most, then click **Predict Credit Score**.  
All remaining features are auto-filled with training-set medians. Works with all 5 trained models.
""")
        st.markdown("---")

        # ---- Determine SHAP-top features -----------------------------------
        if not shap_top.empty and "feature" in shap_top.columns:
            all_shap_feats = shap_top["feature"].tolist()
            primary_feats  = [f for f in all_shap_feats if f in feature_cols][:12]
        else:
            primary_feats = feat_ranges["std"].nlargest(12).index.tolist()

        primary_num = [f for f in primary_feats if f in numeric_features]
        primary_cat = [f for f in primary_feats if f in categorical_features]
        if not primary_cat:
            primary_cat = categorical_features[:]   # always expose categorical

        # All remaining numeric features (not in primary)
        secondary_num = [f for f in numeric_features if f not in primary_num]

        # =========================================================
        # STEP 1 — Model selector
        # =========================================================
        st.subheader("Step 1 — Choose a model")
        MODEL_DESCRIPTIONS = {
            "Linear Regression": "Fast baseline. Assumes linear feature relationships. Interpretable coefficients.",
            "Decision Tree":     "Single tree with GridSearchCV-tuned depth. Highly interpretable splits.",
            "Random Forest":     "200-tree ensemble. Robust to noise, strong on tabular data.",
            "XGBoost":           "Best model (RMSE {:.2f}). Gradient-boosted ensemble with 5-param tuning.".format(
                                    comparison_df.loc[best_model_key, "rmse"] if best_model_key in comparison_df.index else 0),
            "MLP (Neural Net)": "Keras neural net: 3 hidden layers (128→64→32). Adam + MSE + EarlyStopping.",
        }
        col_sel, col_desc = st.columns([1, 2])
        with col_sel:
            chosen_model = st.selectbox(
                "Model",
                MODEL_DISPLAY_LIST,
                index=MODEL_DISPLAY_LIST.index(
                    MODEL_DISPLAY_MAP.get(best_model_key, MODEL_DISPLAY_LIST[3])
                ),
                label_visibility="collapsed",
            )
        with col_desc:
            st.info(MODEL_DESCRIPTIONS.get(chosen_model, ""))

        # Warn if MLP selected but TF not available in this environment
        if chosen_model == "MLP (Neural Net)":
            _mlp_available = la.load_keras_model() is not None
            if not _mlp_available:
                st.warning(
                    "⚠️ **MLP (Neural Net) is not available** in this deployment because "
                    "TensorFlow is not installed (it exceeds Streamlit Cloud's package limits). "
                    "All four other models work normally — please select one of those to predict."
                )

        st.markdown("---")

        # =========================================================
        # STEP 2 — Feature inputs
        # =========================================================
        st.subheader("Step 2 — Adjust feature values")

        user_inputs: dict = {}

        # ---- Reset button -----------------------------------------------
        if st.button("↺  Reset all to training defaults", key="reset_btn"):
            for key in list(st.session_state.keys()):
                if key.startswith(("sl_", "ni_", "cat_")):
                    del st.session_state[key]
            st.rerun()

        # ---- Categorical selectboxes ------------------------------------
        st.markdown("#### Categorical / Binary Features")
        all_cat_feats = categorical_features[:]   # CAT_GAMBLING always first
        # Also show binary int flags (CAT_DEBT, CAT_CREDIT_CARD, etc.)
        binary_flags  = [f for f in numeric_features if f.startswith("CAT_")]
        cat_row       = st.columns(min(len(all_cat_feats) + len(binary_flags), 6))
        col_i = 0
        for feat in all_cat_feats:
            cats = cat_values.get(feat, ["No", "Low", "High"])
            fdef_idx = 0
            with cat_row[col_i % len(cat_row)]:
                lbl = FEAT_DESC.get(feat, feat)
                user_inputs[feat] = st.selectbox(lbl, cats, key=f"cat_{feat}")
            col_i += 1
        for feat in binary_flags:
            fdef = int(round(float(feat_defaults.loc[feat, "default"]))) if feat in feat_defaults.index else 0
            with cat_row[col_i % len(cat_row)]:
                lbl = FEAT_DESC.get(feat, feat)
                user_inputs[feat] = st.selectbox(lbl, [0, 1], index=fdef, key=f"cat_{feat}")
            col_i += 1

        st.markdown("&nbsp;")

        # ---- Primary numeric sliders (top SHAP features) ----------------
        st.markdown(f"#### Top {len(primary_num)} Features by SHAP Importance")
        st.caption("These features have the greatest impact on CREDIT_SCORE predictions.")

        for row_start in range(0, len(primary_num), 3):
            row_feats = primary_num[row_start : row_start + 3]
            cols_ui   = st.columns(3)
            for col_ui, feat in zip(cols_ui, row_feats):
                with col_ui:
                    fmin = float(feat_ranges.loc[feat, "min"])    if feat in feat_ranges.index else 0.0
                    fmax = float(feat_ranges.loc[feat, "max"])    if feat in feat_ranges.index else 1.0
                    fmed = float(feat_defaults.loc[feat, "default"]) if feat in feat_defaults.index else fmin
                    fdef = float(np.clip(fmed, fmin, fmax))
                    lbl  = FEAT_DESC.get(feat, feat)

                    if abs(fmax - fmin) < 1e-9:
                        val = st.number_input(lbl, value=fdef, key=f"ni_{feat}",
                                              help=f"Range: {fmin:.3f} – {fmax:.3f} | Median: {fmed:.3f}")
                    else:
                        step = round((fmax - fmin) / 200, 6)
                        val = st.slider(lbl, min_value=fmin, max_value=fmax,
                                        value=fdef, step=step, format="%.4f",
                                        key=f"sl_{feat}",
                                        help=f"Median: {fmed:.4f} | Range: [{fmin:.3f}, {fmax:.3f}]")
                    user_inputs[feat] = val

        st.markdown("&nbsp;")

        # ---- Advanced: all remaining features by category ---------------
        with st.expander(f"Advanced: customize all {len(secondary_num)} additional numeric features"):
            st.caption(
                "These features are grouped by type. Adjust any to explore their impact on the prediction. "
                "Unchanged features keep their training-set median."
            )
            # Group secondary features
            grp_map: dict[str, list[str]] = {}
            for f in secondary_num:
                grp_map.setdefault(_feature_group(f), []).append(f)

            for grp_name, grp_feats in sorted(grp_map.items()):
                st.markdown(f"**{grp_name}** ({len(grp_feats)} features)")
                adv_cols = st.columns(4)
                for ci, feat in enumerate(grp_feats):
                    with adv_cols[ci % 4]:
                        fmin = float(feat_ranges.loc[feat, "min"])    if feat in feat_ranges.index else 0.0
                        fmax = float(feat_ranges.loc[feat, "max"])    if feat in feat_ranges.index else 1.0
                        fmed = float(feat_defaults.loc[feat, "default"]) if feat in feat_defaults.index else fmin
                        fdef = float(np.clip(fmed, fmin, fmax))
                        val  = st.number_input(
                            feat, value=fdef,
                            min_value=fmin, max_value=fmax,
                            step=round(max((fmax - fmin) / 100, 1e-6), 6),
                            format="%.4f",
                            key=f"adv_{feat}",
                            help=f"Range: [{fmin:.3f}, {fmax:.3f}] | Median: {fmed:.4f}",
                            label_visibility="visible",
                        )
                        user_inputs[feat] = val

        st.markdown("---")

        # =========================================================
        # STEP 3 — Predict
        # =========================================================
        st.subheader("Step 3 — Predict")
        predict_col, _ = st.columns([2, 1])
        with predict_col:
            run_pred = st.button(
                "🔮  Predict Credit Score",
                use_container_width=True,
                type="primary",
            )

        if run_pred:
            with st.spinner("Computing prediction …"):
                try:
                    row_df = build_input_row(
                        user_inputs, feature_cols, feat_defaults, cat_values
                    )
                    score = predict(chosen_model, row_df, preprocessor_scaled, preprocessor_tree)
                    score_int   = int(round(np.clip(score, 300, 800)))
                    score_pct   = (score_int - 300) / 500   # 300–800 range

                    if   score_int >= 740: tier, color, bg = "Excellent", "#1b5e20", "#e8f5e9"
                    elif score_int >= 670: tier, color, bg = "Good",      "#0d47a1", "#e3f2fd"
                    elif score_int >= 580: tier, color, bg = "Fair",      "#e65100", "#fff3e0"
                    else:                  tier, color, bg = "Poor",      "#b71c1c", "#ffebee"

                    delta = score_int - int(round(DATASET_MEAN))
                    delta_str = f"+{delta}" if delta >= 0 else str(delta)
                    delta_label = "above" if delta >= 0 else "below"

                    # ---- Score card -----------------------------------------
                    st.markdown(f"""
<div style="background:{bg}; border:2px solid {color}; border-radius:14px;
    padding:22px 30px; margin:10px 0 18px 0;">
  <div style="display:flex; align-items:center; gap:24px;">
    <div style="text-align:center; min-width:120px;">
      <div style="font-size:52px; font-weight:900; color:{color}; line-height:1">{score_int}</div>
      <div style="font-size:14px; color:#555">out of 800</div>
    </div>
    <div>
      <div style="font-size:22px; font-weight:700; color:{color}">{tier} Credit</div>
      <div style="font-size:14px; color:#444; margin-top:4px">
        {delta_str} points {delta_label} the dataset average ({int(round(DATASET_MEAN))})
      </div>
      <div style="font-size:13px; color:#666; margin-top:4px">
        Model: <b>{chosen_model}</b>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                    # ---- Score bar (300–800) ---------------------------------
                    tiers_html = (
                        '<div style="display:flex; gap:0; border-radius:6px; overflow:hidden; height:16px; margin:4px 0 14px 0;">'
                        '<div style="flex:0.28; background:#ffcdd2;" title="Poor (300–579)"></div>'
                        '<div style="flex:0.20; background:#ffe0b2;" title="Fair (580–669)"></div>'
                        '<div style="flex:0.14; background:#bbdefb;" title="Good (670–739)"></div>'
                        '<div style="flex:0.12; background:#c8e6c9;" title="Excellent (740–800)"></div>'
                        '</div>'
                    )
                    st.markdown(
                        f"<div style='font-size:12px;color:#888;margin-bottom:2px'>"
                        f"300 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        f"Poor &nbsp;&nbsp; Fair &nbsp;&nbsp; Good &nbsp;&nbsp; Excellent"
                        f" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 800</div>"
                        f"{tiers_html}",
                        unsafe_allow_html=True,
                    )
                    st.progress(float(np.clip(score_pct, 0, 1)))
                    st.markdown("&nbsp;")

                    # ---- Details columns ------------------------------------
                    col_inp, col_notes = st.columns([1, 1])

                    with col_inp:
                        st.markdown("**Your input values**")
                        # Separate categorical from numeric for cleaner display
                        inp_rows = []
                        for feat, val in user_inputs.items():
                            median_val = (
                                float(feat_defaults.loc[feat, "default"])
                                if feat in feat_defaults.index else "N/A"
                            )
                            changed = (
                                str(val) != str(median_val)
                                if isinstance(val, str)
                                else abs(float(val) - float(median_val)) > 1e-6
                                if isinstance(median_val, float) else False
                            )
                            inp_rows.append({
                                "Feature": FEAT_DESC.get(feat, feat),
                                "Your Value": val,
                                "Median Default": median_val,
                                "Changed?": "Yes" if changed else "—",
                            })
                        inp_df = pd.DataFrame(inp_rows)
                        st.dataframe(
                            inp_df.style.apply(
                                lambda col: [
                                    "color: #c62828; font-weight:bold"
                                    if v == "Yes" else "" for v in col
                                ] if col.name == "Changed?" else [""] * len(col),
                                axis=0,
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )

                    with col_notes:
                        st.markdown("**Prediction summary**")
                        n_changed = sum(
                            1 for r in inp_rows if r["Changed?"] == "Yes"
                        )
                        st.markdown(f"""
| Property | Value |
|----------|-------|
| Model | **{chosen_model}** |
| Predicted score | **{score_int}** / 800 |
| Credit tier | **{tier}** |
| Dataset average | **{int(round(DATASET_MEAN))}** |
| Difference | **{delta_str} pts** {delta_label} average |
| Features you changed | **{n_changed}** of {len(user_inputs)} shown |
| Auto-filled (median) | **{len(feature_cols) - len(user_inputs)}** features |
""")
                        st.caption(
                            "Score tiers: Poor < 580 · Fair 580–669 · "
                            "Good 670–739 · Excellent ≥ 740"
                        )

                    # ---- Local SHAP waterfall --------------------------------
                    st.markdown("---")
                    chosen_key_pred = MODEL_FILE_MAP.get(chosen_model, "")
                    if chosen_key_pred in TREE_MODEL_KEYS:
                        with st.expander(
                            f"🔎 Why {score_int}? — Local SHAP explanation ({chosen_model})",
                            expanded=True,
                        ):
                            with st.spinner("Computing local SHAP …"):
                                try:
                                    feat_names_tree = la.load_preprocessed_feature_names("tree")
                                    shap_result = compute_local_shap(
                                        chosen_model, row_df, feat_names_tree
                                    )
                                    if shap_result is not None:
                                        exp, base_val = shap_result
                                        import shap as _shap
                                        import matplotlib
                                        matplotlib.use("Agg")
                                        import matplotlib.pyplot as plt
                                        _shap.plots.waterfall(exp, max_display=14, show=False)
                                        fig = plt.gcf()
                                        fig.suptitle(
                                            f"CREDIT_SCORE = {score_int} — feature contributions",
                                            fontsize=11, fontweight="bold", y=1.02,
                                        )
                                        st.pyplot(fig, use_container_width=True)
                                        plt.close("all")
                                        st.caption(
                                            f"Starting from the baseline score of **{base_val:.1f}** "
                                            "(dataset average), each feature pushes the prediction "
                                            "up (red) or down (blue) to reach **{:.1f}**.".format(score)
                                        )
                                    else:
                                        st.info("Local SHAP could not be computed.")
                                except Exception as shap_err:
                                    st.info(f"Local SHAP unavailable: {shap_err}")
                    else:
                        st.info(
                            "💡 **Local SHAP explanation** is available for tree-based models "
                            "(Decision Tree, Random Forest, XGBoost). "
                            "Switch to one of those models to see a personalised waterfall chart "
                            "explaining exactly why this score was predicted."
                        )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "📊 **Credit Score Predictor** | MSIS 522 HW1 | "
    "Models trained offline — artefacts loaded from `Backend/outputs/` — no retraining at runtime. "
    "Built with Streamlit · scikit-learn · XGBoost · SHAP · (MLP trained with TensorFlow/Keras)."
)
