# Credit Score Predictor — MSIS 522 HW1

End-to-end machine learning pipeline: EDA, 5 regression models, hyperparameter tuning, SHAP explainability, and an interactive Streamlit web app.

---

## Dataset

| Property | Value |
|----------|-------|
| File | `Data/credit_score.csv` |
| Rows | 1,000 customers |
| Features | 84 (82 numeric + 1 categorical: `CAT_GAMBLING`) |
| Target | `CREDIT_SCORE` (regression, range 300–800) |
| ID column (excluded) | `CUST_ID` |
| Leakage column (excluded) | `DEFAULT` |

---

## Project Structure

```
MSIS_522_HW1/
├── Data/
│   └── credit_score.csv
├── Backend/
│   ├── src/
│   │   ├── config.py            # Central config: paths, grids, constants
│   │   ├── data_loader.py       # Load CSV, split 80/20, save summaries
│   │   ├── preprocessing.py     # ColumnTransformer (impute, scale, OHE)
│   │   ├── eda.py               # EDA plots + interpretation text
│   │   ├── train_baseline.py    # Linear Regression
│   │   ├── train_tree.py        # Decision Tree + GridSearchCV
│   │   ├── train_forest.py      # Random Forest + GridSearchCV
│   │   ├── train_boosting.py    # XGBoost + GridSearchCV
│   │   ├── train_mlp.py         # Keras MLP + early stopping
│   │   ├── evaluate.py          # Metrics + comparison charts
│   │   ├── shap_analysis.py     # SHAP TreeExplainer
│   │   └── utils.py             # Shared I/O helpers
│   ├── outputs/                 # All generated artefacts
│   │   ├── data_summary/        # JSON/CSV overviews, feature metadata
│   │   ├── plots/               # EDA + model diagnostic PNGs
│   │   ├── metrics/             # model_comparison.csv/json, summary.txt
│   │   ├── models/              # Serialised models + preprocessors
│   │   └── shap/                # SHAP plots + feature importance
│   ├── run_pipeline.py          # Single-command orchestrator
│   └── requirements.txt
├── Frontend/
│   ├── app.py                   # Streamlit application
│   ├── utils/
│   │   ├── load_artifacts.py    # Loaders for all artefact types
│   │   ├── prediction.py        # Input assembly + model routing
│   │   └── display_helpers.py   # Streamlit UI components
│   └── requirements.txt
└── README.md
```

---

## How to Run the Backend Pipeline

### 1. Install backend dependencies

```bash
pip install -r Backend/requirements.txt
```

### 2. Run the full pipeline (from repo root)

```bash
python Backend/run_pipeline.py
```

This will:
- Load `Data/credit_score.csv` and split 80/20
- Save dataset summaries → `Backend/outputs/data_summary/`
- Generate all EDA plots → `Backend/outputs/plots/`
- Train all 5 models (Linear Regression, Decision Tree, Random Forest, XGBoost, MLP)
- Tune DT / RF / XGBoost with GridSearchCV (5-fold CV, RMSE scoring)
- Evaluate all models on the held-out test set → `Backend/outputs/metrics/`
- Run SHAP on the best tree model → `Backend/outputs/shap/`

**Expected runtime:** ~5–15 minutes (GridSearchCV dominates on larger grids).

---

## How to Run the Frontend App

### 1. Install frontend dependencies

```bash
pip install -r Frontend/requirements.txt
```

### 2. Launch Streamlit (from repo root)

```bash
streamlit run Frontend/app.py
```

The app opens at `http://localhost:8501`. It **does not retrain models** — it loads all artefacts from `Backend/outputs/`.

---

## Output Artefacts

| Location | Contents |
|----------|----------|
| `Backend/outputs/data_summary/` | `dataset_overview.json`, `describe.csv`, `missing_values.csv`, `feature_ranges.csv`, `feature_defaults.csv`, `plot_interpretations.json` |
| `Backend/outputs/plots/` | EDA PNGs, model comparison charts, residual/prediction plots, MLP history |
| `Backend/outputs/metrics/` | `model_comparison.csv/json`, `model_summary.txt`, `mlp_history.json` |
| `Backend/outputs/models/` | `linear_regression.pkl`, `decision_tree.pkl`, `random_forest.pkl`, `xgboost.pkl`, `mlp_model.keras`, preprocessors, best-param JSONs |
| `Backend/outputs/shap/` | SHAP summary/bar/waterfall PNGs, `shap_top_features.csv`, `shap_metadata.json`, `shap_interpretation.txt` |

---

## Models

| Model | Tuning | Preprocessor |
|-------|--------|--------------|
| Linear Regression | — | Impute + StandardScaler + OHE |
| Decision Tree | GridSearchCV 5-fold | Impute + OHE |
| Random Forest | GridSearchCV 5-fold | Impute + OHE |
| XGBoost | GridSearchCV 5-fold (3+ hyperparams) | Impute + OHE |
| MLP (Keras) | EarlyStopping | Impute + StandardScaler + OHE |

---

## Deployment (Streamlit Community Cloud)

1. Push the full repo (including `Backend/outputs/`) to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect repo.
3. Set **Main file path** to `Frontend/app.py`.
4. Set **Requirements file** to `Frontend/requirements.txt`.
5. Ensure `Backend/outputs/` is committed (use Git LFS for large `.pkl` / `.keras` files if needed).

> The app must have all artefacts committed. It does **not** run the backend pipeline on the cloud.
