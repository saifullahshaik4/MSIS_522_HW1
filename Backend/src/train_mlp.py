"""Train a Keras MLP regressor on scaled/preprocessed data."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    MODELS_DIR, METRICS_DIR, PLOTS_DIR,
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_VALIDATION_SPLIT,
    MLP_EARLY_STOPPING_PATIENCE, RANDOM_STATE,
)
from .utils import save_json, ensure_dirs


def train_mlp(
    X_train: pd.DataFrame,
    y_train,
    preprocessor_scaled,
) -> object:
    """Preprocess, build, train, and save a Keras Sequential MLP.

    The model is saved standalone (no preprocessing baked in).
    Preprocessing is handled by the saved preprocessor_scaled.pkl.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential   # type: ignore[attr-defined]
    from tensorflow.keras.layers import Dense, Dropout  # type: ignore[attr-defined]
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[attr-defined]

    ensure_dirs(MODELS_DIR, METRICS_DIR, PLOTS_DIR)

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Transform once
    X_arr = preprocessor_scaled.transform(X_train)
    y_arr = np.array(y_train, dtype=np.float32)

    n_features = X_arr.shape[1]
    print(f"[train_mlp] Input features after preprocessing: {n_features}")

    model = Sequential([
        Dense(128, activation="relu", input_shape=(n_features,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=MLP_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("[train_mlp] Training …")
    history = model.fit(
        X_arr, y_arr,
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        validation_split=MLP_VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(str(MODELS_DIR / "mlp_model.keras"))
    print(f"  Saved → {MODELS_DIR / 'mlp_model.keras'}")

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    save_json(hist_dict, METRICS_DIR / "mlp_history.json")

    _plot_history(hist_dict)
    print("[train_mlp] Done.")
    return model


def _plot_history(hist: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist["loss"]) + 1)

    axes[0].plot(epochs, hist["loss"],     label="Train MSE")
    axes[0].plot(epochs, hist["val_loss"], label="Val MSE")
    axes[0].set_title("MLP Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    if "mae" in hist:
        axes[1].plot(epochs, hist["mae"],     label="Train MAE")
        axes[1].plot(epochs, hist["val_mae"], label="Val MAE")
        axes[1].set_title("MLP MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.suptitle("MLP Training History", fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "mlp_training_history.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
