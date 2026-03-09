"""Shared utility helpers used across the pipeline."""
from __future__ import annotations

import json
import joblib
import numpy as np
from pathlib import Path


def ensure_dirs(*dirs: Path) -> None:
    """Create directories (and parents) if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"  Saved → {path}")


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"  Saved → {path}")


def load_joblib(path: Path):
    return joblib.load(path)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"  Saved → {path}")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serialisable: {type(obj)}")
