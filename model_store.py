"""Save / load the trained model together with the feature contract it was trained on."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import sklearn

from .features import FEATURE_COLUMNS
from .model import TARGET_MODE
from .training import TrainingResult


class ModelNotFoundError(FileNotFoundError):
    pass


class ModelMismatchError(RuntimeError):
    pass


def save_artifacts(result: TrainingResult, model_path: Path, metrics_path: Path) -> None:
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result.model,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_mode": TARGET_MODE,
        "lag_hours": result.metrics["model"]["lag_hours"],
        "max_horizon_hours": result.metrics["model"]["max_horizon_hours"],
        "last_observation": result.metrics["last_observation"],
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, model_path, compress=3)
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=False) + "\n")


def load_bundle(model_path: Path) -> dict:
    """Load the model bundle; refuse if its feature contract differs from the current code."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(f"No trained model at {model_path}. Run `python -m scripts.train_model` first.")
    bundle = joblib.load(model_path)
    if bundle["feature_columns"] != list(FEATURE_COLUMNS):
        raise ModelMismatchError(
            "Saved model was trained with different features than the current code; retrain the model.")
    if bundle.get("target_mode") != TARGET_MODE:  # e.g. a model saved before the target formulation changed
        raise ModelMismatchError(
            f"Saved model uses target mode {bundle.get('target_mode')!r}, code expects {TARGET_MODE!r}; retrain the model.")
    if bundle["sklearn_version"].split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ModelMismatchError(
            f"Model trained with scikit-learn {bundle['sklearn_version']} but running {sklearn.__version__}; retrain.")
    return bundle


def load_metrics(metrics_path: Path) -> dict:
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise ModelNotFoundError(f"No metrics file at {metrics_path}. Run `python -m scripts.train_model` first.")
    return json.loads(metrics_path.read_text())
