"""Model evaluation module for computing and reporting metrics."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a trained model on test data.

    Args:
        model: Fitted sklearn Pipeline.
        X_test: Test feature matrix.
        y_test: Test target array.
        target_names: Human-readable class names for the report.

    Returns:
        Dictionary containing accuracy, classification report, and
        confusion matrix.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=target_names,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, predictions)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    logger.info("Evaluation complete: accuracy=%.4f", accuracy)
    return metrics


def save_metrics(metrics: dict[str, Any], path: str | None = None) -> str:
    """Write evaluation metrics to a JSON file.

    Args:
        metrics: Metrics dictionary from ``evaluate_model``.
        path: Output file path. Defaults to ``metrics/latest_metrics.json``
            under the project root.

    Returns:
        Path where metrics were written.
    """
    if path is None:
        config = load_config()
        metrics_dir = get_project_root() / config["paths"]["metrics_dir"]
    else:
        metrics_dir = Path(path).parent

    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "latest_metrics.json" if path is None else Path(path)
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", out_path)
    return str(out_path)


if __name__ == "__main__":
    from src.data.loader import load_data
    from src.models.train import load_model

    logging.basicConfig(level=logging.INFO)

    _, X_test, _, y_test, _, target_names = load_data()
    model = load_model()
    metrics = evaluate_model(model, X_test, y_test, target_names=target_names)
    save_metrics(metrics)
    logger.info("Evaluation pipeline complete")
