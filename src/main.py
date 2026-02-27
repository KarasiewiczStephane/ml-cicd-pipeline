"""Entry point for the ML CI/CD pipeline.

Orchestrates the full workflow: load data, train model, evaluate, and save
artifacts.
"""

import logging
import sys

from src.data.loader import export_sample_data, load_data
from src.models.evaluate import evaluate_model, save_metrics
from src.models.train import save_model, train_model
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_pipeline() -> dict:
    """Execute the complete ML pipeline.

    Returns:
        Dictionary with evaluation metrics.
    """
    config = load_config()

    logger.info("Starting ML pipeline (env=%s)", config.get("environment", "dev"))

    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()

    # Train
    model = train_model(X_train, y_train)
    save_model(model)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, target_names=target_names)
    save_metrics(metrics)

    logger.info("Pipeline complete: accuracy=%.4f", metrics["accuracy"])

    # Export sample data for reproducibility
    export_sample_data()

    return metrics


def main() -> None:
    """CLI entry point."""
    config = load_config()
    log_level = config.get("logging", {}).get("level", "INFO")
    log_format = config.get("logging", {}).get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(level=getattr(logging, log_level), format=log_format)

    metrics = run_pipeline()
    print(f"Pipeline finished. Accuracy: {metrics['accuracy']:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
