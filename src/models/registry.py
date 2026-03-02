"""Model registry for tracking production model metrics and versioning."""

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Tracks model versions and their metrics for production comparison.

    Args:
        registry_path: Path to the registry JSON file. Defaults to
            ``metrics/registry.json`` under the project root.
    """

    def __init__(self, registry_path: str | None = None) -> None:
        if registry_path is None:
            config = load_config()
            registry_path = str(
                get_project_root() / config["paths"]["metrics_dir"] / "registry.json"
            )
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load the registry from disk or return a default structure."""
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {"production": None, "history": []}

    def _save_registry(self) -> None:
        """Persist the registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.registry, indent=2))
        logger.info("Registry saved to %s", self.registry_path)

    def get_production_metrics(self) -> dict[str, Any] | None:
        """Return the current production model metrics.

        Returns:
            Metrics dictionary or None if no model is in production.
        """
        return self.registry.get("production")

    def compare_with_production(
        self,
        new_metrics: dict[str, Any],
        threshold: float | None = None,
    ) -> bool:
        """Check if new metrics pass the performance gate.

        The gate passes when the new accuracy meets or exceeds both the
        configured threshold *and* the current production accuracy.

        Args:
            new_metrics: Metrics from the candidate model.
            threshold: Minimum accuracy threshold. Read from config when
                not provided.

        Returns:
            True if the new model passes the gate.
        """
        if threshold is None:
            config = load_config()
            threshold = config["performance"]["accuracy_threshold"]

        new_accuracy = new_metrics["accuracy"]
        prod = self.get_production_metrics()

        if prod is None:
            passed = new_accuracy >= threshold
            logger.info(
                "No production model; gate %s (accuracy=%.4f, threshold=%.4f)",
                "PASS" if passed else "FAIL",
                new_accuracy,
                threshold,
            )
            return passed

        prod_accuracy = prod["accuracy"]
        passed = new_accuracy >= prod_accuracy and new_accuracy >= threshold
        logger.info(
            "Performance gate %s (new=%.4f, prod=%.4f, threshold=%.4f)",
            "PASS" if passed else "FAIL",
            new_accuracy,
            prod_accuracy,
            threshold,
        )
        return passed

    def promote_to_production(
        self,
        metrics: dict[str, Any],
        model_path: str,
    ) -> None:
        """Promote a model to production by updating the registry.

        Args:
            metrics: Evaluation metrics for the new production model.
            model_path: Path to the model artifact.
        """
        current_prod = self.registry.get("production")
        if current_prod is not None:
            self.registry["history"].append(current_prod)

        self.registry["production"] = {
            **metrics,
            "model_path": model_path,
        }
        self._save_registry()
        logger.info("Model promoted to production: accuracy=%.4f", metrics["accuracy"])

    def get_history(self) -> list[dict[str, Any]]:
        """Return the list of previous production models.

        Returns:
            List of historical production model entries.
        """
        return self.registry.get("history", [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    registry = ModelRegistry()
    metrics_path = get_project_root() / "metrics" / "latest_metrics.json"

    if not metrics_path.exists():
        logger.error("No metrics file found at %s", metrics_path)
        raise SystemExit(1)

    metrics = json.loads(metrics_path.read_text())

    if not registry.compare_with_production(metrics):
        logger.error(
            "Performance gate FAILED: accuracy %.4f below threshold",
            metrics["accuracy"],
        )
        raise SystemExit(1)

    logger.info("Performance gate PASSED: accuracy %.4f", metrics["accuracy"])
    registry.promote_to_production(metrics, model_path="models/model.joblib")
