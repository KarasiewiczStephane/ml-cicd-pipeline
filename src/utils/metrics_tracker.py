"""Metrics history tracker with trend visualization."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks model metrics over time and generates trend charts.

    Args:
        history_path: Path to the history JSON file. Defaults to
            ``metrics/history.json`` under the project root.
    """

    def __init__(self, history_path: str | None = None) -> None:
        if history_path is None:
            config = load_config()
            history_path = str(get_project_root() / config["paths"]["metrics_dir"] / "history.json")
        self.history_path = Path(history_path)
        self.history: list[dict[str, Any]] = self._load_history()

    def _load_history(self) -> list[dict[str, Any]]:
        """Load existing metrics history from disk."""
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return []

    def _save_history(self) -> None:
        """Persist the metrics history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(self.history, indent=2))
        logger.info("Metrics history saved to %s", self.history_path)

    def append_metrics(self, metrics: dict[str, Any]) -> None:
        """Add a new metrics entry to the history.

        Args:
            metrics: Evaluation metrics dictionary from ``evaluate_model``.
        """
        report = metrics.get("classification_report", {})
        weighted = report.get("weighted avg", {})

        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "accuracy": metrics["accuracy"],
            "precision": weighted.get("precision", 0.0),
            "recall": weighted.get("recall", 0.0),
            "f1": weighted.get("f1-score", 0.0),
        }
        self.history.append(entry)
        self._save_history()
        logger.info("Appended metrics entry: accuracy=%.4f", metrics["accuracy"])

    def generate_trend_chart(self, output_path: str | None = None) -> str | None:
        """Generate a performance trend line chart.

        Requires at least two data points. Uses matplotlib for rendering.

        Args:
            output_path: Path for the output PNG. Defaults to
                ``metrics/trend.png``.

        Returns:
            Path to the generated chart, or None if insufficient data.
        """
        if len(self.history) < 2:
            logger.info("Not enough data points for trend chart (need >= 2)")
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if output_path is None:
            config = load_config()
            output_path = str(get_project_root() / config["paths"]["metrics_dir"] / "trend.png")

        recent = self.history[-20:]
        dates = [h["timestamp"][:10] for h in recent]
        accuracy = [h["accuracy"] for h in recent]
        f1 = [h["f1"] for h in recent]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, accuracy, marker="o", label="Accuracy")
        ax.plot(dates, f1, marker="s", label="F1-score")
        ax.set_xlabel("Date")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Trend")
        ax.legend()
        ax.set_ylim(0.0, 1.05)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

        logger.info("Trend chart saved to %s", output_path)
        return output_path

    def get_latest(self) -> dict[str, Any] | None:
        """Return the most recent metrics entry.

        Returns:
            Latest metrics dictionary or None if history is empty.
        """
        if not self.history:
            return None
        return self.history[-1]

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of the metrics history.

        Returns:
            Dictionary with entry count, latest accuracy, and best accuracy.
        """
        if not self.history:
            return {"entries": 0, "latest_accuracy": None, "best_accuracy": None}

        accuracies = [h["accuracy"] for h in self.history]
        return {
            "entries": len(self.history),
            "latest_accuracy": accuracies[-1],
            "best_accuracy": max(accuracies),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    metrics_path = get_project_root() / "metrics" / "latest_metrics.json"
    if not metrics_path.exists():
        logger.error("No metrics file found at %s", metrics_path)
        raise SystemExit(1)

    metrics = json.loads(metrics_path.read_text())
    tracker = MetricsTracker()
    tracker.append_metrics(metrics)
    chart_path = tracker.generate_trend_chart()

    summary = tracker.get_summary()
    logger.info(
        "History: %d entries, best accuracy: %.4f",
        summary["entries"],
        summary["best_accuracy"],
    )
