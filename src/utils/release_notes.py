"""Release notes generator for model releases."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


def generate_release_notes(
    metrics_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Generate markdown release notes from evaluation metrics.

    Args:
        metrics_path: Path to the metrics JSON file. Defaults to
            ``metrics/latest_metrics.json``.
        output_path: Path for the output markdown file. Defaults to
            ``metrics/release_notes.md``.

    Returns:
        Path where the release notes were written.
    """
    config = load_config()
    project_root = get_project_root()

    if metrics_path is None:
        metrics_path = str(project_root / config["paths"]["metrics_dir"] / "latest_metrics.json")
    if output_path is None:
        output_path = str(project_root / config["paths"]["metrics_dir"] / "release_notes.md")

    metrics = _load_metrics(metrics_path)
    notes = _format_notes(metrics)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(notes)
    logger.info("Release notes written to %s", out)
    return str(out)


def _load_metrics(path: str) -> dict[str, Any]:
    """Read metrics from a JSON file.

    Args:
        path: Path to the metrics JSON.

    Returns:
        Parsed metrics dictionary.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(p.read_text())


def _format_notes(metrics: dict[str, Any]) -> str:
    """Build the markdown release notes string.

    Args:
        metrics: Evaluation metrics dictionary.

    Returns:
        Formatted markdown string.
    """
    accuracy = metrics["accuracy"]
    report = metrics.get("classification_report", {})
    weighted = report.get("weighted avg", {})

    return (
        f"# Model Release Notes\n\n"
        f"**Date:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}\n\n"
        f"## Performance Metrics\n"
        f"- **Accuracy:** {accuracy:.4f}\n"
        f"- **Precision (weighted):** {weighted.get('precision', 0):.4f}\n"
        f"- **Recall (weighted):** {weighted.get('recall', 0):.4f}\n"
        f"- **F1-score (weighted):** {weighted.get('f1-score', 0):.4f}\n\n"
        f"## Model Information\n"
        f"- Algorithm: RandomForestClassifier\n"
        f"- Dataset: Iris\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_release_notes()
    print(f"Release notes generated: {path}")
