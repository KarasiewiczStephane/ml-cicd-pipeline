"""Dynamic badge URL generator for README status indicators."""

import json
import logging
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


def generate_badge_url(label: str, value: str, color: str) -> str:
    """Build a shields.io badge URL.

    Args:
        label: Badge label text.
        value: Badge value text.
        color: Badge color name (e.g. 'brightgreen', 'yellow', 'red').

    Returns:
        Full shields.io badge URL.
    """
    label_encoded = urllib.parse.quote(label)
    value_encoded = urllib.parse.quote(value)
    return f"https://img.shields.io/badge/{label_encoded}-{value_encoded}-{color}"


def generate_accuracy_badge(metrics_path: str | None = None) -> str:
    """Create a badge URL showing model accuracy.

    Args:
        metrics_path: Path to the metrics JSON. Defaults to
            ``metrics/latest_metrics.json``.

    Returns:
        Shields.io badge URL for accuracy.
    """
    if metrics_path is None:
        config = load_config()
        metrics_path = str(
            get_project_root() / config["paths"]["metrics_dir"] / "latest_metrics.json"
        )

    metrics = _load_metrics(metrics_path)
    accuracy = metrics["accuracy"] * 100
    color = _accuracy_color(accuracy)
    return generate_badge_url("accuracy", f"{accuracy:.1f}%25", color)


def generate_coverage_badge(coverage_pct: float) -> str:
    """Create a badge URL showing test coverage.

    Args:
        coverage_pct: Coverage percentage (0-100).

    Returns:
        Shields.io badge URL for coverage.
    """
    color = "brightgreen" if coverage_pct >= 80 else "yellow" if coverage_pct >= 60 else "red"
    return generate_badge_url("coverage", f"{coverage_pct:.0f}%25", color)


def generate_trained_badge() -> str:
    """Create a badge URL showing the last training date.

    Returns:
        Shields.io badge URL with today's date.
    """
    date = datetime.now(tz=timezone.utc).strftime("%Y--%m--%d")
    return generate_badge_url("last trained", date, "blue")


def update_readme_badges(
    readme_path: str | None = None,
    badges: dict[str, str] | None = None,
) -> None:
    """Update badge image URLs in the README file.

    Looks for markdown image tags with matching alt text and replaces
    the URL portion.

    Args:
        readme_path: Path to README.md. Defaults to project root.
        badges: Mapping of alt-text to new badge URLs.
    """
    if readme_path is None:
        readme_path = str(get_project_root() / "README.md")
    if badges is None:
        return

    path = Path(readme_path)
    if not path.exists():
        logger.warning("README not found at %s", readme_path)
        return

    content = path.read_text()
    for alt_text, url in badges.items():
        pattern = rf"(!\[{re.escape(alt_text)}\])\([^)]+\)"
        replacement = rf"\1({url})"
        content = re.sub(pattern, replacement, content)

    path.write_text(content)
    logger.info("Updated %d badge(s) in %s", len(badges), readme_path)


def _load_metrics(path: str) -> dict[str, Any]:
    """Load metrics from a JSON file."""
    return json.loads(Path(path).read_text())


def _accuracy_color(accuracy_pct: float) -> str:
    """Determine badge color from accuracy percentage."""
    if accuracy_pct >= 90:
        return "brightgreen"
    if accuracy_pct >= 80:
        return "yellow"
    return "red"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print("Accuracy:", generate_accuracy_badge())
    except FileNotFoundError:
        logger.warning("Metrics file not found, skipping accuracy badge")
    print("Coverage:", generate_coverage_badge(85.0))
    print("Trained:", generate_trained_badge())
