"""Deployment version tracker for rollback support."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import get_project_root

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_FILE = ".deployment_history.json"


class DeploymentTracker:
    """Tracks deployment versions and supports rollback lookups.

    Args:
        history_path: Path to the deployment history file.
    """

    def __init__(self, history_path: str | None = None) -> None:
        if history_path is None:
            history_path = str(get_project_root() / DEFAULT_HISTORY_FILE)
        self.history_path = Path(history_path)
        self.history = self._load_history()

    def _load_history(self) -> list[dict[str, Any]]:
        """Load deployment history from disk."""
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return []

    def _save_history(self) -> None:
        """Persist deployment history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(self.history, indent=2))
        logger.info("Deployment history saved to %s", self.history_path)

    def record_deployment(self, tag: str, environment: str = "production") -> None:
        """Record a successful deployment.

        Args:
            tag: Docker image tag or commit SHA deployed.
            environment: Target environment name.
        """
        entry = {
            "tag": tag,
            "environment": environment,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": "deployed",
        }
        self.history.append(entry)
        self._save_history()
        logger.info("Recorded deployment: tag=%s, env=%s", tag, environment)

    def get_previous_tag(self, environment: str = "production") -> str | None:
        """Retrieve the tag of the deployment before the latest one.

        Args:
            environment: Target environment to look up.

        Returns:
            Previous deployment tag, or None if no history exists.
        """
        env_history = [h for h in self.history if h["environment"] == environment]
        if len(env_history) < 2:
            return None
        return env_history[-2]["tag"]

    def get_latest_tag(self, environment: str = "production") -> str | None:
        """Retrieve the tag of the most recent deployment.

        Args:
            environment: Target environment to look up.

        Returns:
            Latest deployment tag, or None if no history exists.
        """
        env_history = [h for h in self.history if h["environment"] == environment]
        if not env_history:
            return None
        return env_history[-1]["tag"]

    def mark_rollback(self, tag: str, environment: str = "production") -> None:
        """Record a rollback event.

        Args:
            tag: Tag that was rolled back to.
            environment: Environment where rollback occurred.
        """
        entry = {
            "tag": tag,
            "environment": environment,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": "rollback",
        }
        self.history.append(entry)
        self._save_history()
        logger.info("Recorded rollback: tag=%s, env=%s", tag, environment)
