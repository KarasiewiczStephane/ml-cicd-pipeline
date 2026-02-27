"""Configuration loader for environment-specific YAML configs."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(env: str | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file based on the environment.

    Args:
        env: Environment name (dev, staging, prod). Defaults to the
            ENVIRONMENT env var, falling back to 'dev'.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file for the given environment
            does not exist.
    """
    env = env or os.getenv("ENVIRONMENT", "dev")
    project_root = get_project_root()
    config_path = project_root / "configs" / f"{env}.yaml"

    # Fall back to generic config.yaml when env-specific file is missing
    if not config_path.exists():
        config_path = project_root / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config for environment: %s", env)
    return config


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns:
        Path to the project root.
    """
    return Path(__file__).resolve().parent.parent.parent
