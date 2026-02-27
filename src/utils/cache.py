"""Training cache for skipping redundant model retraining.

Uses content hashing of source and data directories to determine
whether the training pipeline needs to run again.
"""

import hashlib
import json
import logging
from pathlib import Path

from src.utils.config import get_project_root

logger = logging.getLogger(__name__)

DEFAULT_WATCHED_PATHS = ["src/models/", "src/data/", "data/"]
DEFAULT_CACHE_FILE = ".training_cache.json"


def compute_hash(paths: list[str] | None = None) -> str:
    """Compute a SHA-256 hash over the contents of the watched paths.

    Args:
        paths: Directories and files to include in the hash. Defaults
            to model source, data source, and data directories.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    if paths is None:
        paths = DEFAULT_WATCHED_PATHS

    project_root = get_project_root()
    hasher = hashlib.sha256()

    for rel_path in sorted(paths):
        p = project_root / rel_path
        if p.is_file():
            hasher.update(p.read_bytes())
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f):
                    hasher.update(f.read_bytes())

    digest = hasher.hexdigest()
    logger.debug("Computed content hash: %s", digest[:12])
    return digest


def should_retrain(cache_file: str | None = None) -> bool:
    """Check whether training should run based on content changes.

    Args:
        cache_file: Path to the cache state file. Defaults to
            ``.training_cache.json`` in the project root.

    Returns:
        True if code or data has changed since the last training run.
    """
    if cache_file is None:
        cache_file = str(get_project_root() / DEFAULT_CACHE_FILE)

    current_hash = compute_hash()
    cache_path = Path(cache_file)

    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("hash") == current_hash:
            logger.info("Cache hit — no changes detected, training can be skipped")
            return False

    logger.info("Cache miss — changes detected, training required")
    return True


def update_cache(cache_file: str | None = None) -> None:
    """Write the current content hash to the cache file.

    Args:
        cache_file: Path to the cache state file.
    """
    if cache_file is None:
        cache_file = str(get_project_root() / DEFAULT_CACHE_FILE)

    current_hash = compute_hash()
    Path(cache_file).write_text(json.dumps({"hash": current_hash}))
    logger.info("Training cache updated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if should_retrain():
        print("skip=false")
    else:
        print("skip=true")
