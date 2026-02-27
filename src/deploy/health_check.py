"""Health check module for deployment verification."""

import logging
from typing import Any

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


def check_health() -> bool:
    """Verify that the model is loadable and can produce predictions.

    Returns:
        True if the model loads and runs inference successfully.
    """
    try:
        import joblib

        config = load_config()
        model_path = get_project_root() / config["model"]["path"]

        if not model_path.exists():
            logger.warning("Model file not found: %s", model_path)
            return False

        model = joblib.load(model_path)
        # Smoke test with a single Iris-shaped sample
        test_input = [[5.1, 3.5, 1.4, 0.2]]
        prediction = model.predict(test_input)

        if len(prediction) != 1:
            logger.warning("Unexpected prediction shape: %s", prediction)
            return False

        logger.info("Health check passed: prediction=%s", prediction)
        return True
    except Exception:
        logger.exception("Health check failed")
        return False


def check_model_loaded() -> dict[str, Any]:
    """Return a structured health status for API responses.

    Returns:
        Dictionary with health status and model state.
    """
    healthy = check_health()
    return {"status": "healthy" if healthy else "unhealthy", "model_loaded": healthy}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = check_model_loaded()
    print(result)
    if not result["model_loaded"]:
        raise SystemExit(1)
