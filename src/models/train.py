"""Model training pipeline for the ML CI/CD project."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


def create_pipeline(
    n_estimators: int | None = None,
    random_state: int | None = None,
) -> Pipeline:
    """Create an sklearn Pipeline with scaling and a RandomForest classifier.

    Args:
        n_estimators: Number of trees. Read from config when not provided.
        random_state: Random seed. Read from config when not provided.

    Returns:
        An unfitted sklearn Pipeline.
    """
    config = load_config()
    if n_estimators is None:
        n_estimators = config["model"]["n_estimators"]
    if random_state is None:
        random_state = config["model"]["random_state"]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                ),
            ),
        ]
    )
    logger.info(
        "Created pipeline: n_estimators=%d, random_state=%d",
        n_estimators,
        random_state,
    )
    return pipeline


def train_model(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    pipeline: Pipeline | None = None,
) -> Pipeline:
    """Fit the pipeline on training data.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        pipeline: Pre-built pipeline. A new one is created when not given.

    Returns:
        The fitted Pipeline.
    """
    if pipeline is None:
        pipeline = create_pipeline()

    pipeline.fit(X_train, y_train)
    logger.info("Model training complete on %d samples", len(X_train))
    return pipeline


def save_model(model: Pipeline, path: str | None = None) -> str:
    """Persist a trained model to disk.

    Args:
        model: Fitted sklearn Pipeline to save.
        path: Output file path. Read from config when not provided.

    Returns:
        Path where the model was saved.
    """
    config = load_config()
    if path is None:
        path = str(get_project_root() / config["model"]["path"])

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    logger.info("Model saved to %s", output)
    return str(output)


def load_model(path: str | None = None) -> Pipeline:
    """Load a persisted model from disk.

    Args:
        path: Path to the model file. Read from config when not provided.

    Returns:
        The loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    config = load_config()
    if path is None:
        path = str(get_project_root() / config["model"]["path"])

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


if __name__ == "__main__":
    from src.data.loader import load_data

    logging.basicConfig(level=logging.INFO)

    X_train, X_test, y_train, y_test, _, _ = load_data()
    model = train_model(X_train, y_train)
    save_model(model)
    logger.info("Training pipeline complete")
