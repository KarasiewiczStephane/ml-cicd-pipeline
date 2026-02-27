"""Data loading utilities for the ML pipeline."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def load_data(
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the Iris dataset and split into train/test sets.

    Args:
        test_size: Fraction of data reserved for testing. Read from config
            when not provided.
        random_state: Random seed for reproducibility. Read from config
            when not provided.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names,
        target_names).
    """
    config = load_config()
    test_size = test_size or config["data"]["test_size"]
    random_state = random_state or config["data"]["random_state"]

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(
        "Loaded Iris dataset: train=%d, test=%d samples",
        len(X_train),
        len(X_test),
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        list(iris.feature_names),
        list(iris.target_names),
    )


def load_data_as_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Iris dataset as a pandas DataFrame.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    iris: Any = load_iris(as_frame=True)
    logger.info("Loaded Iris dataset as DataFrame: %d rows", len(iris.data))
    return iris.data, iris.target


def export_sample_data(output_dir: str = "data/sample") -> str:
    """Export the Iris dataset to a CSV file for reproducibility.

    Args:
        output_dir: Directory to write the CSV file into.

    Returns:
        Path to the written CSV file.
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    csv_path = output_path / "iris.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Exported sample data to %s", csv_path)
    return str(csv_path)
