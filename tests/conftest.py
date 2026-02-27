"""Shared test fixtures for the ML CI/CD pipeline test suite."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@pytest.fixture()
def iris_data():
    """Provide a train/test split of the Iris dataset."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(iris.feature_names),
        "target_names": list(iris.target_names),
    }


@pytest.fixture()
def sample_predictions():
    """Provide sample predictions and ground truth for metric tests."""
    return {
        "y_true": np.array([0, 0, 1, 1, 2, 2]),
        "y_pred": np.array([0, 0, 1, 1, 2, 2]),
    }
