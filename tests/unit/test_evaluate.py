"""Tests for the model evaluation module."""

import json

from src.models.evaluate import evaluate_model, save_metrics
from src.models.train import train_model


class TestEvaluateModel:
    """Tests for evaluate_model()."""

    def test_returns_accuracy(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        metrics = evaluate_model(model, iris_data["X_test"], iris_data["y_test"])
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_returns_classification_report(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        metrics = evaluate_model(model, iris_data["X_test"], iris_data["y_test"])
        assert "classification_report" in metrics
        assert "weighted avg" in metrics["classification_report"]

    def test_returns_confusion_matrix(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        metrics = evaluate_model(model, iris_data["X_test"], iris_data["y_test"])
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 3

    def test_accuracy_above_threshold(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        metrics = evaluate_model(model, iris_data["X_test"], iris_data["y_test"])
        assert metrics["accuracy"] > 0.8

    def test_with_target_names(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        metrics = evaluate_model(
            model,
            iris_data["X_test"],
            iris_data["y_test"],
            target_names=iris_data["target_names"],
        )
        assert "setosa" in metrics["classification_report"]


class TestSaveMetrics:
    """Tests for save_metrics()."""

    def test_saves_json(self, iris_data, tmp_path):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        metrics = evaluate_model(model, iris_data["X_test"], iris_data["y_test"])
        path = str(tmp_path / "metrics.json")
        save_metrics(metrics, path=path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == metrics["accuracy"]
