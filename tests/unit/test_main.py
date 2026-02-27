"""Tests for the main pipeline orchestration module."""

from src.main import run_pipeline


class TestRunPipeline:
    """Tests for run_pipeline()."""

    def test_returns_metrics(self):
        metrics = run_pipeline()
        assert "accuracy" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics

    def test_accuracy_above_threshold(self):
        metrics = run_pipeline()
        assert metrics["accuracy"] > 0.8

    def test_metrics_values_are_valid(self):
        metrics = run_pipeline()
        assert 0.0 <= metrics["accuracy"] <= 1.0
        report = metrics["classification_report"]
        assert "weighted avg" in report
        assert 0.0 <= report["weighted avg"]["f1-score"] <= 1.0
