"""Tests for the metrics history tracker."""

import json

from src.utils.metrics_tracker import MetricsTracker


def _make_metrics(accuracy: float = 0.95) -> dict:
    return {
        "accuracy": accuracy,
        "classification_report": {
            "weighted avg": {
                "precision": accuracy,
                "recall": accuracy,
                "f1-score": accuracy,
            }
        },
    }


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_empty_history(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        assert tracker.get_latest() is None
        assert tracker.get_summary()["entries"] == 0

    def test_append_metrics(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        tracker.append_metrics(_make_metrics(0.92))
        assert len(tracker.history) == 1
        assert tracker.history[0]["accuracy"] == 0.92

    def test_append_multiple(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        tracker.append_metrics(_make_metrics(0.90))
        tracker.append_metrics(_make_metrics(0.95))
        assert len(tracker.history) == 2

    def test_get_latest(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        tracker.append_metrics(_make_metrics(0.90))
        tracker.append_metrics(_make_metrics(0.95))
        latest = tracker.get_latest()
        assert latest["accuracy"] == 0.95

    def test_get_summary(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        tracker.append_metrics(_make_metrics(0.85))
        tracker.append_metrics(_make_metrics(0.95))
        tracker.append_metrics(_make_metrics(0.90))
        summary = tracker.get_summary()
        assert summary["entries"] == 3
        assert summary["latest_accuracy"] == 0.90
        assert summary["best_accuracy"] == 0.95

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "history.json")
        tracker = MetricsTracker(path)
        tracker.append_metrics(_make_metrics(0.92))

        tracker2 = MetricsTracker(path)
        assert len(tracker2.history) == 1
        assert tracker2.history[0]["accuracy"] == 0.92

    def test_loads_existing(self, tmp_path):
        path = tmp_path / "history.json"
        path.write_text(json.dumps([{"timestamp": "2024-01-01", "accuracy": 0.88}]))
        tracker = MetricsTracker(str(path))
        assert len(tracker.history) == 1

    def test_no_chart_with_single_point(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        tracker.append_metrics(_make_metrics(0.90))
        result = tracker.generate_trend_chart(str(tmp_path / "trend.png"))
        assert result is None

    def test_generates_chart(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        tracker.append_metrics(_make_metrics(0.90))
        tracker.append_metrics(_make_metrics(0.95))
        chart_path = str(tmp_path / "trend.png")
        result = tracker.generate_trend_chart(chart_path)
        assert result == chart_path
        assert (tmp_path / "trend.png").exists()

    def test_entry_has_all_fields(self, tmp_path):
        tracker = MetricsTracker(str(tmp_path / "history.json"))
        tracker.append_metrics(_make_metrics(0.92))
        entry = tracker.history[0]
        assert "timestamp" in entry
        assert "accuracy" in entry
        assert "precision" in entry
        assert "recall" in entry
        assert "f1" in entry
