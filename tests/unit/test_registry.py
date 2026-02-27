"""Tests for the model registry module."""

import json

from src.models.registry import ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def _make_metrics(self, accuracy: float = 0.95) -> dict:
        return {
            "accuracy": accuracy,
            "classification_report": {
                "weighted avg": {"precision": 0.95, "recall": 0.95, "f1-score": 0.95}
            },
        }

    def test_loads_empty_registry(self, tmp_path):
        path = str(tmp_path / "registry.json")
        registry = ModelRegistry(registry_path=path)
        assert registry.get_production_metrics() is None
        assert registry.get_history() == []

    def test_loads_existing_registry(self, tmp_path):
        path = tmp_path / "registry.json"
        data = {"production": {"accuracy": 0.9}, "history": []}
        path.write_text(json.dumps(data))
        registry = ModelRegistry(registry_path=str(path))
        assert registry.get_production_metrics()["accuracy"] == 0.9

    def test_compare_no_production_passes(self, tmp_path):
        path = str(tmp_path / "registry.json")
        registry = ModelRegistry(registry_path=path)
        assert registry.compare_with_production(self._make_metrics(0.90), threshold=0.85) is True

    def test_compare_no_production_fails(self, tmp_path):
        path = str(tmp_path / "registry.json")
        registry = ModelRegistry(registry_path=path)
        assert registry.compare_with_production(self._make_metrics(0.50), threshold=0.85) is False

    def test_compare_better_than_production(self, tmp_path):
        path = tmp_path / "registry.json"
        data = {"production": {"accuracy": 0.85}, "history": []}
        path.write_text(json.dumps(data))
        registry = ModelRegistry(registry_path=str(path))
        assert registry.compare_with_production(self._make_metrics(0.90), threshold=0.80) is True

    def test_compare_worse_than_production(self, tmp_path):
        path = tmp_path / "registry.json"
        data = {"production": {"accuracy": 0.95}, "history": []}
        path.write_text(json.dumps(data))
        registry = ModelRegistry(registry_path=str(path))
        assert registry.compare_with_production(self._make_metrics(0.90), threshold=0.80) is False

    def test_compare_equal_accuracy(self, tmp_path):
        path = tmp_path / "registry.json"
        data = {"production": {"accuracy": 0.90}, "history": []}
        path.write_text(json.dumps(data))
        registry = ModelRegistry(registry_path=str(path))
        assert registry.compare_with_production(self._make_metrics(0.90), threshold=0.80) is True

    def test_promote_to_production(self, tmp_path):
        path = str(tmp_path / "registry.json")
        registry = ModelRegistry(registry_path=path)
        registry.promote_to_production(self._make_metrics(0.95), model_path="models/model.joblib")
        assert registry.get_production_metrics()["accuracy"] == 0.95
        assert registry.get_production_metrics()["model_path"] == "models/model.joblib"

    def test_promote_tracks_history(self, tmp_path):
        path = str(tmp_path / "registry.json")
        registry = ModelRegistry(registry_path=path)
        registry.promote_to_production(self._make_metrics(0.90), model_path="v1.joblib")
        registry.promote_to_production(self._make_metrics(0.95), model_path="v2.joblib")
        history = registry.get_history()
        assert len(history) == 1
        assert history[0]["accuracy"] == 0.90

    def test_registry_persists(self, tmp_path):
        path = str(tmp_path / "registry.json")
        registry = ModelRegistry(registry_path=path)
        registry.promote_to_production(self._make_metrics(0.95), model_path="v1.joblib")

        registry2 = ModelRegistry(registry_path=path)
        assert registry2.get_production_metrics()["accuracy"] == 0.95
