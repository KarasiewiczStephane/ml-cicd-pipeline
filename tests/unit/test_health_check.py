"""Tests for the health check module."""

from src.deploy.health_check import check_health, check_model_loaded
from src.models.train import save_model, train_model


class TestCheckHealth:
    """Tests for check_health()."""

    def test_returns_true_with_model(self, iris_data, tmp_path, monkeypatch):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        model_path = tmp_path / "model.joblib"
        save_model(model, path=str(model_path))

        monkeypatch.setattr(
            "src.deploy.health_check.get_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.deploy.health_check.load_config",
            lambda: {"model": {"path": "model.joblib"}},
        )
        assert check_health() is True

    def test_returns_false_without_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.deploy.health_check.get_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.deploy.health_check.load_config",
            lambda: {"model": {"path": "missing.joblib"}},
        )
        assert check_health() is False


class TestCheckModelLoaded:
    """Tests for check_model_loaded()."""

    def test_healthy_status(self, iris_data, tmp_path, monkeypatch):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        model_path = tmp_path / "model.joblib"
        save_model(model, path=str(model_path))

        monkeypatch.setattr(
            "src.deploy.health_check.get_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.deploy.health_check.load_config",
            lambda: {"model": {"path": "model.joblib"}},
        )
        result = check_model_loaded()
        assert result["status"] == "healthy"
        assert result["model_loaded"] is True

    def test_unhealthy_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.deploy.health_check.get_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.deploy.health_check.load_config",
            lambda: {"model": {"path": "missing.joblib"}},
        )
        result = check_model_loaded()
        assert result["status"] == "unhealthy"
        assert result["model_loaded"] is False
