"""Integration tests for deployment health checks.

These tests verify the health check module works end-to-end with a
real trained model. The Docker-based integration tests run only in
CI via the deploy workflow.
"""

from src.data.loader import load_data
from src.deploy.health_check import check_health, check_model_loaded
from src.models.train import save_model, train_model


class TestDeploymentIntegration:
    """End-to-end deployment verification tests."""

    def test_health_check_with_trained_model(self, tmp_path, monkeypatch):
        """Train a model, save it, and verify health check passes."""
        X_train, _, y_train, _, _, _ = load_data()
        model = train_model(X_train, y_train)
        save_model(model, path=str(tmp_path / "model.joblib"))

        monkeypatch.setattr(
            "src.deploy.health_check.get_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.deploy.health_check.load_config",
            lambda: {"model": {"path": "model.joblib"}},
        )

        assert check_health() is True
        status = check_model_loaded()
        assert status["status"] == "healthy"

    def test_health_check_without_model(self, tmp_path, monkeypatch):
        """Verify health check fails gracefully when model is missing."""
        monkeypatch.setattr(
            "src.deploy.health_check.get_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.deploy.health_check.load_config",
            lambda: {"model": {"path": "nonexistent.joblib"}},
        )

        assert check_health() is False
        status = check_model_loaded()
        assert status["status"] == "unhealthy"

    def test_full_pipeline_then_health_check(self, tmp_path, monkeypatch):
        """Run the complete pipeline and verify deployment readiness."""
        X_train, X_test, y_train, y_test, _, _ = load_data()
        model = train_model(X_train, y_train)
        model_path = tmp_path / "model.joblib"
        save_model(model, path=str(model_path))

        # Verify model can predict
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

        monkeypatch.setattr(
            "src.deploy.health_check.get_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.deploy.health_check.load_config",
            lambda: {"model": {"path": "model.joblib"}},
        )

        assert check_health() is True
