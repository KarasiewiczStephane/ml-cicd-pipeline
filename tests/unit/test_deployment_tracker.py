"""Tests for the deployment version tracker."""

import json

from src.utils.deployment_tracker import DeploymentTracker


class TestDeploymentTracker:
    """Tests for DeploymentTracker."""

    def test_empty_history(self, tmp_path):
        tracker = DeploymentTracker(str(tmp_path / "history.json"))
        assert tracker.get_latest_tag() is None
        assert tracker.get_previous_tag() is None

    def test_record_deployment(self, tmp_path):
        tracker = DeploymentTracker(str(tmp_path / "history.json"))
        tracker.record_deployment("abc123")
        assert tracker.get_latest_tag() == "abc123"

    def test_get_previous_tag(self, tmp_path):
        tracker = DeploymentTracker(str(tmp_path / "history.json"))
        tracker.record_deployment("v1")
        tracker.record_deployment("v2")
        assert tracker.get_previous_tag() == "v1"
        assert tracker.get_latest_tag() == "v2"

    def test_no_previous_with_single_deployment(self, tmp_path):
        tracker = DeploymentTracker(str(tmp_path / "history.json"))
        tracker.record_deployment("v1")
        assert tracker.get_previous_tag() is None

    def test_environment_filtering(self, tmp_path):
        tracker = DeploymentTracker(str(tmp_path / "history.json"))
        tracker.record_deployment("v1", environment="staging")
        tracker.record_deployment("v2", environment="production")
        assert tracker.get_latest_tag(environment="staging") == "v1"
        assert tracker.get_latest_tag(environment="production") == "v2"

    def test_mark_rollback(self, tmp_path):
        tracker = DeploymentTracker(str(tmp_path / "history.json"))
        tracker.record_deployment("v1")
        tracker.record_deployment("v2")
        tracker.mark_rollback("v1")
        assert len(tracker.history) == 3
        assert tracker.history[-1]["status"] == "rollback"
        assert tracker.history[-1]["tag"] == "v1"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "history.json")
        tracker = DeploymentTracker(path)
        tracker.record_deployment("v1")

        tracker2 = DeploymentTracker(path)
        assert tracker2.get_latest_tag() == "v1"

    def test_loads_existing_history(self, tmp_path):
        path = tmp_path / "history.json"
        path.write_text(
            json.dumps([{"tag": "existing", "environment": "production", "status": "deployed"}])
        )
        tracker = DeploymentTracker(str(path))
        assert tracker.get_latest_tag() == "existing"
