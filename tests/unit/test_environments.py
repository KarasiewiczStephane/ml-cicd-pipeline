"""Tests for multi-environment configuration loading."""

from src.utils.config import load_config


class TestEnvironmentConfigs:
    """Verify each environment config loads correctly with expected values."""

    def test_dev_config(self):
        config = load_config(env="dev")
        assert config["environment"] == "dev"
        assert config["performance"]["accuracy_threshold"] == 0.80
        assert config["server"]["debug"] is True
        assert config["logging"]["level"] == "DEBUG"

    def test_staging_config(self):
        config = load_config(env="staging")
        assert config["environment"] == "staging"
        assert config["performance"]["accuracy_threshold"] == 0.85
        assert config["server"]["debug"] is False
        assert config["logging"]["level"] == "INFO"

    def test_prod_config(self):
        config = load_config(env="prod")
        assert config["environment"] == "prod"
        assert config["performance"]["accuracy_threshold"] == 0.90
        assert config["server"]["debug"] is False
        assert config["logging"]["level"] == "WARNING"

    def test_stricter_thresholds_in_higher_envs(self):
        dev = load_config(env="dev")
        staging = load_config(env="staging")
        prod = load_config(env="prod")
        assert (
            dev["performance"]["accuracy_threshold"]
            < staging["performance"]["accuracy_threshold"]
            < prod["performance"]["accuracy_threshold"]
        )

    def test_all_envs_have_same_keys(self):
        dev = load_config(env="dev")
        staging = load_config(env="staging")
        prod = load_config(env="prod")
        assert set(dev.keys()) == set(staging.keys()) == set(prod.keys())

    def test_default_env_fallback(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        config = load_config()
        assert config is not None
