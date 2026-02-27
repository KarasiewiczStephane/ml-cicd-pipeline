"""Tests for the config loading utility."""

from pathlib import Path

import pytest

from src.utils.config import get_project_root, load_config


class TestLoadConfig:
    """Tests for load_config()."""

    def test_loads_default_config(self):
        config = load_config()
        assert "data" in config
        assert "model" in config
        assert "logging" in config

    def test_config_has_expected_keys(self):
        config = load_config()
        assert config["data"]["test_size"] == 0.2
        assert config["data"]["random_state"] == 42
        assert config["model"]["n_estimators"] == 100

    def test_missing_env_config_falls_back(self):
        config = load_config(env="nonexistent")
        assert config is not None

    def test_missing_all_configs_raises(self, tmp_path, monkeypatch):
        def fake_get_project_root():
            return tmp_path / "nonexistent"

        monkeypatch.setattr("src.utils.config.get_project_root", fake_get_project_root)
        with pytest.raises(FileNotFoundError):
            load_config(env="totally_missing")


class TestGetProjectRoot:
    """Tests for get_project_root()."""

    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_contains_configs(self):
        root = get_project_root()
        assert (root / "configs").exists()
