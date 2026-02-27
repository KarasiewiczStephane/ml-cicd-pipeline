"""Tests for the training cache module."""

import json

from src.utils.cache import compute_hash, should_retrain, update_cache


class TestComputeHash:
    """Tests for compute_hash()."""

    def test_returns_hex_string(self):
        digest = compute_hash()
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_deterministic(self):
        h1 = compute_hash()
        h2 = compute_hash()
        assert h1 == h2

    def test_different_paths_different_hash(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        h1 = compute_hash([str(tmp_path / "a.txt")])
        (tmp_path / "a.txt").write_text("world")
        h2 = compute_hash([str(tmp_path / "a.txt")])
        assert h1 != h2


class TestShouldRetrain:
    """Tests for should_retrain()."""

    def test_no_cache_returns_true(self, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        assert should_retrain(cache_file=cache_file) is True

    def test_matching_cache_returns_false(self, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        update_cache(cache_file=cache_file)
        assert should_retrain(cache_file=cache_file) is False

    def test_stale_cache_returns_true(self, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        (tmp_path / "cache.json").write_text(json.dumps({"hash": "stale_hash"}))
        assert should_retrain(cache_file=cache_file) is True


class TestUpdateCache:
    """Tests for update_cache()."""

    def test_creates_cache_file(self, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        update_cache(cache_file=cache_file)
        data = json.loads((tmp_path / "cache.json").read_text())
        assert "hash" in data
        assert len(data["hash"]) == 64
