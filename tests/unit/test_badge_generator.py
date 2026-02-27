"""Tests for the badge generator module."""

import json

from src.utils.badge_generator import (
    generate_accuracy_badge,
    generate_badge_url,
    generate_coverage_badge,
    generate_trained_badge,
    update_readme_badges,
)


class TestGenerateBadgeUrl:
    """Tests for generate_badge_url()."""

    def test_basic_url(self):
        url = generate_badge_url("status", "passing", "green")
        assert "img.shields.io/badge" in url
        assert "status" in url
        assert "passing" in url
        assert "green" in url

    def test_encodes_special_chars(self):
        url = generate_badge_url("my label", "99%", "blue")
        assert "my%20label" in url


class TestGenerateAccuracyBadge:
    """Tests for generate_accuracy_badge()."""

    def test_high_accuracy(self, tmp_path):
        metrics = {"accuracy": 0.95}
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(metrics))
        url = generate_accuracy_badge(str(path))
        assert "brightgreen" in url
        assert "95.0" in url

    def test_medium_accuracy(self, tmp_path):
        metrics = {"accuracy": 0.82}
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(metrics))
        url = generate_accuracy_badge(str(path))
        assert "yellow" in url

    def test_low_accuracy(self, tmp_path):
        metrics = {"accuracy": 0.50}
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(metrics))
        url = generate_accuracy_badge(str(path))
        assert "red" in url


class TestGenerateCoverageBadge:
    """Tests for generate_coverage_badge()."""

    def test_high_coverage(self):
        url = generate_coverage_badge(90.0)
        assert "brightgreen" in url
        assert "90" in url

    def test_medium_coverage(self):
        url = generate_coverage_badge(70.0)
        assert "yellow" in url

    def test_low_coverage(self):
        url = generate_coverage_badge(40.0)
        assert "red" in url

    def test_exact_threshold(self):
        url = generate_coverage_badge(80.0)
        assert "brightgreen" in url


class TestGenerateTrainedBadge:
    """Tests for generate_trained_badge()."""

    def test_contains_date(self):
        url = generate_trained_badge()
        assert "blue" in url
        assert "last%20trained" in url


class TestUpdateReadmeBadges:
    """Tests for update_readme_badges()."""

    def test_updates_badge_url(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# Test\n![Coverage](https://old-url)\n")
        update_readme_badges(
            str(readme),
            {"Coverage": "https://new-url"},
        )
        content = readme.read_text()
        assert "https://new-url" in content
        assert "https://old-url" not in content

    def test_preserves_other_content(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# Title\nSome text\n![Badge](https://old)\nMore text\n")
        update_readme_badges(str(readme), {"Badge": "https://new"})
        content = readme.read_text()
        assert "# Title" in content
        assert "Some text" in content
        assert "More text" in content

    def test_no_badges_is_noop(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# Hello\n")
        update_readme_badges(str(readme), None)
        assert readme.read_text() == "# Hello\n"
