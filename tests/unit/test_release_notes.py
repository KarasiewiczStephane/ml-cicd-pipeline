"""Tests for the release notes generator."""

import json

from src.utils.release_notes import _format_notes, generate_release_notes


class TestFormatNotes:
    """Tests for _format_notes()."""

    def _make_metrics(self) -> dict:
        return {
            "accuracy": 0.9667,
            "classification_report": {
                "weighted avg": {
                    "precision": 0.97,
                    "recall": 0.9667,
                    "f1-score": 0.9665,
                }
            },
        }

    def test_contains_accuracy(self):
        notes = _format_notes(self._make_metrics())
        assert "0.9667" in notes

    def test_contains_headers(self):
        notes = _format_notes(self._make_metrics())
        assert "# Model Release Notes" in notes
        assert "## Performance Metrics" in notes
        assert "## Model Information" in notes

    def test_contains_weighted_avg(self):
        notes = _format_notes(self._make_metrics())
        assert "Precision" in notes
        assert "Recall" in notes
        assert "F1-score" in notes

    def test_handles_missing_weighted_avg(self):
        metrics = {"accuracy": 0.90, "classification_report": {}}
        notes = _format_notes(metrics)
        assert "0.9000" in notes


class TestGenerateReleaseNotes:
    """Tests for generate_release_notes()."""

    def test_writes_file(self, tmp_path):
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "accuracy": 0.95,
                    "classification_report": {
                        "weighted avg": {
                            "precision": 0.95,
                            "recall": 0.95,
                            "f1-score": 0.95,
                        }
                    },
                }
            )
        )
        output_path = str(tmp_path / "notes.md")
        result = generate_release_notes(metrics_path=str(metrics_path), output_path=output_path)
        assert result == output_path
        with open(output_path) as f:
            content = f.read()
        assert "0.9500" in content

    def test_missing_metrics_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            generate_release_notes(
                metrics_path=str(tmp_path / "nonexistent.json"),
                output_path=str(tmp_path / "notes.md"),
            )
