"""Tests for the data validation module."""

import json

import numpy as np
import pandas as pd

from src.data.validator import DataValidator, run_validation


class TestDataValidator:
    """Tests for the DataValidator class."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [5.0, 6.0, 7.0, 8.0],
                "target": [0, 1, 0, 1],
            }
        )

    def test_validate_schema_pass(self):
        validator = DataValidator({})
        df = self._make_df()
        assert validator.validate_schema(df, ["a", "b", "target"]) is True

    def test_validate_schema_fail(self):
        validator = DataValidator({})
        df = self._make_df()
        assert validator.validate_schema(df, ["a", "b", "missing_col"]) is False

    def test_validate_missing_values_pass(self):
        validator = DataValidator({})
        df = self._make_df()
        assert validator.validate_missing_values(df, threshold=0.1) is True

    def test_validate_missing_values_fail(self):
        validator = DataValidator({})
        df = self._make_df()
        df.loc[0, "a"] = None
        df.loc[1, "a"] = None
        assert validator.validate_missing_values(df, threshold=0.1) is False

    def test_validate_row_count_pass(self):
        validator = DataValidator({})
        df = self._make_df()
        assert validator.validate_row_count(df, min_rows=1, max_rows=100) is True

    def test_validate_row_count_fail_too_few(self):
        validator = DataValidator({})
        df = self._make_df()
        assert validator.validate_row_count(df, min_rows=10, max_rows=100) is False

    def test_validate_row_count_fail_too_many(self):
        validator = DataValidator({})
        df = self._make_df()
        assert validator.validate_row_count(df, min_rows=1, max_rows=2) is False

    def test_validate_distribution_pass(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0, 1, 100)
        validator = DataValidator({})
        assert validator.validate_distribution(a, b, threshold=0.05) is True

    def test_validate_distribution_fail(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(5, 1, 100)
        validator = DataValidator({})
        assert validator.validate_distribution(a, b, threshold=0.05) is False

    def test_generate_report(self):
        validator = DataValidator({})
        df = self._make_df()
        validator.validate_schema(df, ["a", "b", "target"])
        validator.validate_missing_values(df)
        report = validator.generate_report()
        assert report["passed"] is True
        assert len(report["results"]) == 2
        assert "timestamp" in report

    def test_generate_report_empty(self):
        validator = DataValidator({})
        report = validator.generate_report()
        assert report["passed"] is False

    def test_save_report(self, tmp_path):
        validator = DataValidator({})
        df = self._make_df()
        validator.validate_schema(df, ["a", "b", "target"])
        path = str(tmp_path / "report.json")
        validator.save_report(path=path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["passed"] is True


class TestRunValidation:
    """Tests for the run_validation() entry point."""

    def test_runs_successfully(self):
        report = run_validation()
        assert report["passed"] is True
        assert len(report["results"]) >= 3
