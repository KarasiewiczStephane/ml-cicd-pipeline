"""Data validation module with schema, distribution, and quality checks."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates datasets against configurable quality rules.

    Args:
        config: Validation configuration dictionary. Loaded from
            ``configs/config.yaml`` when not provided.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            full_config = load_config()
            config = full_config.get("data_validation", {})
        self.config = config
        self.results: list[dict[str, Any]] = []

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: list[str],
    ) -> bool:
        """Check that all expected columns are present.

        Args:
            df: DataFrame to validate.
            expected_columns: Column names that must exist.

        Returns:
            True if all expected columns are present.
        """
        missing = set(expected_columns) - set(df.columns)
        passed = len(missing) == 0
        self.results.append(
            {
                "check": "schema_validation",
                "passed": passed,
                "details": {"missing_columns": list(missing)},
            }
        )
        logger.info("Schema validation: %s (missing=%s)", "PASS" if passed else "FAIL", missing)
        return passed

    def validate_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.1,
    ) -> bool:
        """Verify that missing-value ratios stay below a threshold.

        Args:
            df: DataFrame to validate.
            threshold: Maximum allowed fraction of missing values per column.

        Returns:
            True if all columns are within the threshold.
        """
        missing_ratio = df.isnull().sum() / len(df)
        passed = bool((missing_ratio <= threshold).all())
        self.results.append(
            {
                "check": "missing_values",
                "passed": passed,
                "details": {"max_missing_ratio": float(missing_ratio.max())},
            }
        )
        logger.info("Missing values check: %s", "PASS" if passed else "FAIL")
        return passed

    def validate_row_count(
        self,
        df: pd.DataFrame,
        min_rows: int = 1,
        max_rows: int = 1_000_000,
    ) -> bool:
        """Assert row count is within bounds.

        Args:
            df: DataFrame to validate.
            min_rows: Minimum acceptable row count.
            max_rows: Maximum acceptable row count.

        Returns:
            True if row count is within [min_rows, max_rows].
        """
        count = len(df)
        passed = min_rows <= count <= max_rows
        self.results.append(
            {
                "check": "row_count",
                "passed": passed,
                "details": {"row_count": count, "min": min_rows, "max": max_rows},
            }
        )
        logger.info("Row count check: %s (%d rows)", "PASS" if passed else "FAIL", count)
        return passed

    def validate_distribution(
        self,
        current: np.ndarray,
        reference: np.ndarray,
        threshold: float = 0.05,
    ) -> bool:
        """Run a two-sample Kolmogorov-Smirnov test for distribution drift.

        Args:
            current: Current data distribution sample.
            reference: Reference (baseline) distribution sample.
            threshold: P-value threshold below which drift is flagged.

        Returns:
            True if the distributions are not significantly different.
        """
        statistic, p_value = stats.ks_2samp(current, reference)
        passed = bool(p_value > threshold)
        self.results.append(
            {
                "check": "distribution_ks_test",
                "passed": passed,
                "details": {"statistic": float(statistic), "p_value": float(p_value)},
            }
        )
        logger.info("Distribution check: %s (p=%.4f)", "PASS" if passed else "FAIL", p_value)
        return passed

    def generate_report(self) -> dict[str, Any]:
        """Build a summary report from all validation results.

        Returns:
            Dictionary with timestamp, individual results, and overall
            pass/fail status.
        """
        report = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "results": self.results,
            "passed": all(r["passed"] for r in self.results) if self.results else False,
        }
        logger.info("Validation report: overall=%s", "PASS" if report["passed"] else "FAIL")
        return report

    def save_report(self, path: str | None = None) -> str:
        """Write the validation report to a JSON file.

        Args:
            path: Output file path. Defaults to
                ``reports/data_validation_report.json``.

        Returns:
            Path where the report was written.
        """
        if path is None:
            reports_dir = get_project_root() / "reports"
        else:
            reports_dir = Path(path).parent

        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "data_validation_report.json" if path is None else Path(path)

        report = self.generate_report()
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("Validation report saved to %s", out_path)
        return str(out_path)


def run_validation() -> dict[str, Any]:
    """Execute the full data validation pipeline.

    Loads the Iris dataset, applies schema, missing-value, and row-count
    checks, then saves the report.

    Returns:
        The generated validation report dictionary.
    """
    from src.data.loader import load_data_as_dataframe

    features, target = load_data_as_dataframe()
    df = features.copy()
    df["target"] = target

    config = load_config()
    val_config = config.get("data_validation", {})

    validator = DataValidator(val_config)

    expected_columns = val_config.get(
        "expected_columns",
        list(features.columns) + ["target"],
    )
    validator.validate_schema(df, expected_columns)
    validator.validate_missing_values(
        df,
        threshold=val_config.get("missing_threshold", 0.1),
    )
    validator.validate_row_count(
        df,
        min_rows=val_config.get("min_rows", 100),
        max_rows=val_config.get("max_rows", 10000),
    )

    report = validator.generate_report()
    validator.save_report()
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_validation()
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"Data validation {status}")
    if not report["passed"]:
        raise SystemExit(1)
