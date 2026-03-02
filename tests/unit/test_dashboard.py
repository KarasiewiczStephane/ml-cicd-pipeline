"""Tests for the ML CI/CD pipeline dashboard data generators."""

from unittest.mock import MagicMock, patch

from src.dashboard.app import (
    generate_demo_gate_results,
    generate_demo_metrics_history,
    generate_demo_pipeline_runs,
    generate_demo_registry_history,
    load_real_data,
    render_accuracy_trend,
    render_gate_results,
    render_header,
    render_pipeline_timeline,
    render_registry_table,
    render_summary_metrics,
)


class TestDemoRegistryHistory:
    """Tests for the registry history generator."""

    def test_returns_list(self) -> None:
        data = generate_demo_registry_history()
        assert isinstance(data, list)

    def test_has_entries(self) -> None:
        data = generate_demo_registry_history()
        assert len(data) > 0

    def test_entry_keys(self) -> None:
        data = generate_demo_registry_history()
        required_keys = {
            "version",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "model_path",
            "timestamp",
            "status",
        }
        for entry in data:
            assert required_keys.issubset(entry.keys())

    def test_accuracy_range(self) -> None:
        data = generate_demo_registry_history()
        for entry in data:
            assert 0.0 <= entry["accuracy"] <= 1.0

    def test_last_entry_is_production(self) -> None:
        data = generate_demo_registry_history()
        assert data[-1]["status"] == "production"


class TestDemoMetricsHistory:
    """Tests for the metrics history generator."""

    def test_returns_list(self) -> None:
        data = generate_demo_metrics_history()
        assert isinstance(data, list)

    def test_has_twenty_entries(self) -> None:
        data = generate_demo_metrics_history()
        assert len(data) == 20

    def test_entry_keys(self) -> None:
        data = generate_demo_metrics_history()
        required_keys = {"timestamp", "accuracy", "precision", "recall", "f1"}
        for entry in data:
            assert required_keys.issubset(entry.keys())

    def test_all_scores_bounded(self) -> None:
        data = generate_demo_metrics_history()
        for entry in data:
            for key in ("accuracy", "precision", "recall", "f1"):
                assert 0.0 <= entry[key] <= 1.0


class TestDemoPipelineRuns:
    """Tests for the pipeline runs generator."""

    def test_returns_list(self) -> None:
        data = generate_demo_pipeline_runs()
        assert isinstance(data, list)

    def test_has_entries(self) -> None:
        data = generate_demo_pipeline_runs()
        assert len(data) > 0

    def test_entry_keys(self) -> None:
        data = generate_demo_pipeline_runs()
        required_keys = {"run_id", "stage", "start", "end", "duration_min", "status"}
        for entry in data:
            assert required_keys.issubset(entry.keys())

    def test_has_failed_run(self) -> None:
        data = generate_demo_pipeline_runs()
        statuses = {entry["status"] for entry in data}
        assert "failed" in statuses

    def test_has_success_run(self) -> None:
        data = generate_demo_pipeline_runs()
        statuses = {entry["status"] for entry in data}
        assert "success" in statuses


class TestDemoGateResults:
    """Tests for the gate results generator."""

    def test_returns_list(self) -> None:
        data = generate_demo_gate_results()
        assert isinstance(data, list)

    def test_has_entries(self) -> None:
        data = generate_demo_gate_results()
        assert len(data) == 6

    def test_entry_keys(self) -> None:
        data = generate_demo_gate_results()
        required_keys = {
            "run_id",
            "timestamp",
            "new_accuracy",
            "prod_accuracy",
            "threshold",
            "passed",
        }
        for entry in data:
            assert required_keys.issubset(entry.keys())

    def test_has_pass_and_fail(self) -> None:
        data = generate_demo_gate_results()
        outcomes = {entry["passed"] for entry in data}
        assert True in outcomes
        assert False in outcomes


class TestLoadRealData:
    """Tests for the real data loader fallback."""

    def test_returns_none_tuple_on_import_error(self) -> None:
        metrics, registry = load_real_data()
        assert metrics is None or isinstance(metrics, list)
        assert registry is None or isinstance(registry, list)


@patch("src.dashboard.app.st")
class TestRenderFunctions:
    """Tests for dashboard render functions using mocked Streamlit."""

    def test_render_header(self, mock_st: MagicMock) -> None:
        render_header()
        mock_st.title.assert_called_once()
        mock_st.caption.assert_called_once()

    def test_render_summary_metrics(self, mock_st: MagicMock) -> None:
        mock_st.columns.return_value = [MagicMock() for _ in range(4)]
        registry = generate_demo_registry_history()
        metrics = generate_demo_metrics_history()
        render_summary_metrics(registry, metrics)
        mock_st.columns.assert_called_once_with(4)

    def test_render_accuracy_trend(self, mock_st: MagicMock) -> None:
        metrics = generate_demo_metrics_history()
        render_accuracy_trend(metrics)
        mock_st.subheader.assert_called_once()
        mock_st.plotly_chart.assert_called_once()

    def test_render_registry_table(self, mock_st: MagicMock) -> None:
        registry = generate_demo_registry_history()
        render_registry_table(registry)
        mock_st.subheader.assert_called_once()
        mock_st.dataframe.assert_called_once()

    def test_render_pipeline_timeline(self, mock_st: MagicMock) -> None:
        pipeline = generate_demo_pipeline_runs()
        render_pipeline_timeline(pipeline)
        mock_st.subheader.assert_called_once()
        mock_st.plotly_chart.assert_called_once()

    def test_render_gate_results(self, mock_st: MagicMock) -> None:
        mock_st.columns.return_value = [MagicMock() for _ in range(4)]
        gates = generate_demo_gate_results()
        render_gate_results(gates)
        assert mock_st.columns.call_count == len(gates)
