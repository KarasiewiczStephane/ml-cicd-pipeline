"""Streamlit dashboard for the ML CI/CD pipeline.

Visualizes model registry history, accuracy trends, pipeline run timelines,
and performance gate pass/fail indicators using demo or real data.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML CI/CD Pipeline Dashboard",
    page_icon="🔄",
    layout="wide",
)


@st.cache_data
def generate_demo_registry_history() -> list[dict]:
    """Generate synthetic model registry history entries."""
    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    versions = []
    accuracy = 0.82
    for i in range(8):
        accuracy = min(accuracy + 0.015 * (i % 3), 0.98)
        versions.append(
            {
                "version": f"v1.{i}",
                "accuracy": round(accuracy, 4),
                "precision": round(accuracy - 0.02, 4),
                "recall": round(accuracy - 0.01, 4),
                "f1": round(accuracy - 0.015, 4),
                "model_path": f"models/model_v1.{i}.joblib",
                "timestamp": (base_time + timedelta(days=i * 7)).isoformat(),
                "status": "production" if i == 7 else "archived",
            }
        )
    return versions


@st.cache_data
def generate_demo_metrics_history() -> list[dict]:
    """Generate synthetic metrics history entries for trend visualization."""
    base_time = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    entries = []
    accuracy = 0.80
    for i in range(20):
        noise = (i % 4 - 1.5) * 0.008
        accuracy = min(max(accuracy + 0.005 + noise, 0.75), 0.97)
        entries.append(
            {
                "timestamp": (base_time + timedelta(days=i * 3)).isoformat(),
                "accuracy": round(accuracy, 4),
                "precision": round(accuracy - 0.015, 4),
                "recall": round(accuracy - 0.008, 4),
                "f1": round(accuracy - 0.012, 4),
            }
        )
    return entries


@st.cache_data
def generate_demo_pipeline_runs() -> list[dict]:
    """Generate synthetic pipeline run timeline data."""
    base_time = datetime(2025, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
    stages = ["data_validation", "training", "evaluation", "gate_check", "deployment"]
    runs = []
    for i in range(6):
        run_start = base_time + timedelta(days=i * 5)
        success = i != 3
        for j, stage in enumerate(stages):
            stage_start = run_start + timedelta(minutes=j * 12)
            duration = [2, 25, 8, 1, 5][j]
            stage_failed = not success and stage == "gate_check"
            runs.append(
                {
                    "run_id": f"run-{i + 1:03d}",
                    "stage": stage,
                    "start": stage_start.isoformat(),
                    "end": (stage_start + timedelta(minutes=duration)).isoformat(),
                    "duration_min": duration,
                    "status": "failed" if stage_failed else "success",
                }
            )
            if stage_failed:
                break
    return runs


@st.cache_data
def generate_demo_gate_results() -> list[dict]:
    """Generate synthetic performance gate check results."""
    base_time = datetime(2025, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    results = []
    for i in range(6):
        accuracy = 0.85 + 0.02 * (i % 4)
        threshold = 0.88
        passed = accuracy >= threshold
        results.append(
            {
                "run_id": f"run-{i + 1:03d}",
                "timestamp": (base_time + timedelta(days=i * 5)).isoformat(),
                "new_accuracy": round(accuracy, 4),
                "prod_accuracy": round(threshold - 0.01, 4),
                "threshold": threshold,
                "passed": passed,
            }
        )
    return results


def load_real_data() -> tuple[list[dict] | None, list[dict] | None]:
    """Attempt to load real data from the project's metrics tracker and registry.

    Returns:
        Tuple of (metrics_history, registry_history) or (None, None) if unavailable.
    """
    try:
        from src.models.registry import ModelRegistry
        from src.utils.metrics_tracker import MetricsTracker

        tracker = MetricsTracker()
        registry = ModelRegistry()

        metrics_history = tracker.history if tracker.history else None
        registry_history = registry.get_history() if registry.get_history() else None
        return metrics_history, registry_history
    except Exception:
        return None, None


def render_header() -> None:
    """Render the dashboard header with summary metrics."""
    st.title("ML CI/CD Pipeline Dashboard")
    st.caption("Model versioning, performance tracking, and deployment pipeline monitoring")


def render_summary_metrics(registry_data: list[dict], metrics_data: list[dict]) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    latest = registry_data[-1] if registry_data else {}
    prev = registry_data[-2] if len(registry_data) >= 2 else {}

    col1.metric(
        "Current Model",
        latest.get("version", "N/A"),
        delta=f"{latest.get('accuracy', 0) - prev.get('accuracy', 0):+.4f}" if prev else None,
    )
    col2.metric(
        "Accuracy",
        f"{latest.get('accuracy', 0):.2%}",
        delta=f"{latest.get('accuracy', 0) - prev.get('accuracy', 0):+.2%}" if prev else None,
    )
    col3.metric("Total Versions", len(registry_data))

    best_accuracy = max((r["accuracy"] for r in metrics_data), default=0)
    col4.metric("Best Accuracy", f"{best_accuracy:.2%}")


def render_accuracy_trend(metrics_data: list[dict]) -> None:
    """Render accuracy and F1-score trend line chart."""
    st.subheader("Performance Trend")

    df = pd.DataFrame(metrics_data)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line={"color": "#2196F3", "width": 2},
            marker={"size": 6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["f1"],
            mode="lines+markers",
            name="F1 Score",
            line={"color": "#FF9800", "width": 2},
            marker={"size": 6},
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis={"range": [0.7, 1.0]},
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_registry_table(registry_data: list[dict]) -> None:
    """Render the model registry history as a data table."""
    st.subheader("Model Registry History")

    df = pd.DataFrame(registry_data)
    display_cols = ["version", "accuracy", "precision", "recall", "f1", "status", "timestamp"]
    available_cols = [c for c in display_cols if c in df.columns]

    styled_df = df[available_cols].sort_values("timestamp", ascending=False)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_pipeline_timeline(pipeline_runs: list[dict]) -> None:
    """Render pipeline run timeline as a Gantt-style chart."""
    st.subheader("Pipeline Run Timeline")

    df = pd.DataFrame(pipeline_runs)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    color_map = {"success": "#4CAF50", "failed": "#F44336"}

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="run_id",
        color="status",
        hover_data=["stage", "duration_min"],
        color_discrete_map=color_map,
    )
    fig.update_layout(
        height=350,
        yaxis_title="Pipeline Run",
        xaxis_title="Time",
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_gate_results(gate_results: list[dict]) -> None:
    """Render performance gate pass/fail indicators."""
    st.subheader("Performance Gate Results")

    for result in reversed(gate_results):
        status_icon = "PASS" if result["passed"] else "FAIL"
        status_color = "green" if result["passed"] else "red"

        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        col1.markdown(f"**{result['run_id']}**")
        col2.markdown(
            f"New: `{result['new_accuracy']:.4f}` | Prod: `{result['prod_accuracy']:.4f}`"
        )
        col3.markdown(f"Threshold: `{result['threshold']:.2f}`")
        col4.markdown(f":{status_color}[**{status_icon}**]")


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    real_metrics, real_registry = load_real_data()

    use_real = st.sidebar.checkbox("Use real data (if available)", value=False)
    data_source = "demo"

    if use_real and real_metrics and real_registry:
        metrics_data = real_metrics
        registry_data = real_registry
        data_source = "real"
    else:
        metrics_data = generate_demo_metrics_history()
        registry_data = generate_demo_registry_history()

    st.sidebar.info(f"Data source: **{data_source}**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    show_trend = st.sidebar.checkbox("Show performance trend", value=True)
    show_registry = st.sidebar.checkbox("Show registry table", value=True)
    show_pipeline = st.sidebar.checkbox("Show pipeline timeline", value=True)
    show_gates = st.sidebar.checkbox("Show gate results", value=True)

    render_summary_metrics(registry_data, metrics_data)
    st.markdown("---")

    if show_trend:
        render_accuracy_trend(metrics_data)

    col_left, col_right = st.columns(2)

    with col_left:
        if show_registry:
            render_registry_table(registry_data)

    with col_right:
        if show_gates:
            gate_results = generate_demo_gate_results()
            render_gate_results(gate_results)

    if show_pipeline:
        st.markdown("---")
        pipeline_runs = generate_demo_pipeline_runs()
        render_pipeline_timeline(pipeline_runs)


if __name__ == "__main__":
    main()
