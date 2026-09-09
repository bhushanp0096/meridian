"""
frontend/components/timeline_view.py
====================================
Timeline decomposition visualizer (Baseline outcome vs Media incremental outcome over time).
Renders 104 weekly intervals with posterior credible intervals.
"""

from typing import Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from frontend.styles import PLOTLY_LAYOUT_DEFAULTS


def render_timeline_view(
    baseline_vs_media: list[dict[str, Any]],
    model_spec: dict[str, Any],
):
    """
    Render weekly time-series decomposition chart with posterior uncertainty ribbons.
    """
    st.subheader("⏱️ Weekly Outcome Decomposition (Baseline vs. Media)")
    st.markdown(
        "Observe how target outcomes decompose week-by-week into organic/baseline volume vs. media-driven volume. "
        "Shaded ribbons represent the **90% posterior credible interval**."
    )

    if not baseline_vs_media:
        st.warning("No timeline decomposition data available.")
        return

    df = pd.DataFrame(baseline_vs_media)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    kpi_name = model_spec.get("kpi_name", "Bookings")
    kpi_units = model_spec.get("kpi_units", "units")

    # Time Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        date_min = df["date"].min().date()
        date_max = df["date"].max().date()
        date_range = st.date_input(
            "Filter Date Range:",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key="timeline_date_picker",
        )
    with col2:
        show_uncertainty = st.checkbox("Show 90% Uncertainty Ribbons", value=True, key="timeline_ci_chk")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df[(df["date"] >= start_d) & (df["date"] <= end_d)].sort_values(by="date")
    else:
        df_filtered = df.sort_values(by="date")

    dates = df_filtered["date"].tolist()
    baseline_m = df_filtered["baseline_mean"].tolist()
    media_m = df_filtered["media_mean"].tolist()
    actuals = df_filtered["actual"].tolist() if "actual" in df_filtered.columns else None

    # Total predicted = baseline + media
    total_pred = [b + m for b, m in zip(baseline_m, media_m)]

    fig = go.Figure()

    # 1. Baseline uncertainty ribbon
    if show_uncertainty and "baseline_ci_lower" in df_filtered.columns:
        b_lo = df_filtered["baseline_ci_lower"].tolist()
        b_hi = df_filtered["baseline_ci_upper"].tolist()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=b_lo,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=b_hi,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.15)",
                name="Baseline 90% CI",
                hoverinfo="skip",
            )
        )

    # 2. Baseline line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=baseline_m,
            mode="lines",
            line=dict(color="#64748B", width=2, dash="dot"),
            name="Baseline Outcome",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Baseline: %{y:,.0f}<extra></extra>",
        )
    )

    # 3. Media line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=media_m,
            mode="lines",
            line=dict(color="#38BDF8", width=2.5),
            name="Media Incremental Outcome",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Media Incremental: %{y:,.0f}<extra></extra>",
        )
    )

    # 4. Total Model Predicted
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=total_pred,
            mode="lines",
            line=dict(color="#10B981", width=2),
            name="Total Model Predicted",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Total Predicted: %{y:,.0f}<extra></extra>",
        )
    )

    # 5. Actual observations (if present)
    if actuals and any(a is not None for a in actuals):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=actuals,
                mode="markers",
                marker=dict(size=4, color="#F59E0B"),
                name=f"Actual {kpi_name}",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"Weekly Historical Breakdown: Baseline vs. Incremental {kpi_name}", x=0),
        xaxis_title="Week",
        yaxis_title=f"{kpi_name} ({kpi_units})",
        height=480,
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")

    st.plotly_chart(fig, use_container_width=True)
