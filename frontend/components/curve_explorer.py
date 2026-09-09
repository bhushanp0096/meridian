"""
frontend/components/curve_explorer.py
=====================================
Response / saturation curve explorer with shaded posterior uncertainty bands
and historical spend reference marker.
"""

from typing import Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from frontend.styles import CHANNEL_COLORS, PLOTLY_LAYOUT_DEFAULTS, get_channel_color


def render_curve_explorer(
    response_curves: dict[str, list[dict[str, Any]]],
    spend_summary: list[dict[str, Any]],
    model_spec: dict[str, Any],
):
    """
    Render response and saturation curve visualizer with shaded credible intervals.
    """
    st.subheader("📈 Response & Saturation Curves")
    st.markdown(
        "Examine diminishing returns across channels. "
        "The **shaded band denotes the 90% posterior credible interval**, "
        "and the **dashed marker indicates current historical spend**."
    )

    if not response_curves:
        st.warning("No response curves available.")
        return

    channels = sorted(list(response_curves.keys()))
    spend_map = {item["channel"]: item.get("total_spend", 0.0) for item in spend_summary}
    kpi_name = model_spec.get("kpi_name", "Bookings")
    kpi_units = model_spec.get("kpi_units", "units")

    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Curve Controls**")
        selected_channel = st.selectbox(
            "Select Channel:",
            options=channels,
            index=0,
            key="curve_channel_select",
        )

        x_axis_mode = st.radio(
            "X-Axis Representation:",
            options=["Spend Multiplier (0% – 200%)", "Absolute Spend (₹)"],
            index=1,
            key="curve_xaxis_radio",
        )

        st.markdown("---")
        # Display channel taxonomy & quick facts
        tax = model_spec.get("channel_taxonomy", {}).get(selected_channel, {})
        cat = tax.get("category", "General")
        metric_label = tax.get("execution_metric", "spend").replace("_", " ").title()
        cost_unit = tax.get("cost_unit", "")

        st.caption(f"Category: **{cat}**")
        metric_disp = f"{metric_label} · {cost_unit}" if cost_unit else metric_label
        st.caption(f"Metric: **{metric_disp}**")

        if tax.get("has_optimal_frequency"):
            st.info("💡 Optimal frequency solved during budget optimization.")

        ch_spend = spend_map.get(selected_channel, 0.0)
        st.metric("Historical Spend", f"₹{ch_spend:,.0f}")

        # Find outcome at 1.0x
        raw_pts = response_curves.get(selected_channel, [])
        cur_pt = next((p for p in raw_pts if abs(p.get("spend_multiplier", 0.0) - 1.0) < 0.01), None)
        if cur_pt:
            st.metric(
                f"Historical {kpi_name}",
                f"{cur_pt['incremental_outcome_mean']:,.0f}",
                help=f"90% CI: [{cur_pt['ci_lower']:,.0f} – {cur_pt['ci_upper']:,.0f}] {kpi_units}",
            )

    pts = response_curves.get(selected_channel, [])
    if not pts:
        st.error(f"No curve data points found for {selected_channel}")
        return

    df_curve = pd.DataFrame(pts).sort_values(by="spend_multiplier")
    base_spend = spend_map.get(selected_channel, 1.0)

    if "Absolute" in x_axis_mode:
        x_vals = [m * base_spend for m in df_curve["spend_multiplier"]]
        x_label = "Spend (₹)"
        current_x = base_spend
        x_format = "₹%{x:,.0f}"
    else:
        x_vals = df_curve["spend_pct_of_current"].tolist()
        x_label = "Spend (% of Historical)"
        current_x = 100.0
        x_format = "%{x:.0f}%"

    y_mean = df_curve["incremental_outcome_mean"].tolist()
    y_lo = df_curve["ci_lower"].tolist()
    y_hi = df_curve["ci_upper"].tolist()

    ch_color = get_channel_color(selected_channel)
    fill_color = f"rgba{tuple(list(int(ch_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.18])}"

    fig = go.Figure()

    # 1. Lower bound (invisible line for fill)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_lo,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 2. Upper bound with shaded fill
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_hi,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=fill_color,
            name="90% Credible Interval",
            hoverinfo="skip",
        )
    )

    # 3. Posterior mean line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_mean,
            mode="lines",
            line=dict(color=ch_color, width=3),
            name="Posterior Mean",
            hovertemplate=(
                f"<b>{selected_channel}</b><br>"
                + f"Spend: {x_format}<br>"
                + f"Incremental {kpi_name}: %{{y:,.0f}}<br>"
                + "90% CI: [%{customdata[0]:,.0f} – %{customdata[1]:,.0f}]<extra></extra>"
            ),
            customdata=list(zip(y_lo, y_hi)),
        )
    )

    # 4. Marker for historical spend
    if cur_pt:
        cur_y = cur_pt["incremental_outcome_mean"]
        fig.add_trace(
            go.Scatter(
                x=[current_x],
                y=[cur_y],
                mode="markers",
                marker=dict(size=11, color="#EF4444", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                name="Historical Spend (1.0x)",
                hovertemplate=(
                    f"<b>Current Historical Position</b><br>"
                    + f"Spend: {x_format}<br>"
                    + f"Incremental {kpi_name}: %{{y:,.0f}}<extra></extra>"
                ),
            )
        )

        # Add vertical line for current spend
        fig.add_vline(
            x=current_x,
            line_width=1.5,
            line_dash="dash",
            line_color="rgba(239, 68, 68, 0.7)",
            annotation_text="Current Spend",
            annotation_position="top left",
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title=dict(text=f"{selected_channel} — Response Curve with 90% Uncertainty Band", x=0),
        xaxis_title=x_label,
        yaxis_title=f"Incremental {kpi_name} ({kpi_units})",
        height=480,
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")

    with col_chart:
        st.plotly_chart(fig, width="stretch")

    # Multi-channel saturation overlay tab
    with st.expander("🔍 Compare All Channel Curves on Relative Spend Scale (0% – 200%)"):
        fig_all = go.Figure()
        for ch in channels:
            ch_pts = response_curves.get(ch, [])
            if not ch_pts:
                continue
            df_ch = pd.DataFrame(ch_pts).sort_values(by="spend_multiplier")
            fig_all.add_trace(
                go.Scatter(
                    x=df_ch["spend_pct_of_current"],
                    y=df_ch["incremental_outcome_mean"],
                    mode="lines",
                    name=ch,
                    line=dict(color=get_channel_color(ch), width=2),
                    hovertemplate=f"<b>{ch}</b><br>Spend: %{{x:.0f}}%<br>Outcome: %{{y:,.0f}}<extra></extra>",
                )
            )

        fig_all.add_vline(x=100.0, line_dash="dash", line_color="rgba(255,255,255,0.4)", annotation_text="100% Historical")
        fig_all.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            title=dict(text="All Media Channels Saturation Profiles Comparison", x=0),
            xaxis_title="Spend (% of Historical Baseline)",
            yaxis_title=f"Incremental {kpi_name} ({kpi_units})",
            height=420,
        )
        fig_all.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        fig_all.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_all, width="stretch")
