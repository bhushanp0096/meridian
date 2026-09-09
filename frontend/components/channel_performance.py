"""
frontend/components/channel_performance.py
==========================================
Channel performance visualizations: ROI and Contribution with posterior credible intervals.
"""

from typing import Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from frontend.styles import CHANNEL_COLORS, PLOTLY_LAYOUT_DEFAULTS


def render_channel_performance(
    roi_summary: list[dict[str, Any]],
    contributions: list[dict[str, Any]],
    model_spec: dict[str, Any],
):
    """
    Render Channel ROI and Contribution charts with error bars + detailed data table.
    """
    st.subheader("📊 Channel Effectiveness & Attribution Overview")
    st.markdown(
        "Compare media channels by Return on Investment (ROI) and incremental outcome. "
        "**Error bars represent the 90% posterior credible interval**, reflecting uncertainty in attribution."
    )

    if not roi_summary or not contributions:
        st.warning("No channel performance data available.")
        return

    df_roi = pd.DataFrame(roi_summary)
    df_contrib = pd.DataFrame(contributions)

    kpi_name = model_spec.get("kpi_name", "Bookings")
    kpi_units = model_spec.get("kpi_units", "units")

    tab1, tab2, tab3 = st.tabs(["📈 ROI & Marginal ROI", "🏆 Incremental Contribution", "📋 Detailed Data Table"])

    # -----------------------------------------------------------------------
    # Tab 1: Channel ROI with Error Bars
    # -----------------------------------------------------------------------
    with tab1:
        col_ctrl, col_chart = st.columns([1, 4])
        with col_ctrl:
            st.markdown("**Metric Display**")
            show_metric = st.radio(
                "Metric to plot:",
                options=["ROI (Average)", "mROI (Marginal)"],
                index=0,
                key="roi_chart_metric_radio",
            )
            sort_by = st.selectbox(
                "Sort channels by:",
                options=["Highest Mean", "Channel Name", "Historical Spend"],
                index=0,
                key="roi_sort_select",
            )

        metric_prefix = "roi" if "Average" in show_metric else "mroi"
        mean_col = f"{metric_prefix}_mean"
        lo_col = f"{metric_prefix}_ci_lower"
        hi_col = f"{metric_prefix}_ci_upper"

        # Prepare and sort data
        plot_df = df_roi.copy()
        if sort_by == "Highest Mean":
            plot_df = plot_df.sort_values(by=mean_col, ascending=False)
        elif sort_by == "Historical Spend":
            plot_df = plot_df.sort_values(by="spend_mean", ascending=False)
        else:
            plot_df = plot_df.sort_values(by="channel")

        channels = plot_df["channel"].tolist()
        means = plot_df[mean_col].tolist()
        los = plot_df[lo_col].tolist()
        his = plot_df[hi_col].tolist()

        # Asymmetric error bars: array = hi - mean, arrayminus = mean - lo
        error_plus = [max(0.0, h - m) if h is not None and m is not None else 0.0 for h, m in zip(his, means)]
        error_minus = [max(0.0, m - l) if l is not None and m is not None else 0.0 for l, m in zip(los, means)]

        bar_colors = [CHANNEL_COLORS.get(ch, "#3B82F6") for ch in channels]

        fig_roi = go.Figure()
        fig_roi.add_trace(
            go.Bar(
                x=channels,
                y=means,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=error_plus,
                    arrayminus=error_minus,
                    thickness=2,
                    width=6,
                    color="#E2E8F0",
                ),
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + f"{show_metric}: %{{y:.2f}}<br>"
                    + "90% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>"
                ),
                customdata=list(zip(los, his)),
            )
        )

        fig_roi.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            title=dict(text=f"Channel {show_metric} with 90% Credible Intervals", x=0),
            xaxis_title="Media Channel",
            yaxis_title=f"{show_metric} (Outcome per Unit Spend)",
            height=430,
        )
        fig_roi.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
        fig_roi.update_xaxes(tickangle=-15)

        with col_chart:
            st.plotly_chart(fig_roi, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tab 2: Channel Contribution Share & Incremental Volume
    # -----------------------------------------------------------------------
    with tab2:
        col_c_ctrl, col_c_chart = st.columns([1, 4])
        with col_c_ctrl:
            contrib_mode = st.radio(
                "Contribution View:",
                options=["Absolute Incremental Outcome", "Percentage Share of Media (%)"],
                index=0,
                key="contrib_view_mode",
            )

        c_df = df_contrib.copy().sort_values(by="mean", ascending=False)
        c_channels = c_df["channel"].tolist()
        c_means = c_df["mean"].tolist()
        c_los = c_df["ci_lower"].tolist()
        c_his = c_df["ci_upper"].tolist()

        c_err_plus = [max(0.0, h - m) if h is not None and m is not None else 0.0 for h, m in zip(c_his, c_means)]
        c_err_minus = [max(0.0, m - l) if l is not None and m is not None else 0.0 for l, m in zip(c_los, c_means)]
        c_colors = [CHANNEL_COLORS.get(ch, "#3B82F6") for ch in c_channels]

        fig_contrib = go.Figure()

        if "Percentage" in contrib_mode:
            pct_means = c_df["pct_of_total_mean"].tolist()
            fig_contrib.add_trace(
                go.Bar(
                    x=c_channels,
                    y=pct_means,
                    marker_color=c_colors,
                    hovertemplate="<b>%{x}</b><br>Share of Total: %{y:.1f}%<extra></extra>",
                )
            )
            fig_contrib.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                title=dict(text="Channel Share of Total Incremental Outcome (%)", x=0),
                xaxis_title="Channel",
                yaxis_title="Share (%)",
                height=430,
            )
        else:
            fig_contrib.add_trace(
                go.Bar(
                    x=c_channels,
                    y=c_means,
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=c_err_plus,
                        arrayminus=c_err_minus,
                        thickness=2,
                        width=6,
                        color="#E2E8F0",
                    ),
                    marker_color=c_colors,
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>"
                        + f"Incremental {kpi_name}: %{{y:,.0f}}<br>"
                        + "90% CI: [%{customdata[0]:,.0f}, %{customdata[1]:,.0f}]<extra></extra>"
                    ),
                    customdata=list(zip(c_los, c_his)),
                )
            )
            fig_contrib.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                title=dict(text=f"Incremental {kpi_name} Contribution by Channel (90% CI)", x=0),
                xaxis_title="Channel",
                yaxis_title=f"Incremental {kpi_name} ({kpi_units})",
                height=430,
            )

        fig_contrib.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        fig_contrib.update_xaxes(tickangle=-15)

        with col_c_chart:
            st.plotly_chart(fig_contrib, use_container_width=True)

    # -----------------------------------------------------------------------
    # Tab 3: Detailed Data Table
    # -----------------------------------------------------------------------
    with tab3:
        # Merge spend, ROI, and contribution for an exhaustive table
        merged = df_roi.copy()
        if "channel" in merged.columns and "channel" in df_contrib.columns:
            merged = merged.merge(
                df_contrib[["channel", "pct_of_total_mean"]],
                on="channel",
                how="left",
            )

        table_rows = []
        for _, row in merged.iterrows():
            ch = row["channel"]
            spend = row.get("spend_mean", 0.0) or 0.0
            roi_m = row.get("roi_mean", 0.0) or 0.0
            roi_l = row.get("roi_ci_lower", 0.0) or 0.0
            roi_h = row.get("roi_ci_upper", 0.0) or 0.0
            mroi_m = row.get("mroi_mean", 0.0) or 0.0
            mroi_l = row.get("mroi_ci_lower", 0.0) or 0.0
            mroi_h = row.get("mroi_ci_upper", 0.0) or 0.0
            inc_m = row.get("incremental_outcome_mean", 0.0) or 0.0
            cpik = row.get("cpik_mean", 0.0) or 0.0
            pct_share = row.get("pct_of_total_mean", 0.0) or 0.0

            table_rows.append({
                "Channel": ch,
                "Spend (₹)": f"₹{spend:,.0f}",
                "ROI (Mean)": f"{roi_m:.2f}x",
                "ROI 90% CI": f"[{roi_l:.2f} – {roi_h:.2f}]",
                "mROI (Mean)": f"{mroi_m:.2f}x",
                "mROI 90% CI": f"[{mroi_l:.2f} – {mroi_h:.2f}]",
                f"Incremental {kpi_name}": f"{inc_m:,.0f}",
                "Share (%)": f"{pct_share:.1f}%",
                "Cost / KPI (₹)": f"₹{cpik:.1f}" if cpik > 0 else "—",
            })

        display_df = pd.DataFrame(table_rows)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
