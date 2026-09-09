"""
frontend/components/channel_performance.py
==========================================
Channel performance visualizations: ROI and Contribution with posterior credible intervals.
"""

from typing import Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from frontend.styles import CHANNEL_COLORS, PLOTLY_LAYOUT_DEFAULTS, get_channel_color


def render_channel_performance(
    roi_summary: list[dict[str, Any]],
    contributions: list[dict[str, Any]],
    model_spec: dict[str, Any],
):
    """
    Render Channel ROI and Contribution charts with error bars, category filters, and taxonomy breakdown.
    """
    st.subheader("📊 Channel Effectiveness & Attribution Overview")
    st.markdown(
        "Compare media channels by Return on Investment (ROI) and incremental outcome across execution metrics. "
        "**Error bars represent the 90% posterior credible interval**, reflecting uncertainty in attribution."
    )

    if not roi_summary or not contributions:
        st.warning("No channel performance data available.")
        return

    taxonomy = model_spec.get("channel_taxonomy", {})
    categories = model_spec.get("categories", [])

    # Category filter control
    cat_options = ["All Categories"] + sorted(list(set(categories)))
    col_f1, col_f2 = st.columns([2, 3])
    with col_f1:
        selected_cat = st.selectbox(
            "Filter Channels by Category:",
            options=cat_options,
            index=0,
            key="perf_category_filter",
        )

    # Filter dataframes if a category is selected
    df_roi = pd.DataFrame(roi_summary)
    df_contrib = pd.DataFrame(contributions)

    if selected_cat != "All Categories":
        df_roi = df_roi[df_roi["channel"].apply(lambda ch: taxonomy.get(ch, {}).get("category") == selected_cat)]
        df_contrib = df_contrib[df_contrib["channel"].apply(lambda ch: taxonomy.get(ch, {}).get("category") == selected_cat)]

    if df_roi.empty:
        st.info(f"No channels found under category '{selected_cat}'.")
        return

    kpi_name = model_spec.get("kpi_name", "Bookings")
    kpi_units = model_spec.get("kpi_units", "units")

    tab1, tab2, tab3 = st.tabs(["📈 ROI & Marginal ROI", "🏆 Incremental Contribution", "📋 Detailed Data Table"])

    # -----------------------------------------------------------------------
    # Tab 1: Channel ROI with Error Bars & Metric Tags
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

        error_plus = [max(0.0, h - m) if h is not None and m is not None else 0.0 for h, m in zip(his, means)]
        error_minus = [max(0.0, m - l) if l is not None and m is not None else 0.0 for l, m in zip(los, means)]
        bar_colors = [get_channel_color(ch) for ch in channels]

        # Channel display labels with metric chips
        display_names = []
        custom_data = []
        for ch, l, h in zip(channels, los, his):
            tax = taxonomy.get(ch, {})
            unit = tax.get("cost_unit") or tax.get("execution_metric", "")
            badge = f" [{unit}]" if unit else ""
            display_names.append(f"{ch}{badge}")
            custom_data.append((
                l,
                h,
                tax.get("execution_metric", "standard"),
                tax.get("category", "General"),
                tax.get("cost_unit", "—"),
            ))

        fig_roi = go.Figure()
        fig_roi.add_trace(
            go.Bar(
                x=display_names,
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
                    + "90% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<br>"
                    + "Category: %{customdata[3]}<br>"
                    + "Metric: %{customdata[2]} (%{customdata[4]})<extra></extra>"
                ),
                customdata=custom_data,
            )
        )

        fig_roi.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            title=dict(text=f"Channel {show_metric} with 90% Credible Intervals", x=0),
            xaxis_title="Media Channel [Execution Metric]",
            yaxis_title=f"{show_metric} (Outcome per Unit Spend)",
            height=430,
        )
        fig_roi.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
        fig_roi.update_xaxes(tickangle=-15)

        with col_chart:
            st.plotly_chart(fig_roi, width="stretch")

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
        c_colors = [get_channel_color(ch) for ch in c_channels]

        c_display_names = []
        c_custom_data = []
        for ch, l, h in zip(c_channels, c_los, c_his):
            tax = taxonomy.get(ch, {})
            unit = tax.get("cost_unit") or ("Organic" if tax.get("execution_metric") == "organic" else "")
            badge = f" [{unit}]" if unit else ""
            c_display_names.append(f"{ch}{badge}")
            c_custom_data.append((
                l,
                h,
                tax.get("execution_metric", "standard"),
                tax.get("category", "General"),
            ))

        fig_contrib = go.Figure()

        if "Percentage" in contrib_mode:
            pct_means = c_df["pct_of_total_mean"].tolist()
            fig_contrib.add_trace(
                go.Bar(
                    x=c_display_names,
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
                    x=c_display_names,
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
                        + "90% CI: [%{customdata[0]:,.0f}, %{customdata[1]:,.0f}]<br>"
                        + "Category: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=c_custom_data,
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
            st.plotly_chart(fig_contrib, width="stretch")

    # -----------------------------------------------------------------------
    # Tab 3: Detailed Multi-Metric Data Table
    # -----------------------------------------------------------------------
    with tab3:
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
            tax = taxonomy.get(ch, {})
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

            metric_label = tax.get("execution_metric", "standard").replace("_", " ").title()
            cost_unit = tax.get("cost_unit")
            unit_cost_str = f"₹{cpik:.1f} ({cost_unit})" if (cpik > 0 and cost_unit) else (f"₹{cpik:.1f}" if cpik > 0 else "—")

            table_rows.append({
                "Channel": ch,
                "Category": tax.get("category", "General"),
                "Execution Metric": metric_label,
                "Spend (₹)": f"₹{spend:,.0f}" if spend > 0 else "Organic",
                "ROI (Mean)": f"{roi_m:.2f}x" if spend > 0 else "—",
                "ROI 90% CI": f"[{roi_l:.2f} – {roi_h:.2f}]" if spend > 0 else "—",
                "mROI (Mean)": f"{mroi_m:.2f}x" if spend > 0 else "—",
                "mROI 90% CI": f"[{mroi_l:.2f} – {mroi_h:.2f}]" if spend > 0 else "—",
                f"Incremental {kpi_name}": f"{inc_m:,.0f}",
                "Share (%)": f"{pct_share:.1f}%",
                "Unit Efficiency": unit_cost_str,
            })

        display_df = pd.DataFrame(table_rows)
        st.dataframe(display_df, width="stretch", hide_index=True)
