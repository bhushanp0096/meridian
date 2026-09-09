"""
frontend/components/budget_planner.py
=====================================
Interactive budget planner and scenario optimization view.
Connects to POST /optimize with loading states, uncertainty bounds, and side-by-side allocation comparisons.
"""

from typing import Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from frontend.styles import CHANNEL_COLORS, PLOTLY_LAYOUT_DEFAULTS


def render_budget_planner(
    api_client,
    spend_summary: list[dict[str, Any]],
    model_spec: dict[str, Any],
):
    """
    Render interactive budget optimization interface and results visualizer.
    """
    st.subheader("🎯 Budget Optimizer & Reallocation Planner")
    st.markdown(
        "Solve for the optimal budget allocation across media channels to maximize outcome under diminishing returns. "
        "Default scenario loads in **<5ms** from precomputed posterior cache."
    )

    if not spend_summary:
        st.warning("Spend summary data unavailable.")
        return

    hist_spend_map = {item["channel"]: item["total_spend"] for item in spend_summary}
    total_hist_spend = sum(hist_spend_map.values())
    channels = sorted(list(hist_spend_map.keys()))
    kpi_name = model_spec.get("kpi_name", "Bookings")
    kpi_units = model_spec.get("kpi_units", "units")

    # -----------------------------------------------------------------------
    # Setup Scenario Parameters Form
    # -----------------------------------------------------------------------
    with st.expander("⚙️ Scenario Constraints & Optimization Settings", expanded=True):
        col_type, col_budget = st.columns([1, 2])
        with col_type:
            opt_type = st.radio(
                "Optimization Objective:",
                options=["Fixed Budget (Maximize Outcome)", "Flexible Budget (Target ROI)"],
                index=0,
                key="budget_opt_type_radio",
            )

        with col_budget:
            if "Fixed" in opt_type:
                target_budget = st.number_input(
                    "Total Budget to Allocate (₹):",
                    min_value=100_000.0,
                    max_value=total_hist_spend * 3.0,
                    value=float(total_hist_spend),
                    step=50_000.0,
                    format="%.0f",
                    help="Default matches historical total spend across the 104-week period.",
                    key="budget_total_input",
                )
                target_roi = None
                target_mroi = None
            else:
                target_budget = None
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    target_roi = st.number_input(
                        "Target ROI:",
                        min_value=0.1,
                        max_value=50.0,
                        value=3.5,
                        step=0.5,
                        key="budget_target_roi",
                    )
                with col_r2:
                    target_mroi = st.number_input(
                        "Target Marginal ROI (Optional):",
                        min_value=0.0,
                        max_value=20.0,
                        value=0.0,
                        step=0.2,
                        key="budget_target_mroi",
                    )
                    if target_mroi == 0.0:
                        target_mroi = None

        st.markdown("**Channel Spend Shift Constraints**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            lower_bound_pct = st.slider(
                "Maximum Spend Decrease (%):",
                min_value=5,
                max_value=80,
                value=30,
                step=5,
                help="Maximum allowable reduction from historical spend per channel (default 30%).",
                key="budget_lower_slider",
            )
        with col_c2:
            upper_bound_pct = st.slider(
                "Maximum Spend Increase (%):",
                min_value=5,
                max_value=100,
                value=30,
                step=5,
                help="Maximum allowable increase from historical spend per channel (default 30%).",
                key="budget_upper_slider",
            )

        # Advanced per-channel constraints
        with st.expander("Advanced: Override Specific Channel Bounds"):
            st.caption("Optionally override min/max absolute spend for individual channels.")
            channel_overrides = {}
            col_ov = st.columns(len(channels))
            for i, ch in enumerate(channels):
                with col_ov[i % len(col_ov)]:
                    base = hist_spend_map[ch]
                    apply_ov = st.checkbox(f"Constrain {ch}", key=f"ov_chk_{ch}")
                    if apply_ov:
                        c_min = st.number_input(
                            f"Min ₹ ({ch})",
                            value=float(round(base * (1 - lower_bound_pct / 100.0))),
                            key=f"min_{ch}",
                        )
                        c_max = st.number_input(
                            f"Max ₹ ({ch})",
                            value=float(round(base * (1 + upper_bound_pct / 100.0))),
                            key=f"max_{ch}",
                        )
                        channel_overrides[ch] = (c_min, c_max)

    # Optimize Action Button
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

    with col_info:
        st.caption("Queries the Meridian serving backend. If using default constraints, returns instantaneously from cache.")

    # Initialize session state for optimization results
    if "opt_results" not in st.session_state:
        # Load default optimization initially
        try:
            st.session_state.opt_results = api_client.post_optimize(
                {
                    "total_budget": None,
                    "spend_constraint_lower": 0.3,
                    "spend_constraint_upper": 0.3,
                    "channel_constraints": None,
                    "target_roi": None,
                    "target_mroi": None,
                }
            )
        except Exception as init_exc:
            st.session_state.opt_results = None

    if run_optimization:
        scenario_req = {
            "total_budget": target_budget,
            "spend_constraint_lower": lower_bound_pct / 100.0,
            "spend_constraint_upper": upper_bound_pct / 100.0,
            "channel_constraints": channel_overrides if channel_overrides else None,
            "target_roi": target_roi,
            "target_mroi": target_mroi,
        }
        with st.spinner("Optimizing budget allocation across Meridian posterior distributions..."):
            try:
                result = api_client.post_optimize(scenario_req)
                st.session_state.opt_results = result
                st.success("Optimization completed successfully!")
            except Exception as opt_err:
                st.error(f"Optimization failed: {opt_err}")

    # -----------------------------------------------------------------------
    # Display Optimization Results
    # -----------------------------------------------------------------------
    result_data = st.session_state.get("opt_results")
    if not result_data:
        st.info("Click 'Run Optimization' to compute budget allocations.")
        return

    channels_data = result_data.get("channels", [])
    if not channels_data:
        st.warning("No channel recommendations returned by optimizer.")
        return

    df_res = pd.DataFrame(channels_data)
    total_opt_spend = result_data.get("total_optimized_spend", 0.0) or sum(df_res["optimized_spend"])
    total_curr_spend = sum(df_res["current_spend"])
    net_spend_delta = total_opt_spend - total_curr_spend
    net_spend_delta_pct = (net_spend_delta / total_curr_spend * 100.0) if total_curr_spend > 0 else 0.0

    # Total expected lift
    net_lift = sum(c.get("lift_mean", 0.0) for c in channels_data)
    net_lift_lo = sum(c.get("lift_ci_lower", 0.0) for c in channels_data)
    net_lift_hi = sum(c.get("lift_ci_upper", 0.0) for c in channels_data)

    # Check confidence flags
    flagged_channels = [c["channel"] for c in channels_data if c.get("confidence_flagged", False)]

    st.markdown("---")
    st.markdown("### 📊 Recommended Allocation Summary")

    # Key Result Metrics Cards
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Optimized Total Budget</div>
                <div class="metric-value">₹{total_opt_spend:,.0f}</div>
                <div class="metric-sub">Delta vs Current: {'+' if net_spend_delta >= 0 else ''}₹{net_spend_delta:,.0f} ({net_spend_delta_pct:+.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Expected Incremental Lift</div>
                <div class="metric-value">+{net_lift:,.0f}</div>
                <div class="metric-sub highlight">90% CI: [{net_lift_lo:,.0f} – {net_lift_hi:,.0f}] {kpi_units}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        flag_text = f"⚠️ {len(flagged_channels)} Channel(s) Borderline" if flagged_channels else "✅ High Confidence"
        flag_sub = f"Zero-crossing CI: {', '.join(flagged_channels)}" if flagged_channels else "All lifts strictly positive"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Confidence Rating</div>
                <div class="metric-value">{flag_text}</div>
                <div class="metric-sub">{flag_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if flagged_channels:
        st.warning(
            f"**Caveat / Confidence Notice:** The 90% credible interval for incremental lift crosses zero for: "
            f"**{', '.join(flagged_channels)}**. Treat these specific shifts as directional rather than definitive."
        )

    # Visual Comparison Tabs
    opt_tab1, opt_tab2, opt_tab3 = st.tabs(
        ["📊 Spend Reallocation (Before vs. After)", "🚀 Incremental Lift by Channel", "📋 Detailed Table"]
    )

    with opt_tab1:
        ch_names = df_res["channel"].tolist()
        curr_spends = df_res["current_spend"].tolist()
        opt_spends = df_res["optimized_spend"].tolist()

        fig_spend = go.Figure()
        fig_spend.add_trace(
            go.Bar(
                x=ch_names,
                y=curr_spends,
                name="Current Spend",
                marker_color="#64748B",
                hovertemplate="<b>%{x}</b><br>Current: ₹%{y:,.0f}<extra></extra>",
            )
        )
        fig_spend.add_trace(
            go.Bar(
                x=ch_names,
                y=opt_spends,
                name="Optimized Spend",
                marker_color="#10B981",
                hovertemplate="<b>%{x}</b><br>Optimized: ₹%{y:,.0f}<extra></extra>",
            )
        )
        fig_spend.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            barmode="group",
            title=dict(text="Current vs. Optimized Budget Allocation by Channel", x=0),
            xaxis_title="Media Channel",
            yaxis_title="Total Budget (₹)",
            height=430,
        )
        fig_spend.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        fig_spend.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_spend, use_container_width=True)

    with opt_tab2:
        # Expected lift per channel with credible intervals
        lift_means = df_res["lift_mean"].tolist()
        lift_los = df_res["lift_ci_lower"].tolist()
        lift_his = df_res["lift_ci_upper"].tolist()

        err_plus = [max(0.0, h - m) if h is not None and m is not None else 0.0 for h, m in zip(lift_his, lift_means)]
        err_minus = [max(0.0, m - l) if l is not None and m is not None else 0.0 for l, m in zip(lift_los, lift_means)]

        bar_colors = ["#10B981" if m >= 0 else "#EF4444" for m in lift_means]

        fig_lift = go.Figure()
        fig_lift.add_trace(
            go.Bar(
                x=ch_names,
                y=lift_means,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_plus,
                    arrayminus=err_minus,
                    thickness=2,
                    width=6,
                    color="#E2E8F0",
                ),
                marker_color=bar_colors,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    + f"Expected Lift: %{{y:,.0f}} {kpi_units}<br>"
                    + "90% CI: [%{customdata[0]:,.0f} – %{customdata[1]:,.0f}]<extra></extra>"
                ),
                customdata=list(zip(lift_los, lift_his)),
            )
        )
        fig_lift.add_hline(y=0.0, line_color="rgba(255,255,255,0.4)", line_dash="dash")
        fig_lift.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            title=dict(text=f"Expected Incremental {kpi_name} Lift by Channel (90% CI)", x=0),
            xaxis_title="Media Channel",
            yaxis_title=f"Lift in {kpi_name} ({kpi_units})",
            height=430,
        )
        fig_lift.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        fig_lift.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_lift, use_container_width=True)

    with opt_tab3:
        table_rows = []
        for _, row in df_res.iterrows():
            ch = row["channel"]
            c_sp = row.get("current_spend", 0.0) or 0.0
            o_sp = row.get("optimized_spend", 0.0) or 0.0
            sp_del = row.get("spend_delta", 0.0) or 0.0
            sp_pct = row.get("spend_delta_pct", 0.0) or 0.0
            opt_roi = row.get("roi_mean", 0.0) or 0.0
            opt_roi_l = row.get("roi_ci_lower", 0.0) or 0.0
            opt_roi_h = row.get("roi_ci_upper", 0.0) or 0.0
            l_m = row.get("lift_mean", 0.0) or 0.0
            l_l = row.get("lift_ci_lower", 0.0) or 0.0
            l_h = row.get("lift_ci_upper", 0.0) or 0.0
            flagged = row.get("confidence_flagged", False)

            table_rows.append({
                "Channel": ch,
                "Current Spend (₹)": f"₹{c_sp:,.0f}",
                "Optimized Spend (₹)": f"₹{o_sp:,.0f}",
                "Shift (₹)": f"{'+' if sp_del >= 0 else ''}₹{sp_del:,.0f}",
                "Shift (%)": f"{sp_pct:+.1f}%",
                "Optimized ROI (90% CI)": f"{opt_roi:.2f}x [{opt_roi_l:.2f} – {opt_roi_h:.2f}]",
                f"Expected Lift ({kpi_units})": f"{l_m:+,.0f} [{l_l:,.0f} – {l_h:,.0f}]",
                "Confidence": "⚠️ Flagged" if flagged else "✅ Confident",
            })

        tbl_df = pd.DataFrame(table_rows)
        st.dataframe(tbl_df, use_container_width=True, hide_index=True)
        csv_data = df_res.to_csv(index=False)
        st.download_button(
            label="📥 Export Optimized Allocation (CSV)",
            data=csv_data,
            file_name="meridian_optimized_budget.csv",
            mime="text/csv",
        )
