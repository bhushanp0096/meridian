"""
frontend/components/kpi_cards.py
================================
Executive summary KPI metric cards.
"""

from typing import Any
import streamlit as st
import pandas as pd


def render_kpi_cards(
    model_spec: dict[str, Any],
    roi_summary: list[dict[str, Any]],
    spend_summary: list[dict[str, Any]],
    contributions: list[dict[str, Any]],
    accuracy: dict[str, Any],
):
    """
    Render executive KPI summary cards in a clean 4-column responsive grid.
    """
    total_spend = sum(item.get("total_spend", 0.0) for item in spend_summary)
    
    # Calculate overall weighted media ROI or total incremental outcome
    total_outcome = sum(c.get("mean", 0.0) for c in contributions if c.get("channel") != "Email_Opens")
    total_outcome_lo = sum(c.get("ci_lower", 0.0) for c in contributions if c.get("channel") != "Email_Opens")
    total_outcome_hi = sum(c.get("ci_upper", 0.0) for c in contributions if c.get("channel") != "Email_Opens")

    overall_roi = (total_outcome / total_spend) if total_spend > 0 else 0.0
    overall_roi_lo = (total_outcome_lo / total_spend) if total_spend > 0 else 0.0
    overall_roi_hi = (total_outcome_hi / total_spend) if total_spend > 0 else 0.0

    kpi_name = model_spec.get("kpi_name", "KPI")
    kpi_units = model_spec.get("kpi_units", "units")
    r_squared = accuracy.get("r_squared")
    mape = accuracy.get("mape")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Historical Spend</div>
                <div class="metric-value">₹{total_spend:,.0f}</div>
                <div class="metric-sub">Across {len(spend_summary)} paid channels (104 wks)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Media Incremental {kpi_name}</div>
                <div class="metric-value">{total_outcome:,.0f}</div>
                <div class="metric-sub">90% CI: [{total_outcome_lo:,.0f} – {total_outcome_hi:,.0f}] {kpi_units}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Overall Media Efficiency</div>
                <div class="metric-value">{overall_roi:.2f}x</div>
                <div class="metric-sub highlight">90% CI: [{overall_roi_lo:.2f} – {overall_roi_hi:.2f}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        fit_sub = f"R²: {r_squared:.3f}" if r_squared is not None else "Model Status: Calibrated"
        if mape is not None:
            fit_sub += f" | MAPE: {mape*100:.1f}%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Goodness of Fit</div>
                <div class="metric-value">90% HDI</div>
                <div class="metric-sub">{fit_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
