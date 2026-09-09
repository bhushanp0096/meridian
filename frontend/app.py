"""
frontend/app.py
===============
Google Meridian MMM — Production Interactive Serving Application (Streamlit MVP)

Features:
- Executive Summary KPI metric cards
- Channel Performance Overview (ROI, mROI, Incremental Volume with 90% CIs)
- Response / Saturation Curve Explorer with posterior uncertainty ribbons
- Budget Planner with interactive sliders, constraints, and side-by-side reallocation
- Baseline vs. Media Time-Series Decomposition
- What-If Scenario Simulator with custom spend & CPM multipliers
"""

import sys
from pathlib import Path

# Add project root to sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from frontend.styles import (
    apply_custom_css,
    COLOR_PRIMARY,
    PLOTLY_LAYOUT_DEFAULTS,
    CHANNEL_COLORS,
)
from frontend.api_client import MeridianApiClient
from frontend.components.kpi_cards import render_kpi_cards
from frontend.components.channel_performance import render_channel_performance
from frontend.components.curve_explorer import render_curve_explorer
from frontend.components.budget_planner import render_budget_planner
from frontend.components.timeline_view import render_timeline_view

# Streamlit Page Configuration
st.set_page_config(
    page_title="Google Meridian MMM | Serving Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS design system
apply_custom_css()


@st.cache_resource(show_spinner=False)
def get_client() -> MeridianApiClient:
    """Initialize and cache the API client."""
    return MeridianApiClient()


@st.cache_data(ttl=300, show_spinner=False)
def load_app_data(_client: MeridianApiClient):
    """
    Fetch all baseline model specifications and cached summaries.
    Cached for fast page transitions.
    """
    spec = _client.get_model_spec()
    health = _client.get_health()
    roi = _client.get_roi_summary()
    contribs = _client.get_contributions()
    curves = _client.get_response_curves()
    spends = _client.get_spend_summary()
    timeline = _client.get_baseline_vs_media()
    accuracy = _client.get_accuracy()
    return {
        "spec": spec,
        "health": health,
        "roi": roi,
        "contribs": contribs,
        "curves": curves,
        "spends": spends,
        "timeline": timeline,
        "accuracy": accuracy,
    }


def render_whatif_simulator(client: MeridianApiClient, spends: list[dict], curves: dict, spec: dict):
    """Render What-If simulation engine."""
    st.subheader("🧪 What-If Scenario & Cost Simulator")
    st.markdown(
        "Simulate the impact of shifting channel budgets or changing CPM/cost-per-unit rates. "
        "Projected outcomes are computed across response curves with **posterior credible bounds**."
    )

    channels = [item["channel"] for item in spends]
    spend_map = {item["channel"]: item["total_spend"] for item in spends}
    kpi_name = spec.get("kpi_name", "Bookings")
    kpi_units = spec.get("kpi_units", "units")

    with st.expander("🛠️ Configure Scenario Adjustments", expanded=True):
        scen_name = st.text_input("Scenario Label:", value="Q4 Planned Shift", key="whatif_scen_name")
        st.caption("Adjust spend multiplier (e.g., 1.2 = +20%) or CPM cost multiplier (e.g., 1.5 = +50% cost):")

        adjustments = {}
        cols = st.columns(min(len(channels), 4))
        for i, ch in enumerate(channels):
            with cols[i % len(cols)]:
                st.markdown(f"**{ch}**")
                base = spend_map.get(ch, 0.0)
                st.caption(f"Base: ₹{base:,.0f}")
                s_mult = st.slider(
                    f"Spend Multiplier",
                    min_value=0.2,
                    max_value=2.0,
                    value=1.0,
                    step=0.05,
                    key=f"whatif_smult_{ch}",
                )
                c_mult = st.slider(
                    f"CPM Multiplier",
                    min_value=0.5,
                    max_value=2.5,
                    value=1.0,
                    step=0.1,
                    key=f"whatif_cmult_{ch}",
                )
                adjustments[ch] = {
                    "spend_multiplier": s_mult,
                    "cpm_multiplier": c_mult,
                }

        run_sim = st.button("Simulate Scenario", type="primary", key="whatif_run_btn")

    if run_sim or "whatif_res" in st.session_state:
        if run_sim:
            with st.spinner("Computing what-if projections..."):
                payload = {
                    "scenario_name": scen_name,
                    "channel_adjustments": adjustments,
                    "use_kpi": False,
                    "ci_level": 0.9,
                }
                st.session_state.whatif_res = client.post_whatif(payload)

        res = st.session_state.whatif_res
        st.markdown("---")
        st.markdown(f"### Results: {res.get('scenario_name', 'Simulation')}")

        w1, w2, w3 = st.columns(3)
        b_sp = res.get("total_baseline_spend", 0.0)
        s_sp = res.get("total_scenario_spend", 0.0)
        sp_del = s_sp - b_sp
        sp_del_pct = (sp_del / b_sp * 100.0) if b_sp > 0 else 0.0
        out_m = res.get("total_outcome_mean", 0.0)
        out_l = res.get("total_outcome_ci_lower", 0.0)
        out_h = res.get("total_outcome_ci_upper", 0.0)

        with w1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Baseline Spend</div>
                    <div class="metric-value">₹{b_sp:,.0f}</div>
                    <div class="metric-sub">Across {len(channels)} channels</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with w2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Simulated Spend</div>
                    <div class="metric-value">₹{s_sp:,.0f}</div>
                    <div class="metric-sub">Delta: {'+' if sp_del >= 0 else ''}₹{sp_del:,.0f} ({sp_del_pct:+.1f}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with w3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Projected Incremental {kpi_name}</div>
                    <div class="metric-value">{out_m:,.0f}</div>
                    <div class="metric-sub highlight">90% CI: [{out_l:,.0f} – {out_h:,.0f}] {kpi_units}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Table of channel what-if breakdown
        ch_items = res.get("channels", [])
        if ch_items:
            tbl = []
            for r in ch_items:
                tbl.append({
                    "Channel": r["channel"],
                    "Baseline Spend": f"₹{r['baseline_spend']:,.0f}",
                    "Scenario Spend": f"₹{r['scenario_spend']:,.0f}",
                    "Spend Delta": f"{r['spend_delta_pct']:+.1f}%",
                    "CPM Multiplier": f"{r['cpm_multiplier']:.1f}x",
                    f"Projected {kpi_name}": f"{r['expected_outcome_mean']:,.0f}",
                    "90% Credible Interval": f"[{r['ci_lower']:,.0f} – {r['ci_upper']:,.0f}]",
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)


def main():
    client = get_client()

    # Load data with user-friendly loading state
    with st.spinner("Connecting to Meridian serving API and loading cache..."):
        try:
            data = load_app_data(client)
        except Exception as err:
            st.error(
                f"⚠️ Unable to load model data: {err}\n\n"
                "Ensure that precomputed cache exists (`python jobs/precompute.py`) "
                "or that FastAPI is active."
            )
            return

    spec = data["spec"]
    health = data["health"]
    roi = data["roi"]
    contribs = data["contribs"]
    curves = data["curves"]
    spends = data["spends"]
    timeline = data["timeline"]
    accuracy = data["accuracy"]

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📊 Google Meridian MMM")
        st.markdown("**Production Serving Application**")

        # System Health Status Chip
        status_label = health.get("status", "healthy").capitalize()
        badge_cls = "badge-healthy" if status_label == "Healthy" else "badge-warning"
        st.markdown(
            f"""
            <span class="badge {badge_cls}">● Serving Status: {status_label}</span>
            <span class="badge badge-info">Cache: Ready</span>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Navigation**")
        selected_view = st.radio(
            "Go to View:",
            options=[
                "📊 Performance Overview",
                "📈 Response Curves",
                "🎯 Budget Planner",
                "⏱️ Timeline Decomposition",
                "🧪 What-If Simulator",
            ],
            index=0,
            key="main_navigation_radio",
        )

        st.markdown("---")
        st.markdown("**Model Specs**")
        st.caption(f"**Target KPI**: `{spec.get('kpi_name')}` ({spec.get('kpi_units')})")
        st.caption(f"**Geos ({spec.get('n_geos')})**: {', '.join(spec.get('geo_names', []))}")
        st.caption(f"**Time**: {spec.get('n_times')} weeks ({spec.get('date_start')} → {spec.get('date_end')})")
        st.caption(f"**Media Channels ({spec.get('n_media_channels')})**: {', '.join(spec.get('media_channel_names', []))}")
        st.caption(f"**RF Channels ({spec.get('n_rf_channels')})**: {', '.join(spec.get('rf_channel_names', []))}")

        st.markdown("---")
        st.caption("Google Meridian 2.0.0 | JAX / CPU Serving | Stage 5 MVP")

    # -----------------------------------------------------------------------
    # Main Dashboard Body
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="app-header">
            <h1>Google Meridian MMM — Mix Studio</h1>
            <p>Posterior-grounded causal media mix analytics and budget optimization platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Executive KPI Cards (rendered across all views)
    render_kpi_cards(
        model_spec=spec,
        roi_summary=roi,
        spend_summary=spends,
        contributions=contribs,
        accuracy=accuracy,
    )

    st.markdown("---")

    # Router logic
    if selected_view == "📊 Performance Overview":
        render_channel_performance(roi_summary=roi, contributions=contribs, model_spec=spec)
    elif selected_view == "📈 Response Curves":
        render_curve_explorer(response_curves=curves, spend_summary=spends, model_spec=spec)
    elif selected_view == "🎯 Budget Planner":
        render_budget_planner(api_client=client, spend_summary=spends, model_spec=spec)
    elif selected_view == "⏱️ Timeline Decomposition":
        render_timeline_view(baseline_vs_media=timeline, model_spec=spec)
    elif selected_view == "🧪 What-If Simulator":
        render_whatif_simulator(client=client, spends=spends, curves=curves, spec=spec)


if __name__ == "__main__":
    main()
