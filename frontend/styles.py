"""
frontend/styles.py
==================
Theme tokens, custom CSS, and styling helpers for the Meridian Streamlit App.
Provides a modern Mix Studio executive aesthetic.
"""

import streamlit as st

# Color Palette Constants
COLOR_PRIMARY = "#1E88E5"      # Meridian Blue
COLOR_SECONDARY = "#00ACC1"    # Cyan / Teal accent
COLOR_SUCCESS = "#2E7D32"      # Muted emerald green
COLOR_WARNING = "#F57C00"      # Amber alert
COLOR_DANGER = "#D32F2F"       # Red alert
COLOR_NEUTRAL_DARK = "#1E293B" # Dark Slate
COLOR_NEUTRAL_LIGHT = "#F8FAFC"# Light Slate
COLOR_MUTED = "#64748B"        # Cool Grey
COLOR_CARD_BG = "rgba(255, 255, 255, 0.05)"
COLOR_CARD_BORDER = "rgba(226, 232, 240, 0.15)"

# Channel color map for consistent visualization across all views
CHANNEL_COLORS = {
    "Online_Video": "#4F46E5",     # Indigo
    "Display": "#06B6D4",          # Cyan
    "Paid_Social": "#EC4899",      # Pink
    "Paid_Search": "#F59E0B",      # Amber
    "Affiliate": "#10B981",        # Emerald
    "CTV": "#8B5CF6",              # Purple
    "Linear_TV": "#3B82F6",        # Sky Blue
    "Email_Opens": "#64748B",      # Slate (organic)
}

DEFAULT_CHANNEL_PALETTE = [
    "#4F46E5",  # Indigo
    "#06B6D4",  # Cyan
    "#EC4899",  # Pink
    "#F59E0B",  # Amber
    "#10B981",  # Emerald
    "#8B5CF6",  # Purple
    "#3B82F6",  # Sky Blue
    "#64748B",  # Slate
    "#E11D48",  # Rose
    "#D97706",  # Ochre
    "#059669",  # Mint
    "#7C3AED",  # Violet
]


def get_channel_color(channel: str, fallback_index: int = 0) -> str:
    """Return consistent, harmonious color for any channel (fixed or discovered)."""
    if channel in CHANNEL_COLORS:
        return CHANNEL_COLORS[channel]
    hash_val = sum(ord(c) for c in channel) + fallback_index
    return DEFAULT_CHANNEL_PALETTE[hash_val % len(DEFAULT_CHANNEL_PALETTE)]


def get_metric_badge_html(metric: str, cost_unit: str | None = None) -> str:
    """Return styled HTML chip for an execution metric."""
    metric_map = {
        "reach_and_frequency": ("badge-rf", f"R&F · {cost_unit or 'CPR'}"),
        "clicks": ("badge-clicks", f"Clicks · {cost_unit or 'CPC'}"),
        "impressions": ("badge-impressions", f"Impr · {cost_unit or 'CPM'}"),
        "organic": ("badge-organic", "Organic"),
        "spend": ("badge-spend", f"Spend · {cost_unit or 'Spend'}"),
    }
    cls, label = metric_map.get(metric, ("badge-info", metric))
    return f'<span class="badge {cls}">{label}</span>'


PLOTLY_LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)


def apply_custom_css():
    """Inject polished custom CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Import modern typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Top Header & Subhead */
        .app-header {
            margin-bottom: 1.5rem;
        }
        .app-header h1 {
            font-weight: 700;
            font-size: 2.1rem;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .app-header p {
            color: #64748B;
            font-size: 1.05rem;
            margin-top: 0;
        }

        /* Metric cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(226, 232, 240, 0.12);
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: #38BDF8;
            transform: translateY(-2px);
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #F8FAFC;
        }
        .metric-sub {
            font-size: 0.82rem;
            color: #64748B;
            margin-top: 0.25rem;
        }
        .metric-sub.highlight {
            color: #10B981;
            font-weight: 600;
        }

        /* Badge chips */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .badge-healthy {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-info {
            background-color: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .badge-warning {
            background-color: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .badge-danger {
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .badge-rf {
            background-color: rgba(139, 92, 246, 0.15);
            color: #A78BFA;
            border: 1px solid rgba(139, 92, 246, 0.35);
        }
        .badge-clicks {
            background-color: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .badge-impressions {
            background-color: rgba(79, 70, 229, 0.15);
            color: #818CF8;
            border: 1px solid rgba(79, 70, 229, 0.35);
        }
        .badge-organic {
            background-color: rgba(100, 116, 139, 0.15);
            color: #94A3B8;
            border: 1px solid rgba(100, 116, 139, 0.35);
        }
        .badge-spend {
            background-color: rgba(20, 184, 166, 0.15);
            color: #2DD4BF;
            border: 1px solid rgba(20, 184, 166, 0.35);
        }
        .badge-category {
            background-color: rgba(30, 41, 59, 0.7);
            color: #E2E8F0;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        /* Callout notification container */
        .callout-box {
            padding: 1rem 1.25rem;
            border-radius: 8px;
            background-color: rgba(30, 41, 59, 0.4);
            border-left: 4px solid #38BDF8;
            margin-bottom: 1.2rem;
        }

        /* Clean tab formatting */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
