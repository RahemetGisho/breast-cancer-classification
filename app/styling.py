"""Visual identity for the Streamlit dashboard.

Deliberately restrained: one accent color (a muted indigo), a warm
neutral background instead of stark white, and no red/green
traffic-light coding for malignant/benign -- this is a clinical
context, not a stoplight, and that's the same design note already
written into reports/powerbi/POWERBI_DASHBOARD.md for consistency
across both dashboards this project produces.
"""

import streamlit as st

ACCENT = "#5B5BD6"  # muted indigo -- used for "malignant" / risk emphasis only
ACCENT_SOFT = "#EDEDFB"  # accent tint, for card backgrounds
INK = "#26291A"  # near-black body text
MUTED = "#6B7280"  # secondary text
BORDER = "#E4E4E7"  # hairline borders
BG = "#FAFAF8"  # warm off-white page background
SURFACE = "#FFFFFF"  # card surfaces
NEUTRAL_BADGE = (
    "#F1F1F0"  # "benign" badge background -- deliberately neutral, not green
)

PLOTLY_TEMPLATE = {
    "layout": {
        "font": {
            "family": "Inter, -apple-system, sans-serif",
            "color": INK,
            "size": 13,
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "colorway": [ACCENT, MUTED, "#A5A5E8", "#D1D1F0"],
        "xaxis": {
            "gridcolor": BORDER,
            "zerolinecolor": BORDER,
            # "color" sets axis line + ticks + label text all at once --
            # this is the fix for washed-out gray axis text. Relying only
            # on the top-level "font" setting above doesn't reliably
            # cascade onto axis ticks/titles in every Plotly version.
            "color": INK,
            "title": {"font": {"color": INK}},
            "tickfont": {"color": INK},
        },
        "yaxis": {
            "gridcolor": BORDER,
            "zerolinecolor": BORDER,
            "color": INK,
            "title": {"font": {"color": INK}},
            "tickfont": {"color": INK},
        },
        "legend": {"font": {"color": INK}},
        "margin": {"t": 40, "r": 20, "b": 40, "l": 60},
    }
}


def apply_theme() -> None:
    font_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Inter:wght@400;500;600;700&"
        "family=JetBrains+Mono:wght@400;500&display=swap"
    )
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{font_url}" rel="stylesheet">
        <style>
            html, body, [class*="css"] {{
                font-family: 'Inter', -apple-system, sans-serif;
            }}
            .stApp {{
                background-color: {BG};
            }}
            .block-container {{
                padding-top: 2.5rem;
                max-width: 1200px;
            }}
            #MainMenu, footer, header {{ visibility: hidden; }}

            .dash-title {{
                font-size: 1.9rem;
                font-weight: 700;
                color: {INK};
                letter-spacing: -0.02em;
                margin-bottom: 0.1rem;
            }}
            .dash-subtitle {{
                color: {MUTED};
                font-size: 0.95rem;
                margin-bottom: 1.6rem;
            }}

            .metric-card {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
                padding: 1.1rem 1.3rem;
                height: 100%;
            }}
            .metric-label {{
                color: {MUTED};
                font-size: 0.78rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.35rem;
            }}
            .metric-value {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.65rem;
                font-weight: 500;
                color: {INK};
            }}
            .metric-sub {{
                color: {MUTED};
                font-size: 0.78rem;
                margin-top: 0.2rem;
            }}

            .badge {{
                display: inline-block;
                padding: 0.28rem 0.75rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 600;
            }}
            .badge-malignant {{
                background: {ACCENT_SOFT};
                color: {ACCENT};
            }}
            .badge-benign {{
                background: {NEUTRAL_BADGE};
                color: {INK};
            }}

            .caveat-banner {{
                background: {ACCENT_SOFT};
                border-left: 3px solid {ACCENT};
                border-radius: 6px;
                padding: 0.7rem 1rem;
                font-size: 0.85rem;
                color: {INK};
                margin-bottom: 1.2rem;
            }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """


def diagnosis_badge(label: str) -> str:
    cls = "badge-malignant" if label == "malignant" else "badge-benign"
    return f'<span class="badge {cls}">{label.capitalize()}</span>'
