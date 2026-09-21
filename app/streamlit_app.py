"""Breast Cancer Classification -- model results dashboard + live demo.

Run with: streamlit run app/streamlit_app.py

Four sections, switched via a horizontal radio (styled as a tab row --
plain st.tabs was replaced because Streamlit collapses an overflowing
tab row into a hover-to-scroll strip; st.radio with horizontal=True
has no such overflow behavior, so all four are always visible):
  Overview        -- KPIs, confusion matrix, interactive recall/precision-vs-threshold
  Explainability  -- global SHAP importance + per-case local explanation
  Monitoring      -- the API's own prediction log (labeled honestly)
  Try It          -- live prediction against the actual registered pipeline

Every number shown here is read from the same reports/powerbi/ exports
and the same registered model the API and REPORT.md use -- nothing is
recomputed independently or hand-typed, so this dashboard can never
silently drift from what's actually registered.
"""

import sys
from pathlib import Path

# Streamlit only adds this script's own folder (app/) to sys.path, not
# the project root above it -- without this, `from app.loaders import
# ...` fails with ModuleNotFoundError, since Python has no idea a
# project root containing an `app` package exists at all.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.loaders import (  # noqa: E402
    explain_row,
    load_model_metrics,
    load_prediction_log,
    load_prediction_log_contributors,
    load_shap_importance,
    load_test_predictions,
    predict_row,
)
from app.styling import (
    PLOTLY_TEMPLATE,
    apply_theme,
    diagnosis_badge,
    metric_card,
)  # noqa: E402
from src.data.schema import FEATURE_COLUMNS  # noqa: E402

st.set_page_config(
    page_title="Breast Cancer Classifier -- Dashboard",
    page_icon="\U0001fa7a",
    layout="wide",
)
apply_theme()


def _fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


metrics = load_model_metrics()
test_df = load_test_predictions()

st.markdown(
    '<div class="dash-title">Breast Cancer Classification</div>', unsafe_allow_html=True
)
st.markdown(
    f'<div class="dash-subtitle">Model: <b>{metrics["model_name"]}</b> &middot; '
    f'trained {metrics["trained_at_utc"][:10]} &middot; '
    "decision-support tool, not a diagnostic replacement</div>",
    unsafe_allow_html=True,
)

if "selected_section" not in st.session_state:
    st.session_state.selected_section = "Overview"

sections = ["Overview", "Explainability", "Monitoring", "Try It"]
nav_cols = st.columns(len(sections))
for col, section in zip(nav_cols, sections):
    with col:
        is_active = st.session_state.selected_section == section
        if st.button(
            section,
            key=f"nav_{section}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            if not is_active:
                st.session_state.selected_section = section
                st.rerun()

selected = st.session_state.selected_section

# ---------------------------------------------------------------- Overview
if selected == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card(
                "Recall (malignant)",
                f"{metrics['recall_malignant']:.1%}",
                "primary metric",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card(
                "Precision (malignant)", f"{metrics['precision_malignant']:.1%}"
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("ROC-AUC", f"{metrics['roc_auc']:.1%}"), unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            metric_card(
                "False negatives",
                str(int(metrics["false_negatives"])),
                "missed malignant cases",
            ),
            unsafe_allow_html=True,
        )

    st.write("")
    col_cm, col_thresh = st.columns([1, 1.4])

    with col_cm:
        st.markdown("##### Confusion matrix (held-out test set)")
        order = ["benign", "malignant"]
        cm = pd.crosstab(
            test_df["actual_label"], test_df["predicted_label"], dropna=False
        ).reindex(index=order, columns=order, fill_value=0)
        fig = go.Figure(
            data=go.Heatmap(
                z=cm.values,
                x=[f"Predicted {c}" for c in order],
                y=[f"Actual {r}" for r in order],
                colorscale=[[0, "#FFFFFF"], [1, "#5B5BD6"]],
                text=cm.values,
                texttemplate="%{text}",
                textfont={"size": 18},
                showscale=False,
            )
        )
        fig.update_layout(height=320)
        st.plotly_chart(_fig(fig), use_container_width=True)

    with col_thresh:
        st.markdown("##### Recall / precision vs. classification threshold")
        threshold = st.slider(
            "Decision threshold (probability of malignant)", 0.0, 1.0, 0.5, 0.01
        )

        thresholds = np.linspace(0.01, 0.99, 99)
        y_true = (test_df["actual_label"] == "malignant").astype(int).to_numpy()
        proba = test_df["probability_malignant"].to_numpy()

        recalls, precisions = [], []
        for t in thresholds:
            pred = (proba >= t).astype(int)
            tp = ((y_true == 1) & (pred == 1)).sum()
            fn = ((y_true == 1) & (pred == 0)).sum()
            fp = ((y_true == 0) & (pred == 1)).sum()
            recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
            precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=thresholds, y=recalls, name="Recall", line=dict(width=2.5))
        )
        fig2.add_trace(
            go.Scatter(
                x=thresholds,
                y=precisions,
                name="Precision",
                line=dict(width=2.5, dash="dot"),
            )
        )
        fig2.add_vline(x=threshold, line_dash="dash", line_color="#9CA3AF")
        fig2.update_layout(
            height=320,
            xaxis_title="Threshold",
            yaxis_title="Score",
            yaxis_range=[0, 1.02],
        )
        st.plotly_chart(_fig(fig2), use_container_width=True)

        pred_at_t = (proba >= threshold).astype(int)
        tp = int(((y_true == 1) & (pred_at_t == 1)).sum())
        fn = int(((y_true == 1) & (pred_at_t == 0)).sum())
        fp = int(((y_true == 0) & (pred_at_t == 1)).sum())
        r = tp / (tp + fn) if (tp + fn) else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        st.caption(
            f"At threshold **{threshold:.2f}**: recall **{r:.1%}**, precision **{p:.1%}**, "
            f"{fn} false negative(s), {fp} false positive(s)."
        )

# ------------------------------------------------------------ Explainability
if selected == "Explainability":
    st.markdown("##### Global feature importance (mean absolute SHAP value)")
    shap_df = load_shap_importance().sort_values("mean_abs_shap_value").tail(15)
    fig3 = px.bar(
        shap_df,
        x="mean_abs_shap_value",
        y="feature",
        orientation="h",
        color_discrete_sequence=["#5B5BD6"],
    )
    fig3.update_layout(height=420, xaxis_title="Mean |SHAP value|", yaxis_title="")
    st.plotly_chart(_fig(fig3), use_container_width=True)

    st.write("")
    st.markdown("##### Explain one test case")
    options = test_df.apply(
        lambda r: f"#{r['prediction_id']} -- actual: {r['actual_label']}, "
        f"predicted: {r['predicted_label']} ({r['probability_malignant']:.0%})",
        axis=1,
    ).tolist()
    choice = st.selectbox("Pick a held-out test case", options)
    row_id = int(choice.split("#")[1].split(" ")[0])
    row = test_df[test_df["prediction_id"] == row_id]

    contributors = explain_row(row[FEATURE_COLUMNS])
    contrib_df = pd.DataFrame(contributors).sort_values("shap_value")
    fig4 = px.bar(
        contrib_df,
        x="shap_value",
        y="feature",
        orientation="h",
        color=contrib_df["shap_value"] > 0,
        color_discrete_map={True: "#5B5BD6", False: "#C7C7E8"},
    )
    fig4.update_layout(
        height=280,
        xaxis_title="SHAP contribution (+ toward malignant)",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(_fig(fig4), use_container_width=True)

# ----------------------------------------------------------------- Monitoring
if selected == "Monitoring":
    log_df = load_prediction_log()
    contrib_log_df = load_prediction_log_contributors()

    if log_df is None or len(log_df) == 0:
        st.info(
            "No prediction log found -- run the API and send a few requests, "
            "then rerun the export."
        )
    else:
        st.markdown(
            '<div class="caveat-banner">This reflects local test traffic only, not real '
            "production volume -- generated while manually testing the API during "
            "development.</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"##### Requests logged: {len(log_df)}")
            fig5 = px.histogram(
                log_df,
                x="probability_malignant",
                nbins=10,
                color_discrete_sequence=["#5B5BD6"],
            )
            fig5.update_layout(
                height=300, xaxis_title="Predicted probability (malignant)"
            )
            st.plotly_chart(_fig(fig5), use_container_width=True)
        with c2:
            st.markdown("##### Requests over time")
            log_df_sorted = log_df.sort_values("timestamp_utc")
            fig6 = px.scatter(
                log_df_sorted,
                x="timestamp_utc",
                y="probability_malignant",
                color_discrete_sequence=["#5B5BD6"],
            )
            fig6.update_layout(height=300, xaxis_title="", yaxis_title="P(malignant)")
            st.plotly_chart(_fig(fig6), use_container_width=True)

        st.markdown("##### Logged requests")
        display_cols = [
            "prediction_id",
            "timestamp_utc",
            "prediction_label",
            "probability_malignant",
        ]
        st.dataframe(log_df[display_cols], use_container_width=True, hide_index=True)

        if contrib_log_df is not None:
            selected_id = st.selectbox(
                "Inspect a logged prediction's top contributors",
                log_df["prediction_id"].tolist(),
            )
            sub = contrib_log_df[
                contrib_log_df["prediction_id"] == selected_id
            ].sort_values("rank")
            st.dataframe(
                sub[["rank", "feature", "shap_value", "direction"]],
                use_container_width=True,
                hide_index=True,
            )

# -------------------------------------------------------------------- Try It
if selected == "Try It":
    st.markdown(
        "Enter measurements manually, or load a real held-out case to start from realistic values."
    )

    if "form_values" not in st.session_state:
        st.session_state.form_values = test_df.iloc[0][FEATURE_COLUMNS].to_dict()

    if st.button("Load a random held-out case"):
        sample_row = test_df.sample(1).iloc[0]
        st.session_state.form_values = sample_row[FEATURE_COLUMNS].to_dict()

    mean_cols = [c for c in FEATURE_COLUMNS if c.endswith("_mean")]
    se_cols = [c for c in FEATURE_COLUMNS if c.endswith("_se")]
    worst_cols = [c for c in FEATURE_COLUMNS if c.endswith("_worst")]

    col_a, col_b, col_c = st.columns(3)
    values = {}
    for col_group, group_cols in [
        (col_a, mean_cols),
        (col_b, se_cols),
        (col_c, worst_cols),
    ]:
        with col_group:
            st.caption(group_cols[0].split("_")[-1].upper())
            for feat in group_cols:
                values[feat] = st.number_input(
                    feat,
                    min_value=0.0,
                    value=float(st.session_state.form_values[feat]),
                    format="%.4f",
                    key=f"input_{feat}",
                )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])[FEATURE_COLUMNS]
        result = predict_row(row)

        st.write("")
        col_result, col_contrib = st.columns([1, 1.4])
        with col_result:
            st.markdown(
                diagnosis_badge(result["prediction_label"]), unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="margin-top:0.8rem" class="metric-value">'
                f'{result["probability_malignant"]:.1%}</div>'
                f'<div class="metric-sub">probability of malignancy</div>',
                unsafe_allow_html=True,
            )
        with col_contrib:
            contrib_df = pd.DataFrame(result["top_contributors"]).sort_values(
                "shap_value"
            )
            fig7 = px.bar(
                contrib_df,
                x="shap_value",
                y="feature",
                orientation="h",
                color=contrib_df["shap_value"] > 0,
                color_discrete_map={True: "#5B5BD6", False: "#C7C7E8"},
            )
            fig7.update_layout(
                height=260,
                showlegend=False,
                xaxis_title="SHAP contribution",
                yaxis_title="",
            )
            st.plotly_chart(_fig(fig7), use_container_width=True)
