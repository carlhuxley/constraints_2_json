"""Batch evaluation: metric rates, latency, failure drill-down and model comparison."""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dash_common import BAD, GOOD, pct, setting
from src import monitoring_data as md

st.title("Batch evaluation")
root = setting("reports_dir", "docs/experiments")
st.caption(f"Reports in `{root}` — read-only; the dashboard never writes to this folder.")


@st.cache_data
def load_all(root: str, mtimes: tuple) -> list:
    return [md.load_report(path, root) for path, _ in mtimes]


paths = md.find_reports(root) if os.path.isdir(root) else []
if not paths:
    st.info(f"No reports found in `{root}`. Run `python -m scripts.run_experiments "
            "--output-dir <folder>` and point the sidebar at that folder.")
    st.stop()

reports = load_all(root, tuple((p, os.path.getmtime(p)) for p in paths))
by_label = {r["label"]: r for r in reports}
chosen = st.multiselect("Conditions", list(by_label), default=list(by_label))
selected = [by_label[label] for label in chosen]
if not selected:
    st.stop()

# --- Metric rates -------------------------------------------------------------------
st.subheader("Metric rates")
rates = pd.DataFrame([
    {"Condition": r["label"], "Metric": metric, "Rate": value}
    for r in selected for metric, value in r["rates"].items() if value is not None
])
fig = px.bar(rates, x="Metric", y="Rate", color="Condition", barmode="group",
             text_auto=".1%", range_y=[0, 1.05])
fig.update_layout(yaxis_tickformat=".0%", legend_title_text="", height=420)
st.plotly_chart(fig, width="stretch")

# --- Latency ----------------------------------------------------------------------
st.subheader("Latency distribution")
with_cases = [r for r in selected if r["cases"]]
without_cases = [r for r in selected if not r["cases"]]
fig = go.Figure()
for r in with_cases:
    fig.add_trace(go.Box(y=[c["latency_ms"] for c in r["cases"]], name=r["label"],
                         boxpoints="all", jitter=0.4, pointpos=0, marker_size=4))
for r in without_cases:
    if r["median_latency_ms"] is not None:
        fig.add_trace(go.Scatter(
            x=[r["label"]] * 2, y=[r["median_latency_ms"], r["p99_latency_ms"]],
            mode="markers+text", text=["median", "p99"], textposition="middle right",
            marker=dict(size=10, symbol=["circle", "triangle-up"]), name=r["label"],
            showlegend=False))
fig.update_layout(yaxis_type="log", yaxis_title="ms per rule (log scale)", height=420,
                  showlegend=False)
st.plotly_chart(fig, width="stretch")
if without_cases:
    st.caption("Reports written before per-case rows were added show median and p99 only: "
               + ", ".join(r["label"] for r in without_cases) + ".")

# --- Failure drill-down -------------------------------------------------------------
st.subheader("Failures (semantic match)")
report = by_label[st.selectbox("Condition", chosen, key="drill_condition")]
failures = pd.DataFrame(md.semantic_failures(report))
if failures.empty:
    st.success("No semantic failures.")
else:
    cols = st.columns(2)
    types = sorted(t for t in failures["constraint_type"].dropna().unique())
    sources = sorted(s for s in failures["source"].dropna().unique())
    pick_types = cols[0].multiselect("Constraint type", types) if types else []
    pick_sources = cols[1].multiselect("Answered by", sources) if sources else []
    shown = failures
    if pick_types:
        shown = shown[shown["constraint_type"].isin(pick_types)]
    if pick_sources:
        shown = shown[shown["source"].isin(pick_sources)]
    st.caption(f"{len(shown)} of {len(failures)} failures out of {report['total_tests']} rules")
    st.dataframe(shown.dropna(axis=1, how="all"), width="stretch", hide_index=True)

# --- Comparison ---------------------------------------------------------------------
st.subheader("Compare two conditions")
cols = st.columns(2)
label_a = cols[0].selectbox("Baseline (A)", list(by_label), key="cmp_a")
label_b = cols[1].selectbox("Compared (B)", list(by_label),
                            index=min(1, len(by_label) - 1), key="cmp_b")
if label_a == label_b:
    st.info("Pick two different conditions.")
    st.stop()

result = md.compare(by_label[label_a], by_label[label_b])
if not result["same_dataset"]:
    st.warning("These reports were not scored on the same dataset (or don't record it): "
               "the paired comparison is not meaningful.")
m = st.columns(4)
m[0].metric("Fixed by B", len(result["fixed"]))
m[1].metric("Broken by B", len(result["broken"]))
m[2].metric("Wrong in both", result["still_wrong"])
m[3].metric("McNemar exact p", f"{result['p_value']:.3g}")
st.caption("Fixed = A wrong, B right; broken = A right, B wrong. A small p means the "
           "difference is unlikely to be noise on this many rules.")

deltas = pd.DataFrame([r for r in result["by_type"] if r["delta"] is not None])
if not deltas.empty:
    deltas = deltas.sort_values("delta")
    fig = go.Figure(go.Bar(
        x=deltas["delta"], y=deltas["constraint_type"], orientation="h",
        marker_color=[GOOD if d > 0 else BAD for d in deltas["delta"]],
        text=[f"{pct(a, 0)} → {pct(b, 0)} (n={n})" for a, b, n in
              zip(deltas["a"], deltas["b"], deltas["total_tests"])],
        textposition="outside"))
    fig.update_layout(title=f"Semantic match change by constraint type ({label_b} − {label_a})",
                      xaxis_tickformat="+.0%", height=40 * len(deltas) + 140,
                      margin=dict(l=10, r=160))
    st.plotly_chart(fig, width="stretch")

for title, rows in (("Fixed by B", result["fixed"]), ("Broken by B", result["broken"])):
    with st.expander(f"{title} ({len(rows)})"):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
