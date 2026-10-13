"""Live monitoring: tails the interpreter event log."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dash_common import SOURCE_COLOURS, dollars, pct, setting
from src import monitoring_data as md
from src.monitoring import DEFAULT_EVENT_LOG

st.title("Live monitoring")
path = setting("event_log", DEFAULT_EVENT_LOG)
window = st.slider("Rolling window (calls)", 5, 200, 20)
latest_only = st.toggle("Latest run only", value=False)


def new_events() -> list:
    """All events so far; reads only what was appended since the last refresh."""
    state = st.session_state
    if state.get("live_path") != path:
        state.live_path, state.live_offset, state.live_events = path, 0, []
    previous = state.live_offset
    events, state.live_offset = md.read_events(path, previous)
    # A smaller offset means the log was truncated or replaced: start over
    state.live_events = events if state.live_offset < previous else state.live_events + events
    return state.live_events


@st.fragment(run_every=setting("refresh_s", 5))
def live_view():
    events = new_events()
    calls = [e for e in events if e.get("type") == "interpret"]
    if latest_only and calls:
        run = calls[-1]["run_id"]
        events = [e for e in events if e.get("run_id") == run]
        calls = [e for e in calls if e["run_id"] == run]

    if not calls:
        st.info(f"No interpret events in `{path}` yet. Run the CLI with a T5 adapter, e.g.\n\n"
                "`python -m src.main -s examples/financial_schema.json "
                "-d examples/financial_dictionary.csv "
                "--adapter adapters/adapter_financial_v2/final_adapter "
                "--confidence-threshold 0.7 -o enriched.json`")
        return

    s = md.summarize(events, rules_per_day=setting("rules_per_day", 10_000))
    k = st.columns(6)
    k[0].metric("Rules interpreted", s["calls"], help=f"{s['runs']} run(s)")
    k[1].metric("T5 answered", pct(s["t5_success_rate"]))
    k[2].metric("Fallback rate", pct(s["fallback_rate"]))
    k[3].metric("LLM error rate", pct(s["llm_error_rate"]),
                help=f"{s['llm_errors']} of {s['llm_calls']} LLM calls failed")
    k[4].metric("LLM spend", dollars(s["spend"]))
    k[5].metric("Projected / month", dollars(s["projected_monthly_cost"]),
                help=f"Cost per rule so far × {setting('rules_per_day', 10_000):,} rules/day × 30")
    if s["unanswered"] or s["t5_kept_after_llm_failure"]:
        st.warning(f"{s['unanswered']} rule(s) got no answer; {s['t5_kept_after_llm_failure']} "
                   "kept a low-confidence T5 answer because the LLM call failed.")

    df = pd.DataFrame(calls)
    df["n"] = range(1, len(df) + 1)
    df["answered_by"] = df["source"].fillna("none")

    left, right = st.columns(2)
    share = pd.DataFrame({
        "call": df["n"],
        **{src: (df["answered_by"] == src).rolling(window, min_periods=1).mean()
           for src in ("t5", "llm", "none")},
    }).melt(id_vars="call", var_name="answered by", value_name="share")
    fig = px.line(share, x="call", y="share", color="answered by",
                  color_discrete_map=SOURCE_COLOURS,
                  title=f"Who answered, rolling over {window} calls")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.02], height=360)
    left.plotly_chart(fig, width="stretch")

    reasons = pd.DataFrame(sorted(s["fallback_reasons"].items(), key=lambda kv: kv[1]),
                           columns=["reason", "count"])
    fig = px.bar(reasons, x="count", y="reason", orientation="h",
                 title="Why T5 was not used")
    fig.update_traces(marker_color=SOURCE_COLOURS["llm"])
    fig.update_layout(height=360)
    if reasons.empty:
        right.success("No fallbacks: T5 answered every rule.")
    else:
        right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    conf = df.dropna(subset=["t5_confidence"]) if "t5_confidence" in df else df.iloc[0:0]
    if not conf.empty:
        fig = px.histogram(conf, x="t5_confidence", color="answered_by", nbins=20,
                           color_discrete_map=SOURCE_COLOURS, range_x=[0, 1],
                           title="T5 confidence (least certain token)")
        threshold = conf["confidence_threshold"].dropna()
        if not threshold.empty:
            fig.add_vline(x=threshold.iloc[-1], line_dash="dash",
                          annotation_text=f"threshold {threshold.iloc[-1]}")
        fig.update_layout(height=360, bargap=0.05)
        left.plotly_chart(fig, width="stretch")

    fig = px.box(df, x="answered_by", y="total_latency_ms", color="answered_by",
                 color_discrete_map=SOURCE_COLOURS, points="all", log_y=True,
                 title="Latency per rule by source (log scale)")
    fig.update_layout(height=360, showlegend=False, yaxis_title="ms")
    right.plotly_chart(fig, width="stretch")

    st.subheader("Failures stream")
    stream = md.failure_stream(events, limit=100)
    if stream:
        cols = ["ts", "field_name", "rule", "fallback_reason", "t5_confidence", "t5_output",
                "llm_output", "source", "llm_error", "validation_errors"]
        table = pd.DataFrame(stream).reindex(columns=cols)
        for col in ("t5_output", "llm_output", "validation_errors"):
            table[col] = table[col].map(lambda v: "" if v is None else str(v))
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.success("No fallbacks or unanswered rules.")


live_view()
