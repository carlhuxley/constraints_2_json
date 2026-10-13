"""
Monitoring dashboard for constraints_2_json.

    streamlit run dashboard/app.py

Pages: batch evaluation reports (read-only), live interpreter monitoring from the
event log, and the training-data collection process.
"""
import os

import streamlit as st

import dash_common  # noqa: F401  (puts the repo root on sys.path for src imports)
from src.monitoring import STREAMS

st.set_page_config(page_title="constraints_2_json monitoring", layout="wide")


def use_stream() -> None:
    """Point the live pages at the chosen stream's files."""
    st.session_state.event_log, st.session_state.collected_path = (
        STREAMS[st.session_state.stream])


# First run of a session: Real, unless C2J_EVENT_LOG / C2J_COLLECTED say otherwise
if "stream" not in st.session_state:
    event_log = os.environ.get("C2J_EVENT_LOG", STREAMS["Real"][0])
    collected = os.environ.get("C2J_COLLECTED", STREAMS["Real"][1])
    st.session_state.stream = next(
        (name for name, paths in STREAMS.items() if paths[0] == event_log), "Real")
    st.session_state.event_log, st.session_state.collected_path = event_log, collected

with st.sidebar:
    st.header("Settings")
    st.radio("Stream", list(STREAMS), key="stream", horizontal=True, on_change=use_stream,
             help="Real: events from CLI runs. Simulated: the replay written by "
                  "`python -m scripts.simulate_live`.")
    st.text_input("Event log", key="event_log")
    st.text_input("Collected training data", key="collected_path")
    st.text_input("Reports folder", "docs/experiments", key="reports_dir",
                  help="Experiment reports to load. They are only read, never changed.")
    st.number_input("Rules per day (cost projection)", min_value=1, value=10_000,
                    step=1_000, key="rules_per_day")
    st.number_input("Refresh every (s)", min_value=2, value=5, key="refresh_s")

pages = st.navigation([
    st.Page("views/batch.py", title="Batch evaluation", icon=":material/bar_chart:"),
    st.Page("views/live.py", title="Live monitoring", icon=":material/monitor_heart:"),
    st.Page("views/training.py", title="Training data", icon=":material/dataset:"),
])
pages.run()
