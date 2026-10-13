"""
Monitoring dashboard for constraints_2_json.

    streamlit run dashboard/app.py

Pages: batch evaluation reports (read-only), live interpreter monitoring from the
event log, and the training-data collection process.
"""
import os

import streamlit as st

import dash_common  # noqa: F401  (puts the repo root on sys.path for src imports)
from src.monitoring import DEFAULT_EVENT_LOG

st.set_page_config(page_title="constraints_2_json monitoring", layout="wide")

with st.sidebar:
    st.header("Settings")
    st.text_input("Reports folder", "docs/experiments", key="reports_dir",
                  help="Experiment reports to load. They are only read, never changed.")
    # Env vars let a launcher (e.g. scripts/simulate_live.py) point the pages elsewhere
    st.text_input("Event log", os.environ.get("C2J_EVENT_LOG", DEFAULT_EVENT_LOG),
                  key="event_log")
    st.text_input("Collected training data",
                  os.environ.get("C2J_COLLECTED", "training_data_collected.json"),
                  key="collected_path")
    st.number_input("Rules per day (cost projection)", min_value=1, value=10_000,
                    step=1_000, key="rules_per_day")
    st.number_input("Refresh every (s)", min_value=2, value=5, key="refresh_s")

pages = st.navigation([
    st.Page("views/batch.py", title="Batch evaluation", icon=":material/bar_chart:"),
    st.Page("views/live.py", title="Live monitoring", icon=":material/monitor_heart:"),
    st.Page("views/training.py", title="Training data", icon=":material/dataset:"),
])
pages.run()
