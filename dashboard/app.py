"""
Monitoring dashboard for constraints_2_json.

    streamlit run dashboard/app.py

Pages: batch evaluation reports (read-only), live interpreter monitoring from the
event log, and the training-data collection process.
"""
import os
import sys

import streamlit as st

# Pages import src.*; streamlit only puts dashboard/ on the path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.monitoring import DEFAULT_EVENT_LOG  # noqa: E402

st.set_page_config(page_title="constraints_2_json monitoring", layout="wide")

with st.sidebar:
    st.header("Settings")
    st.text_input("Reports folder", "docs/experiments", key="reports_dir",
                  help="Experiment reports to load. They are only read, never changed.")
    st.text_input("Event log", DEFAULT_EVENT_LOG, key="event_log")
    st.text_input("Collected training data", "training_data_collected.json",
                  key="collected_path")
    st.number_input("Rules per day (cost projection)", min_value=1, value=10_000,
                    step=1_000, key="rules_per_day")
    st.number_input("Refresh every (s)", min_value=2, value=5, key="refresh_s")

pages = st.navigation([
    st.Page("pages/batch.py", title="Batch evaluation", icon=":material/bar_chart:"),
    st.Page("pages/live.py", title="Live monitoring", icon=":material/monitor_heart:"),
    st.Page("pages/training.py", title="Training data", icon=":material/dataset:"),
])
pages.run()
