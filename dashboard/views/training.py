"""Training-data collection: what the hybrid has gathered for the next retrain."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dash_common import SOURCE_COLOURS, setting
from src import monitoring_data as md
from src.monitoring import DEFAULT_EVENT_LOG

st.title("Training data collection")
st.caption("Examples are collected when the CLI runs with `--collect-training-data`. "
           "LLM answers also generate paraphrased variations (`llm_variation`).")
target = st.number_input("Retrain when this many new examples are collected",
                         min_value=1, value=200, step=50)


@st.cache_data(ttl=300)
def known_inputs() -> set:
    return md.known_training_inputs()


@st.fragment(run_every=setting("refresh_s", 5) * 2)
def training_view():
    events, _ = md.read_events(setting("event_log", DEFAULT_EVENT_LOG))
    s = md.training_summary(events, setting("collected_path", "training_data_collected.json"),
                            known_inputs())

    new_examples = s["events"] - s["duplicates_of_training_data"]
    k = st.columns(4)
    k[0].metric("Collected (event log)", s["events"])
    k[1].metric("Already in training data", s["duplicates_of_training_data"],
                help="Same rule text (normalised) as an example in training_data/*.json")
    k[2].metric("New examples", new_examples)
    k[3].metric("In collected file", "–" if s["collected_file_total"] is None
                else s["collected_file_total"],
                help="Includes examples collected before event logging existed")
    st.progress(min(new_examples / target, 1.0),
                text=f"{new_examples} of {target} new examples towards the next retrain")

    if not s["recent"]:
        st.info("No training examples in the event log yet.")
        return

    df = pd.DataFrame(reversed(s["recent"]))  # oldest first for the cumulative chart
    df["ts"] = pd.to_datetime(df["ts"])
    df["collected"] = df.groupby("source").cumcount() + 1
    fig = px.line(df, x="ts", y="collected", color="source", markers=True,
                  color_discrete_map=SOURCE_COLOURS, line_shape="hv",
                  title="Examples collected over time, by source")
    fig.update_layout(height=360, xaxis_title="", yaxis_title="examples")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Most recent examples")
    table = pd.DataFrame(s["recent"][:200])
    table["output"] = table["output"].map(str)
    st.dataframe(table, width="stretch", hide_index=True)


training_view()
