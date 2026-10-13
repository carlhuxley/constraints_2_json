"""Shared colours and helpers for the dashboard pages.

Every page imports this first: it puts the repo root on sys.path so `src` imports
work however Streamlit runs the page (Streamlit only adds dashboard/ itself).
"""
import os
import sys

import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The same source keeps the same colour on every chart
SOURCE_COLOURS = {"t5": "#2b6cb0", "llm": "#dd6b20", "none": "#a0aec0",
                  "llm_variation": "#f6ad55"}
GOOD, BAD = "#2f855a", "#c53030"


def pct(value, digits=1):
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def dollars(value):
    """Small LLM spends need more decimals than whole-month projections."""
    if value is None:
        return "–"
    return f"${value:.5f}" if value < 0.01 else f"${value:,.2f}"


def setting(key, default=None):
    """A sidebar setting from app.py (falls back when a page is run on its own)."""
    return st.session_state.get(key, default)
