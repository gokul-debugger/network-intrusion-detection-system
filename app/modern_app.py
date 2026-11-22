"""Deployment entrypoint for the modern-upgrade branch's NSL-KDD reference preview."""

import os
import logging
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src import dashboard_ui as ui

st.set_page_config(page_title="Intrusion Monitor", page_icon=":material/shield_lock:", layout="wide")
st.html("""<style>
  .stMainBlockContainer {max-width:1280px;padding-top:2rem;padding-bottom:3rem;}
  h1 {font-size:2rem !important;letter-spacing:0;} h2 {font-size:1.4rem !important;letter-spacing:0;}
  [data-testid="stMetricValue"] {font-size:1.8rem;}
  [data-testid="stSidebar"] h1 {font-size:1.35rem;}
  @media(max-width:700px){.stMainBlockContainer{padding-left:1rem;padding-right:1rem;}}
</style>""")

with st.sidebar:
    st.title("Intrusion Monitor")
    st.caption("REFERENCE EDITION · NSL-KDD")

try:
    ui.bundle()
except Exception:
    logging.getLogger(__name__).exception("Reference bundle initialization failed")
    st.error("The reference release could not be loaded. Deployment dependencies or assets need verification.")
    st.stop()

page = st.navigation([
    st.Page("pages/connections.py", title="Connections", icon=":material/radar:", default=True),
    st.Page("pages/traffic.py", title="Traffic logs", icon=":material/monitoring:"),
    st.Page("pages/evidence.py", title="Model evidence", icon=":material/fact_check:"),
    st.Page("pages/explanations.py", title="Explanations", icon=":material/insights:"),
])

with st.sidebar:
    st.divider()
    st.caption("LightGBM · frozen evaluation v3")
    st.link_button("Source repository", ui.REPO, icon=":material/code:")
    if st.button("Clear session data", icon=":material/delete:"):
        for key in list(st.session_state):
            if key.startswith("private_"): del st.session_state[key]
        st.session_state["upload_epoch"] = st.session_state.get("upload_epoch", 0) + 1
        st.rerun()

page.run()
