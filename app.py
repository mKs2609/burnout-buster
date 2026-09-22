"""
app.py — Burnout Buster, a student wellness screening tool for VIPS-TC.

This module is only the shell: page setup, session state, chrome and tab routing.
Each tab lives in views/, shared pieces in ui.py / charts.py / utils.py, the model
in scoring.py and storage in database.py.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Burnout Buster",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui import inject_css, render_footer, render_navbar
from views import analytics, counselor, home, model_card, portal, survey

TABS = [
    ("Home",            home),
    ("Take Survey",     survey),
    ("My Portal",       portal),
    ("Counselor",       counselor),
    ("Analytics",       analytics),
    ("About the Model", model_card),
]

def main():
    inject_css()
    for key, default in [("counselor_logged_in", False),
                         ("student_logged_in", False),
                         ("student_data", {})]:
        st.session_state.setdefault(key, default)

    render_navbar()
    for tab, (_, view) in zip(st.tabs([label for label, _ in TABS]), TABS):
        with tab:
            view.render()
    render_footer()

main()
