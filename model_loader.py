"""model_loader.py — loads the trained model once per Streamlit session."""
import json
import joblib
import streamlit as st


@st.cache_resource
def load_model():
    """Returns (model, meta). Cached, so the pickle is read once per server process."""
    model = joblib.load("burnout_model.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta
