"""Q-MedAI combined faculty demonstration."""

from pathlib import Path

import streamlit as st

from src.demo.pages import render_app


PROJECT_ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Q-MedAI", page_icon="🧬", layout="wide")


if __name__ == "__main__":
    render_app(PROJECT_ROOT)
