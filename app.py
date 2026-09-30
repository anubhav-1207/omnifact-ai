import os
import json
import time
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING (Dark Theme Hackathon UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OmniExtract AI | Data Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }

    .status-badge {
        background-color: #238636;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        font-weight: bold;
        border-radius: 6px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
</style>
""", unsafe_allow_html=True)
