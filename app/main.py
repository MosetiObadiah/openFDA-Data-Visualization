import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from drug_page import (
    display_drug_reports
)

def display_home_page():
    st.markdown("# OpenFDA Data Visualization Dashboard")

    st.markdown("## Overview")
    st.markdown("Interactive dashboard analyzing adverse drug events from the [OpenFDA API](https://open.fda.gov/apis/drug/event/). Features include sentiment analysis, demographic insights, and trend visualization.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Key Features")
        st.markdown("""
        - **Data Analysis**
          - 1,000+ adverse event reports
          - Patient demographics
          - Monthly trends
          - Common reactions

        - **AI Integration**
          - Sentiment analysis
          - Pattern recognition
          - Risk scoring
        """)

    with col2:
        st.markdown("### Dashboard Sections")
        st.markdown("""
        - **Drug Reports**
          - Adverse reactions
          - Monthly trends
          - Event summaries

        - **Demographics**
          - Age/sex distribution
          - Geographic patterns

        - **Sentiment Analysis**
          - Reaction sentiment
          - Risk assessment
        """)

    st.markdown("---")

st.set_page_config(
    page_title="OpenFDA Dashboard",
    layout="wide"
    )

tabs = st.tabs(["Home", "Drugs", "Devices", "Food", "Tobacco"])

with tabs[0]:
    display_home_page()

with tabs[1]:
    st.title("Drug Reports")
    display_drug_reports()

with tabs[2]:
    st.title("Device Reports")

with tabs[3]:
    st.title("Food Reports")


with tabs[4]:
    st.title("Tobacco Reports")
