import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from drug_page import display_drug_reports
from device_page import display_device_reports

def display_home_page():
    st.markdown("# OpenFDA Data Visualization Dashboard")

    st.markdown("## Overview")
    st.markdown("Interactive dashboard analyzing adverse events from the [OpenFDA API](https://open.fda.gov/apis/). Features include device analysis, drug events, and trend visualization.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Key Features")
        st.markdown("""
        - **Data Analysis**
          - Device class distribution
          - Manufacturer analysis
          - Geographic patterns
          - Problem trends

        - **AI Integration**
          - Pattern recognition
          - Risk assessment
          - Market insights
        """)

    with col2:
        st.markdown("### Dashboard Sections")
        st.markdown("""
        - **Device Reports**
          - Class distribution
          - Problem analysis
          - Manufacturer insights
          - Geographic patterns

        - **Drug Reports**
          - Adverse reactions
          - Monthly trends
          - Event summaries
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
    display_drug_reports()

with tabs[2]:
    display_device_reports()

with tabs[3]:
    st.title("Food Reports")

with tabs[4]:
    st.title("Tobacco Reports")
