import streamlit as st
import pandas as pd
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.device_page import display_device_reports
from app.food_page import display_food_reports
from app.drug_page import display_drug_reports
from app.tobacco_page import display_tobacco_reports

def display_home_page():
    st.title("OpenFDA Data Visualization Dashboard")
    st.write("""
    Welcome to the OpenFDA Data Visualization Dashboard. This application provides comprehensive insights into adverse events and recalls across various FDA-regulated products.

    ### Key Features
    - Interactive visualizations
    - Real-time data analysis
    - AI-powered insights
    - Detailed statistics and trends

    ### Dashboard Sections
    - **Devices**: Analysis of medical device reports and recalls
    - **Drugs**: Analysis of drug adverse events and safety data
    - **Food**: Analysis of food recalls and safety events
    - **Tobacco**: Analysis of tobacco product data and compliance
    """)

def main():
    st.set_page_config(
        page_title="OpenFDA Dashboard",
        page_icon="🏥",
        layout="wide"
    )

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Devices", "Drugs", "Food", "Tobacco"])

    with tab1:
        display_home_page()

    with tab2:
        display_device_reports()

    with tab3:
        display_drug_reports()

    with tab4:
        display_food_reports()

    with tab5:
        display_tobacco_reports()

if __name__ == "__main__":
    main()
