import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.device_page import display_device_reports
from app.food_page import display_food_reports
from app.drug_page import display_drug_reports
from app.tobacco_page import display_tobacco_reports

# Initialize session state for global parameters if they don't exist
if 'sample_size' not in st.session_state:
    st.session_state.sample_size = 1000
if 'start_date' not in st.session_state:
    st.session_state.start_date = '2004-01-01'
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now().strftime('%Y-%m-%d')
if 'top_n_results' not in st.session_state:
    st.session_state.top_n_results = 20

def display_home():
    st.title("FDA Data Analysis Dashboard")

    # Global Controls Section
    st.header("Global Controls")

    # Create two columns for controls
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Sampling")
        # Sample size slider
        st.session_state.sample_size = st.slider(
            "Maximum Sample Size",
            min_value=100,
            max_value=10000,
            value=st.session_state.sample_size,
            step=100,
            help="Maximum number of records to fetch for each query"
        )

        # Top N results slider
        st.session_state.top_n_results = st.slider(
            "Number of Top Results to Display",
            min_value=5,
            max_value=50,
            value=st.session_state.top_n_results,
            step=5,
            help="Number of top results to show in charts and tables"
        )

    with col2:
        st.subheader("Date Range")
        # Date range picker
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(st.session_state.start_date, '%Y-%m-%d'),
            min_value=datetime(2004, 1, 1),
            max_value=datetime.now(),
            help="Start date for data analysis"
        )
        st.session_state.start_date = start_date.strftime('%Y-%m-%d')

        end_date = st.date_input(
            "End Date",
            value=datetime.strptime(st.session_state.end_date, '%Y-%m-%d'),
            min_value=datetime(2004, 1, 1),
            max_value=datetime.now(),
            help="End date for data analysis"
        )
        st.session_state.end_date = end_date.strftime('%Y-%m-%d')

    # Dashboard Description
    st.header("Dashboard Overview")
    st.markdown("""
    This dashboard provides comprehensive analysis of FDA data across multiple categories:

    - **Devices**: Analysis of medical device events, including device class distribution, problems, and manufacturer analysis
    - **Drugs**: Analysis of drug adverse events and safety data
    - **Food**: Analysis of food recalls and safety data
    - **Tobacco**: Analysis of tobacco product reports and safety data

    Use the global controls above to adjust the data sampling and date range for all analyses.
    """)

    # Display current settings
    st.subheader("Current Settings")
    st.info(f"""
    - Maximum Sample Size: {st.session_state.sample_size:,} records
    - Top Results to Display: {st.session_state.top_n_results}
    - Date Range: {st.session_state.start_date} to {st.session_state.end_date}
    """)

def main():
    st.set_page_config(
        page_title="FDA Data Analysis Dashboard",
        page_icon="🏥",
        layout="wide"
    )

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Home", "Devices", "Drugs", "Food", "Tobacco"
    ])

    with tab1:
        display_home()

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
