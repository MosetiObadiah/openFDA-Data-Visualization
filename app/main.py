import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.device_page import display_device_reports
from app.food_page import display_food_reports
from app.drug_page import display_drug_reports
from app.tobacco_page import display_tobacco_reports
from app.other_page import display_other_data
from app.healthcare_trends import display_healthcare_trends
from app.correlation_analysis import display_correlation_analysis
from src.data_utils import clear_cache

def display_home():
    # Global Controls Section
    st.header("Global Controls")

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
            help="Maximum number of records to fetch for each query",
            key="home_sample_size"
        )

        # Top N results slider
        st.session_state.top_n_results = st.slider(
            "Number of Top Results to Display",
            min_value=5,
            max_value=50,
            value=st.session_state.top_n_results,
            step=5,
            help="Number of top results to show in charts and tables",
            key="home_top_n_results"
        )

    with col2:
        st.subheader("Date Range")
        # Date range picker
        st.session_state.start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            min_value=date(2020, 1, 1),
            max_value=date.today(),
            help="Start date for data analysis",
            key="home_start_date"
        )
        st.session_state.end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            min_value=st.session_state.start_date,
            max_value=date.today(),
            help="End date for data analysis",
            key="home_end_date"
        )

    # Dashboard Description
    st.header("Dashboard Overview")
    st.markdown("""
    This dashboard provides comprehensive analysis of FDA data across multiple categories:

    - **Devices**: Analysis of medical device events, including device class distribution, problems, and manufacturer analysis
    - **Drugs**: Analysis of drug adverse events and safety data
    - **Food**: Analysis of food recalls and safety data
    - **Tobacco**: Analysis of tobacco product reports and safety data
    - **Other**: Other FDA datasets including substance and NSDE data
    - **Healthcare Trends**: AI-powered prediction of future healthcare trends based on FDA data
    - **Correlation Analysis**: Cross-category analysis that identifies relationships between different data domains

    Use the global controls above to adjust the data sampling and date range for all analyses.
    Each tab features subtabs with specialized analyses and visualizations.
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
        page_icon="",
        layout="wide"
    )

    # Initializing session state variables if they don't exist
    if "sample_size" not in st.session_state:
        st.session_state.sample_size = 1000
    if "top_n_results" not in st.session_state:
        st.session_state.top_n_results = 10
    if "start_date" not in st.session_state:
        st.session_state.start_date = date(2020, 1, 1)
    if "end_date" not in st.session_state:
        st.session_state.end_date = date.today()

    # Sidebar for global controls
    with st.sidebar:
        st.header("Global Controls")

        # Sample size control
        st.session_state.sample_size = st.slider(
            "Maximum Sample Size",
            min_value=100,
            max_value=10000,
            value=st.session_state.sample_size,
            step=100,
            help="Maximum number of records to fetch from the database",
            key="sidebar_sample_size"
        )

        # Top N results control
        st.session_state.top_n_results = st.slider(
            "Top N Results",
            min_value=5,
            max_value=50,
            value=st.session_state.top_n_results,
            step=5,
            help="Number of top results to display in charts and tables",
            key="sidebar_top_n_results"
        )

        # Date range controls
        st.session_state.start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            min_value=date(2018, 1, 1),
            max_value=date.today(),
            help="Start date for data analysis",
            key="sidebar_start_date"
        )

        st.session_state.end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            min_value=st.session_state.start_date,
            max_value=date.today(),
            help="End date for data analysis",
            key="sidebar_end_date"
        )

        st.subheader("Current Settings")
        st.write(f"Sample Size: {st.session_state.sample_size:,} records")
        st.write(f"Top N Results: {st.session_state.top_n_results}")
        st.write(f"Date Range: {st.session_state.start_date} to {st.session_state.end_date}")

        st.subheader("Data Source")
        st.write("All data is retrieved in real-time from the [OpenFDA API](https://open.fda.gov/apis/).")

        st.subheader("About")
        st.write("Created with Streamlit & OpenFDA API © 2024")

        st.subheader("Troubleshooting")
        st.caption("If you get a value error of any kind, try connecting to the internet again or use a vpn if available")
        st.caption("If data is not loading or tables appear broken, try clearing the cache.")
        if st.button("Clear Cache", key="sidebar_clear_cache"):
            clear_cache()
            st.success("Cache cleared successfully!")

    # Main content area
    st.title("Data Analysis Dashboard")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Overview",
        "Drug Reports",
        "Device Reports",
        "Food Reports",
        "Tobacco Reports",
        "Other Data",
        "Healthcare Trends",
        "Correlation Analysis"
    ])

    with tab1:
        display_home()

    with tab2:
        display_drug_reports()

    with tab3:
        display_device_reports()

    with tab4:
        display_food_reports()

    with tab5:
        display_tobacco_reports()

    with tab6:
        display_other_data()

    with tab7:
        display_healthcare_trends()

    with tab8:
        display_correlation_analysis()

if __name__ == "__main__":
    main()
