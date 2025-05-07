import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
from src.device_events import (
    device_class_distribution,
    device_problems_by_year,
    device_manufacturer_analysis,
    get_top_device_classes,
    get_device_problems_trend,
    get_manufacturer_market_share
)
from src.utils import get_state_abbreviations
from src.components import (
    render_metric_header,
    render_data_table
)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_insights_from_data(df: pd.DataFrame, context: str, custom_question: str = None) -> str:
    """Use Gemini API to generate insights from the DataFrame."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    if custom_question:
        prompt = f"""
        Based on the following data about {context}:

        {df.to_string()}

        Answer the following question in 5-7 sentences, focusing on data-driven insights and potential implications:
        {custom_question}
        """
    else:
        prompt = f"""
        Analyze the following data about {context}:

        {df.to_string()}

        Provide a comprehensive analysis in 5-7 sentences, focusing on:
        1. Key trends and patterns
        2. Notable findings
        3. Potential implications
        4. Recommendations
        5. Areas for further investigation
        """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def display_device_class_distribution():
    """Display device class distribution with visualizations."""
    metric_title = "Device Class Distribution"
    metric_description = (
        "This section shows the distribution of medical device classes, "
        "helping identify which types of devices are most commonly involved in events."
    )

    render_metric_header(metric_title, metric_description)

    try:
        df = device_class_distribution()
        if df.empty:
            st.warning("No data available for device class distribution.")
            return

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for top 10 device classes
            top_10 = get_top_device_classes(df)
            fig_bar = px.bar(
                top_10,
                x="Device Class",
                y="Count",
                text="Percentage",
                title="Top 10 Device Classes",
                labels={
                    "Device Class": "Device Class",
                    "Count": "Number of Events",
                    "Percentage": "Percentage"
                }
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create pie chart for device class distribution
            fig_pie = px.pie(
                df,
                values="Count",
                names="Device Class",
                title="Device Class Distribution",
                hover_data=["Percentage"]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Display detailed data
        st.markdown("### Detailed Statistics")
        render_data_table(df)

        # AI Insights
        st.markdown("### Insights")
        if st.button("Get General Insights for Device Classes"):
            insights = get_insights_from_data(df, "device class distribution")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about device classes", key="device_class_question")
        if custom_question:
            insights = get_insights_from_data(df, "device class distribution", custom_question)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load device class data: {e}")

def display_device_problems():
    """Display device problems analysis with visualizations."""
    metric_title = "Device Problems Analysis"
    metric_description = (
        "This section analyzes the types of problems reported with medical devices, "
        "helping identify common issues and trends."
    )

    render_metric_header(metric_title, metric_description)

    try:
        # Add year range selector
        current_year = datetime.now().year
        start_year = st.slider("Start Year", 2010, current_year, 2010)
        end_year = st.slider("End Year", start_year, current_year, current_year)

        df = device_problems_by_year(start_year, end_year)
        if df.empty:
            st.warning("No data available for device problems.")
            return

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            # Create horizontal bar chart for top problems
            top_problems = get_device_problems_trend(df).head(10)
            fig_bar = px.bar(
                top_problems,
                y="Problem",
                x="Count",
                text="Percentage",
                title="Top 10 Device Problems",
                orientation='h'
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create treemap for problem distribution
            fig_treemap = px.treemap(
                df,
                path=["Problem"],
                values="Count",
                title="Device Problems Distribution"
            )
            st.plotly_chart(fig_treemap, use_container_width=True)

        # Display detailed data
        st.markdown("### Detailed Statistics")
        render_data_table(df)

        # AI Insights
        st.markdown("### Insights")
        if st.button("Get General Insights for Device Problems"):
            insights = get_insights_from_data(df, "device problems")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about device problems", key="device_problems_question")
        if custom_question:
            insights = get_insights_from_data(df, "device problems", custom_question)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load device problems data: {e}")

def display_manufacturer_analysis():
    """Display manufacturer analysis with visualizations."""
    metric_title = "Manufacturer Analysis"
    metric_description = (
        "This section analyzes medical device manufacturers, "
        "showing market share and event distribution across different companies."
    )

    render_metric_header(metric_title, metric_description)

    try:
        df = device_manufacturer_analysis()
        if df.empty:
            st.warning("No data available for manufacturer analysis.")
            return

        # Calculate market share
        df = get_manufacturer_market_share(df)

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for top manufacturers
            fig_bar = px.bar(
                df,
                x="Manufacturer",
                y="Market Share",
                text="Count",
                title="Top 20 Manufacturers by Market Share",
                labels={
                    "Manufacturer": "Manufacturer",
                    "Market Share": "Market Share (%)",
                    "Count": "Number of Events"
                }
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create pie chart for manufacturer distribution
            fig_pie = px.pie(
                df,
                values="Count",
                names="Manufacturer",
                title="Manufacturer Distribution",
                hover_data=["Market Share"]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Display detailed data
        st.markdown("### Detailed Statistics")
        render_data_table(df)

        # AI Insights
        st.markdown("### Insights")
        if st.button("Get General Insights for Manufacturers"):
            insights = get_insights_from_data(df, "device manufacturers")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about manufacturers", key="manufacturer_question")
        if custom_question:
            insights = get_insights_from_data(df, "device manufacturers", custom_question)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load manufacturer data: {e}")

def display_device_reports():
    st.title("Device Reports")

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "Device Class Distribution",
        "Device Problems",
        "Manufacturer Analysis"
    ])

    with tab1:
        display_device_class_distribution()

    with tab2:
        display_device_problems()

    with tab3:
        display_manufacturer_analysis()
