import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
from src.data_loader import fetch_api_data
from src.device_events import (
    device_class_distribution,
    device_problems_by_year,
    device_manufacturer_analysis,
    get_device_events_by_age
)

# Configure Gemini
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))

def get_insights_from_data(df: pd.DataFrame, context: str, custom_question: str = None) -> str:
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
    st.subheader("Device Class Distribution")

    # global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

    # Fetch and process data
    df = device_class_distribution()

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Device Class",
            y="Count",
            title="Device Class Distribution",
            labels={"Device Class": "Class", "Count": "Number of Devices"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Count",
            names="Device Class",
            title="Distribution of Device Classes"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    render_ai_insights_section(df, (start_str, end_str), "device_class")

def display_device_problems():
    st.subheader("Device Problems")

    # global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

    # Fetch and process data
    df = device_problems_by_year(st.session_state.start_date.year, st.session_state.end_date.year)

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Problem",
            y="Count",
            title="Device Problems",
            labels={"Problem": "Problem Type", "Count": "Number of Cases"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Count",
            names="Problem",
            title="Distribution of Device Problems"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    render_ai_insights_section(df, (start_str, end_str), "device_problems")

def display_manufacturer_analysis():
    st.subheader("Manufacturer Analysis")

    # global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

    # Fetch and process data
    df = device_manufacturer_analysis()

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Manufacturer",
            y="Count",
            title="Manufacturer Distribution",
            labels={"Manufacturer": "Manufacturer", "Count": "Number of Devices"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Count",
            names="Manufacturer",
            title="Distribution of Manufacturers"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    render_ai_insights_section(df, (start_str, end_str), "manufacturer")

def display_device_events_by_age():
    st.subheader("Device Events by Age")

    # global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

    # Fetch and process data
    df = get_device_events_by_age()

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Age Group",
            y="Count",
            title="Device Events by Age Group",
            labels={"Age Group": "Age Group", "Count": "Number of Events"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Count",
            names="Age Group",
            title="Distribution of Device Events by Age Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    render_ai_insights_section(df, (start_str, end_str), "device_age")

def display_device_reports():
    st.title("Device Reports")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Advisory Committees",
        "Manufacturers",
        "Enforcement Reports",
        "Recall Analysis"
    ])

    with tab1:
        display_advisory_committees()

    with tab2:
        display_manufacturers()

    with tab3:
        display_enforcement_reports()

    with tab4:
        display_recall_analysis()

def display_advisory_committees():
    st.subheader("Advisory Committee Distribution")

    endpoint = "device/510k.json"
    params = {
        "count": "advisory_committee_description",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        st.warning("No data available for advisory committees.")
        return

    df = pd.DataFrame(data["results"])
    df.columns = ["Committee", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        # Horizontal bar chart
        fig = px.bar(
            df,
            y="Committee",
            x="Count",
            title="Advisory Committee Distribution",
            orientation='h'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Treemap for hierarchical view
        fig = px.treemap(
            df,
            path=['Committee'],
            values='Count',
            title="Advisory Committee Distribution (Treemap)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics in expander
    with st.expander("Detailed Statistics", expanded=False):
        st.dataframe(df)

    render_ai_insights_section(df, "Advisory Committees", "advisory_committees")

def display_manufacturers():
    st.subheader("Manufacturer Analysis")

    endpoint = "device/510k.json"
    params = {
        "count": "applicant.exact",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        st.warning("No data available for manufacturers.")
        return

    df = pd.DataFrame(data["results"])
    df.columns = ["Manufacturer", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        # Horizontal bar chart for top manufacturers
        fig = px.bar(
            df.head(20),  # Show top 20 manufacturers
            y="Manufacturer",
            x="Count",
            title="Top 20 Manufacturers",
            orientation='h'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart for market share
        fig = px.pie(
            df.head(10),  # Show top 10 manufacturers
            values="Count",
            names="Manufacturer",
            title="Top 10 Manufacturers Market Share"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics in expander
    with st.expander("Detailed Statistics", expanded=False):
        st.dataframe(df)

    render_ai_insights_section(df, "Manufacturers", "device_manufacturers")

def display_enforcement_reports():
    st.subheader("Enforcement Reports Timeline")

    endpoint = "device/enforcement.json"
    params = {
        "count": "report_date",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        st.warning("No data available for enforcement reports.")
        return

    df = pd.DataFrame(data["results"])
    df.columns = ["Date", "Count"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    col1, col2 = st.columns(2)

    with col1:
        # Line chart for trend analysis
        fig = px.line(
            df,
            x="Date",
            y="Count",
            title="Enforcement Reports Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Area chart for cumulative view
        fig = px.area(
            df,
            x="Date",
            y="Count",
            title="Cumulative Enforcement Reports"
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed Statistics", expanded=False):
        st.dataframe(df)

    render_ai_insights_section(df, "Enforcement Reports", "enforcement_reports")

def display_recall_analysis():
    st.subheader("Recall Analysis")

    endpoint = "device/recall.json"
    params = {
        "count": "root_cause_description.exact",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        st.warning("No data available for recall analysis.")
        return

    df = pd.DataFrame(data["results"])
    df.columns = ["Root Cause", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        # Horizontal bar chart for root causes
        fig = px.bar(
            df,
            y="Root Cause",
            x="Count",
            title="Recall Root Causes",
            orientation='h'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sunburst chart for hierarchical view
        fig = px.sunburst(
            df,
            path=['Root Cause'],
            values='Count',
            title="Recall Root Causes (Sunburst)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics in expander
    with st.expander("Detailed Statistics", expanded=False):
        st.dataframe(df)

    render_ai_insights_section(df, "Recall Analysis", "recall_analysis")

def render_ai_insights_section(df, context, key_prefix):
    st.subheader("AI Insights")
    question = st.text_input("Custom question (optional)", key=f"{key_prefix}_question")
    if st.button("Generate Insights", key=f"{key_prefix}_insights"):
        with st.spinner("Generating insights..."):
            st.write(get_insights_from_data(df, context, question or ""))
