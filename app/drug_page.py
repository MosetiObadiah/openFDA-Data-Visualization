import streamlit as st
import pandas as pd
from datetime import date
import google.generativeai as genai
import os
from dotenv import load_dotenv
import plotly.express as px

from src.drug_events import (
    adverse_events_by_patient_age_group_within_data_range,
    adverse_events_by_drug_within_data_range,
    recall_frequency_by_year,
    most_common_recalled_drugs,
    recall_reasons_over_time,
    get_aggregated_age_data,
    get_top_drugs,
    get_recall_reasons_pivot,
    get_actions_taken_with_drug,
    adverse_events_by_country
)
from src.utils import get_state_abbreviations
from src.components import (
    render_metric_header,
    render_date_picker,
    render_data_table,
    render_age_filter,
    render_bar_chart,
    render_line_chart
)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_insights_from_data(df: pd.DataFrame, filter_range: tuple, filter_col: str, custom_question: str = None) -> str:
    """Use Gemini API to generate insights from the DataFrame, with an optional custom question."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    if custom_question:
        prompt = f"""
        Based on the following data about {filter_col}:

        {df.to_string()}

        Answer the following question in a concise manner (3-5 sentences) using a conversational tone as if explaining to a non-expert:
        {custom_question}
        """
    else:
        prompt = f"""
        Analyze the following data about {filter_col}:

        {df.to_string()}

        Provide a concise explanation (3-5 sentences) of what the user is seeing, highlighting key trends, patterns, or notable points in the data. Use a conversational tone as if explaining to a non-expert.
        """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def display_adverse_events_by_age():
    """Display adverse events by age group."""
    st.subheader("Adverse Events by Age Group")

    # Use global date range from session state
    start_str = st.session_state.start_date
    end_str = st.session_state.end_date

    # Fetch and process data
    df = adverse_events_by_patient_age_group_within_data_range(start_str, end_str)

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Patient Age",
            y="Adverse Event Count",
            title="Adverse Events by Age Group",
            labels={"Patient Age": "Age Group", "Adverse Event Count": "Number of Events"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Adverse Event Count",
            names="Patient Age",
            title="Distribution of Adverse Events by Age Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    # AI Insights section
    st.subheader("AI Insights")
    if st.button("Generate Insights", key="age_insights"):
        insights = get_insights_from_data(df)
        st.write(insights)

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="age_question")
    if question and st.button("Get Answer", key="age_answer"):
        insights = get_insights_from_data(df, question)
        st.write(insights)

def display_adverse_events_by_drug():
    """Display adverse events by drug."""
    st.subheader("Adverse Events by Drug")

    # Use global date range from session state
    start_str = st.session_state.start_date
    end_str = st.session_state.end_date

    # Fetch and process data
    df = adverse_events_by_drug_within_data_range(start_str, end_str)

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Drug Name",
            y="Adverse Event Count",
            title="Adverse Events by Drug",
            labels={"Drug Name": "Drug", "Adverse Event Count": "Number of Events"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Adverse Event Count",
            names="Drug Name",
            title="Distribution of Adverse Events by Drug"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    # AI Insights section
    st.subheader("AI Insights")
    if st.button("Generate Insights", key="drug_insights"):
        insights = get_insights_from_data(df)
        st.write(insights)

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="drug_question")
    if question and st.button("Get Answer", key="drug_answer"):
        insights = get_insights_from_data(df, question)
        st.write(insights)

def display_global_adverse_events():
    """Display global adverse events distribution."""
    st.subheader("Global Adverse Events Distribution")

    # Fetch and process data
    df = adverse_events_by_country()

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Country",
            y="Count",
            title="Adverse Events by Country",
            labels={"Country": "Country", "Count": "Number of Events"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Count",
            names="Country",
            title="Distribution of Adverse Events by Country"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    # AI Insights section
    st.subheader("AI Insights")
    if st.button("Generate Insights", key="global_insights"):
        insights = get_insights_from_data(df)
        st.write(insights)

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="global_question")
    if question and st.button("Get Answer", key="global_answer"):
        insights = get_insights_from_data(df, question)
        st.write(insights)

def display_actions_taken_with_drug():
    """Display actions taken with drug data."""
    st.subheader("Actions Taken with Drug")

    # Fetch and process data
    df = get_actions_taken_with_drug()

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            df,
            x="Action",
            y="Count",
            title="Actions Taken with Drug",
            labels={"Action": "Action", "Count": "Number of Cases"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            df,
            values="Count",
            names="Action",
            title="Distribution of Actions Taken with Drug"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display detailed statistics
    st.subheader("Detailed Statistics")
    st.dataframe(df)

    # AI Insights section
    st.subheader("AI Insights")
    if st.button("Generate Insights", key="actions_insights"):
        insights = get_insights_from_data(df)
        st.write(insights)

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="actions_question")
    if question and st.button("Get Answer", key="actions_answer"):
        insights = get_insights_from_data(df, question)
        st.write(insights)

def display_drug_reports():
    """Display drug reports with various visualizations."""
    st.title("Drug Reports")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Adverse Events by Age",
        "Adverse Events by Drug",
        "Global Adverse Events",
        "Actions Taken with Drug"
    ])

    with tab1:
        display_adverse_events_by_age()

    with tab2:
        display_adverse_events_by_drug()

    with tab3:
        display_global_adverse_events()

    with tab4:
        display_actions_taken_with_drug()
