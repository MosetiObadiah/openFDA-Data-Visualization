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
    get_top_drugs,
    get_recall_reasons_pivot,
    get_actions_taken_with_drug,
    adverse_events_by_country,
    get_drug_events_by_substance,
    get_drug_events_by_action,
    get_drug_events_by_patient_sex,
    get_drug_events_by_patient_weight,
    get_drug_events_by_reaction_outcome,
    get_drug_events_by_reporter_qualification
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

def get_insights_from_data(df: pd.DataFrame, context: str, custom_question: str = None) -> str:
    if df.empty:
        return "No data available for insights."
    # Summarize the top rows for context
    summary = df.head(10).to_string(index=False)
    if custom_question:
        prompt = (
            f"Given the following data about {context}:\n\n"
            f"{summary}\n\n"
            f"Answer this question in 3-5 sentences, focusing on data-driven insights:\n"
            f"{custom_question}"
        )
    else:
        prompt = (
            f"Analyze the following data about {context}:\n\n"
            f"{summary}\n\n"
            "Provide a concise summary (3-5 sentences) of key trends, patterns, and notable findings."
        )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def display_adverse_events_by_age():
    """Display adverse events by age group."""
    st.subheader("Adverse Events by Age Group")

    # Use global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

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
        st.write(get_insights_from_data(df, "Adverse Events by Age Group"))

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="age_question")
    if question and st.button("Get Answer", key="age_answer"):
        st.write(get_insights_from_data(df, "Adverse Events by Age Group", question))

def display_adverse_events_by_drug():
    """Display adverse events by drug."""
    st.subheader("Adverse Events by Drug")

    # Use global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

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
        st.write(get_insights_from_data(df, "Adverse Events by Drug"))

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="drug_question")
    if question and st.button("Get Answer", key="drug_answer"):
        st.write(get_insights_from_data(df, "Adverse Events by Drug", question))

def display_global_adverse_events():
    """Display global adverse events distribution."""
    st.subheader("Global Adverse Events Distribution")

    # Use global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

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
        st.write(get_insights_from_data(df, "Global Adverse Events Distribution"))

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="global_question")
    if question and st.button("Get Answer", key="global_answer"):
        st.write(get_insights_from_data(df, "Global Adverse Events Distribution", question))

def display_actions_taken_with_drug():
    """Display actions taken with drug data."""
    st.subheader("Actions Taken with Drug")

    # Use global date range from session state
    start_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_str = st.session_state.end_date.strftime('%Y-%m-%d')

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
        st.write(get_insights_from_data(df, "Actions Taken with Drug"))

    # Custom question input
    question = st.text_input("Ask a specific question about the data", key="actions_question")
    if question and st.button("Get Answer", key="actions_answer"):
        st.write(get_insights_from_data(df, "Actions Taken with Drug", question))

def display_drug_reports():
    st.title("Drug Reports")
    tab_names = [
        "Active Ingredient (Substance)",
        "Patient Weight"
    ]
    tabs = st.tabs(tab_names)

    # 1. Active Ingredient (Substance)
    with tabs[0]:
        st.subheader("Active Ingredient (Substance)")
        df = get_drug_events_by_substance()
        if df.empty:
            st.warning("No data available.")
        else:
            top_n = st.session_state.top_n_results if "top_n_results" in st.session_state else 20
            top_df = df.head(top_n)
            # Generate a color map for the top N ingredients
            color_seq = px.colors.qualitative.Plotly
            color_map = {name: color_seq[i % len(color_seq)] for i, name in enumerate(top_df["Substance"])}
            # Bar chart for top N
            fig_bar = px.bar(top_df, x="Substance", y="Count", title=f"Top {top_n} Active Ingredients (Bar Chart)", color="Substance", color_discrete_map=color_map)
            fig_bar.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            # Treemap for all results, using the same color map for top N
            fig_all = px.treemap(df, path=["Substance"], values="Count", title="All Active Ingredients (Treemap)", color="Substance", color_discrete_map=color_map)
            st.plotly_chart(fig_all, use_container_width=True)
            with st.expander("Detailed Statistics", expanded=False):
                st.dataframe(df)
            st.subheader("AI Insights")
            question = st.text_input("Custom question (optional)", key="substance_question")
            if st.button("Generate Insights", key="substance_insights"):
                st.write(get_insights_from_data(df, "Active Ingredient (Substance)", question or ""))

    # 2. Patient Weight
    with tabs[1]:
        st.subheader("Patient Weight")
        df_weight = get_drug_events_by_patient_weight()
        if df_weight.empty:
            st.warning("No data available for patient weight.")
        else:
            fig_weight = px.bar(df_weight, x="Weight Group", y="Count", title="Distribution by Patient Weight Group")
            st.plotly_chart(fig_weight, use_container_width=True)
            with st.expander("Detailed Weight Statistics", expanded=False):
                st.dataframe(df_weight)
        st.subheader("AI Insights")
        if st.button("Generate Insights", key="weight_insights"):
            st.write(get_insights_from_data(df_weight, "Patient Weight"))
        question = st.text_input("Ask a specific question about the data", key="weight_question")
        if question and st.button("Get Answer", key="weight_answer"):
            st.write(get_insights_from_data(df_weight, "Patient Weight", question))
