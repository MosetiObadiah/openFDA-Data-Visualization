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
    get_actions_taken_with_drug
)
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
    """Display adverse events by patient age with bar and line charts."""
    metric_title = "Adverse Events by Patient Age"
    metric_description = (
        "This metric summarizes the number of reported adverse drug events, grouped by patient age, "
        "within a specified date range, highlighting age-related health risks."
    )

    render_metric_header(metric_title, metric_description)
    start_str, end_str = render_date_picker()
    try:
        df = adverse_events_by_patient_age_group_within_data_range(start_str, end_str)
        if df.empty:
            st.warning("No data available for the selected date range.")
            return
    except Exception as e:
        st.error(f"Failed to load data for Adverse Events by Age: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        render_data_table(df)

    df = get_aggregated_age_data(df)
    filtered_df = render_age_filter(df)

    with col2:
        st.subheader("Bar Chart")
        render_bar_chart(
            filtered_df,
            x_col="Patient Age",
            y_col="Adverse Event Count",
            title="Adverse Events by Patient Age",
            x_label="Age of Patient",
            y_label="Number of Adverse Events"
        )

    st.subheader("Line Chart")
    render_line_chart(
        filtered_df,
        x_col="Patient Age",
        y_col="Adverse Event Count",
        title="Adverse Events by Patient Age",
        x_label="Age of Patient",
        y_label="Number of Adverse Events"
    )

    # AI Insights with Custom Question
    st.markdown("### Insights")
    if st.button("Get General Insights for Age Data"):
        insights = get_insights_from_data(filtered_df, (int(filtered_df["Patient Age"].min()), int(filtered_df["Patient Age"].max())), "Patient Age")
        st.write(insights)

    custom_question = st.text_input("Ask a specific question about this data (e.g., 'Which age group has the most adverse events?')", key="age_insight_question")
    if custom_question:
        insights = get_insights_from_data(filtered_df, (int(filtered_df["Patient Age"].min()), int(filtered_df["Patient Age"].max())), "Patient Age", custom_question)
        st.write(insights)

def display_adverse_events_by_drug():
    """Display adverse events by drug with a bar chart."""
    metric_title = "Adverse Events by Drug"
    metric_description = (
        "This metric shows the number of reported adverse drug events, grouped by drug name, "
        "within a specified date range, highlighting drugs with higher risks."
    )

    render_metric_header(metric_title, metric_description)
    start_str, end_str = render_date_picker(key_prefix="drug_")
    try:
        df_drug = adverse_events_by_drug_within_data_range(start_str, end_str)
        if df_drug.empty:
            st.warning("No data available for the selected date range.")
            return
    except Exception as e:
        st.error(f"Failed to load data for Adverse Events by Drug: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        render_data_table(df_drug)

    df_drug = get_top_drugs(df_drug)

    with col2:
        st.subheader("Bar Chart")
        render_bar_chart(
            df_drug,
            x_col="Drug Name",
            y_col="Adverse Event Count",
            title="Adverse Events by Drug (Top 20)",
            x_label="Drug Name",
            y_label="Number of Adverse Events"
        )

    # AI Insights with Custom Question
    st.markdown("### Insights")
    if st.button("Get General Insights for Drug Data"):
        insights = get_insights_from_data(df_drug, (df_drug["Drug Name"].iloc[0], df_drug["Drug Name"].iloc[-1]), "Drug Name")
        st.write(insights)

    custom_question = st.text_input("Ask a specific question about this data (e.g., 'Which drug has the highest adverse events?')", key="drug_insight_question")
    if custom_question:
        insights = get_insights_from_data(df_drug, (df_drug["Drug Name"].iloc[0], df_drug["Drug Name"].iloc[-1]), "Drug Name", custom_question)
        st.write(insights)

def display_drug_recall_trends():
    """Display drug recall trends including frequency, common drugs, and reasons over time."""
    metric_title = "Drug Recall Trends"
    metric_description = (
        "This section analyzes drug recall trends from the OpenFDA Drug Enforcement Reports, "
        "including frequency of recalls by year, most commonly recalled drugs, and reasons for recalls over time."
    )

    render_metric_header(metric_title, metric_description)

    # Frequency of Recalls by Year
    st.subheader("Frequency of Recalls by Year")
    try:
        df_recall_freq = recall_frequency_by_year()
        if df_recall_freq.empty:
            st.warning("No recall frequency data available.")
        else:
            render_data_table(df_recall_freq)
            render_bar_chart(
                df_recall_freq,
                x_col="Year",
                y_col="Recall Count",
                title="Drug Recalls by Year",
                x_label="Year",
                y_label="Number of Recalls"
            )

            # AI Insights with Custom Question
            st.markdown("### Insights for Recall Frequency")
            if st.button("Get General Insights for Recall Frequency"):
                insights = get_insights_from_data(df_recall_freq, (int(df_recall_freq["Year"].min()), int(df_recall_freq["Year"].max())), "Year")
                st.write(insights)

            custom_question = st.text_input("Ask a specific question about this data (e.g., 'Which year had the most recalls?')", key="recall_freq_insight_question")
            if custom_question:
                insights = get_insights_from_data(df_recall_freq, (int(df_recall_freq["Year"].min()), int(df_recall_freq["Year"].max())), "Year", custom_question)
                st.write(insights)
    except Exception as e:
        st.error(f"Failed to load data for Recall Frequency: {e}")

    # Most Commonly Recalled Drugs
    st.subheader("Most Commonly Recalled Drugs")
    try:
        df_recall_drugs = most_common_recalled_drugs()
        if df_recall_drugs.empty:
            st.warning("No data available for recalled drugs.")
        else:
            render_data_table(df_recall_drugs)
            render_bar_chart(
                df_recall_drugs,
                x_col="Product Description",
                y_col="Recall Count",
                title="Most Commonly Recalled Drugs (Top 20)",
                x_label="Drug/Product",
                y_label="Number of Recalls"
            )

            # AI Insights with Custom Question
            st.markdown("### Insights for Recalled Drugs")
            if st.button("Get General Insights for Recalled Drugs"):
                insights = get_insights_from_data(df_recall_drugs, (df_recall_drugs["Product Description"].iloc[0], df_recall_drugs["Product Description"].iloc[-1]), "Product Description")
                st.write(insights)

            custom_question = st.text_input("Ask a specific question about this data (e.g., 'Which drug was recalled the most?')", key="recall_drugs_insight_question")
            if custom_question:
                insights = get_insights_from_data(df_recall_drugs, (df_recall_drugs["Product Description"].iloc[0], df_recall_drugs["Product Description"].iloc[-1]), "Product Description", custom_question)
                st.write(insights)
    except Exception as e:
        st.error(f"Failed to load data for Most Commonly Recalled Drugs: {e}")

    # Recall Reasons Over Time
    st.subheader("Recall Reasons Over Time")
    try:
        df_recall_reasons = recall_reasons_over_time()
        if df_recall_reasons.empty:
            st.warning("No recall reasons data available.")
        else:
            df_pivot = get_recall_reasons_pivot(df_recall_reasons)
            render_data_table(df_pivot)
            render_line_chart(
                df_pivot.melt(id_vars=["Year"], var_name="Reason Category", value_name="Recall Count"),
                x_col="Year",
                y_col="Recall Count",
                title="Recall Reasons Over Time",
                x_label="Year",
                y_label="Number of Recalls"
            )

            # AI Insights with Custom Question
            st.markdown("### Insights for Recall Reasons")
            if st.button("Get General Insights for Recall Reasons"):
                insights = get_insights_from_data(df_pivot, (int(df_pivot["Year"].min()), int(df_pivot["Year"].max())), "Year")
                st.write(insights)

            custom_question = st.text_input("Ask a specific question about this data (e.g., 'What is the most common reason for recalls over time?')", key="recall_reasons_insight_question")
            if custom_question:
                insights = get_insights_from_data(df_pivot, (int(df_pivot["Year"].min()), int(df_pivot["Year"].max())), "Year", custom_question)
                st.write(insights)
    except Exception as e:
        st.error(f"Failed to load data for Recall Reasons: {e}")

def display_actions_taken_with_drug():
    """Display actions taken with the drug after adverse events as a pie chart."""
    metric_title = "Actions Taken with the Drug After Adverse Events"
    metric_description = (
        "This metric shows the actions taken with the drug following an adverse event, such as withdrawal, dose adjustment, or no change, "
        "based on data from 2004 to 2025."
    )

    render_metric_header(metric_title, metric_description)

    try:
        df = get_actions_taken_with_drug()
        if df.empty:
            st.warning("No data available for Actions Taken with Drug.")
            return

        # Display the raw data
        render_data_table(df)

        # Create and display the pie chart
        fig = px.pie(
            df,
            names="Action",
            values="Count",
            title="Actions Taken with the Drug",
            labels={"Action": "Action Taken", "Count": "Number of Records"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # AI Insights with Custom Question
        st.markdown("### Insights")
        if st.button("Get General Insights for Actions Taken"):
            insights = get_insights_from_data(df, (df["Action"].iloc[0], df["Action"].iloc[-1]), "Actions Taken with the Drug")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about this data (e.g., 'What is the most common action taken?')", key="actions_taken_insight_question")
        if custom_question:
            insights = get_insights_from_data(df, (df["Action"].iloc[0], df["Action"].iloc[-1]), "Actions Taken with the Drug", custom_question)
            st.write(insights)
    except Exception as e:
        st.error(f"Failed to load data for Actions Taken with Drug: {e}")

def display_drug_reports():
    """Display all drug-related reports with navigation tabs."""
    st.title("Drug Reports")
    tabs = st.tabs([
        "Adverse Events by Age",
        "Adverse Events by Drug",
        "Drug Recall Trends",
        "Actions Taken with the Drug"
    ])

    with tabs[0]:
        display_adverse_events_by_age()
    with tabs[1]:
        display_adverse_events_by_drug()
    with tabs[2]:
        display_drug_recall_trends()
    with tabs[3]:
        display_actions_taken_with_drug()
