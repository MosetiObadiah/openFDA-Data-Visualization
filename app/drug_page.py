import streamlit as st
import pandas as pd
from datetime import date
import google.generativeai as genai
import os
from dotenv import load_dotenv

from src.drug_events import (
    adverse_events_by_patient_age_group_within_data_range,
    adverse_events_by_drug_within_data_range
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

def get_insights_from_data(df: pd.DataFrame, filter_range: tuple, filter_col: str, custom_instructions: str = "") -> str:
    """Use Gemini API to generate insights from the DataFrame and filter range."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Do{custom_instructions}. Also analyze the following data about adverse drug events by {filter_col}, filtered for {filter_col} from {filter_range[0]} to {filter_range[1]}:

    {df.to_string()}

    Provide a concise explanation (5-10 sentences) of what the user is seeing, highlighting key trends, patterns, or notable points in the data. Use a conversational tone as if explaining to a non-expert. If possible, provide your findings in a list, otherwise use a paragraph.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def display_drug_reports():
    # Adverse Events by Patient Age
    metric_title = "Adverse Events by Patient Age"
    metric_description = (
        "This metric summarizes the number of reported adverse drug events, grouped by patient age, "
        "within a specified date range, highlighting age-related health risks."
    )

    render_metric_header(metric_title, metric_description)
    start_str, end_str = render_date_picker()
    df = adverse_events_by_patient_age_group_within_data_range(start_str, end_str)

    col1, col2 = st.columns(2)
    with col1:
        render_data_table(df)

    df = df.groupby("Patient Age", as_index=False).agg({"Adverse Event Count": "sum"})
    df = df.sort_values("Patient Age")
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

    # AI Insights section
    st.markdown("---")
    st.subheader("AI Insights")

    if not filtered_df.empty:
        with st.expander("View AI reference table data", expanded=False):
            st.dataframe(filtered_df)

        # Add custom instructions section
        use_custom_instructions = st.toggle("Add custom instructions", value=False)
        custom_instructions = ""
        if use_custom_instructions:
            custom_instructions = st.text_area(
                "Enter your questions or instructions for the AI analysis",
                placeholder="Example: Focus on age groups with highest risk or Compare with previous year's data",
                help="Add specific questions or instructions to guide the AI analysis"
            )

        if st.button("Get Insights for Drug Data"):
            insights = get_insights_from_data(
                filtered_df,
                (int(filtered_df["Patient Age"].min()), int(filtered_df["Patient Age"].max())),
                "Patient Age",
                custom_instructions
            )
            st.markdown("### Insights")
            st.write(insights)
    else:
        st.warning("No data available for the selected date range.")
