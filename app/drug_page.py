import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date

from src.drug_events import adverse_events_by_patient_age_group_within_data_range

def display_drug_reports():
    st.subheader("Adverse Events by Patient Age")
    st.write("This metric summarizes the number of reported adverse drug events, grouped by patient age, within a specified date range, highlighting age-related health risks.")

    # Date range selection side by side
    col1, col2 = st.columns(2)

    with col1:
        start = st.date_input("Data start date", value=date(2010, 1, 1), min_value=date(2010, 1, 1), max_value=date(2025, 1, 31), key="start_date")
    with col2:
        end = st.date_input("Data end date", value=date(2025, 1, 31), min_value=date(2010, 1, 1), max_value=date(2025, 1, 31), key="end_date")

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    # Api call with selected date range
    df = adverse_events_by_patient_age_group_within_data_range(start_str, end_str)

    with col1:
        with st.expander("See raw table data", expanded=True):
            st.dataframe(data=df, width=500, use_container_width=False)

    # Group and sort
    df = df.groupby("Patient Age", as_index=False).agg({"Adverse Event Count": "sum"})
    df = df.sort_values("Patient Age")

    # Add slider to filter age
    min_age = int(df["Patient Age"].min())
    max_age = int(df["Patient Age"].max())

    age_range = st.slider(
        "Filter by Patient Age",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )

    filtered_df = df[(df["Patient Age"] >= age_range[0]) & (df["Patient Age"] <= age_range[1])]

    with col2:
        st.subheader("Bar Chart")
        # Bar chart
        fig_bar = px.bar(
            filtered_df,
            x="Patient Age",
            y="Adverse Event Count",
            title="Adverse Events by Patient Age",
            labels={
                "Patient Age": "Age of Patient",
                "Adverse Event Count": "Number of Adverse Events"
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Line Chart")
    # Line chart
    fig_line = px.line(
        filtered_df,
        x="Patient Age",
        y="Adverse Event Count",
        title="Adverse Events by Patient Age",
        labels={
            "Patient Age": "Age of Patient",
            "Adverse Event Count": "Number of Adverse Events"
        }
    )
    st.plotly_chart(fig_line, use_container_width=True)
