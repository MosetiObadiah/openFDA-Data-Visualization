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

def display_global_adverse_events():
    """Display global distribution of adverse drug events with world map and heatmap."""
    metric_title = "Global Distribution of Adverse Drug Events"
    metric_description = (
        "This section shows the global distribution of adverse drug events, "
        "highlighting countries with the highest reported incidents as a percentage of total events."
    )

    render_metric_header(metric_title, metric_description)

    # Initialize df_country outside try block so it can be used in insights
    df_country = None

    try:
        df_country = adverse_events_by_country()
        if df_country.empty:
            st.warning("No data available for adverse events by country.")
            return

        # Convert percentage strings back to numbers for plotting
        df_country["Percentage_Value"] = df_country["Percentage"].str.rstrip("%").astype(float)

        # Create three columns for the visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for top 10 countries
            top_10 = df_country.head(10)
            fig_bar = px.bar(
                top_10,
                x="Country",
                y="Percentage_Value",
                text="Percentage",
                title="Top 10 Countries by Adverse Events",
                labels={
                    "Country": "Country",
                    "Percentage_Value": "Percentage of Total Events",
                    "Percentage": "Percentage"
                }
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create pie chart for top 5 countries vs rest of the world
            top_5 = df_country.head(5)
            rest_of_world = pd.DataFrame({
                'Country': ['Rest of World'],
                'Percentage_Value': [df_country['Percentage_Value'][5:].sum()],
                'Percentage': [f"{df_country['Percentage_Value'][5:].sum():.2f}%"]
            })
            pie_data = pd.concat([top_5, rest_of_world])

            fig_pie = px.pie(
                pie_data,
                values="Percentage_Value",
                names="Country",
                title="Distribution of Adverse Events: Top 5 Countries vs Rest of World",
                hover_data=["Percentage"]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Create world map
        fig_map = px.choropleth(
            df_country,
            locations="Country",
            locationmode="country names",
            color="Percentage_Value",
            hover_name="Country",
            hover_data={
                "Count": True,
                "Percentage": True,
                "Country": False,
                "Percentage_Value": False
            },
            color_continuous_scale="Viridis",
            range_color=[0, df_country["Percentage_Value"].quantile(0.95)],
            title="Global Distribution of Adverse Drug Events (Percentage)"
        )
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Display detailed statistics in a collapsible section
        st.markdown("### Detailed Statistics")
        display_df = df_country.drop(columns=["Percentage_Value"])
        render_data_table(display_df)

    except Exception as e:
        st.error(f"Failed to load data for Global Adverse Events: {e}")
        return

    # AI Insights section (outside the try block)
    if df_country is not None:
        st.markdown("### Insights")
        if st.button("Get General Insights for Global Adverse Events"):
            prompt = f"""
            Analyze the following global adverse drug events data and provide 5-7 key insights:

            Top 5 Countries and their percentages:
            {df_country.head().to_string()}

            Total number of countries: {len(df_country)}
            Total number of events: {df_country['Count'].sum()}

            Focus on:
            1. Geographic distribution patterns
            2. Concentration of events
            3. Notable regional differences
            4. Potential implications for drug safety monitoring
            5. Recommendations for global drug safety

            Provide a concise analysis in 5-7 sentences.
            """
            insights = get_insights_from_data(df_country, (df_country["Country"].iloc[0], df_country["Country"].iloc[-1]), "Country", prompt)
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about this data (e.g., 'Which country has the highest percentage of adverse events?')", key="country_insight_question")
        if custom_question:
            prompt = f"""
            Based on the following data about global adverse drug events:
            {df_country.head().to_string()}

            Answer this specific question in 5-7 sentences, focusing on data-driven insights and potential implications:
            {custom_question}
            """
            insights = get_insights_from_data(df_country, (df_country["Country"].iloc[0], df_country["Country"].iloc[-1]), "Country", prompt)
            st.write(insights)

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

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            # Display the raw data
            st.markdown("### Data Table")
            render_data_table(df)

        with col2:
            # Create and display the pie chart
            fig = px.pie(
                df,
                names="Action",
                values="Count",
                title="Actions Taken with the Drug",
                labels={"Action": "Action Taken", "Count": "Number of Records"},
                hover_data=["Count"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        # AI Insights with Custom Question
        st.markdown("### Insights")
        if st.button("Get General Insights for Actions Taken"):
            prompt = f"""
            Analyze the following data about actions taken with drugs after adverse events:
            {df.to_string()}

            Focus on:
            1. Most common actions taken
            2. Distribution of different actions
            3. Implications for drug safety
            4. Potential impact on patient care
            5. Recommendations for drug monitoring

            Provide a comprehensive analysis in 5-7 sentences.
            """
            insights = get_insights_from_data(df, (df["Action"].iloc[0], df["Action"].iloc[-1]), "Actions Taken with the Drug", prompt)
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about this data (e.g., 'What is the most common action taken?')", key="actions_taken_insight_question")
        if custom_question:
            prompt = f"""
            Based on the following data about actions taken with drugs:
            {df.to_string()}

            Answer this specific question in 5-7 sentences, focusing on data-driven insights and potential implications:
            {custom_question}
            """
            insights = get_insights_from_data(df, (df["Action"].iloc[0], df["Action"].iloc[-1]), "Actions Taken with the Drug", prompt)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load data for Actions Taken with Drug: {e}")

def display_drug_reports():
    """Display all drug-related reports with navigation tabs."""
    st.title("Drug Reports")
    tabs = st.tabs([
        "Adverse Events by Age",
        "Adverse Events by Drug",
        "Global Adverse Events",
        "Actions Taken with the Drug"
    ])

    with tabs[0]:
        display_adverse_events_by_age()
    with tabs[1]:
        display_adverse_events_by_drug()
    with tabs[2]:
        display_global_adverse_events()
    with tabs[3]:
        display_actions_taken_with_drug()
