import streamlit as st
import pandas as pd
from datetime import date
import google.generativeai as genai
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

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
    get_drug_events_by_reporter_qualification,
    get_top_drug_reactions,
    get_drug_indications,
    get_drug_manufacturer_distribution,
    get_drug_therapeutic_response
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

def get_insights_from_data(df, context: str, custom_question: str = None) -> str:
    """Generate insights from data using Gemini.

    Parameters:
        df: Either a DataFrame or a dictionary containing DataFrames
        context: The context of the data for the prompt
        custom_question: Optional specific question to answer

    Returns:
        String containing generated insights or error message
    """
    # Handle empty data
    if df is None:
        return "No data available for insights."

    # Handle different data types
    if isinstance(df, dict):
        # Process dictionary of DataFrames
        summaries = []
        for key, value in df.items():
            if hasattr(value, 'empty') and not value.empty:
                summaries.append(f"{key.replace('_', ' ').title()}:\n{value.head(5).to_string(index=False)}")

        if not summaries:
            return "No data available for insights."

        summary = "\n\n".join(summaries)
    elif hasattr(df, 'empty'):
        # Process single DataFrame
        if df.empty:
            return "No data available for insights."
        summary = df.head(10).to_string(index=False)
    else:
        # Unknown data type
        return "Cannot generate insights: unsupported data format."

    # Create prompt based on available data
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

    # Generate insights
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def render_ai_insights_section(df, context, key_prefix):
    st.subheader("AI Insights")
    question = st.text_input("Custom question (optional)", key=f"{key_prefix}_question")
    if st.button("Generate Insights", key=f"{key_prefix}_insights"):
        with st.spinner("Generating insights..."):
            st.write(get_insights_from_data(df, context, question or ""))

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
    render_ai_insights_section(df, "Adverse Events by Age Group", "age")

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
    render_ai_insights_section(df, "Adverse Events by Drug", "drug")

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
    render_ai_insights_section(df, "Global Adverse Events Distribution", "global")

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
    render_ai_insights_section(df, "Actions Taken with Drug", "actions")

def display_drug_reports():
    st.title("Drug Reports")

    # Create tabs for different analysis sections
    tab_names = [
        "Adverse Reactions",
        "Drug Indications",
        "Patient Demographics",
        "Manufacturers",
        "Therapeutic Response",
        "Substances"
    ]
    tabs = st.tabs(tab_names)

    # 1. Adverse Reactions Tab
    with tabs[0]:
        st.subheader("Top Adverse Reactions to Drugs")
        df_reactions = get_top_drug_reactions(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if df_reactions.empty:
            st.warning("No adverse reaction data available.")
        else:
            # Display top reactions by count
            top_n = st.slider("Number of top reactions to show", 5, 30, 15, key="drug_reaction_slider")
            top_df = df_reactions.head(top_n)

            # Create visualizations - bar chart first, then pie chart
            # Bar chart for top N reactions
            fig_bar = px.bar(
                top_df,
                x="Count",
                y="Reaction",
                title=f"Top {top_n} Adverse Reactions",
                orientation='h',
                color="Category"
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

            # Pie chart for reaction categories
            category_df = df_reactions.groupby("Category")["Count"].sum().reset_index()
            fig_pie = px.pie(
                category_df,
                values="Count",
                names="Category",
                title="Adverse Reactions by Category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Display detailed statistics in an expander
            with st.expander("Detailed Reaction Statistics", expanded=False):
                st.dataframe(df_reactions.style.highlight_max(subset=["Count"], color='lightgreen'))

            # AI Insights section
            render_ai_insights_section(df_reactions, "Drug Adverse Reactions", "drug_reactions")

    # 2. Drug Indications Tab
    with tabs[1]:
        st.subheader("Common Drug Indications")
        df_indications = get_drug_indications(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if df_indications.empty:
            st.warning("No drug indication data available.")
        else:
            # Top indications
            top_n = st.slider("Number of top indications to show", 5, 30, 15, key="drug_indications_slider")
            top_ind_df = df_indications.head(top_n)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for top indications
                fig_ind = px.bar(
                    top_ind_df,
                    x="Count",
                    y="Indication",
                    orientation='h',
                    title=f"Top {top_n} Medical Conditions Treated with Drugs",
                    color="Therapeutic Area"
                )
                fig_ind.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_ind, use_container_width=True)

            with col2:
                # Treemap by therapeutic area
                area_df = df_indications.groupby("Therapeutic Area")["Count"].sum().reset_index()
                fig_area = px.treemap(
                    area_df,
                    path=["Therapeutic Area"],
                    values="Count",
                    title="Indications by Therapeutic Area"
                )
                st.plotly_chart(fig_area, use_container_width=True)

            # Display detailed statistics in an expander
            with st.expander("Detailed Indication Statistics", expanded=False):
                st.dataframe(df_indications.style.highlight_max(subset=["Count"], color='lightgreen'))

            # AI Insights section
            render_ai_insights_section(df_indications, "Drug Indications", "drug_indications")

    # 3. Patient Demographics Tab
    with tabs[2]:
        st.subheader("Patient Demographics")

        # Create two columns for Sex and Weight
        col1, col2 = st.columns(2)

        with col1:
            # Patient Sex Analysis
            st.subheader("Patient Sex Distribution")
            df_sex = get_drug_events_by_patient_sex(
                st.session_state.start_date,
                st.session_state.end_date
            )

            if df_sex.empty:
                st.warning("No patient sex data available.")
            else:
                # Pie chart for sex distribution
                fig_sex = px.pie(
                    df_sex,
                    values="Count",
                    names="Sex",
                    title="Adverse Events by Patient Sex",
                    hover_data=["Percentage"]
                )
                st.plotly_chart(fig_sex, use_container_width=True)

                # Table with counts and percentages (in an expander)
                with st.expander("View Sex Distribution Details", expanded=False):
                    st.dataframe(df_sex)

        with col2:
            # Patient Weight Analysis
            st.subheader("Patient Weight Distribution")
            df_weight = get_drug_events_by_patient_weight()

            if df_weight.empty:
                st.error("No internet connection. Please check your network and try again.")
            else:
                # Bar chart for weight distribution
                fig_weight = px.bar(
                    df_weight,
                    x="Weight Group",
                    y="Count",
                    title="Adverse Events by Patient Weight Group",
                    color="Weight Group"
                )
                st.plotly_chart(fig_weight, use_container_width=True)

                # Table with weight data (in an expander)
                with st.expander("View Weight Distribution Details", expanded=False):
                    st.dataframe(df_weight)

        # Add combined demographic data for AI insights
        if not df_sex.empty or not df_weight.empty:
            # Combine data for insights or use just one dataset if the other is empty
            demographics_data = pd.DataFrame()
            if not df_sex.empty and not df_weight.empty:
                # Create a simple combined dataset for insights
                demographics_data = {
                    "sex_distribution": df_sex,
                    "weight_distribution": df_weight
                }
            elif not df_sex.empty:
                demographics_data = df_sex
            else:
                demographics_data = df_weight

            # Add AI Insights section
            render_ai_insights_section(demographics_data, "Patient Demographics in Drug Events", "drug_demographics")

    # 4. Manufacturers Tab
    with tabs[3]:
        st.subheader("Drug Manufacturers")
        df_manufacturers = get_drug_manufacturer_distribution(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if df_manufacturers.empty:
            st.warning("No manufacturer data available.")
        else:
            # Top manufacturers
            top_n = st.slider("Number of top manufacturers to show", 5, 30, 15, key="drug_manufacturer_slider")
            top_mfr_df = df_manufacturers.head(top_n)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for top manufacturers
                fig_mfr = px.bar(
                    top_mfr_df,
                    x="Count",
                    y="Manufacturer",
                    orientation='h',
                    title=f"Top {top_n} Drug Manufacturers",
                    color="Manufacturer"
                )
                fig_mfr.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    showlegend=False
                )
                st.plotly_chart(fig_mfr, use_container_width=True)

            with col2:
                # Pie chart for top manufacturers
                fig_pie_mfr = px.pie(
                    top_mfr_df,
                    values="Count",
                    names="Manufacturer",
                    title=f"Market Share of Top {top_n} Manufacturers"
                )
                st.plotly_chart(fig_pie_mfr, use_container_width=True)

            # Market concentration metrics
            total_count = df_manufacturers["Count"].sum()
            top5_share = (top_mfr_df.head(5)["Count"].sum() / total_count * 100).round(1)
            top10_share = (top_mfr_df.head(10)["Count"].sum() / total_count * 100).round(1)

            # Show market concentration metrics
            st.subheader("Market Concentration")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Number of Manufacturers", len(df_manufacturers))
            with metrics_col2:
                st.metric("Top 5 Manufacturers Share", f"{top5_share}%")
            with metrics_col3:
                st.metric("Top 10 Manufacturers Share", f"{top10_share}%")

            # Display detailed statistics in an expander
            with st.expander("Detailed Manufacturer Statistics", expanded=False):
                st.dataframe(df_manufacturers.style.highlight_max(subset=["Count"], color='lightgreen'))

            # AI Insights section
            render_ai_insights_section(df_manufacturers, "Drug Manufacturers", "drug_manufacturers")

    # 5. Therapeutic Response Tab
    with tabs[4]:
        st.subheader("Therapeutic Response to Drugs")
        df_response = get_drug_therapeutic_response(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if df_response.empty:
            st.warning("No therapeutic response data available.")
        else:
            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for responses
                fig_resp = px.bar(
                    df_response,
                    x="Count",
                    y="Response",
                    orientation='h',
                    title="Therapeutic Responses to Drugs",
                    color="Response Category",
                    color_discrete_map={"Positive": "green", "Negative": "red"}
                )
                fig_resp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_resp, use_container_width=True)

            with col2:
                # Pie chart for response categories
                category_df = df_response.groupby("Response Category")["Count"].sum().reset_index()
                fig_pie_resp = px.pie(
                    category_df,
                    values="Count",
                    names="Response Category",
                    title="Positive vs. Negative Therapeutic Responses",
                    color="Response Category",
                    color_discrete_map={"Positive": "green", "Negative": "red"}
                )
                st.plotly_chart(fig_pie_resp, use_container_width=True)

            # Display effectiveness ratio
            total_count = df_response["Count"].sum()
            negative_count = df_response[df_response["Response Category"] == "Negative"]["Count"].sum()
            positive_count = df_response[df_response["Response Category"] == "Positive"]["Count"].sum()

            if positive_count > 0:
                effectiveness_ratio = (negative_count / positive_count).round(2)

                st.subheader("Effectiveness Analysis")
                st.write(f"For every 1 report of positive therapeutic effect, there are {effectiveness_ratio} reports of negative effects.")

                # Create a gauge chart for effectiveness score
                # Scale from 0-10, lower is better (fewer negative reports per positive)
                effectiveness_score = min(10, effectiveness_ratio)

                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = effectiveness_score,
                    title = {'text': "Ineffectiveness Score"},
                    gauge = {
                        'axis': {'range': [0, 10]},
                        'bar': {'color': "darkgrey"},
                        'steps': [
                            {'range': [0, 3], 'color': "green"},
                            {'range': [3, 7], 'color': "yellow"},
                            {'range': [7, 10], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': effectiveness_score
                        }
                    }
                ))

                st.plotly_chart(fig_gauge, use_container_width=True)

            # Display detailed statistics in an expander
            with st.expander("Detailed Response Statistics", expanded=False):
                st.dataframe(df_response.style.highlight_max(subset=["Count"], color='lightgreen'))

            # AI Insights section
            render_ai_insights_section(df_response, "Therapeutic Responses to Drugs", "drug_responses")

    # 6. Substances Tab
    with tabs[5]:
        st.subheader("Active Ingredient (Substance)")
        df_substance = get_drug_events_by_substance()

        if df_substance.empty:
            st.warning("No data available for active ingredients.")
        else:
            top_n = st.slider("Number of top substances to show", 5, 30, 15, key="drug_substance_slider")
            top_df = df_substance.head(top_n)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for top N
                fig_bar = px.bar(
                    top_df,
                    x="Count",
                    y="Substance",
                    title=f"Top {top_n} Active Ingredients",
                    orientation='h',
                    color="Substance"
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # Treemap for all results
                fig_all = px.treemap(
                    top_df,
                    path=["Substance"],
                    values="Count",
                    title=f"Top {top_n} Active Ingredients (Treemap)"
                )
                st.plotly_chart(fig_all, use_container_width=True)

            # Display detailed statistics in an expander
            with st.expander("Detailed Substance Statistics", expanded=False):
                st.dataframe(df_substance.style.highlight_max(subset=["Count"], color='lightgreen'))

            # AI Insights section
            render_ai_insights_section(df_substance, "Active Ingredients (Substances)", "drug_substance")
