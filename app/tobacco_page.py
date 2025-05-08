import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tobacco_endpoints import (
    get_tobacco_reports_by_product,
    get_tobacco_reports_by_problem_type,
    get_tobacco_reports_by_health_effect,
    get_tobacco_reports_by_demographic,
    get_tobacco_reports_over_time
)

# Load API key for Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("Gemini API key not found. AI insights will not be available.")

def get_insights_from_data(df: pd.DataFrame, context: str, custom_question: str = None) -> str:
    """Generate AI insights from data using Gemini"""
    if not GEMINI_API_KEY or df.empty:
        return "No data available for insights or API key not configured."

    # Determine the DataFrame to use for dictionary-type results
    if isinstance(df, dict):
        if "categorized" in df and not df["categorized"].empty:
            df_to_use = df["categorized"]
            summary = df_to_use.head(10).to_string(index=False)
        elif "detailed" in df and not df["detailed"].empty:
            df_to_use = df["detailed"]
            summary = df_to_use.head(10).to_string(index=False)
        else:
            return "No data available for insights."
    else:
        summary = df.head(10).to_string(index=False)

    if custom_question:
        prompt = (
            f"Given the following data about {context} in FDA tobacco reports:\n\n"
            f"{summary}\n\n"
            f"Answer this question in 3-5 sentences, focusing on data-driven insights:\n"
            f"{custom_question}"
        )
    else:
        prompt = (
            f"Analyze the following data about {context} in FDA tobacco reports:\n\n"
            f"{summary}\n\n"
            "Provide a concise summary (3-5 sentences) of key trends, patterns, and notable findings. "
            "Include any potential public health implications or recommendations based on this data."
        )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def render_ai_insights_section(df, context, key_prefix):
    """Render AI insights section with option for custom questions"""
    st.subheader("AI Insights")
    question = st.text_input("Custom question (optional)", key=f"{key_prefix}_question")
    if st.button("Generate Insights", key=f"{key_prefix}_insights"):
        with st.spinner("Generating insights..."):
            insights = get_insights_from_data(df, context, question or "")
            st.write(insights)

def display_product_analysis():
    """Display analysis by tobacco product type"""
    st.subheader("Analysis by Tobacco Product Type")

    # Use global date range from session state
    result_df = get_tobacco_reports_by_product(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    if isinstance(result_df, dict) and "categorized" in result_df and not result_df["categorized"].empty:
        category_df = result_df["categorized"]
        detailed_df = result_df["detailed"]

        # Create visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for categories
            fig_bar = px.bar(
                category_df,
                x="Product Category",
                y="Count",
                title="Tobacco Reports by Product Category",
                color="Product Category",
                text="Count"
            )
            fig_bar.update_layout(xaxis_tickangle=0)
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Pie chart for categories
            fig_pie = px.pie(
                category_df,
                values="Count",
                names="Product Category",
                title="Distribution of Tobacco Reports by Product Category",
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Local filtering
        st.subheader("Explore Specific Product Types")

        # Get selected categories for detailed view
        selected_category = st.selectbox(
            "Select Product Category to Explore",
            options=category_df["Product Category"].unique()
        )

        if selected_category:
            # Filter detailed data by selected category
            filtered_products = detailed_df[detailed_df["Product Category"] == selected_category]

            # Sort by count and limit to top results
            top_products = filtered_products.sort_values("Count", ascending=False).head(10)

            # Create visualization for specific products
            fig_products = px.bar(
                top_products,
                x="Product Type",
                y="Count",
                title=f"Top Products in {selected_category} Category",
                color="Product Type",
                text="Count"
            )
            fig_products.update_layout(xaxis_tickangle=-45)
            fig_products.update_traces(textposition='outside')
            st.plotly_chart(fig_products, use_container_width=True)

            # Show the data
            st.dataframe(top_products, use_container_width=True, hide_index=True)

        # AI Insights section
        render_ai_insights_section(result_df, "tobacco product types", "product")
    else:
        st.warning("No data available for the selected date range.")

def display_problem_analysis():
    """Display analysis by tobacco problem type"""
    st.subheader("Analysis by Problem Type")

    # Use global date range from session state
    result_df = get_tobacco_reports_by_problem_type(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    if isinstance(result_df, dict) and "categorized" in result_df and not result_df["categorized"].empty:
        category_df = result_df["categorized"]
        detailed_df = result_df["detailed"]

        # Create visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for categories
            fig_bar = px.bar(
                category_df,
                x="Problem Category",
                y="Count",
                title="Tobacco Reports by Problem Category",
                color="Problem Category",
                text="Count"
            )
            fig_bar.update_layout(xaxis_tickangle=0)
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Treemap visualization for categories
            fig_treemap = px.treemap(
                category_df,
                path=["Problem Category"],
                values="Count",
                title="Hierarchy of Problem Categories"
            )
            st.plotly_chart(fig_treemap, use_container_width=True)

        # Local filtering
        st.subheader("Explore Specific Problem Types")

        # Get selected categories for detailed view
        selected_category = st.selectbox(
            "Select Problem Category to Explore",
            options=category_df["Problem Category"].unique()
        )

        if selected_category:
            # Filter detailed data by selected category
            filtered_problems = detailed_df[detailed_df["Problem Category"] == selected_category]

            # Sort by count and limit to top results
            top_problems = filtered_problems.sort_values("Count", ascending=False).head(10)

            # Create visualization for specific problems
            fig_problems = px.bar(
                top_problems,
                x="Problem Type",
                y="Count",
                title=f"Top Problems in {selected_category} Category",
                color="Problem Type",
                text="Count"
            )
            fig_problems.update_layout(xaxis_tickangle=-45)
            fig_problems.update_traces(textposition='outside')
            st.plotly_chart(fig_problems, use_container_width=True)

            # Show the data
            st.dataframe(top_problems, use_container_width=True, hide_index=True)

        # AI Insights section
        render_ai_insights_section(result_df, "tobacco problem types", "problem")
    else:
        st.warning("No data available for the selected date range.")

def display_health_effect_analysis():
    """Display analysis by health effects"""
    st.subheader("Analysis by Health Effects")

    # Use global date range from session state
    result_df = get_tobacco_reports_by_health_effect(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    if isinstance(result_df, dict) and "categorized" in result_df and not result_df["categorized"].empty:
        category_df = result_df["categorized"]
        detailed_df = result_df["detailed"]

        # Create visualizations for categories
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for categories
            fig_bar = px.bar(
                category_df,
                x="Effect Category",
                y="Count",
                title="Health Effects by Category",
                color="Effect Category",
                text="Count"
            )
            fig_bar.update_layout(xaxis_tickangle=0)
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Pie chart for categories
            fig_pie = px.pie(
                category_df,
                values="Count",
                names="Effect Category",
                title="Distribution of Health Effects by Category",
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Local filtering
        st.subheader("Explore Specific Health Effects")

        # Get selected categories for detailed view
        selected_category = st.selectbox(
            "Select Health Effect Category to Explore",
            options=category_df["Effect Category"].unique()
        )

        if selected_category:
            # Filter detailed data by selected category
            filtered_effects = detailed_df[detailed_df["Effect Category"] == selected_category]

            # Sort by count and limit to top results
            top_effects = filtered_effects.sort_values("Count", ascending=False).head(10)

            # Create visualization for specific health effects
            fig_effects = px.bar(
                top_effects,
                x="Health Effect",
                y="Count",
                title=f"Top Health Effects in {selected_category} Category",
                color="Health Effect",
                text="Count"
            )
            fig_effects.update_layout(xaxis_tickangle=-45)
            fig_effects.update_traces(textposition='outside')
            st.plotly_chart(fig_effects, use_container_width=True)

            # Show the data
            st.dataframe(top_effects, use_container_width=True, hide_index=True)

        # AI Insights section
        render_ai_insights_section(result_df, "health effects from tobacco products", "health_effect")
    else:
        st.warning("No health effect data available for the selected date range.")

def display_demographic_analysis():
    """Display analysis by demographic information"""
    st.subheader("Analysis by Demographics")

    # Create tabs for different demographics
    tab1, tab2 = st.tabs(["Age Analysis", "Gender Analysis"])

    # Age Analysis
    with tab1:
        st.subheader("Age Distribution")

        # Use global date range from session state
        age_df = get_tobacco_reports_by_demographic(
            "age",
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if age_df.empty:
            st.warning("No age data available for the selected date range.")
        else:
            # Create visualization
            fig_age = px.bar(
                age_df,
                x="Age Group",
                y="Count",
                title="Tobacco Reports by Age Group",
                color="Age Group",
                text="Count"
            )
            fig_age.update_layout(xaxis_tickangle=0)
            fig_age.update_traces(textposition='outside')
            st.plotly_chart(fig_age, use_container_width=True)

            # Age group selection for filtering
            selected_age_groups = st.multiselect(
                "Filter by Age Group",
                options=age_df["Age Group"].unique(),
                default=[]
            )

            # Apply filter if selected
            if selected_age_groups:
                filtered_age_df = age_df[age_df["Age Group"].isin(selected_age_groups)]
            else:
                filtered_age_df = age_df

            # Show the filtered data in a collapsed expander
            with st.expander("View Age Distribution Data", expanded=False):
                st.dataframe(filtered_age_df, use_container_width=True, hide_index=True)

            # AI Insights for age
            render_ai_insights_section(age_df, "age distribution in tobacco reports", "age")

    # Gender Analysis
    with tab2:
        st.subheader("Gender Distribution")

        # Use global date range from session state
        gender_df = get_tobacco_reports_by_demographic(
            "gender",
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if gender_df.empty:
            st.warning("No gender data available for the selected date range.")
        else:
            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart
                fig_gender_bar = px.bar(
                    gender_df,
                    x="Gender",
                    y="Count",
                    title="Tobacco Reports by Gender",
                    color="Gender",
                    text="Count"
                )
                fig_gender_bar.update_layout(xaxis_tickangle=0)
                fig_gender_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_gender_bar, use_container_width=True)

            with col2:
                # Pie chart
                fig_gender_pie = px.pie(
                    gender_df,
                    values="Count",
                    names="Gender",
                    title="Distribution of Tobacco Reports by Gender",
                    hole=0.4
                )
                fig_gender_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_gender_pie, use_container_width=True)

            # Show the data
            st.dataframe(gender_df, use_container_width=True, hide_index=True)

            # AI Insights for gender
            render_ai_insights_section(gender_df, "gender distribution in tobacco reports", "gender")

def display_time_analysis():
    """Display analysis of reports over time"""
    st.subheader("Reports Over Time")

    # Time interval selector
    interval = st.radio(
        "Select Time Interval",
        options=["year", "quarter", "month"],
        format_func=lambda x: x.capitalize(),
        horizontal=True,
        key="tobacco_time_interval"
    )

    # Use global date range from session state
    time_df = get_tobacco_reports_over_time(
        interval,
        st.session_state.start_date,
        st.session_state.end_date
    )

    if time_df.empty:
        st.warning(f"No time-based data available for the selected date range and {interval} interval.")
    else:
        # Create time-series visualization
        fig_time = px.line(
            time_df,
            x="Time Period",
            y="Count",
            title=f"Tobacco Reports Over Time (by {interval.capitalize()})",
            markers=True
        )

        # Add a trend line
        fig_time.add_trace(
            go.Scatter(
                x=time_df["Time Period"],
                y=time_df["Count"].rolling(window=3, min_periods=1).mean(),
                mode='lines',
                name='3-point Moving Average',
                line=dict(color='red', dash='dash')
            )
        )

        st.plotly_chart(fig_time, use_container_width=True)

        # Add yearly comparison if using month or quarter
        if interval in ["month", "quarter"]:
            # Create a grouped bar chart to compare different years
            # This requires some data manipulation first
            time_df["Year"] = time_df["Time Period"].astype(str)

            if interval == "month":
                # Extract year from month names if possible
                time_df["Month"] = time_df["Time Period"]

                # Create grouped bar chart
                fig_comparison = px.bar(
                    time_df,
                    x="Month",
                    y="Count",
                    color="Year",
                    title=f"Monthly Comparison of Tobacco Reports",
                    barmode="group"
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

            elif interval == "quarter":
                # Create grouped bar chart for quarters
                fig_comparison = px.bar(
                    time_df,
                    x="Time Period",
                    y="Count",
                    color="Year",
                    title=f"Quarterly Comparison of Tobacco Reports",
                    barmode="group"
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

        # Show the data
        st.dataframe(time_df, use_container_width=True, hide_index=True)

        # AI Insights for time trends
        render_ai_insights_section(time_df, f"tobacco reports over time (by {interval})", "time")

def display_tobacco_reports():
    """Main function to display the Tobacco page"""
    st.title("Tobacco Data Analysis")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Product Analysis",
        "Problem Type Analysis",
        "Health Effects Analysis",
        "Demographics & Time Trends"
    ])

    with tab1:
        display_product_analysis()

    with tab2:
        display_problem_analysis()

    with tab3:
        display_health_effect_analysis()

    with tab4:
        # Create subtabs for demographics and time trends
        subtab1, subtab2 = st.tabs(["Demographics", "Time Trends"])

        with subtab1:
            display_demographic_analysis()

        with subtab2:
            display_time_analysis()
