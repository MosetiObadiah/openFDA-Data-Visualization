import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tobacco_events import (
    get_tobacco_products_distribution,
    get_health_problems_distribution,
    get_health_problems_count_distribution,
    get_product_problems_distribution,
    _fetch_all_tobacco_data
)
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

def render_ai_insights_section(df, context, key_prefix):
    st.subheader("AI Insights")
    question = st.text_input("Custom question (optional)", key=f"{key_prefix}_question")
    if st.button("Generate Insights", key=f"{key_prefix}_insights"):
        with st.spinner("Generating insights..."):
            st.write(get_insights_from_data(df, context, question or ""))

def display_tobacco_reports():
    st.title("Tobacco Reports")

    # Load data
    with st.spinner("Loading tobacco data..."):
        data = _fetch_all_tobacco_data()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tobacco Products",
        "Health Problems",
        "Health Problems Count",
        "Product Problems"
    ])

    # Tab 1: Tobacco Products
    with tab1:
        st.header("Tobacco Products Distribution")
        if "tobacco_products" in data and not data["tobacco_products"].empty:
            df = data["tobacco_products"]

            # Create a horizontal bar chart for better readability
            fig = px.bar(
                df.head(15),  # Show top 15 products
                x="Count",
                y="Product",
                title="Top 15 Tobacco Products by Number of Reports",
                text="Percentage",
                orientation="h"
            )
            fig.update_layout(
                xaxis_title="Number of Reports",
                yaxis_title="Product",
                height=600  # Make the chart taller to accommodate more products
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Statistics"):
                st.dataframe(df)

            render_ai_insights_section(df, "tobacco products distribution", "tobacco_products")

    # Tab 2: Health Problems
    with tab2:
        st.header("Reported Health Problems")
        if "health_problems" in data and not data["health_problems"].empty:
            df = data["health_problems"]

            # Create two columns for visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for top health problems
                fig = px.bar(
                    df.head(10),
                    x="Health Problem",
                    y="Count",
                    title="Top 10 Health Problems",
                    text="Percentage"
                )
                fig.update_layout(
                    xaxis_title="Health Problem",
                    yaxis_title="Number of Reports",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Pie chart for distribution
                fig = px.pie(
                    df,
                    values="Count",
                    names="Health Problem",
                    title="Distribution of Health Problems",
                    hole=0.4
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Statistics"):
                st.dataframe(df)

            render_ai_insights_section(df, "reported health problems", "health_problems")

    # Tab 3: Health Problems Count
    with tab3:
        st.header("Number of Health Problems per Report")
        if "health_problems_count" in data and not data["health_problems_count"].empty:
            df = data["health_problems_count"]

            # Create two columns for visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for distribution
                fig = px.bar(
                    df,
                    x="Number of Problems",
                    y="Count",
                    title="Distribution of Health Problems per Report",
                    text="Percentage"
                )
                fig.update_layout(
                    xaxis_title="Number of Health Problems",
                    yaxis_title="Number of Reports"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Pie chart for distribution
                fig = px.pie(
                    df,
                    values="Count",
                    names="Number of Problems",
                    title="Distribution of Health Problems per Report",
                    hole=0.4
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Statistics"):
                st.dataframe(df)

            render_ai_insights_section(df, "number of health problems per report", "health_problems_count")

    # Tab 4: Product Problems
    with tab4:
        st.header("Reported Product Problems")
        if "product_problems" in data and not data["product_problems"].empty:
            df = data["product_problems"]

            # Create two columns for visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for top product problems
                fig = px.bar(
                    df.head(10),
                    x="Product Problem",
                    y="Count",
                    title="Top 10 Product Problems",
                    text="Percentage"
                )
                fig.update_layout(
                    xaxis_title="Product Problem",
                    yaxis_title="Number of Reports",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Pie chart for distribution
                fig = px.pie(
                    df,
                    values="Count",
                    names="Product Problem",
                    title="Distribution of Product Problems",
                    hole=0.4
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Statistics"):
                st.dataframe(df)

            render_ai_insights_section(df, "reported product problems", "product_problems")
        else:
            st.warning("No product problems data available")
