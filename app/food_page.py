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

from src.food_events import (
    get_recall_reasons_distribution,
    _fetch_all_food_data
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

def display_food_reports():
    st.title("Food Reports")

    # Load data
    with st.spinner("Loading food data..."):
        data = _fetch_all_food_data()

    st.header("Recall Distribution by Report Date")
    if "recall_reasons" in data and not data["recall_reasons"].empty:
        df = data["recall_reasons"]

        fig = px.line(
            df,
            x="Date",
            y="Count",
            title="Recalls Over Time",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Report Date",
            yaxis_title="Number of Recalls"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed Statistics"):
            st.dataframe(df)

        # AI Insights
        st.markdown("### AI Insights")
        if st.button("Get Insights for Recall Distribution"):
            insights = get_insights_from_data(df, "food recall distribution by report date")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about recall distribution", key="reason_question")
        if custom_question:
            insights = get_insights_from_data(df, "food recall distribution by report date", custom_question)
            st.write(insights)
    else:
        st.warning("No recall distribution data available")
