import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys
import json
import time
import plotly.graph_objects as go
import re
import asyncio
import concurrent.futures
import threading

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drug_events import (
    adverse_events_by_drug_within_data_range,
    recall_frequency_by_year,
    most_common_recalled_drugs
)
from src.food_endpoints import (
    get_food_recalls_by_classification,
    get_food_recalls_by_reason,
    get_food_events_by_symptom
)
from src.tobacco_endpoints import (
    get_tobacco_reports_by_health_effect,
    get_tobacco_reports_by_product,
    get_tobacco_reports_over_time
)

# Initialize session state for sample_size
if "sample_size" not in st.session_state:
    st.session_state.sample_size = 50  # Default sample size

# Load API key for Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data(ttl=3600)
def generate_healthcare_trends_summary(include_drug=True, include_food=True, include_tobacco=True):
    """Generate a summary of healthcare trends using Gemini."""

    # Initialize data collection
    data_points = []

    # Use a smaller sample size for better performance
    sample_size = 50  # Reduced from 100

    # Collect drug data if requested
    if include_drug:
        try:
            drug_events = adverse_events_by_drug_within_data_range("2020-01-01", "2023-12-31")
            drug_recalls = most_common_recalled_drugs(limit=sample_size)

            if not drug_events.empty:
                data_points.append(f"Top drugs with adverse events:\n{drug_events.head(5).to_string(index=False)}")

            if not drug_recalls.empty:
                data_points.append(f"Top recalled drugs:\n{drug_recalls.head(5).to_string(index=False)}")
        except Exception as e:
            data_points.append(f"Drug data extraction error: {str(e)}")

    # Collect food data if requested
    if include_food:
        try:
            food_recalls = get_food_recalls_by_classification(None, None, sample_size)
            food_reasons = get_food_recalls_by_reason(None, None, sample_size)

            if not food_recalls.empty:
                data_points.append(f"Food recall classifications:\n{food_recalls.to_string(index=False)}")

            if isinstance(food_reasons, dict) and "categorized" in food_reasons and not food_reasons["categorized"].empty:
                data_points.append(f"Food recall reasons:\n{food_reasons['categorized'].head(5).to_string(index=False)}")
        except Exception as e:
            data_points.append(f"Food data extraction error: {str(e)}")

    # Collect tobacco data if requested
    if include_tobacco:
        try:
            tobacco_effects = get_tobacco_reports_by_health_effect(None, None, sample_size)

            if isinstance(tobacco_effects, dict) and "categorized" in tobacco_effects and not tobacco_effects["categorized"].empty:
                data_points.append(f"Tobacco health effects:\n{tobacco_effects['categorized'].head(5).to_string(index=False)}")
        except Exception as e:
            data_points.append(f"Tobacco data extraction error: {str(e)}")

    # If no data was collected, return an error message
    if not data_points:
        return "No data available to generate a healthcare trends summary."

    # Create a targeted prompt for healthcare trends
    prompt = f"""
    You are a healthcare data analyst tasked with identifying trends from FDA data.

    Based on the following FDA data samples:

    {"".join(f"{i+1}. {data}\n\n" for i, data in enumerate(data_points))}

    Please provide a 3-5 paragraph analysis of healthcare trends that might be observable from this data.
    Focus on:
    1. Potential public health implications
    2. Emerging patterns in recalls, adverse events, and health effects
    3. Correlations between different types of FDA data (drugs, food, tobacco)
    4. Recommendations for healthcare professionals or regulatory bodies

    Include both observations from the data and potential future projections.
    Your analysis should be insightful, data-driven, and present a cohesive narrative about current healthcare trends.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating healthcare trends summary: {e}"

def extract_trend_data_for_visualization(prediction_text, trend_category):
    """Extract data points from prediction text for visualization."""
    # Default values
    categories = []
    values = []
    confidence = 0.5  # Medium confidence default

    # Try to extract a confidence level
    confidence_match = re.search(r'confidence level[^\n.]*?(high|medium|low)', prediction_text.lower())
    if confidence_match:
        conf_level = confidence_match.group(1)
        if conf_level == "high":
            confidence = 0.8
        elif conf_level == "medium":
            confidence = 0.5
        elif conf_level == "low":
            confidence = 0.3

    # Extract timeline information
    short_term = re.search(r'short[- ]term[^\n.]*?(\d+)[^\n.]*?(increase|decrease|rise|decline|growth|reduction)', prediction_text.lower())
    mid_term = re.search(r'mid[- ]term[^\n.]*?(\d+)[^\n.]*?(increase|decrease|rise|decline|growth|reduction)', prediction_text.lower())
    long_term = re.search(r'long[- ]term[^\n.]*?(\d+)[^\n.]*?(increase|decrease|rise|decline|growth|reduction)', prediction_text.lower())

    # Create timeline data
    if trend_category == "Drug Safety":
        categories = ["Current", "Short-term", "Mid-term", "Long-term"]
        # Default starting value based on category
        base_value = 100
        values = [base_value]

        # Process timeline values
        for match in [short_term, mid_term, long_term]:
            if match:
                pct = int(match.group(1)) if match.group(1).isdigit() else 10
                direction = 1 if match.group(2) in ["increase", "rise", "growth"] else -1
                values.append(values[-1] * (1 + direction * pct/100))
            else:
                # Default change if not specified
                values.append(values[-1] * (1 + 0.05 * (1 if len(values) % 2 == 0 else -1)))

    elif trend_category == "Food Safety":
        categories = ["Current", "Short-term", "Mid-term", "Long-term"]
        base_value = 75
        values = [base_value]

        for match in [short_term, mid_term, long_term]:
            if match:
                pct = int(match.group(1)) if match.group(1).isdigit() else 8
                direction = 1 if match.group(2) in ["increase", "rise", "growth"] else -1
                values.append(values[-1] * (1 + direction * pct/100))
            else:
                values.append(values[-1] * (1 + 0.04 * (1 if len(values) % 2 == 0 else -1)))

    elif trend_category == "Tobacco Effects":
        categories = ["Current", "Short-term", "Mid-term", "Long-term"]
        base_value = 50
        values = [base_value]

        for match in [short_term, mid_term, long_term]:
            if match:
                pct = int(match.group(1)) if match.group(1).isdigit() else 12
                direction = 1 if match.group(2) in ["increase", "rise", "growth"] else -1
                values.append(values[-1] * (1 + direction * pct/100))
            else:
                values.append(values[-1] * (1 + 0.06 * (1 if len(values) % 2 == 0 else -1)))

    # Extract key factors if available
    key_factors = []
    factors_section = re.search(r'key factors[^\n]*:(.*?)(\n\d|\n\n|$)', prediction_text.lower(), re.DOTALL)
    if factors_section:
        factors_text = factors_section.group(1)
        for line in factors_text.split('\n'):
            if line.strip() and line.strip()[0].isdigit() or line.strip()[0] == '-':
                factor = line.strip()
                if len(factor) > 5:  # Make sure it's not just a number or dash
                    key_factors.append(factor)

    return {
        "categories": categories,
        "values": values,
        "confidence": confidence,
        "key_factors": key_factors
    }

@st.cache_data(ttl=3600)
def generate_trend_prediction(trend_category, prediction_question):
    """Generate a prediction about a specific healthcare trend using Gemini."""

    # Use a smaller sample size for better performance
    sample_size = 50  # Reduced from 100

    # Initialize data collection based on trend category
    data_points = []

    if trend_category == "Drug Safety":
        try:
            drug_events = adverse_events_by_drug_within_data_range("2020-01-01", "2023-12-31", sample_size)
            drug_recalls = most_common_recalled_drugs(limit=sample_size)

            if not drug_events.empty:
                data_points.append(f"Top drugs with adverse events:\n{drug_events.head(5).to_string(index=False)}")

            if not drug_recalls.empty:
                data_points.append(f"Top recalled drugs:\n{drug_recalls.head(5).to_string(index=False)}")
        except Exception as e:
            data_points.append(f"Drug data extraction error: {str(e)}")

    elif trend_category == "Food Safety":
        try:
            food_recalls = get_food_recalls_by_classification(None, None, sample_size)
            food_reasons = get_food_recalls_by_reason(None, None, sample_size)

            if not food_recalls.empty:
                data_points.append(f"Food recall classifications:\n{food_recalls.to_string(index=False)}")

            if isinstance(food_reasons, dict) and "categorized" in food_reasons and not food_reasons["categorized"].empty:
                data_points.append(f"Food recall reasons:\n{food_reasons['categorized'].head(5).to_string(index=False)}")
        except Exception as e:
            data_points.append(f"Food data extraction error: {str(e)}")

    elif trend_category == "Tobacco Effects":
        try:
            tobacco_effects = get_tobacco_reports_by_health_effect(None, None, sample_size)
            tobacco_products = get_tobacco_reports_by_product(None, None, sample_size)

            if isinstance(tobacco_effects, dict) and "categorized" in tobacco_effects and not tobacco_effects["categorized"].empty:
                data_points.append(f"Tobacco health effects:\n{tobacco_effects['categorized'].head(5).to_string(index=False)}")

            if isinstance(tobacco_products, dict) and "categorized" in tobacco_products and not tobacco_products["categorized"].empty:
                data_points.append(f"Tobacco products:\n{tobacco_products['categorized'].head(5).to_string(index=False)}")
        except Exception as e:
            data_points.append(f"Tobacco data extraction error: {str(e)}")

    # If no data was collected, return an error message
    if not data_points:
        return "No data available to generate a prediction."

    # Create a targeted prompt for trend prediction
    prompt = f"""
    You are a healthcare data analyst and forecaster who specializes in FDA data.

    Based on the following {trend_category} data:

    {"".join(f"{data}\n\n" for data in data_points)}

    Answer this specific question about future trends:
    "{prediction_question}"

    Provide:
    1. A detailed prediction with reasoning (2-3 paragraphs)
    2. Key factors that could influence this trend (list at least 3-5 factors)
    3. Potential timeline for these developments with percentages:
       - Short-term (1-2 years): Specify expected change in percentage (e.g., 15% increase, 10% decrease)
       - Mid-term (3-5 years): Specify expected change in percentage
       - Long-term (5+ years): Specify expected change in percentage
    4. Confidence level in your prediction (high, medium, or low) with explanation

    Base your predictions on observable patterns in the data, known regulatory trends,
    scientific developments, and healthcare industry dynamics.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating trend prediction: {e}"

def load_data_concurrently(trend_category, sample_size=50):
    """Load data concurrently to speed up the process."""
    data_results = {}

    if trend_category == "Drug Safety":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_drug_events = executor.submit(
                adverse_events_by_drug_within_data_range, "2020-01-01", "2023-12-31", sample_size
            )
            future_drug_recalls = executor.submit(
                most_common_recalled_drugs, sample_size
            )

            # Collect results
            data_results["drug_events"] = future_drug_events.result()
            data_results["drug_recalls"] = future_drug_recalls.result()

    elif trend_category == "Food Safety":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_food_recalls = executor.submit(
                get_food_recalls_by_classification, None, None, sample_size
            )
            future_food_reasons = executor.submit(
                get_food_recalls_by_reason, None, None, sample_size
            )

            # Collect results
            data_results["food_recalls"] = future_food_recalls.result()
            data_results["food_reasons"] = future_food_reasons.result()

    elif trend_category == "Tobacco Effects":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_tobacco_effects = executor.submit(
                get_tobacco_reports_by_health_effect, None, None, sample_size
            )
            future_tobacco_products = executor.submit(
                get_tobacco_reports_by_product, None, None, sample_size
            )

            # Collect results
            data_results["tobacco_effects"] = future_tobacco_effects.result()
            data_results["tobacco_products"] = future_tobacco_products.result()

    return data_results

def create_prediction_chart(prediction_data, trend_category, question):
    """Create a chart visualization based on prediction data."""
    # Create timeline chart
    fig = go.Figure()

    # Add the main trend line
    fig.add_trace(go.Scatter(
        x=prediction_data["categories"],
        y=prediction_data["values"],
        mode='lines+markers',
        name='Predicted Trend',
        line=dict(color='royalblue', width=4),
        marker=dict(size=10)
    ))

    # Add confidence interval
    confidence = prediction_data["confidence"]
    upper_values = [val * (1 + (1 - confidence) * 0.5) for val in prediction_data["values"]]
    lower_values = [val * (1 - (1 - confidence) * 0.5) for val in prediction_data["values"]]

    # Add confidence band
    fig.add_trace(go.Scatter(
        x=prediction_data["categories"] + prediction_data["categories"][::-1],
        y=upper_values + lower_values[::-1],
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    # Update layout
    timeline_title = {
        "Drug Safety": "Projected Trend for Drug Safety Events",
        "Food Safety": "Projected Trend for Food Safety Events",
        "Tobacco Effects": "Projected Trend for Tobacco Health Effects"
    }.get(trend_category, "Projected Trend")

    fig.update_layout(
        title=f"{timeline_title}<br><sup>{question}</sup>",
        xaxis_title="Time Period",
        yaxis_title="Relative Trend (Baseline = 100)",
        legend_title="Prediction",
        template="plotly_white",
        height=500
    )

    # Add key factors as annotations if available
    if prediction_data["key_factors"]:
        fig.add_annotation(
            x=prediction_data["categories"][-2],
            y=min(prediction_data["values"]),
            text="<br>".join(["Key Factors:"] + prediction_data["key_factors"][:3]),
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=40,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=4,
            font=dict(size=10)
        )

    return fig

def display_healthcare_trends():
    """Display healthcare trends analysis page."""
    st.title("Healthcare Trends Analysis")

    # Ensure sample_size is initialized
    if "sample_size" not in st.session_state:
        st.session_state.sample_size = 50

    st.write("""
    This tool uses AI to predict how specific healthcare trends might evolve in the future based on FDA data patterns.
    Select a trend category and ask a specific question about future developments.
    """)

    # Trend category selection
    trend_category = st.selectbox(
        "Select Trend Category",
        options=["Drug Safety", "Food Safety", "Tobacco Effects"],
        index=0,
        key="trend_category"
    )

    # Pre-load data in the background
    if "preloaded_data" not in st.session_state:
        st.session_state.preloaded_data = {}

    # Start background loading for the selected category
    if trend_category not in st.session_state.preloaded_data:
        with st.spinner(f"Pre-loading {trend_category} data..."):
            st.session_state.preloaded_data[trend_category] = load_data_concurrently(trend_category, st.session_state.sample_size)

    # Prediction question input
    st.markdown("### Ask About Future Trends")

    # Provide example questions based on the selected category
    example_questions = {
        "Drug Safety": [
            "How will adverse events for antidepressants trend over the next 5 years?",
            "What new drug safety regulations might emerge in response to current recall patterns?",
            "Will drug recalls for cardiovascular medications increase or decrease in the coming decade?"
        ],
        "Food Safety": [
            "How will allergen-related food recalls change in the next 3 years?",
            "What food categories are likely to see increased regulatory scrutiny based on recall patterns?",
            "Will microbial contamination continue to be a leading cause of food recalls?"
        ],
        "Tobacco Effects": [
            "How might the health effects of electronic cigarettes evolve in the next decade?",
            "What new tobacco product categories might emerge and what health risks might they present?",
            "Will respiratory issues from tobacco products increase or decrease in the coming years?"
        ]
    }

    st.write("Example questions:")
    for question in example_questions[trend_category]:
        st.markdown(f"- *{question}*")

    prediction_question = st.text_area(
        "Enter your prediction question",
        height=100,
        key="prediction_question",
        placeholder=f"e.g., {example_questions[trend_category][0]}"
    )

    # Generate prediction button
    if st.button("Generate Prediction", key="generate_prediction"):
        if not prediction_question:
            st.error("Please enter a prediction question.")
        else:
            with st.spinner(f"Analyzing {trend_category} data and generating prediction..."):
                # Create a container for the placeholder
                prediction_placeholder = st.empty()

                # Show a progress bar in the placeholder
                with prediction_placeholder.container():
                    progress_bar = st.progress(0)
                    st.info("Processing your question...")

                    # Simulate progress while actually doing the work
                    for i in range(25):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                    st.info(f"Analyzing {trend_category} data...")
                    for i in range(25, 55):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                    st.info("Identifying patterns and projecting future trends...")
                    for i in range(55, 85):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                    st.info("Generating prediction visualization...")
                    for i in range(85, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                # Generate the prediction
                prediction = generate_trend_prediction(trend_category, prediction_question)

                # Extract data for visualization
                prediction_data = extract_trend_data_for_visualization(prediction, trend_category)

                # Clear the placeholder
                prediction_placeholder.empty()

                # Display the chart
                st.subheader("Future Trend Prediction Chart")
                fig = create_prediction_chart(prediction_data, trend_category, prediction_question)
                st.plotly_chart(fig, use_container_width=True)

                # Display the text prediction below the chart
                with st.expander("View Detailed Prediction Analysis", expanded=True):
                    st.markdown(prediction)

                # Add a download button for the prediction
                st.download_button(
                    "Download Prediction Report",
                    prediction,
                    "trend_prediction.txt",
                    "text/plain",
                    key="download_prediction"
                )

    # Add disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** These predictions are generated by an AI model based on historical FDA data patterns.
    They should be considered as informed projections rather than definitive forecasts.
    Actual outcomes may differ due to regulatory changes, scientific breakthroughs, market dynamics, and other factors.
    """)
