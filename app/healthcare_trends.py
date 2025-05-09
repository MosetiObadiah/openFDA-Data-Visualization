import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys
import time
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drug_events import (
    adverse_events_by_drug_within_data_range,
    most_common_recalled_drugs
)
from src.food_endpoints import (
    get_food_recalls_by_classification,
    get_food_recalls_by_reason
)
from src.tobacco_endpoints import (
    get_tobacco_reports_by_health_effect,
    get_tobacco_reports_by_product
)

# Initialize session state for sample_size
if "sample_size" not in st.session_state:
    st.session_state.sample_size = 1000  # Default sample size

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data(ttl=3600)
def generate_healthcare_trends_summary(include_drug=True, include_food=True, include_tobacco=True):

    # Initialize data collection
    data_points = []

    # Use a smaller sample size for better performance
    sample_size = 1000

    # It will colledt food drug or tabacco data if requested
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

    # targeted prompt for healthcare trends
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

@st.cache_data(ttl=3600)
def generate_trend_prediction(trend_category, prediction_question):
    sample_size = 1000

    data_points = []
    data_available = False

    # Identify keywords in the question to determine which data to fetch
    question_lower = prediction_question.lower()

    if trend_category == "Drug Safety":
        try:
            # Check for specific keywords to determine which drug data to fetch
            fetched_data = False

            # Adverse events data
            if any(keyword in question_lower for keyword in ["adverse", "side effect", "reaction", "event"]):
                drug_events = adverse_events_by_drug_within_data_range("2020-01-01", "2023-12-31", sample_size)
                if not drug_events.empty:
                    data_points.append(f"Top drugs with adverse events:\n{drug_events.to_string(index=False)}")
                    fetched_data = True
                    data_available = True

            # Recall data
            if any(keyword in question_lower for keyword in ["recall", "withdraw", "safety", "enforcement"]):
                drug_recalls = most_common_recalled_drugs(limit=sample_size)
                if not drug_recalls.empty:
                    data_points.append(f"Top recalled drugs:\n{drug_recalls.to_string(index=False)}")
                    fetched_data = True
                    data_available = True

            # If no specific data matched keywords, fetch general data
            if not fetched_data:
                drug_events = adverse_events_by_drug_within_data_range("2020-01-01", "2023-12-31", sample_size)
                drug_recalls = most_common_recalled_drugs(limit=sample_size)

                if not drug_events.empty:
                    data_points.append(f"Top drugs with adverse events:\n{drug_events.head(5).to_string(index=False)}")
                    data_available = True

                if not drug_recalls.empty:
                    data_points.append(f"Top recalled drugs:\n{drug_recalls.head(5).to_string(index=False)}")
                    data_available = True

        except Exception as e:
            data_points.append(f"Drug data extraction error: {str(e)}")

    elif trend_category == "Food Safety":
        try:
            fetched_data = False

            # Food classification data
            if any(keyword in question_lower for keyword in ["class", "classification", "category", "type"]):
                food_recalls = get_food_recalls_by_classification(None, None, sample_size)
                if not food_recalls.empty:
                    data_points.append(f"Food recall classifications:\n{food_recalls.to_string(index=False)}")
                    fetched_data = True
                    data_available = True

            # Food recall reason data
            if any(keyword in question_lower for keyword in ["reason", "cause", "why", "contamination", "allergen"]):
                food_reasons = get_food_recalls_by_reason(None, None, sample_size)
                if isinstance(food_reasons, dict) and "categorized" in food_reasons and not food_reasons["categorized"].empty:
                    data_points.append(f"Food recall reasons:\n{food_reasons['categorized'].to_string(index=False)}")
                    fetched_data = True
                    data_available = True

            if not fetched_data:
                food_recalls = get_food_recalls_by_classification(None, None, sample_size)
                food_reasons = get_food_recalls_by_reason(None, None, sample_size)

                if not food_recalls.empty:
                    data_points.append(f"Food recall classifications:\n{food_recalls.head(5).to_string(index=False)}")
                    data_available = True

                if isinstance(food_reasons, dict) and "categorized" in food_reasons and not food_reasons["categorized"].empty:
                    data_points.append(f"Food recall reasons:\n{food_reasons['categorized'].head(5).to_string(index=False)}")
                    data_available = True

        except Exception as e:
            data_points.append(f"Food data extraction error: {str(e)}")

    elif trend_category == "Tobacco Effects":
        try:
            fetched_data = False

            # Health effects data
            if any(keyword in question_lower for keyword in ["health", "effect", "impact", "symptom", "condition"]):
                tobacco_effects = get_tobacco_reports_by_health_effect(None, None, sample_size)
                if isinstance(tobacco_effects, dict) and "categorized" in tobacco_effects and not tobacco_effects["categorized"].empty:
                    data_points.append(f"Tobacco health effects:\n{tobacco_effects['categorized'].to_string(index=False)}")
                    fetched_data = True
                    data_available = True

            # Product data
            if any(keyword in question_lower for keyword in ["product", "cigarette", "vape", "cigar", "smokeless"]):
                tobacco_products = get_tobacco_reports_by_product(None, None, sample_size)
                if isinstance(tobacco_products, dict) and "categorized" in tobacco_products and not tobacco_products["categorized"].empty:
                    data_points.append(f"Tobacco products:\n{tobacco_products['categorized'].to_string(index=False)}")
                    fetched_data = True
                    data_available = True

            if not fetched_data:
                tobacco_effects = get_tobacco_reports_by_health_effect(None, None, sample_size)
                tobacco_products = get_tobacco_reports_by_product(None, None, sample_size)

                if isinstance(tobacco_effects, dict) and "categorized" in tobacco_effects and not tobacco_effects["categorized"].empty:
                    data_points.append(f"Tobacco health effects:\n{tobacco_effects['categorized'].head(5).to_string(index=False)}")
                    data_available = True

                if isinstance(tobacco_products, dict) and "categorized" in tobacco_products and not tobacco_products["categorized"].empty:
                    data_points.append(f"Tobacco products:\n{tobacco_products['categorized'].head(5).to_string(index=False)}")
                    data_available = True

        except Exception as e:
            data_points.append(f"Tobacco data extraction error: {str(e)}")

    # targeted prompt for trend prediction based on whether we have data
    if data_available:
        prompt = f"""
        You are a healthcare data analyst and forecaster specializing in FDA data analysis.

        Based on the following {trend_category} data that I'm providing:

        {"".join(f"{data}\n\n" for data in data_points)}

        Answer this specific question about future trends:
        "{prediction_question}"

        If you need additional data beyond what's provided, you may search for and analyze more comprehensive FDA data from the OpenFDA API endpoints.

        Provide:
        1. A substantive and detailed prediction with reasoning (2-3 paragraphs) based on the provided data and any additional data you find necessary
        2. Key factors that could influence this trend (list at least 3-5 factors)
        3. Confidence level in your prediction (high, medium, or low) with explanation
        4. Summary of the data sources and evidence that informed your prediction

        Your response MUST be substantive and data-driven. Never claim that you cannot make a prediction due to insufficient data.
        """
    else:
        # prompt instructing to search for data
        prompt = f"""
        You are a healthcare data analyst and forecaster specializing in FDA data analysis.

        I don't have specific data to provide for your question about {trend_category}:
        "{prediction_question}"

        Please actively search for and analyze comprehensive FDA data from the OpenFDA API endpoints, including but not limited to:
        - https://api.fda.gov/drug/event.json (adverse drug events)
        - https://api.fda.gov/drug/enforcement.json (drug recalls and enforcement actions)
        - https://api.fda.gov/food/enforcement.json (food recalls and safety data)
        - https://api.fda.gov/tobacco/problem.json (tobacco products reports)
        - Utilize other relevant OpenFDA endpoints as needed

        When analyzing OpenFDA data, consider:
        - Historical trends over the past 5-10 years
        - Changes in reporting frequencies
        - Geographic patterns
        - Demographic factors
        - Regulatory actions and their outcomes

        Also supplement with your knowledge of current scientific literature, research trends, and regulatory developments in the healthcare field.

        Provide:
        1. A substantive and detailed prediction with reasoning (2-3 paragraphs) based on the data you find
        2. Key factors that could influence this trend (list at least 3-5 factors)
        3. Confidence level in your prediction (high, medium, or low) with explanation
        4. Summary of the data sources and evidence that informed your prediction

        Your response MUST be substantive and data-driven. Never claim that you cannot make a prediction due to insufficient data.
        """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating trend prediction: {e}"

def load_data_concurrently(trend_category, sample_size=1000):
    data_results = {}

    if trend_category == "Drug Safety":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_drug_events = executor.submit(
                adverse_events_by_drug_within_data_range, "2020-01-01", "2023-12-31", sample_size
            )
            future_drug_recalls = executor.submit(
                most_common_recalled_drugs, sample_size
            )

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

            data_results["tobacco_effects"] = future_tobacco_effects.result()
            data_results["tobacco_products"] = future_tobacco_products.result()

    return data_results

def display_healthcare_trends():
    st.title("Healthcare Trends Analysis")

    # Ensure sample_size is initialized
    if "sample_size" not in st.session_state:
        st.session_state.sample_size = 1000

    st.write("""
    This tool uses AI to predict how healthcare trends might evolve in the future based on FDA data patterns.
    Ask a specific question about future developments in drug safety, food safety, or tobacco effects.
    """)

    # Pre-load data for all categories in the background
    if "preloaded_data" not in st.session_state:
        st.session_state.preloaded_data = {}

    # Start background loading for all categories if not already loaded
    categories = ["Drug Safety", "Food Safety", "Tobacco Effects"]
    for category in categories:
        if category not in st.session_state.preloaded_data:
            with st.spinner(f"Pre-loading {category} data..."):
                st.session_state.preloaded_data[category] = load_data_concurrently(category, st.session_state.sample_size)

    st.markdown("### Ask About Future Trends")

    example_questions = [
        "How will the landscape of cardiovascular drug safety evolve over the next decade?",
        "What will food safety regulations look like in 10 years for allergen management?",
        "How might tobacco product health effects change in the next 15 years with emerging technologies?"
    ]

    st.write("Example questions:")
    for question in example_questions:
        st.markdown(f"- *{question}*")

    prediction_question = st.text_area(
        "Enter your prediction question",
        height=100,
        key="prediction_question",
        placeholder=f"e.g., {example_questions[0]}"
    )

    if st.button("Generate Prediction", key="generate_prediction"):
        if not prediction_question:
            st.error("Please enter a prediction question.")
        else:
            # Determine which category the question is most related to
            drug_keywords = ["drug", "medication", "pharmaceutical", "prescription", "medicine", "pill", "capsule", "tablet", "adverse", "side effect"]
            food_keywords = ["food", "dietary", "nutrition", "ingredient", "allergen", "eat", "consumption", "recall", "contamination", "pathogen"]
            tobacco_keywords = ["tobacco", "smoking", "cigarette", "vape", "vaping", "e-cigarette", "nicotine", "cigar", "smokeless"]

            # Count matches in the question
            question_lower = prediction_question.lower()
            drug_count = sum(1 for keyword in drug_keywords if keyword in question_lower)
            food_count = sum(1 for keyword in food_keywords if keyword in question_lower)
            tobacco_count = sum(1 for keyword in tobacco_keywords if keyword in question_lower)

            # Determine the most relevant category
            counts = [drug_count, food_count, tobacco_count]
            category_index = counts.index(max(counts)) if max(counts) > 0 else 0  # Default to drugs if no matches
            trend_category = categories[category_index]

            with st.spinner(f"Analyzing FDA data and generating prediction..."):
                prediction_placeholder = st.empty()

                # progress bar in the placeholder
                with prediction_placeholder.container():
                    progress_bar = st.progress(0)
                    st.info("Processing your question...")

                    # Simulate progress while actually doing the work
                    for i in range(25):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                    st.info(f"Analyzing FDA data across categories...")
                    for i in range(25, 55):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                    st.info("Identifying patterns and projecting future trends...")
                    for i in range(55, 85):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                    st.info("Generating comprehensive analysis...")
                    for i in range(85, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                # Generate the prediction
                prediction = generate_trend_prediction(trend_category, prediction_question)

                # Clear placeholder
                prediction_placeholder.empty()

                # Display prediction
                st.subheader("Prediction Analysis")
                st.markdown(prediction)

                # download button for the prediction
                st.download_button(
                    "Download Prediction Report",
                    prediction,
                    "trend_prediction.txt",
                    "text/plain",
                    key="download_prediction"
                )

    st.markdown("---")
    st.caption("""
    **Disclaimer:** These predictions are generated by an AI model based on historical FDA data patterns.
    They should be considered as informed projections rather than definitive forecasts.
    Actual outcomes may differ due to regulatory changes, scientific breakthroughs, market dynamics, and other factors.
    """)
