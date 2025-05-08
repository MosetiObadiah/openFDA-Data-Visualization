import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.food_endpoints import (
    get_food_recalls_by_classification,
    get_food_recalls_by_reason,
    get_food_recalls_by_state,
    get_food_recalls_by_product_type,
    get_food_events_by_product,
    get_food_events_by_symptom,
    get_food_events_by_age,
    get_food_events_over_time
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
            f"Given the following data about {context} in FDA food reports:\n\n"
            f"{summary}\n\n"
            f"Answer this question in 3-5 sentences, focusing on data-driven insights:\n"
            f"{custom_question}"
        )
    else:
        prompt = (
            f"Analyze the following data about {context} in FDA food reports:\n\n"
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

def display_food_recall_classification():
    """Display food recalls by classification"""
    st.subheader("Food Recalls by Classification")

    # Use global date range from session state
    df = get_food_recalls_by_classification(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart with hover text showing descriptions
        fig_bar = px.bar(
            df,
            x="Classification",
            y="Count",
            title="Food Recalls by Classification",
            color="Classification",
            text="Count",
            hover_data=["Description"]
        )
        fig_bar.update_layout(xaxis_tickangle=0)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Pie chart
        fig_pie = px.pie(
            df,
            values="Count",
            names="Classification",
            title="Distribution of Food Recalls by Classification",
            hole=0.4,
            hover_data=["Description"]
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Description table
    st.subheader("Classification Descriptions")
    st.dataframe(df[["Classification", "Description", "Count"]], use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(df, "food recall classifications", "recall_class")

def display_food_recall_reason():
    """Display food recalls by reason"""
    st.subheader("Food Recalls by Reason")

    # Use global date range from session state
    result_df = get_food_recalls_by_reason(
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
                x="Category",
                y="Count",
                title="Food Recalls by Reason Category",
                color="Category",
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
                names="Category",
                title="Distribution of Food Recalls by Reason Category",
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Show top specific reasons within each category
        st.subheader("Top Specific Reasons by Category")

        # Get selected categories
        selected_categories = st.multiselect(
            "Select Categories to Explore",
            options=category_df["Category"].unique(),
            default=list(category_df["Category"].unique())[:2]  # Default to first two categories
        )

        if selected_categories:
            filtered_df = detailed_df[detailed_df["Category"].isin(selected_categories)]

            # Group, sort, and limit to top reasons per category
            top_reasons = (filtered_df.groupby(["Category", "Reason"])
                          .sum()
                          .reset_index()
                          .sort_values(["Category", "Count"], ascending=[True, False]))

            # Get top N reasons per category
            top_n = st.slider("Number of top reasons per category", 3, 10, 5)

            # Create a figure for top reasons by category
            fig_reasons = px.bar(
                top_reasons.groupby("Category").head(top_n),
                x="Reason",
                y="Count",
                color="Category",
                facet_col="Category",
                facet_col_wrap=2,  # Two categories per row
                title=f"Top {top_n} Reasons by Category",
                height=300 * (len(selected_categories) + 1) // 2  # Adjust height based on number of categories
            )
            fig_reasons.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_reasons, use_container_width=True)

        # AI Insights section
        render_ai_insights_section(result_df, "food recall reasons", "recall_reason")
    else:
        st.warning("No data available for the selected date range.")

def display_food_recall_geography():
    """Display food recalls by geography"""
    st.subheader("Food Recalls by Geography")

    # Use global date range from session state
    df = get_food_recalls_by_state(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create a choropleth map
    fig_map = px.choropleth(
        df,
        locations="State",
        locationmode="USA-states",
        color="Count",
        scope="usa",
        hover_name="State",
        hover_data=["Count"],
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Food Recalls by State"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Bar chart for top states
    top_n = min(st.session_state.top_n_results, len(df))
    top_states_df = df.sort_values("Count", ascending=False).head(top_n)

    fig_bar = px.bar(
        top_states_df,
        x="State",
        y="Count",
        title=f"Top {top_n} States by Food Recalls",
        color="Count",
        text="Count",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Show full data table
    with st.expander("View Full Data Table"):
        st.dataframe(df.sort_values("Count", ascending=False), use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(df, "food recall geographical distribution", "recall_geo")

def display_food_recall_product():
    """Display food recalls by product type"""
    st.subheader("Food Recalls by Product Type")

    # Use global date range from session state
    df = get_food_recalls_by_product_type(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            df,
            x="Product Category",
            y="Count",
            title="Food Recalls by Product Category",
            color="Product Category",
            text="Count"
        )
        fig_bar.update_layout(xaxis_tickangle=0)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Treemap visualization
        fig_treemap = px.treemap(
            df,
            path=["Product Category"],
            values="Count",
            title="Hierarchy of Product Categories"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    # Show data table
    st.dataframe(df, use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(df, "food recall product categories", "recall_product")

def display_food_adverse_events():
    """Display food adverse events analysis"""
    st.subheader("Food Adverse Events")

    # Create tabs for different analyses
    tabs = st.tabs(["By Product", "By Symptom", "By Consumer Age", "Over Time"])

    # By Product Tab
    with tabs[0]:
        st.subheader("Adverse Events by Product")

        product_df = get_food_events_by_product(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if product_df.empty:
            st.warning("No product data available for the selected date range.")
        else:
            # Sort by count and limit to top N results
            product_df = product_df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

            # Create visualizations
            fig_product = px.bar(
                product_df,
                y="Product",
                x="Count",
                title="Top Products in Food Adverse Events",
                color="Product",
                text="Count",
                orientation='h'  # Horizontal bar chart
            )
            fig_product.update_layout(yaxis=dict(autorange="reversed"))  # Reverse y-axis to show highest at top
            fig_product.update_traces(textposition='outside')
            st.plotly_chart(fig_product, use_container_width=True)

            # Product name search filter
            search_term = st.text_input("Search for Product", key="product_search")

            # Apply filter if search term provided
            if search_term:
                filtered_product_df = product_df[product_df["Product"].str.contains(search_term, case=False)]
            else:
                filtered_product_df = product_df

            # Show the filtered data
            st.dataframe(filtered_product_df, use_container_width=True, hide_index=True)

            # AI Insights for product
            render_ai_insights_section(product_df, "products involved in food adverse events", "food_product")

    # By Symptom Tab
    with tabs[1]:
        st.subheader("Adverse Events by Symptom")

        symptom_result = get_food_events_by_symptom(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if isinstance(symptom_result, dict) and "categorized" in symptom_result and not symptom_result["categorized"].empty:
            category_df = symptom_result["categorized"]
            detailed_df = symptom_result["detailed"]

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for categories
                fig_bar = px.bar(
                    category_df,
                    x="Category",
                    y="Count",
                    title="Symptoms by Category",
                    color="Category",
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
                    names="Category",
                    title="Distribution of Symptoms by Category",
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            # Show top specific symptoms within selected category
            selected_category = st.selectbox(
                "Select Category to See Top Symptoms",
                options=category_df["Category"].unique()
            )

            if selected_category:
                filtered_symptoms = detailed_df[detailed_df["Category"] == selected_category]
                top_symptoms = filtered_symptoms.sort_values("Count", ascending=False).head(10)

                fig_top_symptoms = px.bar(
                    top_symptoms,
                    x="Symptom",
                    y="Count",
                    title=f"Top Symptoms in {selected_category} Category",
                    color="Symptom",
                    text="Count"
                )
                fig_top_symptoms.update_layout(xaxis_tickangle=-45)
                fig_top_symptoms.update_traces(textposition='outside')
                st.plotly_chart(fig_top_symptoms, use_container_width=True)

            # AI Insights for symptoms
            render_ai_insights_section(symptom_result, "symptoms in food adverse events", "food_symptom")
        else:
            st.warning("No symptom data available for the selected date range.")

    # By Consumer Age Tab
    with tabs[2]:
        st.subheader("Adverse Events by Consumer Age")

        age_df = get_food_events_by_age(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if age_df.empty:
            st.warning("No age data available for the selected date range.")
        else:
            # Create visualizations
            fig_age = px.bar(
                age_df,
                x="Age Group",
                y="Count",
                title="Food Adverse Events by Consumer Age Group",
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

            # Show the filtered data
            st.dataframe(filtered_age_df, use_container_width=True, hide_index=True)

            # AI Insights for age
            render_ai_insights_section(age_df, "consumer age distribution in food adverse events", "food_age")

    # Over Time Tab
    with tabs[3]:
        st.subheader("Adverse Events Over Time")

        # Time interval selector
        interval = st.radio(
            "Select Time Interval",
            options=["year", "quarter", "month"],
            format_func=lambda x: x.capitalize(),
            horizontal=True,
            key="food_time_interval"
        )

        time_df = get_food_events_over_time(
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
                title=f"Food Adverse Events Over Time (by {interval.capitalize()})",
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

            # Show the data
            st.dataframe(time_df, use_container_width=True, hide_index=True)

            # AI Insights for time trends
            render_ai_insights_section(time_df, f"food adverse events over time (by {interval})", "food_time")

def display_food_reports():
    """Main function to display the Food page"""
    st.title("Food Data Analysis")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Recall Classification",
        "Recall Reasons",
        "Recall Geography",
        "Adverse Events"
    ])

    with tab1:
        display_food_recall_classification()

    with tab2:
        display_food_recall_reason()

    with tab3:
        display_food_recall_geography()

    with tab4:
        display_food_adverse_events()
