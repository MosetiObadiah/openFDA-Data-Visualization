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
    get_food_events_over_time,
    get_food_events_by_industry,
    get_food_events_by_outcome,
    get_food_recall_trends
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
    """Render AI insights section with option for custom questions using unique keys."""
    st.subheader("AI Insights")
    # Ensure key is unique across the entire app with food_ prefix
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
                y="Reason",  # Change to y-axis for horizontal bars
                x="Count",   # Change to x-axis for horizontal bars
                color="Category",
                facet_col="Category",
                facet_col_wrap=1,  # One category per row for better readability
                title=f"Top {top_n} Reasons by Category",
                orientation='h',   # Horizontal orientation
                height=200 * len(selected_categories),  # Adjust height based on number of categories
                labels={"Reason": "", "Count": "Number of Recalls"}  # Better labels
            )

            # Customize the layout for better text display
            fig_reasons.update_layout(
                margin=dict(l=20, r=20, t=80, b=20),
                yaxis={'categoryorder':'total ascending'},
                yaxis_title="",
                xaxis_title="Number of Recalls"
            )

            # Adjust facet spacing and titles
            fig_reasons.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

            # Customize subplot settings for better appearance
            fig_reasons.update_yaxes(automargin=True, title_text="")

            # Truncate long reason texts if needed
            for i, _ in enumerate(fig_reasons.data):
                if hasattr(fig_reasons.data[i], 'y'):
                    # Add hovertext showing full reason text
                    fig_reasons.data[i].hovertemplate = '%{y}<br>Count: %{x}<extra></extra>'

            st.plotly_chart(fig_reasons, use_container_width=True)

            # Also offer a clean table view for reference
            with st.expander("View detailed reason data", expanded=False):
                st.dataframe(
                    top_reasons.groupby("Category").head(top_n)[["Category", "Reason", "Count"]]
                    .sort_values(["Category", "Count"], ascending=[True, False]),
                    use_container_width=True
                )

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

    # Add full state names for better visualization
    state_names = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia', 'PR': 'Puerto Rico'
    }

    # Create a copy with state names for display
    display_df = df.copy()
    display_df['State Name'] = display_df['State'].map(state_names)

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
    top_states_df = display_df.sort_values("Count", ascending=False).head(top_n)

    fig_bar = px.bar(
        top_states_df,
        x="State Name",  # Use full state names
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
        st.dataframe(display_df[["State Name", "State", "Count"]].sort_values("Count", ascending=False),
                    use_container_width=True, hide_index=True)

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

    # Show data table in collapsed expander
    with st.expander("View Product Category Data", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(df, "food recall product categories", "recall_product")

def display_food_adverse_events():
    """Display food adverse events analysis with enhanced visualizations."""
    st.subheader("Food Adverse Events Analysis")

    # Create tabs for different analyses with unique keys
    event_tabs = st.tabs([
        "By Industry",
        "By Product",
        "By Symptom",
        "By Age",
        "By Outcome"
    ])

    # By Industry Tab
    with event_tabs[0]:
        st.subheader("Adverse Events by Food Industry")

        industry_df = get_food_events_by_industry(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if industry_df.empty:
            st.warning("No industry data available.")
        else:
            # Top industries slider
            top_n = st.slider("Number of top industries to show", 5, 20, 10, key="food_industry_slider")
            top_industry_df = industry_df.head(top_n)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart
                fig_industry = px.bar(
                    top_industry_df,
                    y="Industry",
                    x="Count",
                    title=f"Top {top_n} Food Industries with Adverse Events",
                    orientation='h',
                    color="Industry"
                )
                fig_industry.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_industry, use_container_width=True)

            with col2:
                # Pie chart
                fig_pie = px.pie(
                    top_industry_df,
                    values="Count",
                    names="Industry",
                    title="Distribution by Industry"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Show data table
            with st.expander("View Full Industry Data", expanded=False):
                st.dataframe(industry_df, use_container_width=True)

            # AI Insights
            render_ai_insights_section(industry_df, "food industries involved in adverse events", "food_industry")

    # By Product Tab
    with event_tabs[1]:
        st.subheader("Adverse Events by Product")

        product_df = get_food_events_by_product(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if product_df.empty:
            st.warning("No product data available.")
        else:
            # Top products slider
            top_n = st.slider("Number of top products to show", 5, 20, 10, key="food_product_slider")
            top_product_df = product_df.head(top_n)

            # Create visualizations
            fig_product = px.bar(
                top_product_df,
                y="Product",
                x="Count",
                title=f"Top {top_n} Products with Adverse Events",
                orientation='h',
                color="Product"
            )
            fig_product.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_product, use_container_width=True)

            # Word cloud option if available
            st.subheader("Search Products")
            search_term = st.text_input("Filter by product name", key="food_product_search")

            if search_term:
                filtered_products = product_df[product_df["Product"].str.contains(search_term, case=False)]
                with st.expander("View Filtered Products", expanded=False):
                    st.dataframe(filtered_products, use_container_width=True)
            else:
                with st.expander("View Top Products Data", expanded=False):
                    st.dataframe(top_product_df, use_container_width=True)

            # AI Insights
            render_ai_insights_section(product_df, "products involved in food adverse events", "food_product_insights")

    # By Symptom Tab
    with event_tabs[2]:
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
                    y="Category",
                    x="Count",
                    title="Symptoms by Category",
                    orientation='h',
                    color="Category"
                )
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # Pie chart for categories
                fig_pie = px.pie(
                    category_df,
                    values="Count",
                    names="Category",
                    title="Distribution of Symptoms by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Show top specific symptoms within selected category
            st.subheader("Explore Symptoms by Category")
            selected_category = st.selectbox(
                "Select Category to See Detailed Symptoms",
                options=category_df["Category"].unique(),
                key="food_symptom_category"
            )

            if selected_category:
                filtered_symptoms = detailed_df[detailed_df["Category"] == selected_category]
                top_symptoms = filtered_symptoms.sort_values("Count", ascending=False).head(10)

                fig_top_symptoms = px.bar(
                    top_symptoms,
                    y="Symptom",
                    x="Count",
                    title=f"Top Symptoms in {selected_category} Category",
                    orientation='h',
                    color="Symptom"
                )
                fig_top_symptoms.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top_symptoms, use_container_width=True)

            # AI Insights
            render_ai_insights_section(symptom_result, "symptoms in food adverse events", "food_symptom_insights")
        else:
            st.warning("No symptom data available.")

    # By Age Tab
    with event_tabs[3]:
        st.subheader("Adverse Events by Consumer Age")

        age_df = get_food_events_by_age(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if age_df.empty:
            st.warning("No age data available.")
        else:
            # Create visualizations
            fig_age = px.bar(
                age_df,
                x="Age Group",
                y="Count",
                title="Food Adverse Events by Consumer Age Group",
                color="Age Group"
            )
            st.plotly_chart(fig_age, use_container_width=True)

            # Pie chart
            fig_pie = px.pie(
                age_df,
                values="Count",
                names="Age Group",
                title="Distribution by Age Group"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Show data table
            st.dataframe(age_df, use_container_width=True)

            # AI Insights
            render_ai_insights_section(age_df, "age distribution in food adverse events", "food_age_insights")

    # By Outcome Tab
    with event_tabs[4]:
        st.subheader("Adverse Events by Outcome")

        outcome_df = get_food_events_by_outcome(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if outcome_df.empty:
            st.warning("No outcome data available.")
        else:
            # Top outcomes to show
            top_n = st.slider("Number of top outcomes to show", 5, 15, 10, key="food_outcome_slider")
            top_outcome_df = outcome_df.head(top_n)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart
                fig_bar = px.bar(
                    top_outcome_df,
                    y="Outcome",
                    x="Count",
                    title=f"Top {top_n} Adverse Event Outcomes",
                    orientation='h',
                    color="Outcome"
                )
                fig_bar.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # Pie chart
                fig_pie = px.pie(
                    top_outcome_df,
                    values="Count",
                    names="Outcome",
                    title="Distribution of Adverse Event Outcomes"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Show data table
            with st.expander("View Full Outcome Data", expanded=False):
                st.dataframe(outcome_df, use_container_width=True)

            # AI Insights
            render_ai_insights_section(outcome_df, "outcomes of food adverse events", "food_outcome_insights")

def display_food_reports():
    """Display food reports dashboard with multiple analysis tabs."""
    st.title("Food Reports")

    # Create tabs for different analyses
    tabs = st.tabs([
        "Recall Classification",
        "Recall Reasons",
        "Geographic Distribution",
        "Product Categories",
        "Adverse Events",
        "Trends Over Time"
    ])

    # 1. Food Recalls by Classification
    with tabs[0]:
        display_food_recall_classification()

    # 2. Food Recalls by Reason
    with tabs[1]:
        display_food_recall_reason()

    # 3. Food Recalls by Geography
    with tabs[2]:
        display_food_recall_geography()

    # 4. Food Recalls by Product Type
    with tabs[3]:
        display_food_recall_product()

    # 5. Food Adverse Events Analysis
    with tabs[4]:
        display_food_adverse_events()

    # 6. Trends Over Time
    with tabs[5]:
        display_food_trends()

def display_food_trends():
    """Display food recall and event trends over time."""
    st.subheader("Food Safety Trends Over Time")

    # Create subtabs
    trend_tabs = st.tabs(["Recalls by Year", "Events Timeline"])

    # Recalls by Year
    with trend_tabs[0]:
        st.subheader("Food Recall Trends by Year and Classification")

        # Set year range
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", min_value=2004, max_value=2023, value=2010, key="food_start_year")
        with col2:
            end_year = st.number_input("End Year", min_value=start_year, max_value=2023, value=2023, key="food_end_year")

        # Get data
        trend_df = get_food_recall_trends(start_year, end_year)

        if trend_df.empty:
            st.warning("No trend data available for the selected years.")
        else:
            # Line chart for trends
            fig_trend = px.line(
                trend_df,
                x="Year",
                y=["Class I", "Class II", "Class III", "Total"],
                title="Food Recalls by Classification and Year",
                markers=True,
                labels={"value": "Number of Recalls", "variable": "Classification"}
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Area chart for classification percentage
            class_cols = [col for col in trend_df.columns if col not in ["Year", "Total"]]
            percentage_df = trend_df.copy()
            for col in class_cols:
                percentage_df[col] = percentage_df[col] / percentage_df["Total"] * 100

            fig_area = px.area(
                percentage_df,
                x="Year",
                y=class_cols,
                title="Classification Percentage by Year",
                labels={"value": "Percentage", "variable": "Classification"}
            )
            st.plotly_chart(fig_area, use_container_width=True)

            # Show data table
            with st.expander("View Data Table", expanded=False):
                st.dataframe(trend_df, use_container_width=True)

            # AI Insights
            render_ai_insights_section(trend_df, "food recall trends over time", "food_trends")

    # Events Timeline
    with trend_tabs[1]:
        st.subheader("Food Adverse Events Timeline")

        # Time interval selector with unique key
        interval = st.radio(
            "Select Time Interval",
            options=["year", "quarter", "month"],
            format_func=lambda x: x.capitalize(),
            horizontal=True,
            key="food_time_interval"
        )

        # Get data
        time_df = get_food_events_over_time(
            interval,
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if time_df.empty:
            st.warning("No time series data available for the selected date range.")
        else:
            # Create visualizations
            fig_line = px.line(
                time_df,
            x="Date",
            y="Count",
                title=f"Food Adverse Events Over Time (by {interval.capitalize()})",
            markers=True
        )
            st.plotly_chart(fig_line, use_container_width=True)

            # Bar chart
            fig_bar = px.bar(
                time_df,
                x="Date",
                y="Count",
                title=f"Food Adverse Events by {interval.capitalize()}",
                color="Count",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Show data table
            with st.expander("View Data Table", expanded=False):
                st.dataframe(time_df, use_container_width=True)

        # AI Insights
            render_ai_insights_section(time_df, "food adverse events timeline", "food_events_time")
