import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
from src.device_events import (
    device_class_distribution,
    device_problems_by_year,
    device_manufacturer_analysis,
    device_event_type_distribution,
    device_geographic_distribution,
    get_top_device_classes,
    get_device_problems_trend,
    get_manufacturer_market_share,
    get_event_type_categories,
    device_510k_clearance_types,
    device_510k_advisory_committees,
    device_510k_geographic_distribution,
    device_510k_decision_codes,
    _fetch_all_510k_data
)
from src.utils import get_state_abbreviations
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

def display_device_class_distribution():
    """Display device class distribution with visualizations."""
    metric_title = "Device Class Distribution"
    metric_description = (
        "This section shows the distribution of medical device classes, "
        "helping identify which types of devices are most commonly involved in events."
    )

    render_metric_header(metric_title, metric_description)

    try:
        df = device_class_distribution()
        if df.empty:
            st.warning("No data available for device class distribution.")
            return

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for top 10 device classes
            top_10 = get_top_device_classes(df)
            fig_bar = px.bar(
                top_10,
                x="Device Class",
                y="Count",
                text="Percentage",
                title="Top 10 Device Classes",
                labels={
                    "Device Class": "Device Class",
                    "Count": "Number of Events",
                    "Percentage": "Percentage"
                }
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create pie chart for device class distribution
            fig_pie = px.pie(
                df,
                values="Count",
                names="Device Class",
                title="Device Class Distribution",
                hover_data=["Percentage"]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Display detailed data
        st.markdown("### Detailed Statistics")
        render_data_table(df)

        # AI Insights
        st.markdown("### Insights")
        if st.button("Get General Insights for Device Classes"):
            insights = get_insights_from_data(df, "device class distribution")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about device classes", key="device_class_question")
        if custom_question:
            insights = get_insights_from_data(df, "device class distribution", custom_question)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load device class data: {e}")

def display_device_problems():
    """Display device problems analysis with visualizations."""
    metric_title = "Device Problems Analysis"
    metric_description = (
        "This section analyzes the types of problems reported with medical devices, "
        "helping identify common issues and trends."
    )

    render_metric_header(metric_title, metric_description)

    try:
        # Add year range selector
        current_year = datetime.now().year
        start_year = st.slider("Start Year", 2010, current_year, 2010)
        end_year = st.slider("End Year", start_year, current_year, current_year)

        df = device_problems_by_year(start_year, end_year)
        if df.empty:
            st.warning("No data available for device problems.")
            return

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            # Create horizontal bar chart for top problems
            top_problems = get_device_problems_trend(df).head(10)
            fig_bar = px.bar(
                top_problems,
                y="Problem",
                x="Count",
                text="Percentage",
                title="Top 10 Device Problems",
                orientation='h'
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create treemap for problem distribution
            fig_treemap = px.treemap(
                df,
                path=["Problem"],
                values="Count",
                title="Device Problems Distribution"
            )
            st.plotly_chart(fig_treemap, use_container_width=True)

        # Display detailed data
        st.markdown("### Detailed Statistics")
        render_data_table(df)

        # AI Insights
        st.markdown("### Insights")
        if st.button("Get General Insights for Device Problems"):
            insights = get_insights_from_data(df, "device problems")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about device problems", key="device_problems_question")
        if custom_question:
            insights = get_insights_from_data(df, "device problems", custom_question)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load device problems data: {e}")

def display_manufacturer_analysis():
    """Display manufacturer analysis with visualizations."""
    metric_title = "Manufacturer Analysis"
    metric_description = (
        "This section analyzes medical device manufacturers, "
        "showing market share and event distribution across different companies."
    )

    render_metric_header(metric_title, metric_description)

    try:
        df = device_manufacturer_analysis()
        if df.empty:
            st.warning("No data available for manufacturer analysis.")
            return

        # Calculate market share
        df = get_manufacturer_market_share(df)

        # Create two columns for the layout
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for top manufacturers
            fig_bar = px.bar(
                df,
                x="Manufacturer",
                y="Market Share",
                text="Count",
                title="Top 20 Manufacturers by Market Share",
                labels={
                    "Manufacturer": "Manufacturer",
                    "Market Share": "Market Share (%)",
                    "Count": "Number of Events"
                }
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create pie chart for manufacturer distribution
            fig_pie = px.pie(
                df,
                values="Count",
                names="Manufacturer",
                title="Manufacturer Distribution",
                hover_data=["Market Share"]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Display detailed data
        st.markdown("### Detailed Statistics")
        render_data_table(df)

        # AI Insights
        st.markdown("### Insights")
        if st.button("Get General Insights for Manufacturers"):
            insights = get_insights_from_data(df, "device manufacturers")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about manufacturers", key="manufacturer_question")
        if custom_question:
            insights = get_insights_from_data(df, "device manufacturers", custom_question)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load manufacturer data: {e}")

def display_geographic_distribution():
    """Display geographic distribution of device events."""
    metric_title = "Geographic Distribution"
    metric_description = (
        "This section shows the geographic distribution of device events across the United States, "
        "helping identify regional patterns and concentrations."
    )

    render_metric_header(metric_title, metric_description)

    try:
        df = device_geographic_distribution()
        if df.empty:
            st.warning("No data available for geographic distribution.")
            return

        # Create choropleth map
        fig_map = px.choropleth(
            df,
            locations="State",
            locationmode="USA-states",
            color="Count",
            hover_name="State",
            hover_data=["Count", "Percentage"],
            color_continuous_scale="Viridis",
            title="Device Events by State"
        )
        fig_map.update_layout(
            geo=dict(
                scope="usa",
                showlakes=True,
                lakecolor="rgb(255, 255, 255)"
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Create two columns for additional visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Create bar chart for top states
            top_states = df.nlargest(10, "Count")
            fig_bar = px.bar(
                top_states,
                x="State",
                y="Count",
                text="Percentage",
                title="Top 10 States by Device Events"
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            # Create heatmap for state distribution
            fig_heatmap = px.density_heatmap(
                df,
                x="State",
                y="Count",
                title="State Distribution Heatmap"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Display detailed data
        st.markdown("### Detailed Statistics")
        render_data_table(df)

        # AI Insights
        st.markdown("### Insights")
        if st.button("Get General Insights for Geographic Distribution"):
            insights = get_insights_from_data(df, "geographic distribution of device events")
            st.write(insights)

        custom_question = st.text_input("Ask a specific question about geographic distribution", key="geographic_question")
        if custom_question:
            insights = get_insights_from_data(df, "geographic distribution of device events", custom_question)
            st.write(insights)

    except Exception as e:
        st.error(f"Failed to load geographic data: {e}")

def display_device_reports():
    st.title("Device Reports")

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "510(k) Clearance Types",
        "Advisory Committees",
        "Decision Codes"
    ])

    # Load all data in parallel
    with st.spinner("Loading device data..."):
        data = _fetch_all_510k_data()

    with tab1:
        st.header("510(k) Clearance Types")
        if "clearance_types" in data and not data["clearance_types"].empty:
            df = data["clearance_types"]
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x="Clearance Type",
                    y="Count",
                    title="Distribution of 510(k) Clearance Types",
                    color="Count",
                    color_continuous_scale="Viridis",
                    text="Percentage"
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.pie(
                    df,
                    values="Count",
                    names="Clearance Type",
                    title="Percentage Distribution of Clearance Types",
                    hover_data=["Percentage"]
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Statistics"):
                st.dataframe(df)

            # AI Insights for Clearance Types
            st.markdown("### AI Insights")
            if st.button("Get Insights for Clearance Types", key="clearance_insights"):
                insights = get_insights_from_data(df, "510(k) clearance types")
                st.write(insights)

            custom_question = st.text_input("Ask a specific question about clearance types", key="clearance_question")
            if custom_question:
                insights = get_insights_from_data(df, "510(k) clearance types", custom_question)
                st.write(insights)
        else:
            st.warning("No clearance type data available")

    with tab2:
        st.header("Advisory Committees")
        if "advisory_committees" in data and not data["advisory_committees"].empty:
            df = data["advisory_committees"]
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x="Committee Name",
                    y="Count",
                    title="Distribution by Advisory Committee",
                    color="Count",
                    color_continuous_scale="Viridis",
                    text="Percentage"
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.pie(
                    df,
                    values="Count",
                    names="Committee Name",
                    title="Percentage Distribution by Committee",
                    hover_data=["Percentage"]
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Statistics"):
                st.dataframe(df)

            # AI Insights for Advisory Committees
            st.markdown("### AI Insights")
            if st.button("Get Insights for Advisory Committees", key="committee_insights"):
                insights = get_insights_from_data(df, "advisory committees")
                st.write(insights)

            custom_question = st.text_input("Ask a specific question about advisory committees", key="committee_question")
            if custom_question:
                insights = get_insights_from_data(df, "advisory committees", custom_question)
                st.write(insights)
        else:
            st.warning("No advisory committee data available")

    with tab3:
        st.header("Decision Codes")
        if "decision_codes" in data and not data["decision_codes"].empty:
            df = data["decision_codes"]
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x="Decision Description",
                    y="Count",
                    title="Distribution of Decision Codes",
                    color="Count",
                    color_continuous_scale="Viridis",
                    text="Percentage"
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.pie(
                    df,
                    values="Count",
                    names="Decision Description",
                    title="Percentage Distribution of Decisions",
                    hover_data=["Percentage"]
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed Statistics"):
                st.dataframe(df)

            # AI Insights for Decision Codes
            st.markdown("### AI Insights")
            if st.button("Get Insights for Decision Codes", key="decision_insights"):
                insights = get_insights_from_data(df, "510(k) decision codes")
                st.write(insights)

            custom_question = st.text_input("Ask a specific question about decision codes", key="decision_question")
            if custom_question:
                insights = get_insights_from_data(df, "510(k) decision codes", custom_question)
                st.write(insights)
        else:
            st.warning("No decision code data available")
