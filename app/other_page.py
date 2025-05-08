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

from src.substance_endpoints import (
    get_substance_by_relationship_name,
    get_substance_by_moiety_name,
    get_substance_by_code_system,
    get_substance_by_structure_format,
    search_substance_by_name,
    get_nsde_by_product_type,
    get_nsde_by_marketing_category,
    get_nsde_by_route,
    search_nsde_by_ingredient
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

    # Summarize the data for context
    summary = df.head(10).to_string(index=False)

    if custom_question:
        prompt = (
            f"Given the following data about {context} in FDA substance/NSDE data:\n\n"
            f"{summary}\n\n"
            f"Answer this question in 3-5 sentences, focusing on data-driven insights:\n"
            f"{custom_question}"
        )
    else:
        prompt = (
            f"Analyze the following data about {context} in FDA substance/NSDE data:\n\n"
            f"{summary}\n\n"
            "Provide a concise summary (3-5 sentences) of key patterns and notable findings. "
            "Include any potential implications for healthcare or regulatory considerations."
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

def display_substance_relationship():
    """Display substance data by relationship name"""
    st.subheader("Substance Data by Relationship")

    # Get data with sample size limit
    df = get_substance_by_relationship_name(st.session_state.sample_size)

    if df.empty:
        st.warning("No relationship data available.")
        return

    # Sort by count and limit to top N results
    df = df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            df,
            x="Relationship",
            y="Count",
            title="Substance Relationships",
            color="Relationship",
            text="Count"
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Pie chart
        fig_pie = px.pie(
            df,
            values="Count",
            names="Relationship",
            title="Distribution of Substance Relationships",
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Show data table
    st.dataframe(df, use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(df, "substance relationships", "relationship")

def display_substance_moiety():
    """Display substance data by moiety name"""
    st.subheader("Substance Data by Moiety")

    # Get data with sample size limit
    df = get_substance_by_moiety_name(st.session_state.sample_size)

    if df.empty:
        st.warning("No moiety data available.")
        return

    # Sort by count and limit to top N results
    df = df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            df,
            x="Moiety",
            y="Count",
            title="Substance Moieties",
            color="Moiety",
            text="Count"
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Treemap
        fig_treemap = px.treemap(
            df,
            path=["Moiety"],
            values="Count",
            title="Hierarchy of Substance Moieties"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    # Moiety search
    st.subheader("Search for Moieties")
    search_term = st.text_input("Enter search term", key="moiety_search")

    if search_term:
        filtered_df = df[df["Moiety"].str.contains(search_term, case=False)]
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        # Show full data table
        st.dataframe(df, use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(df, "substance moieties", "moiety")

def display_substance_search():
    """Display substance search functionality"""
    st.subheader("Search Substances by Name")

    # Search form
    search_term = st.text_input("Enter substance name to search", key="substance_search")

    if search_term:
        with st.spinner(f"Searching for '{search_term}'..."):
            df = search_substance_by_name(search_term, st.session_state.top_n_results * 2)

        if df.empty:
            st.warning(f"No substances found matching '{search_term}'.")
        else:
            # Display results
            st.success(f"Found {len(df)} results for '{search_term}'.")

            # Create visualizations if we have enough data
            if len(df) >= 3:
                # Create a table visualization
                st.subheader("Search Results")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Only show molecular formula distribution if we have that data
                if "Molecular Formula" in df.columns and df["Molecular Formula"].notna().any():
                    # Count occurrences of each formula
                    formula_counts = df["Molecular Formula"].value_counts().reset_index()
                    formula_counts.columns = ["Molecular Formula", "Count"]

                    # Only show chart if we have at least 2 different formulas
                    if len(formula_counts) >= 2:
                        st.subheader("Molecular Formula Distribution")
                        fig = px.pie(
                            formula_counts,
                            values="Count",
                            names="Molecular Formula",
                            title="Distribution of Molecular Formulas",
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                # Just show the data
                st.dataframe(df, use_container_width=True, hide_index=True)

def display_nsde_analysis():
    """Display NSDE data analysis"""
    st.subheader("NSDE Data Analysis")

    # Create tabs for different NSDE analyses
    tab1, tab2, tab3 = st.tabs([
        "Product Type",
        "Marketing Category",
        "Route of Administration"
    ])

    # Product Type Tab
    with tab1:
        st.subheader("NSDE Data by Product Type")

        df = get_nsde_by_product_type(st.session_state.sample_size)

        if df.empty:
            st.warning("No product type data available.")
        else:
            # Sort by count and limit to top N results
            df = df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

            # Create visualization
            fig = px.bar(
                df,
                x="Product Type",
                y="Count",
                title="NSDE Data by Product Type",
                color="Product Type",
                text="Count"
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            st.dataframe(df, use_container_width=True, hide_index=True)

            # AI Insights section
            render_ai_insights_section(df, "NSDE product types", "nsde_product")

    # Marketing Category Tab
    with tab2:
        st.subheader("NSDE Data by Marketing Category")

        df = get_nsde_by_marketing_category(st.session_state.sample_size)

        if df.empty:
            st.warning("No marketing category data available.")
        else:
            # Sort by count and limit to top N results
            df = df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart
                fig_bar = px.bar(
                    df,
                    x="Marketing Category",
                    y="Count",
                    title="NSDE Data by Marketing Category",
                    color="Marketing Category",
                    text="Count"
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                fig_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # Pie chart
                fig_pie = px.pie(
                    df,
                    values="Count",
                    names="Marketing Category",
                    title="Distribution of Marketing Categories",
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            # Show data table
            st.dataframe(df, use_container_width=True, hide_index=True)

            # AI Insights section
            render_ai_insights_section(df, "NSDE marketing categories", "nsde_marketing")

    # Route of Administration Tab
    with tab3:
        st.subheader("NSDE Data by Route of Administration")

        df = get_nsde_by_route(st.session_state.sample_size)

        if df.empty:
            st.warning("No route of administration data available.")
        else:
            # Sort by count and limit to top N results
            df = df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

            # Create visualization
            fig = px.bar(
                df,
                x="Route",
                y="Count",
                title="NSDE Data by Route of Administration",
                color="Route",
                text="Count"
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            st.dataframe(df, use_container_width=True, hide_index=True)

            # AI Insights section
            render_ai_insights_section(df, "NSDE routes of administration", "nsde_route")

def display_nsde_search():
    """Display NSDE search functionality"""
    st.subheader("Search NSDE by Ingredient")

    # Search form
    search_term = st.text_input("Enter ingredient to search", key="nsde_ingredient_search")

    if search_term:
        with st.spinner(f"Searching for products with '{search_term}'..."):
            df = search_nsde_by_ingredient(search_term, st.session_state.top_n_results * 2)

        if df.empty:
            st.warning(f"No products found with ingredient '{search_term}'.")
        else:
            # Display results
            st.success(f"Found {len(df)} products containing '{search_term}'.")

            # Create visualizations if we have enough data
            if len(df) >= 3:
                # Group by dosage form and route
                if "Dosage Form" in df.columns and df["Dosage Form"].notna().any():
                    dosage_counts = df["Dosage Form"].value_counts().reset_index()
                    dosage_counts.columns = ["Dosage Form", "Count"]

                    if len(dosage_counts) >= 2:
                        st.subheader("Distribution by Dosage Form")
                        fig = px.pie(
                            dosage_counts,
                            values="Count",
                            names="Dosage Form",
                            title=f"Dosage Forms for Products with '{search_term}'",
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if "Route" in df.columns and df["Route"].notna().any():
                    route_counts = df["Route"].value_counts().reset_index()
                    route_counts.columns = ["Route", "Count"]

                    if len(route_counts) >= 2:
                        st.subheader("Distribution by Route")
                        fig = px.bar(
                            route_counts,
                            x="Route",
                            y="Count",
                            title=f"Routes of Administration for Products with '{search_term}'",
                            color="Route"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

            # Show full data table
            st.subheader("Product Details")
            st.dataframe(df, use_container_width=True, hide_index=True)

def display_other_data():
    """Main function to display the Other Data page"""
    st.title("Other FDA Data Analysis")

    # Create tabs for different data categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Substance Analysis",
        "Substance Search",
        "NSDE Analysis",
        "NSDE Search"
    ])

    with tab1:
        # Create subtabs for substance analysis
        subtab1, subtab2 = st.tabs(["Relationships", "Moieties"])

        with subtab1:
            display_substance_relationship()

        with subtab2:
            display_substance_moiety()

    with tab2:
        display_substance_search()

    with tab3:
        display_nsde_analysis()

    with tab4:
        display_nsde_search()
