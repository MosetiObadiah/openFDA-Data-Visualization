import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import google.generativeai as genai
import os
from dotenv import load_dotenv

from src.animal_endpoints import (
    get_animal_events_by_species,
    get_animal_events_by_breed,
    get_animal_events_by_age,
    get_animal_events_by_weight,
    get_animal_events_by_drug,
    get_animal_events_by_reaction,
    get_animal_events_by_outcome,
    get_animal_events_by_duration
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

    # Summarize the top rows for context
    summary = df.head(10).to_string(index=False)
    if custom_question:
        prompt = (
            f"Given the following data about {context} in animal and veterinary FDA reports:\n\n"
            f"{summary}\n\n"
            f"Answer this question in 3-5 sentences, focusing on data-driven insights:\n"
            f"{custom_question}"
        )
    else:
        prompt = (
            f"Analyze the following data about {context} in animal and veterinary FDA reports:\n\n"
            f"{summary}\n\n"
            "Provide a concise summary (3-5 sentences) of key trends, patterns, and notable findings. "
            "Include any potential public health implications or recommendations for veterinarians based on this data."
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

def display_species_analysis():
    """Display analysis by animal species"""
    st.subheader("Analysis by Animal Species")

    # Use global date range from session state
    df = get_animal_events_by_species(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    if df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Sort by count and limit to top N results
    df = df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            df,
            x="Species",
            y="Count",
            title="Adverse Events by Animal Species",
            color="Species",
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
            names="Species",
            title="Distribution of Adverse Events by Species",
            hole=0.4,
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Local filtering
    st.subheader("Explore the Data")

    # Species selection for filtering
    if not df.empty:
        selected_species = st.multiselect(
            "Filter by Species",
            options=df["Species"].unique(),
            default=[]
        )

        # Apply filter if selected
        if selected_species:
            filtered_df = df[df["Species"].isin(selected_species)]
        else:
            filtered_df = df

        # Show the filtered data
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(df, "animal species in adverse events", "species")

def display_breed_analysis():
    """Display analysis by animal breed"""
    st.subheader("Analysis by Animal Breed")

    # Get all species for filtering
    species_df = get_animal_events_by_species(
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.sample_size
    )

    # Species filter
    species_options = species_df["Species"].unique() if not species_df.empty else []
    selected_species = st.selectbox(
        "Select Species",
        options=["All Species"] + list(species_options),
        index=0
    )

    # Get breed data based on species filter
    if selected_species == "All Species":
        df = get_animal_events_by_breed(None, st.session_state.sample_size)
    else:
        df = get_animal_events_by_breed(selected_species, st.session_state.sample_size)

    if df.empty:
        st.warning("No breed data available for the selected filters.")
        return

    # Sort by count and limit to top N results
    df = df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            df,
            x="Breed",
            y="Count",
            title=f"Adverse Events by Breed{' for ' + selected_species if selected_species != 'All Species' else ''}",
            color="Breed",
            text="Count"
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Treemap visualization
        fig_treemap = px.treemap(
            df,
            path=["Breed"],
            values="Count",
            title=f"Hierarchy of Breeds{' for ' + selected_species if selected_species != 'All Species' else ''}"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    # Show the data
    st.subheader("Breed Data")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # AI Insights section
    render_ai_insights_section(
        df,
        f"animal breeds{' in ' + selected_species if selected_species != 'All Species' else ''} adverse events",
        "breed"
    )

def display_age_weight_analysis():
    """Display analysis by animal age and weight"""
    st.subheader("Analysis by Animal Age and Weight")

    # Create tabs for age and weight
    age_tab, weight_tab = st.tabs(["Age Analysis", "Weight Analysis"])

    # Age Analysis Tab
    with age_tab:
        st.subheader("Age Distribution")

        age_df = get_animal_events_by_age(
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
                title="Adverse Events by Animal Age Group",
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
            render_ai_insights_section(age_df, "animal age distribution in adverse events", "age")

    # Weight Analysis Tab
    with weight_tab:
        st.subheader("Weight Distribution")

        weight_df = get_animal_events_by_weight(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if weight_df.empty:
            st.warning("No weight data available for the selected date range.")
        else:
            # Create visualizations
            fig_weight = px.bar(
                weight_df,
                x="Weight Group",
                y="Count",
                title="Adverse Events by Animal Weight Group",
                color="Weight Group",
                text="Count"
            )
            fig_weight.update_layout(xaxis_tickangle=0)
            fig_weight.update_traces(textposition='outside')
            st.plotly_chart(fig_weight, use_container_width=True)

            # Weight group selection for filtering
            selected_weight_groups = st.multiselect(
                "Filter by Weight Group",
                options=weight_df["Weight Group"].unique(),
                default=[]
            )

            # Apply filter if selected
            if selected_weight_groups:
                filtered_weight_df = weight_df[weight_df["Weight Group"].isin(selected_weight_groups)]
            else:
                filtered_weight_df = weight_df

            # Show the filtered data
            st.dataframe(filtered_weight_df, use_container_width=True, hide_index=True)

            # AI Insights for weight
            render_ai_insights_section(weight_df, "animal weight distribution in adverse events", "weight")

def display_drug_reaction_analysis():
    """Display analysis by drug and reaction"""
    st.subheader("Analysis by Drug and Reaction")

    # Create tabs for drug and reaction
    drug_tab, reaction_tab = st.tabs(["Drug Analysis", "Reaction Analysis"])

    # Drug Analysis Tab
    with drug_tab:
        st.subheader("Drug Distribution")

        drug_df = get_animal_events_by_drug(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if drug_df.empty:
            st.warning("No drug data available for the selected date range.")
        else:
            # Sort by count and limit to top N results
            drug_df = drug_df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

            # Create visualizations
            fig_drug = px.bar(
                drug_df,
                x="Drug Name",
                y="Count",
                title="Top Drugs in Animal Adverse Events",
                color="Drug Name",
                text="Count"
            )
            fig_drug.update_layout(xaxis_tickangle=-45)
            fig_drug.update_traces(textposition='outside')
            st.plotly_chart(fig_drug, use_container_width=True)

            # Drug name search filter
            search_term = st.text_input("Search for Drug Name", key="drug_search")

            # Apply filter if search term provided
            if search_term:
                filtered_drug_df = drug_df[drug_df["Drug Name"].str.contains(search_term, case=False)]
            else:
                filtered_drug_df = drug_df

            # Show the filtered data
            st.dataframe(filtered_drug_df, use_container_width=True, hide_index=True)

            # AI Insights for drug
            render_ai_insights_section(drug_df, "drugs involved in animal adverse events", "drug")

    # Reaction Analysis Tab
    with reaction_tab:
        st.subheader("Reaction Distribution")

        reaction_df = get_animal_events_by_reaction(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if reaction_df.empty:
            st.warning("No reaction data available for the selected date range.")
        else:
            # Sort by count and limit to top N results
            reaction_df = reaction_df.sort_values("Count", ascending=False).head(st.session_state.top_n_results)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart
                fig_reaction_bar = px.bar(
                    reaction_df,
                    x="Reaction",
                    y="Count",
                    title="Top Reactions in Animal Adverse Events",
                    color="Reaction"
                )
                fig_reaction_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_reaction_bar, use_container_width=True)

            with col2:
                # Treemap visualization
                fig_reaction_treemap = px.treemap(
                    reaction_df,
                    path=["Reaction"],
                    values="Count",
                    title="Hierarchy of Reactions"
                )
                st.plotly_chart(fig_reaction_treemap, use_container_width=True)

            # Reaction name search filter
            search_term = st.text_input("Search for Reaction", key="reaction_search")

            # Apply filter if search term provided
            if search_term:
                filtered_reaction_df = reaction_df[reaction_df["Reaction"].str.contains(search_term, case=False)]
            else:
                filtered_reaction_df = reaction_df

            # Show the filtered data
            st.dataframe(filtered_reaction_df, use_container_width=True, hide_index=True)

            # AI Insights for reaction
            render_ai_insights_section(reaction_df, "reactions in animal adverse events", "reaction")

def display_outcome_analysis():
    """Display analysis by outcome and duration"""
    st.subheader("Analysis by Outcome and Duration")

    # Create tabs for outcome and duration
    outcome_tab, duration_tab = st.tabs(["Outcome Analysis", "Duration Analysis"])

    # Outcome Analysis Tab
    with outcome_tab:
        st.subheader("Outcome Distribution")

        outcome_df = get_animal_events_by_outcome(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if outcome_df.empty:
            st.warning("No outcome data available for the selected date range.")
        else:
            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart
                fig_outcome_bar = px.bar(
                    outcome_df,
                    x="Outcome",
                    y="Count",
                    title="Outcomes in Animal Adverse Events",
                    color="Outcome",
                    text="Count"
                )
                fig_outcome_bar.update_layout(xaxis_tickangle=0)
                fig_outcome_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_outcome_bar, use_container_width=True)

            with col2:
                # Pie chart
                fig_outcome_pie = px.pie(
                    outcome_df,
                    values="Count",
                    names="Outcome",
                    title="Distribution of Outcomes",
                    hole=0.4
                )
                fig_outcome_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_outcome_pie, use_container_width=True)

            # Show the data
            st.dataframe(outcome_df, use_container_width=True, hide_index=True)

            # AI Insights for outcome
            render_ai_insights_section(outcome_df, "outcomes in animal adverse events", "outcome")

    # Duration Analysis Tab
    with duration_tab:
        st.subheader("Duration Distribution")

        duration_df = get_animal_events_by_duration(
            st.session_state.start_date,
            st.session_state.end_date,
            st.session_state.sample_size
        )

        if duration_df.empty:
            st.warning("No duration data available for the selected date range.")
        else:
            # Create visualizations
            fig_duration = px.bar(
                duration_df,
                x="Duration Group",
                y="Count",
                title="Duration of Adverse Events",
                color="Duration Group",
                text="Count"
            )
            fig_duration.update_layout(xaxis_tickangle=0)
            fig_duration.update_traces(textposition='outside')
            st.plotly_chart(fig_duration, use_container_width=True)

            # Duration group selection for filtering
            selected_duration_groups = st.multiselect(
                "Filter by Duration Group",
                options=duration_df["Duration Group"].unique(),
                default=[]
            )

            # Apply filter if selected
            if selected_duration_groups:
                filtered_duration_df = duration_df[duration_df["Duration Group"].isin(selected_duration_groups)]
            else:
                filtered_duration_df = duration_df

            # Show the filtered data
            st.dataframe(filtered_duration_df, use_container_width=True, hide_index=True)

            # AI Insights for duration
            render_ai_insights_section(duration_df, "duration of animal adverse events", "duration")

def display_animal_vetdata():
    """Main function to display the Animal & Veterinary page"""
    st.title("Animal & Veterinary Data Analysis")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Species Analysis",
        "Breed Analysis",
        "Age & Weight Analysis",
        "Drug & Reaction Analysis"
    ])

    with tab1:
        display_species_analysis()

    with tab2:
        display_breed_analysis()

    with tab3:
        display_age_weight_analysis()

    with tab4:
        display_drug_reaction_analysis()
