from src.data_loader import fetch_api_data
from src.data_cleaner import (
    clean_age_data,
    clean_recall_frequency_data,
    clean_recall_drug_data,
    clean_recall_reason_data
)
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional

@st.cache_data
def adverse_events_by_patient_age_group_within_data_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and process adverse events data by patient age group."""
    url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.patientonsetage"
    data = fetch_api_data(url, "Patient Age")
    df = clean_age_data(data)
    return df

@st.cache_data
def get_aggregated_age_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate age data for visualization."""
    df = df.groupby("Patient Age", as_index=False).agg({"Adverse Event Count": "sum"})
    return df.sort_values("Patient Age")

@st.cache_data
def adverse_events_by_drug_within_data_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and process adverse events data by drug."""
    url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.drug.medicinalproduct.exact"
    data = fetch_api_data(url, "Drug Name")

    if not data or "results" not in data:
        print(f"API Error or no data returned for URL: {url}")
        return pd.DataFrame(columns=["Drug Name", "Adverse Event Count"])

    df = pd.DataFrame(data["results"], columns=["term", "count"])
    df.columns = ["Drug Name", "Adverse Event Count"]

    # Clean and standardize drug names
    df["Drug Name"] = df["Drug Name"].str.replace(r"\.$", "", regex=True)  # Remove trailing periods
    df["Drug Name"] = df["Drug Name"].str.strip().str.upper()  # Standardize format

    df = df.dropna(subset=["Drug Name", "Adverse Event Count"])
    df["Adverse Event Count"] = pd.to_numeric(df["Adverse Event Count"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["Drug Name"])
    return df

@st.cache_data
def get_top_drugs(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Get top drugs by adverse event count."""
    return df.sort_values("Adverse Event Count", ascending=False).head(limit)

@st.cache_data
def recall_frequency_by_year() -> pd.DataFrame:
    """Fetch and process recall frequency data by year."""
    url = "https://api.fda.gov/drug/enforcement.json?count=recall_initiation_date.year"
    data = fetch_api_data(url, "Recall Frequency by Year")

    if not data or "results" not in data:
        print(f"API Error or no data returned for URL: {url}")
        return pd.DataFrame(columns=["Year", "Recall Count"])

    df = clean_recall_frequency_data(data)
    return df

@st.cache_data
def most_common_recalled_drugs() -> pd.DataFrame:
    """Fetch and process most common recalled drugs data."""
    url = "https://api.fda.gov/drug/enforcement.json?count=product_description.exact&limit=100"
    data = fetch_api_data(url, "Most Common Recalled Drugs")

    if not data or "results" not in data:
        print(f"API Error or no data returned for URL: {url}")
        return pd.DataFrame(columns=["Product Description", "Recall Count"])

    df = clean_recall_drug_data(data)
    return df

@st.cache_data
def recall_reasons_over_time(start_year: int = 2004, end_year: int = 2025) -> pd.DataFrame:
    """Fetch and process recall reasons data over time."""
    all_data = []
    current_year = datetime.now().year
    end_year = min(end_year, current_year)  # Don't query future years
    for year in range(start_year, end_year + 1):
        url = f"https://api.fda.gov/drug/enforcement.json?search=recall_initiation_date:[{year}0101+TO+{year}1231]&count=reason_for_recall.exact&limit=100"
        data = fetch_api_data(url, f"Recall Reasons {year}")
        all_data.append({"year": year, "data": data})
    df = clean_recall_reason_data(all_data)
    return df

@st.cache_data
def get_recall_reasons_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table for recall reasons over time."""
    df_pivot = df.pivot_table(
        index="Year",
        columns="Reason Category",
        values="Recall Count",
        aggfunc="sum",
        fill_value=0
    ).reset_index()
    return df_pivot

@st.cache_data
def get_actions_taken_with_drug() -> pd.DataFrame:
    """Fetch and process actions taken with drug data."""
    url = "https://api.fda.gov/drug/event.json?search=receivedate:[20040101+TO+20250507]&count=patient.drug.actiondrug"
    data = fetch_api_data(url, "Actions Taken with Drug")

    if not data or "results" not in data:
        return pd.DataFrame(columns=["term", "count"])

    df = pd.DataFrame(data["results"], columns=["term", "count"])

    # Map numerical terms to descriptive labels
    action_mapping = {
        "0": "Unknown",
        "1": "Drug withdrawn",
        "2": "Dose not changed",
        "3": "Not applicable",
        "4": "Dose reduced",
        "5": "Dose increased",
    }
    df["Action"] = df["term"].map(action_mapping)
    df = df.dropna(subset=["Action"])  # Remove unmapped entries
    return df
