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
import json

@st.cache_data
def adverse_events_by_patient_age_group_within_data_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and process adverse events data by patient age group."""
    url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.patientonsetage"
    data = fetch_api_data(url, "Patient Age")
    if not data or "results" not in data:
        print("No data returned for adverse events by age")
        return pd.DataFrame(columns=["Patient Age", "Adverse Event Count"])
    return clean_age_data(data)

@st.cache_data
def get_aggregated_age_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate age data for visualization."""
    if df.empty:
        return df
    df = df.groupby("Patient Age", as_index=False).agg({"Adverse Event Count": "sum"})
    return df.sort_values("Patient Age")

@st.cache_data
def adverse_events_by_drug_within_data_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and process adverse events data by drug."""
    url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.drug.medicinalproduct.exact"
    data = fetch_api_data(url, "Drug Name")

    if not data or "results" not in data:
        print("No data returned for adverse events by drug")
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
    if df.empty:
        return df
    return df.sort_values("Adverse Event Count", ascending=False).head(limit)

@st.cache_data
def recall_frequency_by_year() -> pd.DataFrame:
    """Fetch and process recall frequency data by year."""
    url = "https://api.fda.gov/drug/enforcement.json?count=recall_initiation_date.year&limit=100"
    print(f"Fetching recall frequency data from: {url}")
    data = fetch_api_data(url, "Recall Frequency by Year")

    if not data or "results" not in data:
        print("No data returned for recall frequency")
        print(f"API Response: {json.dumps(data, indent=2)}")
        return pd.DataFrame(columns=["Year", "Recall Count"])

    # Convert to DataFrame and process
    df = pd.DataFrame(data["results"])
    df.columns = ["Year", "Recall Count"]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year", "Recall Count"])
    df["Recall Count"] = pd.to_numeric(df["Recall Count"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Year")

    # Create a pivot table for heatmap
    df_pivot = df.pivot_table(
        index=df["Year"].dt.year,
        columns=df["Year"].dt.month,
        values="Recall Count",
        aggfunc="sum",
        fill_value=0
    )

    return df_pivot

@st.cache_data
def most_common_recalled_drugs() -> pd.DataFrame:
    """Fetch and process most common recalled drugs data."""
    url = "https://api.fda.gov/drug/enforcement.json?count=product_description.exact&limit=100"
    print(f"Fetching most common recalled drugs from: {url}")
    data = fetch_api_data(url, "Most Common Recalled Drugs")

    if not data or "results" not in data:
        print("No data returned for most common recalled drugs")
        print(f"API Response: {json.dumps(data, indent=2)}")
        return pd.DataFrame(columns=["Product Description", "Recall Count"])

    df = clean_recall_drug_data(data)
    print(f"Processed recalled drugs data: {len(df)} rows")
    return df

@st.cache_data
def recall_reasons_over_time(start_year: int = 2004, end_year: int = 2025) -> pd.DataFrame:
    """Fetch and process recall reasons data over time."""
    all_data = []
    current_year = datetime.now().year
    end_year = min(end_year, current_year)  # Don't query future years

    print(f"Fetching recall reasons from {start_year} to {end_year}")
    for year in range(start_year, end_year + 1):
        url = f"https://api.fda.gov/drug/enforcement.json?search=recall_initiation_date:[{year}0101+TO+{year}1231]&count=reason_for_recall.exact&limit=100"
        print(f"Fetching data for year {year}")
        data = fetch_api_data(url, f"Recall Reasons {year}")
        if data and "results" in data:
            all_data.append({"year": year, "data": data})
        else:
            print(f"No data returned for year {year}")

    if not all_data:
        print("No recall reasons data found for any year")
        return pd.DataFrame(columns=["Year", "Reason for Recall", "Recall Count", "Reason Category"])

    df = clean_recall_reason_data(all_data)
    print(f"Processed recall reasons data: {len(df)} rows")
    return df

@st.cache_data
def get_recall_reasons_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table for recall reasons over time."""
    if df.empty:
        return pd.DataFrame(columns=["Year"])
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
    url = "https://api.fda.gov/drug/event.json?search=receivedate:[20040101+TO+20250507]&count=patient.drug.actiondrug&limit=100"
    print(f"Fetching actions taken with drug data from: {url}")
    data = fetch_api_data(url, "Actions Taken with Drug")

    if not data or "results" not in data:
        print("No data returned for actions taken with drug")
        print(f"API Response: {json.dumps(data, indent=2)}")
        return pd.DataFrame(columns=["Action", "count"])

    # Create DataFrame from results
    df = pd.DataFrame(data["results"])

    # Map numerical terms to descriptive labels
    action_mapping = {
        0: "Unknown",
        1: "Drug withdrawn",
        2: "Dose not changed",
        3: "Not applicable",
        4: "Dose reduced",
        5: "Dose increased",
        6: "Dose reduced and withdrawn"
    }

    # Convert term to integer and map to action names
    df["term"] = pd.to_numeric(df["term"], errors="coerce")
    df["Action"] = df["term"].map(action_mapping)

    # Drop any rows where mapping failed
    df = df.dropna(subset=["Action"])

    # Rename count column for consistency
    df = df.rename(columns={"count": "Count"})

    # Select only needed columns
    df = df[["Action", "Count"]]

    print(f"Processed actions taken data: {len(df)} rows")
    return df

@st.cache_data
def adverse_events_by_country() -> pd.DataFrame:
    """Fetch and process adverse events data by country."""
    url = "https://api.fda.gov/drug/event.json?count=occurcountry.exact&limit=100"
    print(f"Fetching adverse events by country from: {url}")
    data = fetch_api_data(url, "Adverse Events by Country")

    if not data or "results" not in data:
        print("No data returned for adverse events by country")
        return pd.DataFrame(columns=["Country", "Count", "Percentage"])

    # Convert to DataFrame and process
    df = pd.DataFrame(data["results"])
    df.columns = ["Country", "Count"]
    df = df.dropna(subset=["Country", "Count"])
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int)

    # Standardize country names
    country_mapping = {
        "US": "United States",
        "USA": "United States",
        "UNITED STATES": "United States",
        "UNITED KINGDOM": "United Kingdom",
        "UK": "United Kingdom",
        "GREAT BRITAIN": "United Kingdom",
        "RUSSIAN FEDERATION": "Russia",
        "RUSSIA": "Russia",
        "PEOPLE'S REPUBLIC OF CHINA": "China",
        "CHINA": "China",
        "REPUBLIC OF KOREA": "South Korea",
        "SOUTH KOREA": "South Korea",
        "KOREA": "South Korea",
        "REPUBLIC OF INDIA": "India",
        "INDIA": "India"
    }
    df["Country"] = df["Country"].str.upper().map(lambda x: country_mapping.get(x, x.title()))

    # Group by standardized country names and sum counts
    df = df.groupby("Country", as_index=False)["Count"].sum()

    # Calculate percentage of total
    total_count = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total_count * 100).round(2)

    # Format percentage as string with % symbol
    df["Percentage"] = df["Percentage"].apply(lambda x: f"{x:.2f}%")

    return df.sort_values("Count", ascending=False)
