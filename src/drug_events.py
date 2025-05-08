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
    all_results = []
    skip = 0
    limit = 100  # Maximum allowed per request

    while len(all_results) < st.session_state.sample_size:  # Use global sample size
        url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.patientonsetage&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "Patient Age")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < nlimit:  # No more results available
            break

    if not all_results:
        print("No data returned for adverse events by age")
        return pd.DataFrame(columns=["Patient Age", "Adverse Event Count"])
    df = clean_age_data({"results": all_results})
    return df.head(st.session_state.top_n_results)  # Use global top N results

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
    all_results = []
    skip = 0
    limit = 100  # Maximum allowed per request

    while len(all_results) < st.session_state.sample_size:  # Use global sample size
        url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.drug.medicinalproduct.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "Drug Name")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:  # No more results available
            break

    if not all_results:
        print("No data returned for adverse events by drug")
        return pd.DataFrame(columns=["Drug Name", "Adverse Event Count"])

    df = pd.DataFrame(all_results, columns=["term", "count"])
    df.columns = ["Drug Name", "Adverse Event Count"]

    # Clean and standardize drug names
    df["Drug Name"] = df["Drug Name"].str.replace(r"\.$", "", regex=True)  # Remove trailing periods
    df["Drug Name"] = df["Drug Name"].str.strip().str.upper()  # Standardize format

    df = df.dropna(subset=["Drug Name", "Adverse Event Count"])
    df["Adverse Event Count"] = pd.to_numeric(df["Adverse Event Count"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["Drug Name"])
    return df.head(st.session_state.top_n_results)  # Use global top N results

@st.cache_data
def get_top_drugs(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Get top drugs by adverse event count."""
    if df.empty:
        return df
    return df.sort_values("Adverse Event Count", ascending=False).head(limit)

@st.cache_data
def recall_frequency_by_year() -> pd.DataFrame:
    """Fetch and process recall frequency data by year."""
    all_results = []
    skip = 0
    limit = 100  # Maximum allowed per request

    while True:  # Continue until no more results
        url = f"https://api.fda.gov/drug/enforcement.json?count=recall_initiation_date.year&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "Recall Frequency by Year")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:  # No more results available
            break

    if not all_results:
        print("No data returned for recall frequency")
        return pd.DataFrame(columns=["Year", "Recall Count"])

    # Convert to DataFrame and process
    df = pd.DataFrame(all_results)
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
    all_results = []
    skip = 0
    limit = 100  # Maximum allowed per request

    while True:  # Continue until no more results
        url = f"https://api.fda.gov/drug/enforcement.json?count=product_description.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "Most Common Recalled Drugs")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:  # No more results available
            break

    if not all_results:
        print("No data returned for most common recalled drugs")
        return pd.DataFrame(columns=["Product Description", "Recall Count"])

    df = clean_recall_drug_data({"results": all_results})
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
    all_results = []
    skip = 0
    limit = 100  # Maximum allowed per request

    while len(all_results) < st.session_state.sample_size:  # Use global sample size
        url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{st.session_state.start_date}+TO+{st.session_state.end_date}]&count=patient.drug.actiondrug&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "Actions Taken with Drug")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:  # No more results available
            break

    if not all_results:
        print("No data returned for actions taken with drug")
        return pd.DataFrame(columns=["Action", "count"])

    # Create DataFrame from results
    df = pd.DataFrame(all_results)

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
    return df.head(st.session_state.top_n_results)  # Use global top N results

@st.cache_data
def adverse_events_by_country() -> pd.DataFrame:
    """Fetch and process adverse events data by country."""
    all_results = []
    skip = 0
    limit = 100  # Maximum allowed per request

    while len(all_results) < st.session_state.sample_size:  # Use global sample size
        url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{st.session_state.start_date}+TO+{st.session_state.end_date}]&count=occurcountry.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "Adverse Events by Country")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:  # No more results available
            break

    if not all_results:
        print("No data returned for adverse events by country")
        return pd.DataFrame(columns=["Country", "Count", "Percentage"])

    # Convert to DataFrame and process
    df = pd.DataFrame(all_results)
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

    return df.head(st.session_state.top_n_results)  # Use global top N results

def get_drug_events_by_substance():
    """Get drug events by active ingredient (substance name)."""
    endpoint = "drug/event.json"
    params = {
        "count": "patient.drug.openfda.substance_name.exact",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df.columns = ["Substance", "Count"]
    return df

def get_drug_events_by_action():
    """Get drug events by action taken with the drug."""
    endpoint = "drug/event.json"
    params = {
        "count": "patient.drug.actiondrug",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df.columns = ["Action Code", "Count"]

    # Map action codes to descriptive labels
    action_map = {
        "1": "Drug withdrawn",
        "2": "Dose not changed",
        "3": "Not applicable",
        "4": "Dose reduced",
        "5": "Dose increased",
        "6": "Dose reduced and withdrawn"
    }
    df["Action"] = df["Action Code"].map(action_map)
    return df[["Action", "Count"]]

def get_drug_events_by_patient_sex():
    """Get drug events by patient sex."""
    endpoint = "drug/event.json"
    params = {
        "count": "patient.patientsex",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df.columns = ["Sex Code", "Count"]

    # Map sex codes to labels
    sex_map = {
        "1": "Male",
        "2": "Female",
        "0": "Unknown"
    }
    df["Sex"] = df["Sex Code"].map(sex_map)
    return df[["Sex", "Count"]]

def get_drug_events_by_patient_weight():
    """Get drug events by patient weight."""
    endpoint = "drug/event.json"
    params = {
        "count": "patient.patientweight",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df.columns = ["Weight", "Count"]

    # Convert weight to numeric and create weight groups
    df["Weight"] = pd.to_numeric(df["Weight"], errors='coerce')

    # Create weight groups (in kg)
    bins = [0, 30, 50, 70, 90, 110, 130, 150, float('inf')]
    labels = ['<30', '30-50', '50-70', '70-90', '90-110', '110-130', '130-150', '>150']
    df["Weight Group"] = pd.cut(df["Weight"], bins=bins, labels=labels)

    # Group by weight group and sum counts
    weight_groups = df.groupby("Weight Group")["Count"].sum().reset_index()
    return weight_groups

def get_drug_events_by_reaction_outcome():
    """Get drug events by reaction outcome."""
    endpoint = "drug/event.json"
    params = {
        "count": "patient.reaction.reactionoutcome",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df.columns = ["Outcome Code", "Count"]

    # Map outcome codes to labels
    outcome_map = {
        "1": "Recovered/Resolved",
        "2": "Recovering/Resolving",
        "3": "Not Recovered/Not Resolved",
        "4": "Recovered/Resolved with Sequelae",
        "5": "Fatal",
        "6": "Unknown"
    }
    df["Outcome"] = df["Outcome Code"].map(outcome_map)
    return df[["Outcome", "Count"]]

def get_drug_events_by_reporter_qualification():
    """Get drug events by reporter qualification."""
    endpoint = "drug/event.json"
    params = {
        "count": "primarysource.qualification",
        "limit": "100"
    }
    data = fetch_api_data(endpoint, params)

    if not data or "results" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df.columns = ["Qualification Code", "Count"]

    # Map qualification codes to labels
    qualification_map = {
        "1": "Physician",
        "2": "Pharmacist",
        "3": "Other Health Professional",
        "4": "Lawyer",
        "5": "Consumer or non-health professional"
    }
    df["Qualification"] = df["Qualification Code"].map(qualification_map)
    return df[["Qualification", "Count"]]
