from src.data_loader import fetch_api_data
from src.data_cleaner import clean_age_data
import streamlit as st
import pandas as pd

@st.cache_data
def adverse_events_by_patient_age_group_within_data_range(start_date: str, end_date: str):
    url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.patientonsetage"
    data = fetch_api_data(url, "Patient Age")
    df = clean_age_data(data)
    return df

@st.cache_data
def adverse_events_by_drug_within_data_range(start_date: str, end_date: str):
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
