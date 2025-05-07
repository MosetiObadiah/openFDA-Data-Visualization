from src.data_loader import fetch_api_data
from src.data_cleaner import clean_age_data

import streamlit as st

@st.cache_data
def adverse_events_by_patient_age_group_within_data_range(start_date: str, end_date: str):
    url = f"https://api.fda.gov/drug/event.json?search=receivedate:[{start_date}+TO+{end_date}]&count=patient.patientonsetage"
    data = fetch_api_data(url, "Patient Age")
    df = clean_age_data(data)
    return df
