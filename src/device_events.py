import streamlit as st
import pandas as pd
from typing import Dict, List
from src.data_loader import fetch_api_data
import concurrent.futures

@st.cache_data
def device_class_distribution() -> pd.DataFrame:
    all_results = []
    skip = 0
    limit = 100

    while len(all_results) < st.session_state.sample_size:
        url = f"https://api.fda.gov/device/event.json?count=device_class.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "device class distribution")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:
            break

    if not all_results:
        return pd.DataFrame(columns=["Device Class", "Count", "Percentage"])

    df = pd.DataFrame(all_results)
    df.columns = ["Device Class", "Count"]
    total = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"
    return df.head(st.session_state.top_n_results)

@st.cache_data
def device_problems_by_year(start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
    all_results = []
    skip = 0
    limit = 100

    while len(all_results) < st.session_state.sample_size:
        url = f"https://api.fda.gov/device/event.json?search=date_received:[{start_year}0101+TO+{end_year}1231]&count=device_problem.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "device problems by year")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:
            break

    if not all_results:
        return pd.DataFrame(columns=["Problem", "Count", "Percentage"])

    df = pd.DataFrame(all_results)
    df.columns = ["Problem", "Count"]
    total = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"
    return df.head(st.session_state.top_n_results)

@st.cache_data
def device_manufacturer_analysis() -> pd.DataFrame:
    all_results = []
    skip = 0
    limit = 100

    while len(all_results) < st.session_state.sample_size:
        url = f"https://api.fda.gov/device/event.json?count=manufacturer_name.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "device manufacturer analysis")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:
            break

    if not all_results:
        return pd.DataFrame(columns=["Manufacturer", "Count", "Percentage"])

    df = pd.DataFrame(all_results)
    df.columns = ["Manufacturer", "Count"]
    total = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"
    return df.head(st.session_state.top_n_results)

@st.cache_data
def device_event_type_distribution() -> pd.DataFrame:
    all_results = []
    skip = 0
    limit = 100

    while len(all_results) < st.session_state.sample_size:
        url = f"https://api.fda.gov/device/event.json?count=event_type.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "device event type distribution")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:
            break

    if not all_results:
        return pd.DataFrame(columns=["Event Type", "Count", "Percentage"])

    df = pd.DataFrame(all_results)
    df.columns = ["Event Type", "Count"]
    total = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"
    return df.head(st.session_state.top_n_results)

@st.cache_data
def device_geographic_distribution() -> pd.DataFrame:
    all_results = []
    skip = 0
    limit = 100

    while len(all_results) < st.session_state.sample_size:
        url = f"https://api.fda.gov/device/event.json?count=state.exact&limit={limit}&skip={skip}"
        data = fetch_api_data(url, "device geographic distribution")

        if not data or "results" not in data or not data["results"]:
            break

        all_results.extend(data["results"])
        skip += limit

        if len(data["results"]) < limit:
            break

    if not all_results:
        return pd.DataFrame(columns=["State", "Count", "Percentage"])

    df = pd.DataFrame(all_results)
    df.columns = ["State", "Count"]
    total = df["Count"].sum()
    df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"
    return df.head(st.session_state.top_n_results)

def _process_dataframe(df: pd.DataFrame, columns: List[str], total: int) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=columns + ["Percentage"])
    df.columns = columns
    df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"
    return df

@st.cache_data(ttl=3600)
def _fetch_510k_data(url: str, description: str) -> pd.DataFrame:
    data = fetch_api_data(url, description)
    if not data or "results" not in data or not data["results"]:
        return pd.DataFrame()
    return pd.DataFrame(data["results"])

@st.cache_data(ttl=3600)
def device_510k_clearance_types() -> pd.DataFrame:
    df = _fetch_510k_data(
        "https://api.fda.gov/device/510k.json?count=clearance_type.exact&limit=100",
        "510(k) clearance types"
    )
    return _process_dataframe(df, ["Clearance Type", "Count"], df["Count"].sum() if not df.empty else 0)

@st.cache_data(ttl=3600)
def device_510k_advisory_committees() -> pd.DataFrame:
    df = _fetch_510k_data(
        "https://api.fda.gov/device/510k.json?count=advisory_committee.exact&limit=100",
        "510(k) advisory committees"
    )
    if df.empty:
        return pd.DataFrame(columns=["Advisory Committee", "Count", "Percentage", "Committee Name"])

    df = _process_dataframe(df, ["Advisory Committee", "Count"], df["Count"].sum())

    # Map committee codes to full names
    committee_map = {
        "CV": "Cardiovascular",
        "OR": "Orthopedic",
        "SU": "Surgical",
        "HO": "Hospital",
        "RA": "Radiology",
        "CH": "Chemistry"
    }
    df["Committee Name"] = df["Advisory Committee"].map(committee_map)
    return df

@st.cache_data(ttl=3600)
def device_510k_geographic_distribution() -> pd.DataFrame:
    df = _fetch_510k_data(
        "https://api.fda.gov/device/510k.json?count=state.exact&limit=100",
        "510(k) geographic distribution"
    )
    if df.empty:
        return pd.DataFrame(columns=["State", "Count", "Percentage", "State Name"])

    df = _process_dataframe(df, ["State", "Count"], df["Count"].sum())

    # Add state names
    state_map = get_state_abbreviations()
    df["State Name"] = df["State"].map({v: k for k, v in state_map.items()})
    return df

@st.cache_data(ttl=3600)
def device_510k_decision_codes() -> pd.DataFrame:
    df = _fetch_510k_data(
        "https://api.fda.gov/device/510k.json?count=decision_code.exact&limit=100",
        "510(k) decision codes"
    )
    if df.empty:
        return pd.DataFrame(columns=["Decision Code", "Count", "Percentage", "Decision Description"])

    df = _process_dataframe(df, ["Decision Code", "Count"], df["Count"].sum())

    # Map decision codes to descriptions
    decision_map = {
        "SESE": "Substantially Equivalent",
        "SN": "Substantially Not Equivalent",
        "SESK": "Substantially Equivalent - Special",
        "DENG": "Denied",
        "SESU": "Substantially Equivalent - Summary",
        "SEKD": "Substantially Equivalent - K Number"
    }
    df["Decision Description"] = df["Decision Code"].map(decision_map)
    return df

# Helper functions for data processing
def get_top_device_classes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Device Class", "Count", "Percentage"])
    return df.nlargest(n, "Count")

def get_device_problems_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Problem", "Count", "Percentage"])
    return df.sort_values("Count", ascending=False)

def get_manufacturer_market_share(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Manufacturer", "Count", "Market Share", "Percentage"])
    total = df["Count"].sum()
    df["Market Share"] = (df["Count"] / total * 100).round(2)
    return df.sort_values("Market Share", ascending=False)

def get_event_type_categories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Event Type", "Count", "Percentage"])
    return df.sort_values("Count", ascending=False)

def get_top_clearance_types(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Clearance Type", "Count", "Percentage"])
    return df.nlargest(n, "Count")

def get_committee_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Advisory Committee", "Count", "Percentage", "Committee Name"])
    return df.sort_values("Count", ascending=False)

def get_state_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["State", "Count", "Percentage", "State Name"])
    return df.sort_values("Count", ascending=False)

def get_decision_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Decision Code", "Count", "Percentage", "Decision Description"])
    return df.sort_values("Count", ascending=False)

def get_state_abbreviations() -> Dict[str, str]:
    return {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
        "WI": "Wisconsin", "WY": "Wyoming"
    }

def get_device_events_by_age():
    query = """
    SELECT
        CASE
            WHEN patient_age < 18 THEN 'Under 18'
            WHEN patient_age BETWEEN 18 AND 30 THEN '18-30'
            WHEN patient_age BETWEEN 31 AND 45 THEN '31-45'
            WHEN patient_age BETWEEN 46 AND 60 THEN '46-60'
            WHEN patient_age BETWEEN 61 AND 75 THEN '61-75'
            ELSE 'Over 75'
        END as "Age Group",
        COUNT(*) as "Count"
    FROM device_events
    WHERE patient_age IS NOT NULL
    GROUP BY
        CASE
            WHEN patient_age < 18 THEN 'Under 18'
            WHEN patient_age BETWEEN 18 AND 30 THEN '18-30'
            WHEN patient_age BETWEEN 31 AND 45 THEN '31-45'
            WHEN patient_age BETWEEN 46 AND 60 THEN '46-60'
            WHEN patient_age BETWEEN 61 AND 75 THEN '61-75'
            ELSE 'Over 75'
        END
    ORDER BY
        CASE "Age Group"
            WHEN 'Under 18' THEN 1
            WHEN '18-30' THEN 2
            WHEN '31-45' THEN 3
            WHEN '46-60' THEN 4
            WHEN '61-75' THEN 5
            ELSE 6
        END;
    """

    try:
        df = pd.read_sql_query(query, get_db_connection())
        return df
    except Exception as e:
        st.error(f"Error fetching device events by age: {e}")
        return pd.DataFrame()
