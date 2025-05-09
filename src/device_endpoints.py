import pandas as pd
import streamlit as st

from src.data_utils import (
    fetch_with_cache,
    get_count_data,
    format_date_range
)

# Base endpoints for device data
DEVICE_EVENT_ENDPOINT = "device/event.json"
DEVICE_510K_ENDPOINT = "device/510k.json"
DEVICE_CLASSIFICATION_ENDPOINT = "device/classification.json"
DEVICE_ENFORCEMENT_ENDPOINT = "device/enforcement.json"
DEVICE_PMA_ENDPOINT = "device/pma.json"
DEVICE_RECALL_ENDPOINT = "device/recall.json"
DEVICE_REGISTRATION_ENDPOINT = "device/registrationlisting.json"
DEVICE_UDI_ENDPOINT = "device/udi.json"
DEVICE_COVID19_ENDPOINT = "device/covid19serology.json"

@st.cache_data(ttl=3600)
def get_device_events_by_type(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {}

    # Adds date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_received:{date_range}"

    df = get_count_data(
        DEVICE_EVENT_ENDPOINT,
        "event_type.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Event Type", "Count"]

        # Map event type codes to descriptions
        event_type_map = {
            "M": "Malfunction",
            "I": "Injury",
            "D": "Death",
            "O": "Other"
        }
        df["Event Type"] = df["Event Type"].map(lambda x: event_type_map.get(x, x))

    return df

@st.cache_data(ttl=3600)
def get_device_events_by_manufacturer(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {}

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_received:{date_range}"

    df = get_count_data(
        DEVICE_EVENT_ENDPOINT,
        "manufacturer_d_name.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Manufacturer", "Count"]
        # Clean manufacturer names
        df["Manufacturer"] = df["Manufacturer"].str.title()

    return df

@st.cache_data(ttl=3600)
def get_device_events_by_product_code(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {}

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_received:{date_range}"

    df = get_count_data(
        DEVICE_EVENT_ENDPOINT,
        "device.openfda.device_class",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Device Class", "Count"]

        # Map class codes to descriptions
        class_map = {
            "1": "Class I (Low Risk)",
            "2": "Class II (Moderate Risk)",
            "3": "Class III (High Risk)",
            "U": "Unclassified",
            "F": "HDE"
        }
        df["Device Class"] = df["Device Class"].map(lambda x: class_map.get(x, x))

    return df

@st.cache_data(ttl=3600)
def get_device_events_by_medical_specialty(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {}

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_received:{date_range}"

    df = get_count_data(
        DEVICE_EVENT_ENDPOINT,
        "device.openfda.medical_specialty_description.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Medical Specialty", "Count"]

    return df

@st.cache_data(ttl=3600)
def get_device_510k_by_applicant(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {}

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"decision_date:{date_range}"

    df = get_count_data(
        DEVICE_510K_ENDPOINT,
        "applicant.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Applicant", "Count"]
        # Clean applicant names
        df["Applicant"] = df["Applicant"].str.title()

    return df

@st.cache_data(ttl=3600)
def get_device_510k_decision_over_time(interval: str = "year", start_date=None, end_date=None) -> pd.DataFrame:
    search_params = {}

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"decision_date:{date_range}"

    # Choose time period based on interval
    if interval == "year":
        count_field = "decision_date.year"
    elif interval == "month":
        count_field = "decision_date.month"
    else:
        count_field = "decision_date.year"

    # get decision results
    decision_data = get_count_data(
        DEVICE_510K_ENDPOINT,
        "decision_description.exact",
        search_params,
        limit=100
    )

    time_data = get_count_data(
        DEVICE_510K_ENDPOINT,
        count_field,
        search_params,
        limit=100
    )

    if not time_data.empty:
        time_data.columns = ["Time Period", "Count"]

        # time period formated for readability
        if interval == "month":
            month_names = {
                "1": "January", "2": "February", "3": "March", "4": "April",
                "5": "May", "6": "June", "7": "July", "8": "August",
                "9": "September", "10": "October", "11": "November", "12": "December"
            }
            time_data["Time Period"] = time_data["Time Period"].map(lambda x: month_names.get(x, x))

    return {
        "time_data": time_data,
        "decision_data": decision_data.rename(columns={"term": "Decision", "count": "Count"}) if not decision_data.empty else pd.DataFrame()
    }

@st.cache_data(ttl=3600)
def get_device_recalls_by_class(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {}

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"event_date_initiated:{date_range}"

    df = get_count_data(
        DEVICE_RECALL_ENDPOINT,
        "root_cause_description.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Root Cause", "Count"]

        # Categorize root causes
        cause_categories = {
            "Design": ["design", "specification", "component", "software", "hardware"],
            "Manufacturing": ["manufacturing", "production", "assembly", "process"],
            "Packaging/Labeling": ["packaging", "labeling", "label", "instructions"],
            "Quality Control": ["quality", "testing", "validation", "verification"],
            "Material": ["material", "composition", "durability"],
            "Environmental": ["environment", "storage", "shipping", "temperature"],
            "Electrical": ["electrical", "electronic", "circuit", "battery", "power"]
        }

        def categorize_cause(cause):
            if pd.isna(cause) or cause == "":
                return "Unknown"

            cause_lower = cause.lower()
            for category, keywords in cause_categories.items():
                if any(keyword in cause_lower for keyword in keywords):
                    return category

            return "Other"

        df["Category"] = df["Root Cause"].apply(categorize_cause)

        # Create a category summary
        category_df = df.groupby("Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}

    return {"detailed": pd.DataFrame(), "categorized": pd.DataFrame()}

@st.cache_data(ttl=3600)
def get_device_recalls_by_product_class(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {}

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"event_date_initiated:{date_range}"

    df = get_count_data(
        DEVICE_RECALL_ENDPOINT,
        "device_class",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Device Class", "Count"]

        # Map class codes to descriptions
        class_map = {
            "1": "Class I (Low Risk)",
            "2": "Class II (Moderate Risk)",
            "3": "Class III (High Risk)",
            "U": "Unclassified",
        }
        df["Device Class"] = df["Device Class"].map(lambda x: class_map.get(x, x))

    return df

@st.cache_data(ttl=3600)
def get_device_covid19_tests(limit: int = 100) -> pd.DataFrame:
    search_params = {
        "limit": str(limit)
    }

    data = fetch_with_cache(DEVICE_COVID19_ENDPOINT, search_params)

    if not data or "results" not in data:
        return pd.DataFrame()

    # Extract relevant fields from the data
    results = []
    for item in data["results"]:
        result = {
            "Manufacturer": item.get("manufacturer", ""),
            "Device": item.get("device", ""),
            "Technology": item.get("technology", ""),
            "Target": item.get("target_antigen", ""),
            "Test Date": item.get("date_performed", ""),
            "Sensitivity": item.get("sensitivity", {}).get("combined", 0),
            "Specificity": item.get("specificity", {}).get("combined", 0)
        }
        results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Convert sensitivity and specificity to percentage
    if "Sensitivity" in df.columns:
        df["Sensitivity"] = pd.to_numeric(df["Sensitivity"], errors="coerce") * 100
        df["Sensitivity"] = df["Sensitivity"].round(1).astype(str) + "%"

    if "Specificity" in df.columns:
        df["Specificity"] = pd.to_numeric(df["Specificity"], errors="coerce") * 100
        df["Specificity"] = df["Specificity"].round(1).astype(str) + "%"

    return df
