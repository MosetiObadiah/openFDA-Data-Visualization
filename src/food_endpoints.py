import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Optional, Dict, List

from src.data_utils import (
    fetch_with_cache,
    get_count_data,
    fetch_all_pages,
    search_records,
    format_date_range
)

# Base endpoints for food data
FOOD_ENFORCEMENT_ENDPOINT = "food/enforcement.json"
FOOD_EVENT_ENDPOINT = "food/event.json"

@st.cache_data(ttl=3600)
def get_food_recalls_by_classification(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    """Get food recalls by classification"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"recall_initiation_date:{date_range}"

    df = get_count_data(
        FOOD_ENFORCEMENT_ENDPOINT,
        "classification.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Classification", "Count"]

        # Add classification descriptions
        classification_desc = {
            "Class I": "Dangerous or defective products that predictably could cause serious health problems or death",
            "Class II": "Products that might cause temporary health problem, or slight threat of a serious nature",
            "Class III": "Products that are unlikely to cause any adverse health reaction, but that violate FDA regulations"
        }
        df["Description"] = df["Classification"].map(classification_desc)

    return df

@st.cache_data(ttl=3600)
def get_food_recalls_by_reason(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    """Get food recalls by reason"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"recall_initiation_date:{date_range}"

    df = get_count_data(
        FOOD_ENFORCEMENT_ENDPOINT,
        "reason_for_recall.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Reason", "Count"]

        # Clean reason text
        df["Reason"] = df["Reason"].str.title()

        # Categorize reasons
        reason_categories = {
            "Allergen": ["Allergen", "Allergic", "Allergy"],
            "Contamination": ["Contamination", "Contaminated", "Bacteria", "Bacterial", "Mold", "Salmonella", "Listeria", "E. Coli", "Pathogen"],
            "Foreign Material": ["Foreign", "Metal", "Glass", "Plastic", "Wood", "Stone"],
            "Mislabeling": ["Label", "Labeling", "Misbranded", "Mislabeled", "Undeclared"],
            "Quality Issues": ["Quality", "Deterioration", "Spoilage", "Texture", "Taste", "Appearance"],
            "Manufacturing Issues": ["Manufacturing", "Production", "Process", "Unapproved", "Specification"],
            "Other": []
        }

        # Create function to categorize reasons
        def categorize_reason(reason_text):
            for category, keywords in reason_categories.items():
                if any(keyword.lower() in reason_text.lower() for keyword in keywords):
                    return category
            return "Other"

        df["Category"] = df["Reason"].apply(categorize_reason)

        # Create a category summary
        category_df = df.groupby("Category")["Count"].sum().reset_index()

        # Return both dataframes as a dictionary
        return {"detailed": df, "categorized": category_df}

    return {"detailed": pd.DataFrame(), "categorized": pd.DataFrame()}

@st.cache_data(ttl=3600)
def get_food_recalls_by_state(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    """Get food recalls by state"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"recall_initiation_date:{date_range}"

    df = get_count_data(
        FOOD_ENFORCEMENT_ENDPOINT,
        "state.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["State", "Count"]

        # Convert state abbreviations to full names
        state_map = {
            "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
            "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
            "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
            "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
            "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
            "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
            "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
            "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
            "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
            "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
            "DC": "District of Columbia", "PR": "Puerto Rico"
        }
        df["State"] = df["State"].map(lambda x: state_map.get(x, x))

    return df

@st.cache_data(ttl=3600)
def get_food_recalls_by_product_type(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    """Get food recalls by product type"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"recall_initiation_date:{date_range}"

    # This endpoint doesn't have a specific product type field, so we'll analyze product descriptions
    results = []
    data = fetch_with_cache(FOOD_ENFORCEMENT_ENDPOINT, search_params)

    if "results" in data:
        # Extract product descriptions and analyze them
        for result in data.get("results", [])[:limit]:
            product_description = result.get("product_description", "")
            results.append({"description": product_description})

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Categorize product descriptions
    product_categories = {
        "Dairy": ["milk", "cheese", "yogurt", "cream", "butter", "dairy"],
        "Meat": ["beef", "pork", "chicken", "turkey", "meat", "poultry", "fish", "seafood"],
        "Produce": ["vegetable", "fruit", "produce", "fresh", "salad"],
        "Bakery": ["bread", "pastry", "cake", "cookie", "bakery", "baked"],
        "Snacks": ["snack", "chip", "candy", "chocolate"],
        "Beverages": ["drink", "beverage", "juice", "water", "soda", "coffee", "tea"],
        "Prepared Foods": ["prepared", "ready-to-eat", "meal", "dinner", "lunch", "breakfast"],
        "Nuts & Seeds": ["nut", "seed", "peanut", "almond", "cashew", "walnut"],
        "Condiments": ["sauce", "dressing", "oil", "vinegar", "condiment"],
        "Supplements": ["supplement", "vitamin", "mineral", "dietary"]
    }

    def categorize_product(description):
        if pd.isna(description) or description == "":
            return "Unknown"

        desc_lower = description.lower()

        for category, keywords in product_categories.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category

        return "Other"

    df["Product Category"] = df["description"].apply(categorize_product)

    # Count categories
    category_counts = df["Product Category"].value_counts().reset_index()
    category_counts.columns = ["Product Category", "Count"]

    return category_counts

@st.cache_data(ttl=3600)
def get_food_events_by_product(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    """Get food adverse events by product"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_created:{date_range}"

    df = get_count_data(
        FOOD_EVENT_ENDPOINT,
        "products.name_brand.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Product", "Count"]
        # Clean product names
        df["Product"] = df["Product"].str.title()

    return df

@st.cache_data(ttl=3600)
def get_food_events_by_symptom(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    """Get food adverse events by reported symptoms"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_created:{date_range}"

    df = get_count_data(
        FOOD_EVENT_ENDPOINT,
        "reactions.exact",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Symptom", "Count"]
        # Clean symptom names
        df["Symptom"] = df["Symptom"].str.title()

        # Create symptom categories
        symptom_categories = {
            "Gastrointestinal": ["Nausea", "Vomiting", "Diarrhea", "Stomach", "Abdominal", "Cramp", "Gastric", "Intestinal"],
            "Allergic": ["Allergy", "Allergic", "Rash", "Hive", "Swelling", "Itching", "Itch"],
            "Neurological": ["Headache", "Dizziness", "Migraine", "Neurological", "Brain", "Seizure"],
            "Respiratory": ["Breathing", "Breath", "Respiratory", "Cough", "Wheeze", "Asthma", "Chest", "Lung"],
            "Cardiovascular": ["Heart", "Cardiac", "Pulse", "Blood Pressure", "Palpitation"],
            "General": ["Fatigue", "Weakness", "Fever", "Pain", "Malaise", "Discomfort"]
        }

        def categorize_symptom(symptom):
            if pd.isna(symptom) or symptom == "":
                return "Unknown"

            for category, keywords in symptom_categories.items():
                if any(keyword.lower() in symptom.lower() for keyword in keywords):
                    return category

            return "Other"

        df["Category"] = df["Symptom"].apply(categorize_symptom)

        # Create a category summary
        category_df = df.groupby("Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}

    return {"detailed": pd.DataFrame(), "categorized": pd.DataFrame()}

@st.cache_data(ttl=3600)
def get_food_events_by_age(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    """Get food adverse events by patient age"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_created:{date_range}"

    df = get_count_data(
        FOOD_EVENT_ENDPOINT,
        "consumer.age",
        search_params,
        limit
    )

    if not df.empty:
        df.columns = ["Age", "Count"]

        # Convert to numeric and create age groups
        df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
        df = df.dropna(subset=["Age"])

        # Create age bins for better visualization
        bins = [0, 5, 13, 18, 30, 45, 60, 75, float('inf')]
        labels = ['0-4', '5-12', '13-17', '18-29', '30-44', '45-59', '60-74', '75+']
        df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

        # Group by age group
        df = df.groupby("Age Group")["Count"].sum().reset_index()

    return df

@st.cache_data(ttl=3600)
def get_food_events_over_time(interval: str = "month", start_date=None, end_date=None) -> pd.DataFrame:
    """Get food adverse events over time periods"""
    search_params = {}

    # Add date range if provided
    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] = f"date_created:{date_range}"

    # Choose the right time field based on interval
    time_field = "date_created"
    if interval == "year":
        count_field = f"{time_field}.year"
    elif interval == "month":
        count_field = f"{time_field}.month"
    elif interval == "quarter":
        count_field = f"{time_field}.quarter"
    else:
        count_field = f"{time_field}.year"

    df = get_count_data(
        FOOD_EVENT_ENDPOINT,
        count_field,
        search_params,
        limit=100  # Get all available time periods
    )

    if not df.empty:
        df.columns = ["Time Period", "Count"]

        # Format time period for readability
        if interval == "month":
            month_names = {
                "1": "January", "2": "February", "3": "March", "4": "April",
                "5": "May", "6": "June", "7": "July", "8": "August",
                "9": "September", "10": "October", "11": "November", "12": "December"
            }
            df["Time Period"] = df["Time Period"].map(lambda x: month_names.get(x, x))
        elif interval == "quarter":
            quarter_names = {
                "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"
            }
            df["Time Period"] = df["Time Period"].map(lambda x: quarter_names.get(x, x))

    return df
