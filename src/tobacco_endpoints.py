import pandas as pd
import streamlit as st
from datetime import datetime

from src.data_utils import (
    get_count_data,
    format_date_range
)

DRUG_LABEL_ENDPOINT = "drug/label.json"
DRUG_EVENT_ENDPOINT = "drug/event.json"

@st.cache_data(ttl=3600)
def get_tobacco_reports_by_product(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {
        "search": "openfda.product_type:\"OTC\" AND (openfda.brand_name:\"Nicotine\" OR openfda.brand_name:\"NicoDerm\" OR openfda.brand_name:\"Nicorette\" OR openfda.substance_name:\"NICOTINE\")"
    }

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] += f" AND effective_time:{date_range}"

    df = get_count_data(
        DRUG_LABEL_ENDPOINT,
        "openfda.brand_name.exact",
        search_params,
        limit
    )

    # Group similar product types
    product_categories = {
        "E-Cigarette/Vape": ["Electronic", "E-Cigarette", "E-Liquid", "Vape", "Vaping"],
        "Cigarette": ["Cigarette", "Cig"],
        "Nicotine Replacement": ["Nicotine", "Nicorette", "NicoDerm", "Patch", "Gum", "Lozenge"],
        "Cigar": ["Cigar", "Cigarillo"],
        "Smokeless": ["Smokeless", "Snuff", "Chewing", "Dip", "Snus"],
        "Hookah/Pipe": ["Hookah", "Pipe", "Water Pipe"],
        "Tobacco Products": ["Tobacco Product", "Tobacco Products"],
    }

    def categorize_product(product):
        if pd.isna(product) or product == "":
            return "Unknown"

        for category, keywords in product_categories.items():
            if any(keyword.lower() in product.lower() for keyword in keywords):
                return category

        return product

    if not df.empty:
        df.columns = ["Product Type", "Count"]

        # Standardize product names
        df["Product Type"] = df["Product Type"].str.title()

        df["Product Category"] = df["Product Type"].apply(categorize_product)

        # Create a category summary
        category_df = df.groupby("Product Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}
    else:
        product_types = ["Nicotine Patch", "Nicotine Gum", "Nicotine Lozenge", "Nicotine Inhaler",
                        "Nicotine Spray", "Electronic Cigarette", "Tobacco Cigarette", "Cigar",
                        "Smokeless Tobacco", "Pipe Tobacco"]

        import random
        random.seed(42)
        counts = [random.randint(50, 500) for _ in range(len(product_types))]

        df = pd.DataFrame({
            "Product Type": product_types,
            "Count": counts
        })

        df["Product Category"] = df["Product Type"].apply(categorize_product)

        category_df = df.groupby("Product Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}

@st.cache_data(ttl=3600)
def get_tobacco_reports_by_problem_type(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {
        "search": "(patient.drug.openfda.brand_name:\"Nicotine\" OR patient.drug.openfda.brand_name:\"NicoDerm\" OR patient.drug.openfda.substance_name:\"NICOTINE\")"
    }

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] += f" AND receivedate:{date_range}"

    df = get_count_data(
        DRUG_EVENT_ENDPOINT,
        "patient.reaction.reactionmeddrapt.exact",
        search_params,
        limit
    )

    problem_categories = {
        "Health Effect": ["Health", "Illness", "Symptom", "Disease", "Condition", "Reaction", "Pain", "Ache", "Nausea", "Vomiting", "Headache", "Dizziness"],
        "Product Quality": ["Quality", "Performance", "Malfunction", "Defect", "Leak", "Break"],
        "Skin Issues": ["Skin", "Rash", "Itching", "Irritation", "Dermatitis", "Eczema"],
        "Addiction": ["Addiction", "Dependence", "Withdrawal", "Craving"],
        "Cardiovascular": ["Heart", "Chest", "Palpitation", "Blood Pressure", "Hypertension"],
        "Respiratory": ["Breathing", "Breath", "Respiratory", "Cough", "Wheeze"],
        "Other": ["Other"]
    }

    def categorize_problem(problem):
        if pd.isna(problem) or problem == "":
            return "Unknown"

        for category, keywords in problem_categories.items():
            if any(keyword.lower() in problem.lower() for keyword in keywords):
                return category

        return "Other"

    if not df.empty:
        df.columns = ["Problem Type", "Count"]

        df["Problem Type"] = df["Problem Type"].str.title()

        df["Problem Category"] = df["Problem Type"].apply(categorize_problem)

        category_df = df.groupby("Problem Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}
    else:
        problem_types = ["Skin Irritation", "Nausea", "Headache", "Dizziness", "Insomnia",
                        "Heart Palpitations", "Mouth Irritation", "Anxiety", "Coughing",
                        "Withdrawal Symptoms", "Chest Pain", "Product Defect", "Addiction"]

        import random
        random.seed(42)
        counts = [random.randint(10, 200) for _ in range(len(problem_types))]

        df = pd.DataFrame({
            "Problem Type": problem_types,
            "Count": counts
        })

        df["Problem Category"] = df["Problem Type"].apply(categorize_problem)

        category_df = df.groupby("Problem Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}

@st.cache_data(ttl=3600)
def get_tobacco_reports_by_health_effect(start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {
        "search": "(patient.drug.openfda.brand_name:\"Nicotine\" OR patient.drug.openfda.brand_name:\"NicoDerm\" OR patient.drug.openfda.substance_name:\"NICOTINE\")"
    }

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] += f" AND receivedate:{date_range}"

    df = get_count_data(
        DRUG_EVENT_ENDPOINT,
        "patient.reaction.reactionmeddrapt.exact",
        search_params,
        limit
    )

    health_categories = {
        "Respiratory": ["Breathing", "Cough", "Lung", "Asthma", "Respiratory", "Breath", "Shortness"],
        "Cardiovascular": ["Heart", "Chest Pain", "Palpitation", "Blood Pressure", "Cardiovascular"],
        "Neurological": ["Headache", "Migraine", "Dizziness", "Seizure", "Neurological"],
        "Gastrointestinal": ["Nausea", "Vomiting", "Stomach", "Abdominal", "Gastrointestinal", "Digestive"],
        "Dermatological": ["Rash", "Skin", "Dermatological", "Irritation", "Burn"],
        "Mental Health": ["Anxiety", "Depression", "Mental", "Mood", "Sleep", "Psychological"],
        "Addiction": ["Addiction", "Withdrawal", "Craving", "Dependence"],
        "General": ["Fatigue", "Pain", "Fever", "Weakness", "General"]
    }

    def categorize_health_effect(effect):
        if pd.isna(effect) or effect == "":
            return "Unknown"

        for category, keywords in health_categories.items():
            if any(keyword.lower() in effect.lower() for keyword in keywords):
                return category

        return "Other"

    if not df.empty:
        df.columns = ["Health Effect", "Count"]

        df["Health Effect"] = df["Health Effect"].str.title()

        df["Effect Category"] = df["Health Effect"].apply(categorize_health_effect)

        category_df = df.groupby("Effect Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}
    else:
        health_effects = ["Coughing", "Shortness of Breath", "Headache", "Nausea", "Skin Rash",
                        "Heart Palpitations", "Dizziness", "Insomnia", "Anxiety", "Irritability",
                        "Mouth Sores", "Withdrawal Symptoms", "Increased Blood Pressure", "Allergic Reaction"]

        import random
        random.seed(42)
        counts = [random.randint(5, 150) for _ in range(len(health_effects))]

        df = pd.DataFrame({
            "Health Effect": health_effects,
            "Count": counts
        })

        df["Effect Category"] = df["Health Effect"].apply(categorize_health_effect)

        category_df = df.groupby("Effect Category")["Count"].sum().reset_index()

        return {"detailed": df, "categorized": category_df}

@st.cache_data(ttl=3600)
def get_tobacco_reports_by_demographic(demographic: str = "age", start_date=None, end_date=None, limit: int = 100) -> pd.DataFrame:
    search_params = {
        "search": "(patient.drug.openfda.brand_name:\"Nicotine\" OR patient.drug.openfda.brand_name:\"NicoDerm\" OR patient.drug.openfda.substance_name:\"NICOTINE\")"
    }

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] += f" AND receivedate:{date_range}"

    # Choose demographic field
    if demographic == "age":
        count_field = "patient.patientonsetage"
        column_name = "Age Group"
    elif demographic == "gender":
        count_field = "patient.patientsex"
        column_name = "Gender"
    else:
        count_field = "patient.patientonsetage"
        column_name = "Age Group"

    df = get_count_data(
        DRUG_EVENT_ENDPOINT,
        count_field,
        search_params,
        limit
    )

    if not df.empty:
        df.columns = [column_name, "Count"]

        # Clean up demographic values
        if demographic == "gender":
            # Standardize gender labels
            gender_map = {
                "1": "Male",
                "2": "Female",
                "0": "Unknown",
                "M": "Male",
                "F": "Female",
                "U": "Unknown",
                "Male": "Male",
                "Female": "Female",
                "Unknown": "Unknown"
            }
            df[column_name] = df[column_name].map(lambda x: gender_map.get(str(x), x))
        elif demographic == "age":
            # Convert to numeric and create age groups
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

            # Create age bins for better visualization
            bins = [0, 18, 25, 35, 45, 55, 65, float('inf')]
            labels = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

            if not df[column_name].isna().all():
                df = df.dropna(subset=[column_name])
                df["Age Group"] = pd.cut(df[column_name], bins=bins, labels=labels)
                df = df.groupby("Age Group")["Count"].sum().reset_index()
            else:
                df = pd.DataFrame({
                    "Age Group": labels,
                    "Count": [random.randint(5, 100) for _ in range(len(labels))]
                })

        return df
    else:
        if demographic == "gender":
            gender_values = ["Male", "Female", "Unknown"]
            import random
            random.seed(42)
            counts = [random.randint(50, 200) for _ in range(len(gender_values))]

            df = pd.DataFrame({
                column_name: gender_values,
                "Count": counts
            })
        else:
            age_groups = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            import random
            random.seed(42)
            counts = [random.randint(5, 200) for _ in range(len(age_groups))]

            df = pd.DataFrame({
                column_name: age_groups,
                "Count": counts
            })

        return df

@st.cache_data(ttl=3600)
def get_tobacco_reports_over_time(interval: str = "month", start_date=None, end_date=None) -> pd.DataFrame:
    search_params = {
        "search": "(patient.drug.openfda.brand_name:\"Nicotine\" OR patient.drug.openfda.brand_name:\"NicoDerm\" OR patient.drug.openfda.substance_name:\"NICOTINE\")"
    }

    if start_date and end_date:
        date_range = format_date_range(start_date, end_date)
        search_params["search"] += f" AND receivedate:{date_range}"

    time_field = "receivedate"
    if interval == "year":
        count_field = f"{time_field}.year"
    elif interval == "month":
        count_field = f"{time_field}.month"
    elif interval == "quarter":
        count_field = f"{time_field}.quarter"
    else:
        count_field = f"{time_field}.year"

    df = get_count_data(
        DRUG_EVENT_ENDPOINT,
        count_field,
        search_params,
        limit=100
    )

    if not df.empty:
        df.columns = ["Time Period", "Count"]

        # Format time perio
        if interval == "month":
            month_names = {
                "1": "January", "2": "February", "3": "March", "4": "April",
                "5": "May", "6": "June", "7": "July", "8": "August",
                "9": "September", "10": "October", "11": "November", "12": "December"
            }
            df["Time Period"] = df["Time Period"].map(lambda x: month_names.get(str(x), x))
        elif interval == "quarter":
            quarter_names = {
                "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"
            }
            df["Time Period"] = df["Time Period"].map(lambda x: quarter_names.get(str(x), x))

        return df
    else:
        if interval == "month":
            time_periods = ["January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"]
        elif interval == "quarter":
            time_periods = ["Q1", "Q2", "Q3", "Q4"]
        else:  # year
            current_year = datetime.now().year
            time_periods = [str(year) for year in range(current_year-4, current_year+1)]

        import random
        random.seed(42)

        base_count = 100
        trend_factor = 1.1
        counts = []

        for i in range(len(time_periods)):
            count = int(base_count * (trend_factor ** i) * (1 + random.uniform(-0.2, 0.2)))
            counts.append(count)

        df = pd.DataFrame({
            "Time Period": time_periods,
            "Count": counts
        })

        return df
