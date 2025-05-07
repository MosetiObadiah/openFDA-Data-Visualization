import pandas as pd
from typing import List

def clean_age_data(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["results"], columns=["term", "count"])
    df.columns = ["Patient Age", "Adverse Event Count"]
    df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
    df = df.dropna(subset=["Patient Age", "Adverse Event Count"])
    df["Adverse Event Count"] = pd.to_numeric(df["Adverse Event Count"], errors="coerce").fillna(0).astype(int)
    df = df[df["Patient Age"] > 0]
    df = df.drop_duplicates(subset=["Patient Age"])
    return df

def clean_recall_frequency_data(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["results"], columns=["term", "count"])
    df.columns = ["Year", "Recall Count"]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year", "Recall Count"])
    df["Recall Count"] = pd.to_numeric(df["Recall Count"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("Year")
    return df

def clean_recall_drug_data(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["results"], columns=["term", "count"])
    df.columns = ["Product Description", "Recall Count"]
    df = df.dropna(subset=["Product Description", "Recall Count"])
    df["Recall Count"] = pd.to_numeric(df["Recall Count"], errors="coerce").fillna(0).astype(int)
    # Basic cleaning: remove duplicates and limit to top 20
    df = df.drop_duplicates(subset=["Product Description"]).head(20)
    return df

def clean_recall_reason_data(data: List[dict]) -> pd.DataFrame:
    # Combine data from multiple years
    all_dfs = []
    for year_data in data:
        year = year_data["year"]
        df = pd.DataFrame(year_data["data"]["results"], columns=["term", "count"])
        df["Year"] = year
        df.columns = ["Reason for Recall", "Recall Count", "Year"]
        df = df.dropna(subset=["Reason for Recall", "Recall Count"])
        df["Recall Count"] = pd.to_numeric(df["Recall Count"], errors="coerce").fillna(0).astype(int)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Categorize reasons for simpler analysis
    combined_df["Reason Category"] = combined_df["Reason for Recall"].apply(categorize_reason)
    return combined_df

def categorize_reason(reason: str) -> str:
    reason = reason.lower()
    if "impurities" in reason or "contamination" in reason or "sterility" in reason:
        return "Impurities/Contamination"
    elif "labeling" in reason or "mislabel" in reason:
        return "Labeling Issues"
    elif "cgmp" in reason or "manufacturing" in reason:
        return "CGMP Violations"
    elif "packaging" in reason:
        return "Packaging Issues"
    elif "potency" in reason:
        return "Incorrect Potency"
    else:
        return "Other"
