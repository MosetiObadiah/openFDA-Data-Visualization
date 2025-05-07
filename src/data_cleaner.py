import pandas as pd


def clean_age_data(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["results"], columns=["term", "count"])
    df.columns = ["Patient Age", "Adverse Event Count"]

    df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
    df = df.dropna(subset=["Patient Age", "Adverse Event Count"])
    df["Adverse Event Count"] = pd.to_numeric(df["Adverse Event Count"], errors="coerce").fillna(0).astype(int)
    df = df[df["Patient Age"] > 0]
    df = df.drop_duplicates(subset=["Patient Age"])
    return df
