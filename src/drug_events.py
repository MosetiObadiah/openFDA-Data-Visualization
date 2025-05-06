import inspect
from src.data_loader import fetch_api_data
from src.data_cleaner import clean_drug_data, clean_age_data
from pathlib import Path

CSV_PATH = Path("data/csv_store")
CSV_PATH.mkdir(parents=True, exist_ok=True)

def adverse_events_by_drug_name_within_data_range():
    url = "https://api.fda.gov/drug/event.json?search=receivedate:[20200101+TO+20241231]&count=patient.drug.medicinalproduct.exact"
    data = fetch_api_data(url, "Drug Names")
    df = clean_drug_data(data)

    print(inspect.currentframe().f_code.co_name)
    df.to_csv(CSV_PATH / "drug_adverse_events.csv", index=False)
    return df

def adverse_events_by_patient_age_group_within_data_range():
    url = "https://api.fda.gov/drug/event.json?search=receivedate:[20200101+TO+20241231]&count=patient.patientonsetage"
    data = fetch_api_data(url, "Patient Age")
    df = clean_age_data(data)

    print(inspect.currentframe().f_code.co_name)
    df.to_csv(CSV_PATH / "age_adverse_events.csv", index=False)
    return df
