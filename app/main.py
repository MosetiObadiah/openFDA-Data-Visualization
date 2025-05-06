import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.drug_events import (
    adverse_events_by_drug_name_within_data_range,
    adverse_events_by_patient_age_group_within_data_range
)

st.title("OpenFDA Data Visualization")

def fetch_all_data():
    adverse_events_by_patient_age_group_within_data_range()
    adverse_events_by_drug_name_within_data_range()

if st.button(label="FETCH DATA", help="This gets the latest data from the API"):
    fetch_all_data()
    st.success("Data fetched and saved to CSV.")

result = adverse_events_by_drug_name_within_data_range()
st.dataframe(data=result)

if not isinstance(result, pd.DataFrame):
    st.error("The function did not return a pandas DataFrame")
else:
    if 'Drug Name' in result.columns and 'Adverse Event Count' in result.columns:
        df_chart = result.set_index('Drug Name')[['Adverse Event Count']]
        st.bar_chart(df_chart)
    else:
        st.error("DataFrame must contain 'Drug Name' and 'Adverse Event Count' columns")
