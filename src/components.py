import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from typing import Tuple, Optional

def render_metric_header(title: str, description: str) -> None:
    st.subheader(title)
    st.write(description)

def render_date_picker(
    min_date: date = date(2010, 1, 1),
    max_date: date = date(2025, 1, 31),
    default_start: date = date(2010, 1, 1),
    default_end: date = date(2025, 1, 31),
    key_prefix: str = ""
) -> Tuple[str, str]:
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Data start date",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}start_date"
        )
    with col2:
        end = st.date_input(
            "Data end date",
            value=default_end,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}end_date"
        )
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

def render_data_table(df: pd.DataFrame, width: int = 500, expanded: bool = True) -> None:
    with st.expander("See raw table data", expanded=expanded):
        st.dataframe(data=df, width=width, use_container_width=False)

def render_age_filter(df: pd.DataFrame, column: str = "Patient Age") -> pd.DataFrame:
    min_age = int(df[column].min())
    max_age = int(df[column].max())
    age_range = st.slider(
        "Filter by Patient Age",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )
    return df[(df[column] >= age_range[0]) & (df[column] <= age_range[1])]

def render_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str
) -> None:
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_label, y_col: y_label}
    )
    st.plotly_chart(fig, use_container_width=True)

def render_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str
) -> None:
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_label, y_col: y_label}
    )
    st.plotly_chart(fig, use_container_width=True)
