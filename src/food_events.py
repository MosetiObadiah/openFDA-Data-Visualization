import pandas as pd
import requests
from datetime import datetime, timedelta
from src.data_loader import fetch_api_data
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_recall_reasons_distribution():
    """Analyze the distribution of recall reasons by report date."""
    try:
        data = fetch_api_data("food/enforcement.json", {
            "count": "report_date",
            "limit": 100
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        recalls = []
        for result in data["results"]:
            if "time" in result and "count" in result:
                # Convert YYYYMMDD to datetime
                date = datetime.strptime(result["time"], "%Y%m%d")
                recalls.append({
                    "Date": date,
                    "Count": result["count"]
                })

        df = pd.DataFrame(recalls)
        if not df.empty:
            df = df.sort_values("Date")
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_recall_reasons_distribution: {e}")
        return pd.DataFrame()

def _fetch_all_food_data():
    """Fetch all food-related data in parallel."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_func = {
            executor.submit(get_recall_reasons_distribution): "recall_reasons"
        }

        results = {}
        for future in as_completed(future_to_func):
            func_name = future_to_func[future]
            try:
                results[func_name] = future.result()
            except Exception as e:
                print(f"Error fetching {func_name}: {e}")
                results[func_name] = pd.DataFrame()

def get_recall_trends_by_year():
    """Analyze recall trends over time."""
    try:
        current_year = datetime.now().year
        start_year = current_year - 5

        data = fetch_api_data("food/event.json", {
            "limit": 1000,
            "sort": "recall_initiation_date:desc"
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        recalls = []
        for result in data["results"]:
            if "recall_initiation_date" in result:
                date = datetime.strptime(result["recall_initiation_date"], "%Y%m%d")
                if date.year >= start_year:
                    recalls.append({
                        "Year": date.year,
                        "Month": date.month,
                        "Count": 1
                    })

        df = pd.DataFrame(recalls)
        if not df.empty:
            df = df.groupby(["Year", "Month"]).sum().reset_index()
            df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
            df = df.sort_values("Date")

        return df
    except Exception as e:
        print(f"Error in get_recall_trends_by_year: {e}")
        return pd.DataFrame()

def get_product_type_analysis():
    """Analyze distribution of product types and their recall patterns."""
    try:
        data = fetch_api_data("food/event.json", {
            "limit": 1000,
            "sort": "recall_initiation_date:desc"
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        products = []
        for result in data["results"]:
            if "product_type" in result and "reason_for_recall" in result:
                products.append({
                    "Product Type": result["product_type"],
                    "Reason": result["reason_for_recall"],
                    "Count": 1
                })

        df = pd.DataFrame(products)
        if not df.empty:
            df = df.groupby(["Product Type", "Reason"]).sum().reset_index()
            df = df.sort_values("Count", ascending=False)
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_product_type_analysis: {e}")
        return pd.DataFrame()

def get_recall_status_distribution():
    """Analyze the distribution of recall statuses and their patterns."""
    try:
        data = fetch_api_data("food/event.json", {
            "limit": 1000,
            "sort": "recall_initiation_date:desc"
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        statuses = []
        for result in data["results"]:
            if "status" in result and "voluntary_mandated" in result:
                statuses.append({
                    "Status": result["status"],
                    "Type": result["voluntary_mandated"],
                    "Count": 1
                })

        df = pd.DataFrame(statuses)
        if not df.empty:
            df = df.groupby(["Status", "Type"]).sum().reset_index()
            df = df.sort_values("Count", ascending=False)
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_recall_status_distribution: {e}")
        return pd.DataFrame()

def get_geographic_distribution():
    """Analyze the geographic distribution of recalls."""
    try:
        data = fetch_api_data("food/event.json", {
            "limit": 1000,
            "sort": "recall_initiation_date:desc"
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        locations = []
        for result in data["results"]:
            if "state" in result and "country" in result:
                locations.append({
                    "State": result["state"],
                    "Country": result["country"],
                    "Count": 1
                })

        df = pd.DataFrame(locations)
        if not df.empty:
            df = df.groupby(["State", "Country"]).sum().reset_index()
            df = df.sort_values("Count", ascending=False)
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_geographic_distribution: {e}")
        return pd.DataFrame()

def _fetch_all_food_data():
    """Fetch all food-related data in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_func = {
            executor.submit(get_recall_reasons_distribution): "recall_reasons",
            executor.submit(get_recall_trends_by_year): "recall_trends",
            executor.submit(get_product_type_analysis): "product_types",
            executor.submit(get_recall_status_distribution): "recall_status",
            executor.submit(get_geographic_distribution): "geographic"
        }

        results = {}
        for future in as_completed(future_to_func):
            func_name = future_to_func[future]
            try:
                results[func_name] = future.result()
            except Exception as e:
                print(f"Error fetching {func_name}: {e}")
                results[func_name] = pd.DataFrame()

        return results
