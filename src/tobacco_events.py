import pandas as pd
from datetime import datetime
from src.data_loader import fetch_api_data
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_tobacco_products_distribution():
    """Analyze the distribution of tobacco products."""
    try:
        data = fetch_api_data("tobacco/problem.json", {
            "count": "tobacco_products.exact",
            "limit": 100
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        products = []
        for result in data["results"]:
            if "term" in result and "count" in result:
                products.append({
                    "Product": result["term"],
                    "Count": result["count"]
                })

        df = pd.DataFrame(products)
        if not df.empty:
            df = df.sort_values("Count", ascending=False)
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_tobacco_products_distribution: {e}")
        return pd.DataFrame()

def get_health_problems_distribution():
    """Analyze the distribution of reported health problems."""
    try:
        data = fetch_api_data("tobacco/problem.json", {
            "count": "reported_health_problems.exact",
            "limit": 100
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        problems = []
        for result in data["results"]:
            if "term" in result and "count" in result:
                problems.append({
                    "Health Problem": result["term"],
                    "Count": result["count"]
                })

        df = pd.DataFrame(problems)
        if not df.empty:
            df = df.sort_values("Count", ascending=False)
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_health_problems_distribution: {e}")
        return pd.DataFrame()

def get_health_problems_count_distribution():
    """Analyze the distribution of number of health problems per report."""
    try:
        data = fetch_api_data("tobacco/problem.json", {
            "count": "number_health_problems",
            "limit": 100
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        counts = []
        for result in data["results"]:
            if "term" in result and "count" in result:
                counts.append({
                    "Number of Problems": result["term"],
                    "Count": result["count"]
                })

        df = pd.DataFrame(counts)
        if not df.empty:
            df = df.sort_values("Number of Problems")
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_health_problems_count_distribution: {e}")
        return pd.DataFrame()

def get_product_problems_distribution():
    """Analyze the distribution of reported product problems."""
    try:
        data = fetch_api_data("tobacco/problem.json", {
            "count": "reported_product_problems.exact",
            "limit": 100
        })

        if not data or "results" not in data:
            return pd.DataFrame()

        problems = []
        for result in data["results"]:
            if "term" in result and "count" in result:
                problems.append({
                    "Product Problem": result["term"],
                    "Count": result["count"]
                })

        df = pd.DataFrame(problems)
        if not df.empty:
            df = df.sort_values("Count", ascending=False)
            total = df["Count"].sum()
            df["Percentage"] = (df["Count"] / total * 100).round(2).astype(str) + "%"

        return df
    except Exception as e:
        print(f"Error in get_product_problems_distribution: {e}")
        return pd.DataFrame()

def _fetch_all_tobacco_data():
    """Fetch all tobacco-related data in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_func = {
            executor.submit(get_tobacco_products_distribution): "tobacco_products",
            executor.submit(get_health_problems_distribution): "health_problems",
            executor.submit(get_health_problems_count_distribution): "health_problems_count",
            executor.submit(get_product_problems_distribution): "product_problems"
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
