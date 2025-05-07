import requests
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
api_key = os.getenv("OPENFDA_API_KEY")

def fetch_api_data(url: str, url_goal: Optional[str] = None) -> dict:
    try:
        headers = {'api_key': api_key} if api_key else {}
        print(f"Making API request to: {url}")  # Debug print
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        if "error" in data:
            print(f"API Error: {data['error']}")
            return {"results": []}

        if "results" not in data:
            print(f"No results found in API response for {url_goal or 'unknown context'}")
            return {"results": []}

        return data
    except requests.RequestException as e:
        print(f"Error fetching data for {url_goal or 'unknown context'}: {e}")
        return {"results": []}
