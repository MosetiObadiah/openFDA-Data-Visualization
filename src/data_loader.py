import requests
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
api_key = os.getenv("OPENFDA_API_KEY")

def fetch_api_data(url: str, url_goal: Optional[str] = None) -> dict:
    try:
        headers = {'api_key': api_key} if api_key else {}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data for {url_goal or 'unknown context'}: {e}")
        return {"results": []}
