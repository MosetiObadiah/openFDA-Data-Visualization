import requests
import os
import streamlit as st
from typing import Optional
import json

api_key = st.secrets.get("OPENFDA_API_KEY", "")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")

BASE_URL = "https://api.fda.gov/"

# Cache results for 1 hour
@st.cache_data(ttl=3600)
def fetch_api_data(endpoint: str, params: Optional[dict] = None) -> dict:
    try:
        # Construct the full URL with base URL and parameters
        full_url = BASE_URL + endpoint
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            full_url += f"?{query_string}"

        # Add API key
        full_url += f"{'&' if '?' in full_url else '?'}api_key={api_key}" if api_key else ""

        print(f"\nFetching data for: {params or 'unknown context'}")
        print(f"Full URL: {full_url}")

        response = requests.get(full_url)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        # get response content for debugging
        try:
            response_content = response.text
            print(f"Response Content: {response_content[:500]}...")
        except Exception as e:
            print(f"Could not get response content: {e}")

        response.raise_for_status()

        data = response.json()
        if "error" in data:
            print(f"API Error in response body: {data['error']}")
            return {"results": []}

        if "results" not in data:
            print(f"No results found in API response for {params or 'unknown context'}")
            print(f"Response data: {json.dumps(data, indent=2)}")
            return {"results": []}

        print(f"Successfully retrieved {len(data.get('results', []))} results")
        return data

    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error occurred: {http_err}"
        try:
            response_content = response.text
            error_message += f"\nResponse: {response_content}"
        except Exception:
            pass
        print(error_message)
        return {"results": []}

    except requests.RequestException as e:
        print(f"Network error occurred: {e}")
        return {"results": []}

    except Exception as ex:
        print(f"Unexpected error: {ex}")
        return {"results": []}
