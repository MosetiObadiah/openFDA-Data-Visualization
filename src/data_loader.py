import requests
import os
from dotenv import load_dotenv
from typing import Optional
import json

load_dotenv()
api_key = os.getenv("OPENFDA_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")  # Debug print for API key

def fetch_api_data(url: str, url_goal: Optional[str] = None) -> dict:
    try:
        # Construct the full URL with API key
        full_url = f"{url}{'&' if '?' in url else '?'}api_key={api_key}" if api_key else url
        print(f"\nFetching data for: {url_goal or 'unknown context'}")
        print(f"Full URL: {full_url}")

        response = requests.get(full_url)
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        # Try to get response content for debugging
        try:
            response_content = response.text
            print(f"Response Content: {response_content[:500]}...")  # Print first 500 chars
        except Exception as e:
            print(f"Could not get response content: {e}")

        response.raise_for_status()

        data = response.json()
        if "error" in data:
            print(f"API Error in response body: {data['error']}")
            return {"results": []}

        if "results" not in data:
            print(f"No results found in API response for {url_goal or 'unknown context'}")
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
