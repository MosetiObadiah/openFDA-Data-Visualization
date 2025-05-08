import requests
import os
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
import time
import json
from functools import lru_cache
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("openfda")

# API Configuration
BASE_URL = "https://api.fda.gov/"
API_KEY = os.getenv("OPENFDA_API_KEY", "")

# Cache control
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)
cache_data = {}
cache_timestamps = {}

# Threading control
MAX_THREADS = 5
request_queue = queue.Queue()
response_queue = queue.Queue()

class APIRateLimiter:
    """Rate limiter for API requests to avoid hitting rate limits"""
    def __init__(self, requests_per_min=120):
        self.requests_per_min = requests_per_min
        self.request_times = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if we're exceeding the rate limit"""
        with self.lock:
            now = time.time()
            # Remove timestamps older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]

            if len(self.request_times) >= self.requests_per_min:
                # We need to wait
                oldest = self.request_times[0]
                wait_time = 60 - (now - oldest)
                if wait_time > 0:
                    logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)

            # Add current timestamp
            self.request_times.append(now)

rate_limiter = APIRateLimiter()

def worker():
    """Worker thread for API requests"""
    while True:
        try:
            task = request_queue.get()
            if task is None:  # Sentinel to stop the thread
                break

            endpoint, params, cache_key = task

            # Check if we need to rate-limit
            rate_limiter.wait_if_needed()

            # Make the API request
            try:
                full_url = BASE_URL + endpoint
                if API_KEY:
                    if not params:
                        params = {}
                    params["api_key"] = API_KEY

                logger.info(f"Fetching data from: {full_url}")
                response = requests.get(full_url, params=params)
                response.raise_for_status()
                data = response.json()
                response_queue.put((cache_key, data))

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                response_queue.put((cache_key, {"error": str(e)}))

        except Exception as e:
            logger.error(f"Worker thread error: {e}")
        finally:
            request_queue.task_done()

# Start worker threads
def start_workers():
    logger.info(f"Starting {MAX_THREADS} worker threads")
    for _ in range(MAX_THREADS):
        t = threading.Thread(target=worker, daemon=True)
        t.start()

# Initialize worker threads
start_workers()

def fetch_with_cache(endpoint: str, params: Optional[Dict] = None, force_refresh: bool = False) -> Dict:
    """Fetch data from the API with caching"""
    # Create a cache key from the endpoint and params
    params_str = json.dumps(params, sort_keys=True) if params else ""
    cache_key = f"{endpoint}_{params_str}"

    # Check if we have a valid cached response
    now = time.time()
    if not force_refresh and cache_key in cache_data:
        timestamp = cache_timestamps.get(cache_key, 0)
        if now - timestamp < CACHE_TTL:
            logger.info(f"Cache hit for {cache_key}")
            return cache_data[cache_key]

    # Queue the request
    request_queue.put((endpoint, params, cache_key))

    # Wait for the response
    while True:
        try:
            result_key, data = response_queue.get(timeout=30)
            if result_key == cache_key:
                # Cache the result
                cache_data[cache_key] = data
                cache_timestamps[cache_key] = now
                response_queue.task_done()
                return data
            else:
                # Put it back for another thread
                response_queue.put((result_key, data))
                response_queue.task_done()
        except queue.Empty:
            logger.warning(f"Timeout waiting for response for {cache_key}")
            return {"error": "Timeout waiting for response"}

def fetch_all_pages(endpoint: str, params: Dict, count_field: str, max_records: int = 1000) -> List[Dict]:
    """Fetch all pages of data up to max_records"""
    all_results = []
    current_params = params.copy()
    limit = min(100, max_records)  # API limit is 100 per request

    if "limit" not in current_params:
        current_params["limit"] = str(limit)

    skip = 0
    while len(all_results) < max_records:
        current_params["skip"] = str(skip)
        data = fetch_with_cache(endpoint, current_params)

        if "error" in data or "results" not in data or not data["results"]:
            break

        results = data["results"]
        all_results.extend(results)
        skip += len(results)

        if len(results) < limit:  # No more results
            break

    return all_results[:max_records]

def get_count_data(endpoint: str, count_field: str, search_params: Optional[Dict] = None,
                  limit: int = 100) -> pd.DataFrame:
    """Get count data from an endpoint"""
    params = search_params.copy() if search_params else {}
    params["count"] = count_field
    params["limit"] = str(limit)

    data = fetch_with_cache(endpoint, params)

    if "error" in data or "results" not in data or not data["results"]:
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    if len(df.columns) == 2:
        df.columns = ["term", "count"]

    return df

def search_records(endpoint: str, search_query: str, limit: int = 100,
                  additional_params: Optional[Dict] = None) -> List[Dict]:
    """Search for records using a specific query"""
    params = {"search": search_query, "limit": str(limit)}
    if additional_params:
        params.update(additional_params)

    data = fetch_with_cache(endpoint, params)

    if "error" in data or "results" not in data:
        return []

    return data["results"]

def format_date_range(start_date, end_date):
    """Format date range for FDA API queries"""
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    return f"[{start_str}+TO+{end_str}]"

def clear_cache():
    """Clear the API cache"""
    cache_data.clear()
    cache_timestamps.clear()
    logger.info("Cache cleared")
