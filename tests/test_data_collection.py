"""
Unit test for the weather data collector.

This hits the live Open-Meteo API for a known date/location, so it needs
outbound internet access. If the network is unavailable (e.g. a sandboxed
CI runner with an egress allowlist), the test is skipped rather than
reported as a failure, since that's an environment limitation, not a
defect in the collector.
"""

import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data_collection"))

from weather_collector import WeatherCollector


def test_weather_api_returns_valid_response_for_known_date():
    collector = WeatherCollector()
    try:
        df = collector.fetch_daily("2024-07-01", "2024-07-01")
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"no outbound network access to Open-Meteo in this environment: {exc}")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["temperature_c"] is not None and not pd_isna(row["temperature_c"])
    assert row["rainfall_mm"] is not None and not pd_isna(row["rainfall_mm"])
    assert row["humidity_percent"] is not None and not pd_isna(row["humidity_percent"])


def pd_isna(value):
    import pandas as pd
    return pd.isna(value)
