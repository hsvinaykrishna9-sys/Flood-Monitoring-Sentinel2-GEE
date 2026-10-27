"""
Weather-app-style entry point: auto-detects your current location (like
a phone weather app's "Home" screen) or searches any place by name, and
shows live current conditions, an hourly strip, and a daily outlook -
for ANY location worldwide, via the Open-Meteo API (same provider
already used elsewhere in this project).

Flood-risk prediction is a SEPARATE, extra section shown ONLY when the
location is inside Kodagu district - that's the only place the trained
XGBoost model has ever seen real data for. Applying it anywhere else
would be meaningless (the model has zero training signal for any other
location's terrain/hydrology), so every other location gets weather
only, same as a normal weather app, with no flood-risk section at all.

Usage:
    python src/models/weather_app.py              # auto-detect current location
    python src/models/weather_app.py "Kodagu"      # search - shows flood risk too
    python src/models/weather_app.py "Bengaluru"   # search - weather only
"""

import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))
sys.path.insert(0, os.path.dirname(__file__))

import requests
from xgboost import XGBClassifier

from feature_engineering_daily import FEATURE_COLUMNS, run as build_features
from predict_daily import explain_prediction, print_explanation, risk_band, \
    GREEN, YELLOW, RED, BOLD, DIM, RESET

KODAGU_CENTER = (12.4244, 75.7382)
KODAGU_RADIUS_KM = 40  # covers the whole district from its centroid

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}


def wmo_text(code):
    return WMO_CODES.get(int(code), f"Code {code}")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def is_in_kodagu(lat, lon):
    return haversine_km(lat, lon, *KODAGU_CENTER) <= KODAGU_RADIUS_KM


def get_current_location():
    """IP-based auto-location (approximate, city-level) - the CLI
    equivalent of a phone's 'Home'/current-location weather. Returns
    None on any failure so the caller can ask the user to search
    manually instead."""
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if "latitude" not in data or "longitude" not in data:
            return None
        return {
            "lat": data["latitude"],
            "lon": data["longitude"],
            "name": data.get("city", "Unknown"),
            "admin1": data.get("region", ""),
            "country": data.get("country_name", ""),
        }
    except Exception:
        return None


def geocode(query):
    """Search any place name worldwide. Returns None on failure."""
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1},
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        r = results[0]
        return {
            "lat": r["latitude"],
            "lon": r["longitude"],
            "name": r["name"],
            "admin1": r.get("admin1", ""),
            "country": r.get("country", ""),
        }
    except Exception:
        return None


def fetch_weather(lat, lon):
    """Current conditions + 6-hour strip + 6-day outlook, for ANY
    location worldwide. Returns None on failure."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "hourly": "temperature_2m,precipitation_probability,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def print_weather(location_name, admin1, country, data):
    cur = data["current"]
    daily = data["daily"]
    hourly = data["hourly"]

    header = f"{location_name}" + (f", {admin1}" if admin1 else "") + (f" ({country})" if country else "")
    print("=" * 64)
    print(f" {BOLD}{header}{RESET}")
    print(f" {BOLD}{cur['temperature_2m']:.0f}°C{RESET}   {wmo_text(cur['weather_code'])}")
    print(f" H:{daily['temperature_2m_max'][0]:.0f}°  L:{daily['temperature_2m_min'][0]:.0f}°"
          f"   Humidity {cur['relative_humidity_2m']:.0f}%   Wind {cur['wind_speed_10m']:.0f} km/h")
    print("=" * 64)

    print(f"\n {BOLD}NEXT HOURS{RESET}")
    times = hourly["time"]
    current_time = cur.get("time", "")
    # ISO timestamps sort lexicographically same as chronologically, so the
    # first hourly slot at/after "now" is the first one >= current_time -
    # current_time has minute precision (e.g. ...T09:34), hourly slots are
    # on the hour (...T09:00), so a direct equality match would usually miss
    now_idx = next((i for i, t in enumerate(times) if t >= current_time), 0)
    for i in range(now_idx, min(now_idx + 6, len(times))):
        hour_label = times[i][11:16]
        temp = hourly["temperature_2m"][i]
        pop = hourly["precipitation_probability"][i]
        print(f"   {hour_label}  {temp:>5.0f}°C   rain chance {pop:>3.0f}%   {wmo_text(hourly['weather_code'][i])}")

    print(f"\n {BOLD}NEXT DAYS{RESET}")
    for i in range(len(daily["time"])):
        print(f"   {daily['time'][i]}   {daily['temperature_2m_min'][i]:>4.0f}° - "
              f"{daily['temperature_2m_max'][i]:>4.0f}°C   "
              f"rain {daily['precipitation_sum'][i]:>5.1f}mm   {wmo_text(daily['weather_code'][i])}")


def print_flood_risk_section():
    print(f"\n {BOLD}{RED}⚑ FLOOD RISK (Kodagu-specific model){RESET}")
    print(f" {DIM}This location is inside Kodagu district - the only region this{RESET}")
    print(f" {DIM}model was trained on real historical data for.{RESET}\n")

    df = build_features()
    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)

    latest = df.sort_values("date").iloc[-1]
    X_row = latest[FEATURE_COLUMNS].values.reshape(1, -1)
    prob = xgb.predict_proba(X_row)[0, 1]
    band, color = risk_band(prob)

    print(f" Most recent real data: {latest['date'].date()}")
    print(f" Flood risk: {color}{BOLD}{band}{RESET} ({prob:.0%} probability)\n")
    print_explanation(explain_prediction(xgb, X_row))
    print(f"\n {DIM}For the full 10-day outlook and regional (per-taluk) breakdown, run")
    print(f" predict_daily.py or predict_taluk.py directly.{RESET}")


def run():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        loc = geocode(query)
        if loc is None:
            print(f"Couldn't find '{query}'. Try a different spelling or a nearby larger city.")
            return
    else:
        loc = get_current_location()
        if loc is None:
            print("Couldn't auto-detect your location (no network, or the location service is")
            print("unreachable from here). Search for a place instead:")
            print("  python3 src/models/weather_app.py \"Your City\"")
            return
        print(f" {DIM}Auto-detected current location (IP-based, approximate){RESET}")

    data = fetch_weather(loc["lat"], loc["lon"])
    if data is None:
        print(f"Found {loc['name']}, but couldn't reach the weather service right now.")
        return

    print_weather(loc["name"], loc["admin1"], loc["country"], data)

    if is_in_kodagu(loc["lat"], loc["lon"]):
        print_flood_risk_section()


if __name__ == "__main__":
    run()
