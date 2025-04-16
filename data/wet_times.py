import os
import fastf1
import requests
import pandas as pd
import json

# Defaults for Saudi Arabia Grand Prix (Jeddah Corniche)
FASTF1_TRACK = "SaudiArabia"
LATITUDE = 21.6319
LONGITUDE = 39.1046


def get_weather_conditions(api_key: str, lat: float = LATITUDE, lon: float = LONGITUDE) -> dict:
    """
    Fetch current weather data for given coordinates using OpenWeatherMap.

    Args:
        api_key (str): Your OpenWeatherMap API key.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        dict: Contains temperature (°C) and rain probability (0.0–1.0).
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    temperature = data.get("main", {}).get("temp")
    rain_probability = 1.0 if data.get("rain") else 0.0
    return {"temperature": temperature, "rain_probability": rain_probability}


def compute_wet_performance(
    wet_year: int = 2019,
    dry_year: int = 2023,
    session_code: str = "R",
    api_key: str = None
) -> dict:
    """
    Compare wet vs dry race sessions for the Saudi Arabia GP by default.

    Args:
        wet_year: Year of the wet reference race (default 2019).
        dry_year: Year of the dry reference race (default 2023).
        session_code: Session code ("R" for Race, default).
        api_key: (Optional) OpenWeatherMap key to fetch weather.

    Returns:
        dict: Mapping of driver abbreviations to their WetPerformanceScore.
    """
    # Optional weather display
    if api_key:
        weather = get_weather_conditions(api_key)
        print(f"Weather at Saudi Arabia GP: {weather['temperature']}°C, rain_prob={weather['rain_probability']}")

    # Prepare FastF1 cache
    cache_dir = os.path.join("f1_cache", FASTF1_TRACK, f"{wet_year}_{dry_year}")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    # Load sessions
    print(f"Loading {wet_year} wet session for Saudi Arabia ({session_code})...")
    wet_session = fastf1.get_session(wet_year, FASTF1_TRACK, session_code)
    wet_session.load()
    print(f"Loading {dry_year} dry session for Saudi Arabia ({session_code})...")
    dry_session = fastf1.get_session(dry_year, FASTF1_TRACK, session_code)
    dry_session.load()

    # Extract and convert lap times
    laps_wet = wet_session.laps.dropna(subset=["LapTime"]).loc[:, ["Driver", "LapTime"]].copy()
    laps_dry = dry_session.laps.dropna(subset=["LapTime"]).loc[:, ["Driver", "LapTime"]].copy()
    laps_wet["wet_s"] = laps_wet["LapTime"].dt.total_seconds()
    laps_dry["dry_s"] = laps_dry["LapTime"].dt.total_seconds()

    # Compute averages
    avg_wet = laps_wet.groupby("Driver")["wet_s"].mean()
    avg_dry = laps_dry.groupby("Driver")["dry_s"].mean()
    df = pd.concat([avg_wet, avg_dry], axis=1).dropna()

    # Compute performance scores
    df["diff_s"] = df["wet_s"] - df["dry_s"]
    df["pct_change"] = df["diff_s"] / df["dry_s"] * 100
    df["WetPerformanceScore"] = 1 + df["pct_change"] / 100

    # Format output with six decimal places
    formatted = {drv: float(f"{sc:.6f}") for drv, sc in df["WetPerformanceScore"].to_dict().items()}
    print(json.dumps(formatted, separators=(', ', ': ')))
    return formatted

# Alias
get_wet_score = compute_wet_performance

if __name__ == "__main__":
    # Compute for Saudi Arabia GP with default years 2019 vs 2023
    OPENWEATHER_API_KEY = ""
    compute_wet_performance(api_key=OPENWEATHER_API_KEY)
