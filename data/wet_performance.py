import os
import fastf1
import pandas as pd

def compute_wet_performance(wet_year, dry_year, track, race_code):
    """
    Loads race sessions for the specified wet and dry conditions,
    computes average lap times, and returns a performance dictionary.

    Args:
        wet_year (int): Year of the wet race session.
        dry_year (int): Year of the dry race session.
        track (str): Track name (e.g., "Japan").
        race_code (str): Session code (e.g., "R" for Race).

    Returns:
        dict: Mapping of driver names to their computed WetPerformanceScore.
    """
    cache_dir = r"Z:\Coding Projects\F1Predictions\f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)

    print(f"Loading {wet_year} wet session for {track}...")
    wet_session = fastf1.get_session(wet_year, track, race_code)
    wet_session.load()

    print(f"Loading {dry_year} dry session for {track}...")
    dry_session = fastf1.get_session(dry_year, track, race_code)
    dry_session.load()

    laps_wet = wet_session.laps[["Driver", "LapTime"]].copy()
    laps_dry = dry_session.laps[["Driver", "LapTime"]].copy()
    laps_wet.dropna(inplace=True)
    laps_dry.dropna(inplace=True)
    laps_wet["LapTime (s)"] = laps_wet["LapTime"].dt.total_seconds()
    laps_dry["LapTime (s)"] = laps_dry["LapTime"].dt.total_seconds()

    avg_lap_wet = laps_wet.groupby("Driver")["LapTime (s)"].mean().reset_index()
    avg_lap_dry = laps_dry.groupby("Driver")["LapTime (s)"].mean().reset_index()
    merged = pd.merge(avg_lap_wet, avg_lap_dry, on="Driver", suffixes=("_wet", "_dry"))
    merged["LapTimeDifference (s)"] = merged["LapTime (s)_wet"] - merged["LapTime (s)_dry"]
    merged["PerformanceChange (%)"] = (merged["LapTimeDifference (s)"] / merged["LapTime (s)_dry"]) * 100
    merged["WetPerformanceScore"] = 1 + (merged["PerformanceChange (%)"] / 100)

    print("\nComputed Wet Performance Scores:")
    print(merged[["Driver", "WetPerformanceScore"]])
    performance_dict = dict(zip(merged["Driver"], merged["WetPerformanceScore"]))
    return performance_dict

# Create an alias for easier import
get_wet_score = compute_wet_performance
