import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


fastf1.Cache.enable_cache('f1_cache')

#Load 2024 Jeddah Session 
session_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_2024.load()
laps_2024 = session_2024.laps[{"Driver", "LapTime", "Sectior1Time", "Sector2Time", "Sector3Time"}].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sectior1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f" {col} (s)"] = laps_2024[col].dt.total_seconds()

#Aggregate the lap times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg((
    "Sectror1Time (s)", "mean",
    "Sector2Time (s)", "mean",
    "Sector3Time (s)", "mean",
)).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sectror1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

#2025 Bahrain GP quaili data (we need to change this before Jeddah)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"], 
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.594, 92.283]
})

# Wet Performance from the script
driver_wer_performance = {
    
}
