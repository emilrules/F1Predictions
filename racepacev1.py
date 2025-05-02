import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ——— Configuration ———
fastf1.Cache.enable_cache('f1_cache')
track_length_km   = 6.174    # Length of the track in km
target_compound   = 'Soft'   # e.g. 'Soft', 'Medium', 'Hard'
time_delta_thresh = 5.0      # Min seconds gap to leader ⇒ clean air

# ——— 1) Load session & laps ———
session = fastf1.get_session(2024, "Saudi Arabia", "R")
session.load()

laps = (
    session.laps
    .dropna(subset=["LapTime","Sector1Time","Sector2Time","Sector3Time"])
    .copy()
)

# Print driver numbers as a Python list
driver_nums = laps["DriverNumber"].unique().tolist()
print(driver_nums)

# Print available lap columns as a pandas Index (shows dtype='object')
print(laps.columns)

# ——— 2) Convert times to seconds ———
for col in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
    laps[f"{col}_s"] = laps[col].dt.total_seconds()

# ——— 3) Compute TimeDelta (gap to leader at lap start) ———
laps["TimeDelta"] = (
    laps["LapStartTime"]
      - laps.groupby("LapNumber")["LapStartTime"].transform("min")
).dt.total_seconds()

# ——— 4) Filter race‑pace laps ———
race_laps = laps[
    (laps["LapNumber"] > 1) &
    (laps["IsAccurate"] == True) &
    (laps["Compound"] == target_compound) &
    (laps["TimeDelta"] >= time_delta_thresh)
]

if race_laps.empty:
    print("Race laps are empty, relaxing the conditions.")
    race_laps = laps[
        (laps["LapNumber"] > 1) &
        (laps["IsAccurate"] == True)
    ]

print(f"Found {len(race_laps)} valid race laps.")

# ——— 5) Compute average race pace & normalize ———
avg_race_pace    = race_laps.groupby("Driver")["LapTime_s"].mean().sort_values()
normalized_pace  = avg_race_pace / track_length_km

race_pace_df = pd.DataFrame({
    "Average Race Pace (s)":      avg_race_pace,
    "Normalized Race Pace (s/km)": normalized_pace
})

print("\n=== Race Pace Results ===")
print(race_pace_df.to_string())

# ——— 6) Plot normalized pace ———
plt.figure(figsize=(8, 4))
race_pace_df["Normalized Race Pace (s/km)"].plot.bar()
plt.ylabel("Seconds per km")
plt.title(f"Avg Race Pace on {target_compound} Tyres (Clean Air)")
plt.tight_layout()
plt.show()
