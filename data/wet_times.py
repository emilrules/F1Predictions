import fastf1
import pandas as pd

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# Load the 2023 Canadian GP session data (wet race)
session_2023 = fastf1.get_session(2023, "Saudi Arabia", 'R')
session_2023.load()

# Load the 2022 Canadian GP session data (dry race)
session_2022 = fastf1.get_session(2022, "Saudi Arabia", 'R')
session_2022.load()

# Extract the lap data for both years
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()
laps_2022 = session_2022.laps[["Driver", "LapTime"]].copy()

# Drop NaN values
laps_2023.dropna(subset=["LapTime"], inplace=True)
laps_2022.dropna(subset=["LapTime"], inplace=True)

# Convert LapTime to seconds
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()
laps_2022["LapTime (s)"] = laps_2022["LapTime"].dt.total_seconds()

# Calculate the average lap time for each driver in both races
avg_lap_2023 = laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_lap_2022 = laps_2022.groupby("Driver")["LapTime (s)"].mean().reset_index()

# Merge the data from both races on 'Driver'
merged_data = pd.merge(
    avg_lap_2023,
    avg_lap_2022,
    on="Driver",
    suffixes=("_2023", "_2022")
)

# Calculate the performance difference in lap time between the two years
merged_data["LapTimeDiff (s)"] = (
    merged_data["LapTime (s)_2023"] 
    - merged_data["LapTime (s)_2022"]
)

# Calculate the percentage difference
merged_data["PercentageChange (%)"] = (
    merged_data["LapTimeDiff (s)"] 
    / merged_data["LapTime (s)_2022"] 
    * 100
)

# Now create the wet performance score
merged_data["WetPerformanceScore"] = 1 + (merged_data["PercentageChange (%)"] / 100)

# Round to 6 decimals and convert to dictionary
formatted = {row["Driver"]: round(row["WetPerformanceScore"], 6) for _, row in merged_data.iterrows()}

# Print in your desired format
for driver, score in formatted.items():
    print(f'"{driver}": {score},')

