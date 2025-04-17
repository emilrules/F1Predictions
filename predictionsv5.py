import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
API_KEY = os.getenv("OPENWEATHER_API_KEY")


# Enable cache
fastf1.Cache.enable_cache('f1_cache')

# Load 2024 Saudi GP session
session_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"], 
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.594, 92.283]
})

# Wet performance scores
driver_wet_performance = {
    "ALB": 0.988258, "ALO": 0.96324, "BOT": 1.007717, "GAS": 0.973042, "HAM": 0.952873,
    "HUL": 0.991394, "LEC": 1.004386, "MAG": 0.986487, "NOR": 0.991275, "OCO": 0.984206,
    "PER": 0.966067, "RUS": 0.970804, "SAI": 0.998941, "STR": 0.95901, "VER": 0.97979, "ZHO": 0.994705
}
qualifying_2025["WetPerformanceScore"] = qualifying_2025["Driver"].map(driver_wet_performance)

# Weather
API_KEY = ""
weather_url = f"http://api.openweathermap.org/data/2.5/weather?q=Jeddah&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
rain_probability = weather_data.get("pop", 0)
temperature = weather_data.get("main", {}).get("temp", 20)
print(f"🏁 Bahrain GP Weather → Temp: {temperature}°C, Rain Prob: {rain_probability}")

# Adjust qualifying time if rain is likely
if rain_probability > 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceScore"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Team performance
team_points = {
    "McLaren": 151, "Mercedes": 93, "Red Bull": 71, "Ferrari": 57, "Haas": 20,
    "Williams": 19, "Aston Martin": 10, "Racing Bulls": 7, "Alpine": 6, "Kick Sauber": 6
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}



driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Merge features
merged = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged["RainProbability"] = rain_probability
merged["Temperature"] = temperature

# Feature matrix and target
X = merged[["QualifyingTime (s)", "RainProbability", "Temperature", "TeamPerformanceScore"]].fillna(0)
y_reference = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
final = merged.merge(y_reference, on="Driver", how="left").dropna(subset=["LapTime (s)"])
X = final[["QualifyingTime (s)", "RainProbability", "Temperature", "TeamPerformanceScore"]]
y = final["LapTime (s)"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)
final["PredictedRaceTime (s)"] = model.predict(X)

# Results
final = final.sort_values("PredictedRaceTime (s)")
print("Predicted 2025 Saudi Arabia GP Winner:")
print(final[["Driver", "PredictedRaceTime (s)"]])

# MAE
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Scatter Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    final["TeamPerformanceScore"],
    final["PredictedRaceTime (s)"],
    c=final["QualifyingTime"],
    cmap="viridis"
)
for i, driver in enumerate(final["Driver"]):
    plt.annotate(driver, (final["TeamPerformanceScore"].iloc[i], final["PredictedRaceTime (s)"].iloc[i]), 
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.colorbar(scatter, label="Qualifying Time")
plt.xlabel("Team Performance Score")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Team Performance on Predicted Race Results")
plt.tight_layout()
plt.savefig("team_performance_effect.png")
plt.show()

# Feature Importance Plot
plt.figure(figsize=(10, 5))
plt.barh(X.columns, model.feature_importances_, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()
