import fastf1
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load API key (if available)
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

# --- 1) Load 2024 Bahrain GP session ---
session_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_2024.load()
laps_2024 = session_2024.laps[
    ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
].dropna().copy()

# Convert to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times
sector_times_2024 = (
    laps_2024
    .groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
    .mean()
    .reset_index()
)
sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"]
  + sector_times_2024["Sector2Time (s)"]
  + sector_times_2024["Sector3Time (s)"]
)

# --- 2) This weekend’s qualifying data ---
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER","PIA","RUS","NOR","LEC","ANT","TSU","SAI","GAS","HAM","ALB"],
    "QualifyingTime (s)": [
        87.294, 87.304, 87.407, 87.481, 87.670,
        87.798, 87.990, 88.024, 88.025, 88.102, 88.109
    ]
})

# Wet performance scores (ensure all codes present)
driver_wet_performance = {
    "ALB":0.988258, "ALO":0.963240, "BOT":1.007717, "GAS":0.973042,
    "HAM":0.952873, "HUL":0.991394, "LEC":1.004386, "MAG":0.986487,
    "NOR":0.991275, "OCO":0.984206, "PER":0.966067, "RUS":0.970804,
    "SAI":0.998941, "STR":0.959010, "VER":0.979790, "ZHO":0.994705,
    # default for missing:
    "ANT":1.0
}
qualifying_2025["WetPerformanceScore"] = qualifying_2025["Driver"] \
    .map(driver_wet_performance) \
    .fillna(1.0)

# --- 3) Weather for Bahrain ---
weather_url = (
    "http://api.openweathermap.org/data/2.5/weather"
    f"?q=Manama&appid={API_KEY}&units=metric"
)
resp = requests.get(weather_url).json()
rain_probability = resp.get("pop", 0.0)
temperature      = resp.get("main", {}).get("temp", 20.0)
print(f"🏁 Bahrain GP Weather → Temp: {temperature}°C, Rain Prob: {rain_probability}")

# Adjust qualifying times if wet
qualifying_2025["QualifyingTime"] = np.where(
    rain_probability > 0.75,
    qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceScore"],
    qualifying_2025["QualifyingTime (s)"]
)

# --- 4) Team performance (unchanged) ---
team_points = {
    "McLaren":151,"Mercedes":93,"Red Bull":71,"Ferrari":57,"Haas":20,
    "Williams":19,"Aston Martin":10,"Racing Bulls":7,"Alpine":6,"Kick Sauber":6
}
max_pts = max(team_points.values())
team_performance_score = {team:pts/max_pts for team,pts in team_points.items()}

driver_to_team = {
    "VER":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","RUS":"Mercedes",
    "HAM":"Ferrari","GAS":"Alpine","ALO":"Aston Martin","TSU":"Racing Bulls",
    "SAI":"Ferrari","HUL":"Kick Sauber","OCO":"Alpine","STR":"Aston Martin",
    "ANT":"Aston Martin","ALB":"Williams"
}
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"] \
    .map(team_performance_score) \
    .fillna(np.mean(list(team_performance_score.values())))

# --- 5) Merge features & target ---
merged = qualifying_2025.merge(
    sector_times_2024[["Driver","TotalSectorTime (s)"]],
    on="Driver", how="left"
).assign(
    RainProbability=rain_probability,
    Temperature=temperature
)

# Historical avg lap time
y_ref = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
final = merged.merge(y_ref, on="Driver", how="left").dropna(subset=["LapTime (s)"])

# Feature matrix & target
X = final[["QualifyingTime","RainProbability","Temperature","TeamPerformanceScore","TotalSectorTime (s)"]]
y = final["LapTime (s)"]

# Handle any remaining NaNs with an imputer pipeline
model = make_pipeline(
    SimpleImputer(strategy="mean"),
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)

model.fit(X_train, y_train)

# --- 6) Predict & evaluate ---
final["PredictedRaceTime (s)"] = model.predict(X)
print("Predicted 2025 Bahrain GP Winner:")
print(final.sort_values("PredictedRaceTime (s)")[["Driver","PredictedRaceTime (s)"]])
print(f"Mean Absolute Error: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

# --- 7) Plots ---
fi = model.named_steps["gradientboostingregressor"].feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": fi
}).sort_values("Importance", ascending=False)
print("\nFeature importances:")
print(feat_df.to_string(index=False))

plt.figure(figsize=(6,4))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.scatter(final["TeamPerformanceScore"], final["PredictedRaceTime (s)"],
            c=final["QualifyingTime"], cmap="viridis")
for i, drv in enumerate(final["Driver"]):
    plt.annotate(drv,
                 (final["TeamPerformanceScore"].iloc[i],
                  final["PredictedRaceTime (s)"].iloc[i]),
                 textcoords="offset points", xytext=(5,5), fontsize=8)
plt.colorbar(label="Qualifying Time")
plt.xlabel("Team Performance Score")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Team Performance on Predicted Race Results")
plt.tight_layout()
plt.show()
