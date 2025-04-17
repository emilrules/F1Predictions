import os
import fastf1
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
fastf1.Cache.enable_cache('f1_cache')
DEFAULT_TEMP = 26.0

# 1) Load 2024 Bahrain GP session and compute avg lap & sector times
session = fastf1.get_session(2024, "Bahrain", "R")
session.load()

laps = session.laps.dropna(subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]).copy()
laps["LapTime_s"]      = laps["LapTime"].dt.total_seconds()
laps["Sector1Time_s"]  = laps["Sector1Time"].dt.total_seconds()
laps["Sector2Time_s"]  = laps["Sector2Time"].dt.total_seconds()
laps["Sector3Time_s"]  = laps["Sector3Time"].dt.total_seconds()

# average complete lap
y_df = laps.groupby("Driver")["LapTime_s"].mean().reset_index()

# average sectors
sector_avgs = (
    laps
    .groupby("Driver")[['Sector1Time_s','Sector2Time_s','Sector3Time_s']]
    .mean()
    .reset_index()
)

# 2) 2025 Bahrain qualifying times
qual = pd.DataFrame({
    "Driver": ["VER","NOR","PIA","LEC","RUS","HAM","GAS","ALO","TSU","SAI","HUL","OCO","STR"],
    "QualifyingTime": [90.423,90.267,89.841,90.175,90.009,90.772,90.216,91.886,91.303,90.680,92.067,91.594,92.283]
})

# 3) Wet-performance adjustment
wet_scores = {
    "VER":0.97979,"NOR":0.991275,"PIA":1.000,"LEC":1.004386,"RUS":0.970804,
    "HAM":0.952873,"GAS":0.973042,"ALO":0.963240,"TSU":0.970,"SAI":0.998941,
    "HUL":0.991394,"OCO":0.984206,"STR":0.959010
}
qual["WetScore"] = qual["Driver"].map(wet_scores)

# 4) Race-day weather at Sakhir
resp = requests.get(
    f"https://api.openweathermap.org/data/2.5/weather?q=Sakhir,BH&appid={API_KEY}&units=metric",
    timeout=10
).json()
temp      = resp.get("main", {}).get("temp", DEFAULT_TEMP)
rain_mm   = resp.get("rain", {}).get("1h", 0.0)
rain_prob = 1.0 if rain_mm > 0 else 0.0
print(f"🏁 Bahrain GP Weather → Temp: {temp}°C, Rain Prob: {rain_prob}")

if rain_prob > 0.75:
    qual["QualifyingTime"] *= qual["WetScore"]

# 5) Merge all features: qualifying, sectors, and historical lap
final = (
    qual
    .merge(sector_avgs, on="Driver", how="inner")
    .merge(y_df, on="Driver", how="inner")
)

# 6) Baseline prediction: use qualifying time directly
final["PredLap_s"] = final["QualifyingTime"]

# 7) Output results
predicted = final.sort_values("PredLap_s")[["Driver","PredLap_s"]]
print("\nPredicted 2025 Bahrain GP Order (baseline = qualifying):")
print(predicted.to_string(index=False))

# 8) Report baseline MAE vs 2024 avg lap
mae = mean_absolute_error(final["LapTime_s"], final["PredLap_s"])
print(f"\nBaseline MAE vs 2024 avg lap: {mae:.3f} seconds")

# 9) Show the sector times alongside for reference
print("\nDriver | Sector1_s | Sector2_s | Sector3_s")
print(final[["Driver","Sector1Time_s","Sector2Time_s","Sector3Time_s"]].to_string(index=False))

# 10) Plots
# 10a) Scatter: Qualifying vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(final["QualifyingTime"], final["PredLap_s"], c='blue')
for i, d in enumerate(final["Driver"]):
    plt.annotate(d, (final["QualifyingTime"].iloc[i], final["PredLap_s"].iloc[i]), textcoords="offset points", xytext=(5,5), fontsize=8)
plt.plot([final["QualifyingTime"].min(), final["QualifyingTime"].max()],
         [final["QualifyingTime"].min(), final["QualifyingTime"].max()], 'r--')
plt.xlabel("Qualifying Time (s)")
plt.ylabel("Predicted Race Lap (s)")
plt.title("Qualifying vs Predicted Race Lap")
plt.tight_layout()
plt.show()

# 10b) Bar chart: Avg Sector Times
sector_df = final.set_index('Driver')[['Sector1Time_s','Sector2Time_s','Sector3Time_s']]
sector_df.plot(kind='bar', figsize=(10,6))
plt.ylabel('Time (s)')
plt.title('2024 Bahrain GP Avg Sector Times by Driver')
plt.tight_layout()
plt.show()
