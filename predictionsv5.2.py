import os
import fastf1
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    RandomizedSearchCV,
    train_test_split
)
from sklearn.metrics import mean_absolute_error

from data.driver_performance import get_driver_performance
from data.car_performance import get_car_performance

# --- Config & Cache ---
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
fastf1.Cache.enable_cache("f1_cache")

# --- 1) Load 2024 Race Data ---
session = fastf1.get_session(2024, "Saudi Arabia", "R")
session.load()
laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna()

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps[f"{col} (s)"] = laps[col].dt.total_seconds()

sector_times = (
    laps.groupby("Driver")[
        ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
    ]
    .mean()
    .reset_index()
)
sector_times["TotalSectorTime (s)"] = (
    sector_times["Sector1Time (s)"]
    + sector_times["Sector2Time (s)"]
    + sector_times["Sector3Time (s)"]
)

# --- 2) Qualifying Data & Wet Adjustment ---
qual = pd.DataFrame({
    "Driver": ["VER","PIA","RUS","NOR","LEC","ANT","TSU","SAI","GAS","HAM","ALB"],
    "QualifyingTime (s)": [87.294,87.304,87.407,87.481,87.670,87.798,
                           87.990,88.024,88.025,88.102,88.109]
})

wet_perf = {
    "ALB":0.988258,"ALO":0.963240,"BOT":1.007717,"GAS":0.973042,
    "HAM":0.952873,"HUL":0.991394,"LEC":1.004386,"MAG":0.986487,
    "NOR":0.991275,"OCO":0.984206,"PER":0.966067,"RUS":0.970804,
    "SAI":0.998941,"STR":0.959010,"VER":0.979790,"ZHO":0.994705
}
qual["WetPerformanceScore"] = qual["Driver"].map(wet_perf)

if API_KEY:
    resp = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q=Jeddah&appid={API_KEY}&units=metric"
    ).json()
    rain = resp.get("pop", 0.0)
    temp = resp.get("main", {}).get("temp", 20.0)
else:
    rain, temp = 0.0, 20.0

qual["QualifyingTime"] = np.where(
    rain > 0.75,
    qual["QualifyingTime (s)"] * qual["WetPerformanceScore"],
    qual["QualifyingTime (s)"]
)

# --- 3) Driver & Car Ratings ---
qual["DriverPerf"] = qual["Driver"].map(get_driver_performance)
qual["CarPerf"]    = qual["Driver"].map(get_car_performance)

# --- 4) Merge Features & Build X, y ---
df = (
    qual
    .merge(sector_times[["Driver","TotalSectorTime (s)"]], on="Driver", how="left")
    .assign(RainProbability=rain, Temperature=temp)
)

# Target: average lap time in seconds
y_ref = laps.groupby("Driver")["LapTime (s)"].mean().reset_index()
df = df.merge(y_ref, on="Driver", how="left").dropna(subset=["LapTime (s)"])

FEATURES = [
    "QualifyingTime",
    "RainProbability",
    "Temperature",
    "TotalSectorTime (s)",
    "DriverPerf",
    "CarPerf"
]
X = df[FEATURES]
y = df["LapTime (s)"]

# --- 5) 5‑Fold CV for Baseline MAE ---
pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
)
kf = KFold(n_splits=5, shuffle=True, random_state=38)
neg_maes = cross_val_score(
    pipeline, X, y,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
maes = -neg_maes
print(f"5‑Fold CV MAE: {maes.mean():.3f} ± {maes.std():.3f} s")

# --- 6) Hyperparameter Tuning (RandomizedSearchCV) ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=38
)
param_dist = {
    "gradientboostingregressor__n_estimators": [100, 200, 300],
    "gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1],
    "gradientboostingregressor__max_depth": [3, 5, 7]
}
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=10,
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=38,
    n_jobs=-1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best params:", search.best_params_)
print(f"Best CV MAE: {-search.best_score_:.3f} s")

# Hold‑out evaluation
y_pred_val = best_model.predict(X_val)
print(f"Hold‑out MAE: {mean_absolute_error(y_val, y_pred_val):.3f} s")

# --- 7) Final Predictions & Plot ---
df["PredictedLapTime (s)"] = best_model.predict(X)
df["PredictedRaceTime (s)"] = df["PredictedLapTime (s)"] * 53  # e.g. 53 laps

print("Predicted 2025 Saudi Arabia GP Order:")
print(df.sort_values("PredictedRaceTime (s)")[["Driver","PredictedRaceTime (s)"]])

plt.figure(figsize=(10,6))
plt.scatter(df["QualifyingTime"], df["PredictedLapTime (s)"])
for i, drv in enumerate(df["Driver"]):
    plt.annotate(drv, (df["QualifyingTime"].iloc[i], df["PredictedLapTime (s)"].iloc[i]))
plt.xlabel("Qualifying Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Qualifying vs. Predicted Lap Time")
plt.tight_layout()
plt.show()
