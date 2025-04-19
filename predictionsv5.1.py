import os
import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ——— Configuration ———
API_KEY      = os.getenv("OPENWEATHER_API_KEY", "")
CACHE_DIR    = "f1_cache"
DEFAULT_TEMP = 26.0

fastf1.Cache.enable_cache(CACHE_DIR)

# ——— 1) Load & preprocess 2024 Saudi Arabia GP data ———
session = fastf1.get_session(2024, "Saudi Arabia", "R")
session.load()

laps = (
    session.laps
    .dropna(subset=["LapTime","Sector1Time","Sector2Time","Sector3Time"])
    .copy()
)
# Convert to seconds
laps["LapTime_s"]     = laps["LapTime"].dt.total_seconds()
laps["Sector1Time_s"] = laps["Sector1Time"].dt.total_seconds()
laps["Sector2Time_s"] = laps["Sector2Time"].dt.total_seconds()
laps["Sector3Time_s"] = laps["Sector3Time"].dt.total_seconds()

# Historical average lap time → AvgLapTime_s
y_df = (
    laps.groupby("Driver")["LapTime_s"]
    .mean()
    .reset_index()
    .rename(columns={"LapTime_s": "AvgLapTime_s"})
)

# Average sector times
sector_avgs = (
    laps.groupby("Driver")[["Sector1Time_s","Sector2Time_s","Sector3Time_s"]]
    .mean()
    .reset_index()
)

# ——— 2) This weekend’s qualifying data ———
qual = pd.DataFrame({
    "Driver": ["VER","PIA","RUS","NOR","LEC","ANT","TSU","SAI","GAS","HAM","ALB"],
    "QualifyingTime (s)": [87.294,87.304,87.407,87.481,87.670,87.798,87.990,88.024,88.025,88.102,88.109]
})
qual["QualifyingTime"] = qual["QualifyingTime (s)"]

# ——— 3) Wet‑weather adjustment ———
wet_scores = {
    "VER":0.97979,"NOR":0.991275,"PIA":1.00000,"LEC":1.004386,
    "RUS":0.970804,"HAM":0.952873,"GAS":0.973042,"ALO":0.963240,
    "TSU":0.97000,"SAI":0.998941,"HUL":0.991394,"OCO":0.984206,
    "STR":0.959010,"ANT":1.00000,"ALB":0.988258
}
qual["WetScore"] = qual["Driver"].map(wet_scores).fillna(1.0)

# Weather for Jeddah
try:
    w = requests.get(
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q=Jeddah&appid={API_KEY}&units=metric",
        timeout=5
    ).json()
    temp      = w.get("main",{}).get("temp", DEFAULT_TEMP)
    rain_mm   = w.get("rain",{}).get("1h", 0.0)
    rain_prob = 1.0 if rain_mm>0 else 0.0
except:
    temp, rain_prob = DEFAULT_TEMP, 0.0

print(f"🏁 Saudi GP Weather → Temp: {temp}°C, Rain Prob: {rain_prob}")

if rain_prob > 0.75:
    qual["QualifyingTime"] *= qual["WetScore"]

# ——— 4) Team performance ———
driver_to_team = {
    "VER":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","RUS":"Mercedes",
    "HAM":"Ferrari","GAS":"Alpine","ALO":"Aston Martin","TSU":"Racing Bulls",
    "SAI":"Ferrari","HUL":"Kick Sauber","OCO":"Alpine","STR":"Aston Martin",
    "ANT":"Aston Martin","ALB":"Williams"
}
team_points = {
    "McLaren":151,"Mercedes":93,"Red Bull":71,"Ferrari":57,"Haas":20,
    "Williams":19,"Aston Martin":10,"Racing Bulls":7,"Alpine":6,"Kick Sauber":6
}
maxp = max(team_points.values())
team_perf = {t: pts/maxp for t,pts in team_points.items()}

qual["Team"]            = qual["Driver"].map(driver_to_team)
qual["TeamPerformance"] = qual["Team"].map(team_perf).fillna(np.mean(list(team_perf.values())))

# ——— 5) Assemble final DataFrame ———
df = (
    qual
    .merge(sector_avgs, on="Driver", how="left")
    .merge(y_df,      on="Driver", how="left")
    .assign(Temperature=temp, RainProb=rain_prob)
    .dropna(subset=["AvgLapTime_s"])
)

# Features & target
X = df[[
    "QualifyingTime",
    "Sector1Time_s","Sector2Time_s","Sector3Time_s",
    "Temperature","RainProb","TeamPerformance"
]]
y = df["AvgLapTime_s"]

# ——— 6) Train/test split & model ———
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)
model = make_pipeline(
    SimpleImputer(strategy="mean"),
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
)
model.fit(X_train, y_train)

# ——— 7) Predict & evaluate ———
df["PredLapTime_s"] = model.predict(X)
print("\nPredicted 2025 Saudi Arabia GP Order:")
print(df.sort_values("PredLapTime_s")[["Driver","PredLapTime_s"]].to_string(index=False))
print(f"\nTest MAE: {mean_absolute_error(y_test, model.predict(X_test)):.3f} s")

# ——— 8) Plots ———
# a) Qual vs Pred
plt.figure(figsize=(8,6))
plt.scatter(df["QualifyingTime"], df["PredLapTime_s"], c="C0")
minq, maxq = df["QualifyingTime"].min(), df["QualifyingTime"].max()
plt.plot([minq,maxq],[minq,maxq],"r--")
for _,r in df.iterrows():
    plt.annotate(r.Driver, (r.QualifyingTime,r.PredLapTime_s),
                 textcoords="offset points", xytext=(5,5), fontsize=8)
plt.xlabel("Qualifying Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Qualifying vs Predicted Lap")
plt.tight_layout()
plt.show()

# b) Feature importances
fi = model.named_steps["gradientboostingregressor"].feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns, "Importance": fi
}).sort_values("Importance", ascending=False)
print("\nFeature Importances:")
print(feat_df.to_string(index=False))

plt.figure(figsize=(6,4))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
