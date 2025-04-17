import os
import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load env
API_KEY = os.getenv("OPENWEATHER_API_KEY")
fastf1.Cache.enable_cache('f1_cache')
DEFAULT_TEMP = 26.0

# 1) Load 2024 Bahrain GP data
session = fastf1.get_session(2024, "Bahrain", "R")
session.load()
laps = session.laps.dropna(subset=["LapTime","Sector1Time","Sector2Time","Sector3Time"]).copy()
laps["LapTime_s"]     = laps["LapTime"].dt.total_seconds()
laps["Sector1Time_s"] = laps["Sector1Time"].dt.total_seconds()
laps["Sector2Time_s"] = laps["Sector2Time"].dt.total_seconds()
laps["Sector3Time_s"] = laps["Sector3Time"].dt.total_seconds()

y_df = laps.groupby("Driver")["LapTime_s"].mean().reset_index()
sector_avgs = laps.groupby("Driver")[['Sector1Time_s','Sector2Time_s','Sector3Time_s']].mean().reset_index()

# 2) 2025 Bahrain qualifying
qual = pd.DataFrame({
    "Driver":["VER","NOR","PIA","LEC","RUS","HAM","GAS","ALO","TSU","SAI","HUL","OCO","STR"],
    "QualifyingTime":[90.423,90.267,89.841,90.175,90.009,90.772,90.216,91.886,91.303,90.680,92.067,91.594,92.283]
})

# 3) Wet adjust
wet = {"VER":0.97979,"NOR":0.991275,"PIA":1.0,"LEC":1.004386,"RUS":0.970804,
       "HAM":0.952873,"GAS":0.973042,"ALO":0.96324,"TSU":0.97,"SAI":0.998941,
       "HUL":0.991394,"OCO":0.984206,"STR":0.95901}
qual['WetScore'] = qual['Driver'].map(wet)

# 4) Weather
resp = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q=Sakhir,BH&appid={API_KEY}&units=metric",timeout=10).json()
temp      = resp.get('main',{}).get('temp',DEFAULT_TEMP)
rain_mm   = resp.get('rain',{}).get('1h',0.0)
rain_prob = 1.0 if rain_mm>0 else 0.0
print(f"🏁 Bahrain GP Weather → Temp: {temp}°C, Rain Prob: {rain_prob}")
if rain_prob>0.75:
    qual['QualifyingTime'] *= qual['WetScore']

# 5) Merge
final = qual.merge(sector_avgs,on='Driver').merge(y_df,on='Driver')

# 6) Team score + model
driver_to_team={"VER":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari","RUS":"Mercedes",
                "HAM":"Ferrari","GAS":"Alpine","ALO":"Aston Martin","TSU":"Racing Bulls",
                "SAI":"Ferrari","HUL":"Kick Sauber","OCO":"Alpine","STR":"Aston Martin"}
team_pts={"McLaren":151,"Mercedes":93,"Red Bull":71,"Ferrari":57,"Haas":20,"Williams":19,
          "Aston Martin":10,"Racing Bulls":7,"Alpine":6,"Kick Sauber":6}
maxp=max(team_pts.values());team_perf={t:p/maxp for t,p in team_pts.items()}
final['TeamPerf']=final['Driver'].map(driver_to_team).map(team_perf)

# features & train
final['Temp']=temp; final['RainProb']=rain_prob
X = final[['QualifyingTime','Sector1Time_s','Sector2Time_s','Sector3Time_s','Temp','RainProb','TeamPerf']]
y = final['LapTime_s']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=38)
model=GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,random_state=38)
model.fit(X_train,y_train)
final['PredLap']=model.predict(X)

# results
print("\nPredicted 2025 Bahrain GP Order:")
print(final.sort_values('PredLap')[['Driver','PredLap']].to_string(index=False))
print(f"\nMAE: {mean_absolute_error(y_test,model.predict(X_test)):.3f} s")

# 7) Plots
# a) Qual vs Pred
plt.figure(figsize=(8,6));plt.scatter(final['QualifyingTime'],final['PredLap'],c='blue')
plt.plot([final['QualifyingTime'].min(),final['QualifyingTime'].max()],[final['QualifyingTime'].min(),final['QualifyingTime'].max()],'r--')
for i,d in enumerate(final['Driver']): plt.annotate(d,(final['QualifyingTime'].iloc[i],final['PredLap'].iloc[i]),xytext=(5,5),textcoords='offset points',fontsize=8)
plt.xlabel('Qualifying Time (s)');plt.ylabel('Predicted Lap (s)');plt.title('Qual vs Pred Lap');plt.tight_layout();plt.show()

# b) Avg sectors
sector_df=final.set_index('Driver')[['Sector1Time_s','Sector2Time_s','Sector3Time_s']]
sector_df.plot(kind='bar',figsize=(10,6));plt.ylabel('Time (s)');plt.title('Avg Sector Times (2024)');plt.tight_layout();plt.show()

# c) Team vs Pred
plt.figure(figsize=(12,8));sc=plt.scatter(final['TeamPerf'],final['PredLap'],c=final['QualifyingTime'],cmap='viridis')
for i,d in enumerate(final['Driver']): plt.annotate(d,(final['TeamPerf'].iloc[i],final['PredLap'].iloc[i]),xytext=(5,5),textcoords='offset points',fontsize=8)
plt.colorbar(sc,label='Qual Time');plt.xlabel('Team Perf');plt.ylabel('Pred Lap');plt.title('Team Perf vs Pred');plt.tight_layout();plt.show()

# d) Importance
plt.figure(figsize=(10,5));plt.barh(X.columns,model.feature_importances_,color='skyblue')
plt.xlabel('Importance');plt.title('Feature Importance');plt.tight_layout();plt.show()
