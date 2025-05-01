# 🏁 F1 Race Predictor 2025

Welcome to **F1 Race Predictor 2025** — a machine learning–powered project that uses real-time qualifying data, historical race insights, and live weather to forecast race results for the 2025 Formula 1 season.

## 📦 About the Project

This project trains a **Gradient Boosting Regressor** using historical race data and qualifying results to simulate race outcomes for each Grand Prix on the 2025 calendar. Using the FastF1 API, we aggregate sector times, lap averages, constructor performance, and live weather conditions to build predictive features.

### Key Features:
- Lap and sector time processing from the FastF1 API
- Constructor strength based on current Constructors' Championship standings
- Weather API integration for rain and temperature features
- Dynamic rain adjustment (if chance of rain > 75%)
- Mean Absolute Error (MAE) used to evaluate model accuracy

## 📂 How It Works

1. **Data Retrieval** – Pulls race and qualifying session data using FastF1.
2. **Feature Engineering** – Combines timing, weather, and team statistics.
3. **Model Training** – Fits a regression model to lap time targets.
4. **Predictions** – Forecasts predicted lap times and sorts drivers accordingly.
5. **Visualizations** – Shows team performance influence and feature importance.

## 🔌 Requirements

Install the following packages:

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib requests
```

Store your OpenWeatherMap API key as an environment variable:

```powershell
set OPENWEATHER_API_KEY=your_actual_key_here
```

## 🛠️ Running a Prediction

Run the prediction file for the specific race. Each race file is named according to its number in the 2025 calendar.

```bash
python3 predictionsv1.py  # e.g. for Australia
python3 predictionsv2.py  # e.g. for China
```

### Example Output

```
Predicted 2025 GP Winner:
Driver    PredictedRaceTime (s)
VER       90.221
NOR       90.735
...
Mean Absolute Error: 2.91 seconds
```

## 📊 Visuals

- `team_performance_effect.png`: Impact of team strength on predicted results
- Feature importance bar chart: Understand which factors influenced the prediction

## 💪 Working Versions
```
- predictionsv6 coming up for the Milan race
- all versions of predictionsv5.py (Predicted VER, but didnt win due to penalty) 
- predictionsv4.py
```

## 🔮 Planned Features

- Add pit stop strategies and tire degradation factors
- Integrate wind and humidity into weather features
- Model updates race-by-race as the 2025 season progresses
- Experiment with ensemble and neural network architectures

## 🤝 Credits

Built independently but inspired by [@mar-antaya](https://github.com/mar-antaya/2025_f1_predictions.git). Data sources: FastF1 and OpenWeatherMap.

---

🏎 Built for speed, stats, and simulation.

