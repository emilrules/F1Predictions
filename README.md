# 🏁 F1 Race Predictor 2025

Welcome to **F1 Race Predictor 2025** — a machine learning–powered project that uses real-time qualifying data, historical race insights, and live weather to forecast race results for the 2025 Formula 1 season.

## 📦 About the Project

This project trains a **Gradient Boosting Regressor** on 2024 Saudi Arabian Grand Prix data and 2025 qualifying results to simulate race outcomes. Using the FastF1 API, we aggregate sector times, lap averages, and constructor performance to build features that represent real-world driver and team dynamics.

### Key Features:
- Lap and sector time processing from the FastF1 API
- Constructor strength using 2025 points to date
- Live weather integration (via OpenWeatherMap API)
- Optional wet performance multiplier (if rain chance > 75%)
- Mean Absolute Error (MAE) for model accuracy

## 📂 How It Works

1. **Session Data Loading** – Pulls 2024 Saudi Arabia race data using FastF1.
2. **Feature Assembly** – Combines 2025 qualifying results, team stats, and weather.
3. **Model Training** – Fits a Gradient Boosting Regressor to lap time targets.
4. **Predictions** – Outputs predicted race finish times, sorted by driver.
5. **Visualization** – Generates performance-impact and feature-importance plots.

## 🔌 Requirements

Make sure you have the following libraries installed:

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib requests
```

You'll also need an OpenWeatherMap API key, store it locally through powershell

```
set OPENWEATHER_API_KEY=your_actual_key_here
```

## 🛠️ Running a Prediction

```bash
python3 predictionv#
```

### Example Output

```
Predicted 2025 Saudi Arabia GP Winner:
Driver    PredictedRaceTime (s)
VER       90.221
NOR       90.735
...
Mean Absolute Error: 2.91 seconds
```

## 📊 Visuals

- **`team_performance_effect.png`**: Scatter plot of team score vs. predicted time
- **Bar chart**: Relative importance of each feature used in the model

## 🔮 What’s Next

- Add pit stop strategy data for longer races
- Experiment with weather simulation and wind factor
- Extend to all 2025 GP races as the calendar progresses
- Explore alternative models like XGBoost or LSTM for longer-term learning

## 🤝 Credits

This project is independently built but inspired by [@mar-antaya](https://github.com/mar-antaya/2025_f1_predictions.git)'s open-source repository and forecasting approach. Data sources include FastF1 and OpenWeatherMap.

---

🏎 Built with love for data, racing, and ridiculous sector 3 exits.
