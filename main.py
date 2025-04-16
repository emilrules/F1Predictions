from utils.config_reader import load_config
from data.qualifying_data import get_qualifying_data
from data.load_session import get_sector_times_and_laps
from utils.preprocess import build_feature_dataframe
from utils.formatters import format_time_gap
from pipeline import get_model_pipeline
import fastf1
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

def main():
    # Load configuration from config.json
    config = load_config()
    print(f"Running predictions for Race Year: {config['race_year']}, Track: {config['track']}, Session Code: {config['session_code']}")
    
    # Fetch qualifying data (e.g., manually entered or FastF1-based)
    qual_data = get_qualifying_data(config["qualifying_year"], config["track"])
    print("Fetched Qualifying Data:")
    print(qual_data)
    
    # NOTE: We are no longer enriching qualifying data with driver/car/wet performance or weather features.
    # Instead, we rely solely on the qualifying times and historical session data.
    
    # Load historical session data (lap and sector times)
    sector_times, avg_laptimes, compound_data = get_sector_times_and_laps(
        config["historical_year"], config["track"], config["session_code"]
    )
    
    # Build feature matrix and target variable from qualifying data and historical session data.
    # The build_feature_dataframe function should now rely primarily on:
    # - "QualifyingTime (s)" from qual_data, and
    # - historical lap and sector times (e.g. avg_laptimes, sector_times)
    X, y = build_feature_dataframe(qual_data, sector_times, avg_laptimes, compound_data)
    
    # Data augmentation (optional)
    X_expanded = X.copy()
    y_expanded = y.copy()
    n_aug = 5
    for _ in range(n_aug):
        noise = pd.DataFrame(np.random.normal(0, 0.001, X.shape), columns=X.columns, index=X.index)
        X_aug = X + noise
        y_aug = y * (1 + np.random.normal(0, 0.005, len(y)))
        X_expanded = pd.concat([X_expanded, X_aug], ignore_index=True)
        y_expanded = pd.concat([y_expanded, pd.Series(y_aug)], ignore_index=True)
    
    # Model training using GridSearchCV
    pipeline = get_model_pipeline()
    param_grid = {
        'model__n_estimators': [200, 300, 400],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_expanded, y_expanded)
    print("Best parameters:", grid_search.best_params_)
    print("Best MAE:", -grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    
    # Make predictions using the best model
    predictions = best_model.predict(X)
    lap_count = 53  # Adjust race length as needed
    qual_data["PredictedLapTime (s)"] = predictions
    qual_data["PredictedRaceTime (s)"] = predictions * lap_count
    
    # Calculate gaps and format them for display
    qual_data = qual_data.sort_values(by="PredictedRaceTime (s)")
    qual_data["Gap (s)"] = qual_data["PredictedRaceTime (s)"] - qual_data["PredictedRaceTime (s)"].min()
    qual_data["Gap"] = qual_data["Gap (s)"].apply(lambda x: "" if x < 0.001 else format_time_gap(x))
    
    print("\nPredicted Race Results:\n")
    print(qual_data[["Driver", "PredictedLapTime (s)", "Gap"]].to_string(index=False))
    
    # Evaluate the model using a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=38)
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nModel Mean Absolute Error: {mae:.4f} seconds")

if __name__ == '__main__':
    main()
