# utils/preprocess.py
def build_feature_dataframe(qual_df, sector_times, avg_laptimes, compound_pivot):
    """
    Merge qualifying data with historical session data and compute additional features.
    
    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the target variable.
    """
    # Merge qualifying data with sector times on 'Driver'
    merged = qual_df.merge(sector_times, left_on="Driver", right_index=True, how="left")

    # Merge historical lap times if available
    if not avg_laptimes.empty:
        merged = merged.merge(avg_laptimes, left_on="Driver", right_index=True, how="left")

    # Merge compound-specific data if available
    if not compound_pivot.empty:
        merged = merged.merge(compound_pivot, left_on="Driver", right_index=True, how="left")
    
    # Compute additional performance features
    merged["DriverCarCombined"] = merged["DriverPerformanceRating"] * 0.6 + merged["CarPerformanceRating"] * 0.4
    merged["QualifyingRank"] = merged["QualifyingTime (s)"].rank()
    merged["QualifyingPerformance"] = 1 - ((merged["QualifyingRank"] - 1) / len(merged))
    merged["WetImpact"] = merged["WetPerformanceScore"] * merged["RainProbability"]
    merged["QualToRaceRatio"] = merged["QualifyingPerformance"] * 0.7 + merged["DriverCarCombined"] * 0.3

    # Define target variable:
    # If historical lap time (LapTime (s)) exists, use it; fill missing entries with estimated value.
    if "LapTime (s)" in merged.columns:
        y = merged["LapTime (s)"].copy()
        y.fillna(merged["QualifyingTime (s)"] * 1.025, inplace=True)
    else:
        y = merged["QualifyingTime (s)"] * 1.025

    # Select the features you want to include.
    feature_cols = [
        "QualifyingTime (s)",
        "DriverPerformanceRating",
        "CarPerformanceRating",
        "WetPerformanceScore",
        "RainProbability",
        "Temperature",
        "DriverCarCombined",
        "QualifyingPerformance",
        "WetImpact",
        "QualToRaceRatio"
    ]
    # Optionally include sector times if available.
    for col in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
        if col in merged.columns:
            feature_cols.append(col)
    
    X = merged[feature_cols].copy()
    return X, y
