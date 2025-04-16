import pandas as pd

def get_qualifying_data(year, track):
    """
    Return a DataFrame containing manually entered qualifying data for the given year and track.
    Each driver is represented by a three-letter abbreviation, along with their qualifying time in seconds.
    
    This fallback is used when live data cannot be reliably fetched.
    """
    qualifying_data = pd.DataFrame({
        "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
        "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.594, 92.283]
    })
    return qualifying_data

if __name__ == "__main__":
    # Test the manual qualifying data retrieval
    # (year and track parameters are not used here, but kept for interface consistency)
    qual_df = get_qualifying_data(2025, "Bahrain")
    print("Manually Entered Qualifying Data:")
    print(qual_df)
