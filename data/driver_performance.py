driver_performance = {
    "VER": 9.8,
    "NOR": 9.5,
    "LEC": 9.4,
    "SAI": 9.1,
    "HAM": 9.3,
    "RUS": 9.0,
    "PIA": 8.8,
    "ALO": 9.2,
    "GAS": 8.5,
    "TSU": 8.3,
    "HUL": 8.6,
    "OCO": 8.4,
    "STR": 8.2
}

def get_driver_performance(driver):
    """
    Return the historical performance rating for the specified driver.
    
    Args:
        driver (str): Driver name or abbreviation.
    
    Returns:
        float: Performance rating.
    """
    return driver_performance.get(driver, 8.0)
