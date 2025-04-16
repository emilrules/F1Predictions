car_performance = {
    "VER": 9.7,  # Red Bull
    "PIA": 9.5,  # McLaren
    "NOR": 9.5,  # McLaren
    "LEC": 9.4,  # Ferrari
    "SAI": 9.4,  # Ferrari
    "HAM": 9.2,  # Mercedes
    "RUS": 9.2,  # Mercedes
    "ALO": 8.8,  # Aston Martin
    "STR": 8.8,  # Aston Martin
    "GAS": 8.5,  # Alpine
    "OCO": 8.5,  # Alpine
    "TSU": 8.3,  # RB
    "HUL": 8.0   # Haas
}

def get_car_performance(driver):
    """
    Return the car performance rating for the specified driver/team.
    
    Args:
        driver (str): Driver name or team abbreviation.
    
    Returns:
        float: Car performance rating.
    """
    return car_performance.get(driver, 8.0)
