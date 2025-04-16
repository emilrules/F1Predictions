# data/weather_config.py
import requests

def get_weather_conditions(api_key, lat, lon):
    """
    Fetch current weather data for given coordinates.
    
    Args:
        api_key (str): Your OpenWeatherMap API key.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        
    Returns:
        dict: Contains temperature (in °C) and rain probability.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    
    temperature = data["main"]["temp"]
    # Simple proxy for rain probability (could be refined)
    rain_probability = 0.0  
    if "rain" in data:
        rain_probability = 1.0  # Adjust as needed
    
    return {"temperature": temperature, "rain_probability": rain_probability}

# Default static values (useful as fallback or for offline testing)
temperature = 21.0        # Default temperature in °C
rain_probability = 0.10     # Default rain probability
       
if __name__ == "__main__":
    API_KEY = "3faea8465b80ec261d5a03228fb5068e"
    weather = get_weather_conditions(API_KEY, lat=34.8431, lon=136.5410)
    print(weather)
