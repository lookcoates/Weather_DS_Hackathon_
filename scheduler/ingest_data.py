import requests
import pandas as pd
from datetime import datetime
import os

API_KEY = "YOUR_API_KEY"
CITIES = ["London", "New York", "Tokyo"]
UNITS = "metric"

def fetch_weather():
    """Fetch weather data from API for multiple cities"""
    records = []
    for city in CITIES:
        URL = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units={UNITS}"
        try:
            response = requests.get(URL, timeout=30)  # Increased timeout
            response.raise_for_status()
            data = response.json()
            
            if not data.get('list'):
                print(f"No forecast data for {city}")
                continue

            for forecast in data['list']:
                records.append({
                    'city': city,
                    'timestamp': datetime.strptime(forecast['dt_txt'], '%Y-%m-%d %H:%M:%S'),
                    'temperature': forecast['main']['temp'],
                    'humidity': forecast['main']['humidity'],
                    'wind_speed': forecast['wind']['speed'],
                    'pressure': forecast['main']['pressure'],
                    'precipitation': forecast.get('rain', {}).get('1h', 0),
                    'cloud_coverage': forecast['clouds']['all'],
                    'weather_condition': forecast['weather'][0]['main'],
                    'retrieved_at': datetime.now()
                })
        except Exception as e:
            print(f"Error fetching weather data for {city}: {str(e)}")
            continue

    return pd.DataFrame(records)

def save_data(df):
    """Save data to CSV"""
    try:
        os.makedirs('ml_model/data', exist_ok=True)
        df.to_csv('ml_model/data/weather_data.csv', index=False)
        print(f"Successfully saved data for {df['city'].nunique()} cities")
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        weather_df = fetch_weather()
        if not weather_df.empty:
            save_data(weather_df)
        else:
            print("No weather data was fetched")
    except Exception as e:
        print(f"Failed to complete data ingestion: {str(e)}")
