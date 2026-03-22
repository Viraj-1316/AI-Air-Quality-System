import requests
import pandas as pd
from datetime import datetime
import time
import os

WAQI_TOKEN = "cd8d0a75248fa7f2c0d3bfcb43abfe1f5affea8e"   # replace later
WEATHER_KEY = "53e35a484e829ece1dd7c7e84b937212"
CITY = "Pune,IN"

FILE_NAME = "pune_environment_data.csv"

def fetch_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={WEATHER_KEY}&units=metric"
        data = requests.get(url, timeout=10).json()

        if "main" not in data:
            print("Weather API error:", data)
            return None

        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"]
        }
    except Exception as e:
        print("Weather error:", e)
        return None


def fetch_aqi():
    try:
        url = f"https://api.waqi.info/feed/pune/?token={WAQI_TOKEN}"
        data = requests.get(url, timeout=10).json()

        if data["status"] != "ok":
            print("AQI API error:", data)
            return None

        aqi_data = data["data"]["iaqi"]

        return {
            "aqi": data["data"].get("aqi"),
            "pm25": aqi_data.get("pm25", {}).get("v"),
            "pm10": aqi_data.get("pm10", {}).get("v"),
            "no2": aqi_data.get("no2", {}).get("v"),
            "co": aqi_data.get("co", {}).get("v"),
            "o3": aqi_data.get("o3", {}).get("v"),
        }
    except Exception as e:
        print("AQI error:", e)
        return None


def collect_data():
    weather = fetch_weather()
    aqi = fetch_aqi()

    if weather is None or aqi is None:
        print("Skipping record due to error")
        return

    record = {
        "timestamp": datetime.now(),
        **aqi,
        **weather
    }

    df = pd.DataFrame([record])

    df.to_csv(FILE_NAME,
              mode='a',
              header=not os.path.exists(FILE_NAME),
              index=False)

    print("Saved:", record)


# Run every 5 minutes
while True:
    collect_data()
    time.sleep(300)
