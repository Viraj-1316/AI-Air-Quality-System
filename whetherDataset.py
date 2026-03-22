import requests
import pandas as pd
from datetime import datetime
import time

API_KEY = "53e35a484e829ece1dd7c7e84b937212"
CITY = "Pune,IN"

def fetch_weather():
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()

    if "main" not in data:
        print("API error:", data)
        return

    record = {
        "timestamp": datetime.now(),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"]
    }

    df = pd.DataFrame([record])
    df.to_csv("pune_weather.csv",
              mode='a',
              header=not pd.io.common.file_exists("pune_weather.csv"),
              index=False)

    print("Saved:", record)

while True:
    fetch_weather()
    time.sleep(600)   # every 10 minutes
