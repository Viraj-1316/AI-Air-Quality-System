import json
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
import os
import requests
import joblib
import numpy as np
import pandas as pd

# ================= MODEL DOWNLOAD =================
BASE_DIR = os.path.dirname(__file__)

def download_model(url, filename):
    model_dir = os.path.join(BASE_DIR, "models")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    path = os.path.join(model_dir, filename)

    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

    return path

# ======= 🔥 PUT YOUR GOOGLE DRIVE LINKS HERE =======
# AQI_URL = "https://drive.google.com/file/d/1UKueg89Udbs0ckJGtqyokQKowxUfsWCc/view?usp=sharing"
# CACI_URL = "https://drive.google.com/file/d/1dLKcHoSG2zgZ9fQjXi2tnOmUOnx4WBwv/view?usp=sharing"
# PM25_URL = "https://drive.google.com/file/d/1dXErmzkHx_pyO_J85tcaLy7lToGrLBd3/view?usp=sharing"
# TEMP_URL = "https://drive.google.com/file/d/1GhPG2HzBgrNu4HiP0GJHJwhYgQYAIH3A/view?usp=sharing"
# HUMIDITY_URL = "https://drive.google.com/file/d/1bUoPT1O1v8TbpcOq9TlaxDUSHRGeP__I/view?usp=sharing"

AQI_URL = "https://drive.google.com/uc?export=download&id=1UKueg89Udbs0ckJGtqyokQKowxUfsWCc"
CACI_URL = "https://drive.google.com/uc?export=download&id=1dLKcHoSG2zgZ9fQjXi2tnOmUOnx4WBwv"
PM25_URL = "https://drive.google.com/uc?export=download&id=1dXErmzkHx_pyO_J85tcaLy7lToGrLBd3"
TEMP_URL = "https://drive.google.com/uc?export=download&id=1GhPG2HzBgrNu4HiP0GJHJwhYgQYAIH3A"
HUMIDITY_URL = "https://drive.google.com/uc?export=download&id=1bUoPT1O1v8TbpcOq9TlaxDUSHRGeP__I"

# ================= LOAD MODELS =================
aqi_model = joblib.load(download_model(AQI_URL, "aqi_model.pkl"))
caci_model = joblib.load(download_model(CACI_URL, "caci_model.pkl"))
pm25_model = joblib.load(download_model(PM25_URL, "pm25_model.pkl"))
temp_model = joblib.load(download_model(TEMP_URL, "temp_model.pkl"))
humidity_model = joblib.load(download_model(HUMIDITY_URL, "humidity_model.pkl"))

# ================= THINGSPEAK CONFIG =================
CHANNEL_ID = "3220962"
READ_API_KEY = "TF7VPOAMFV8XK33V"

SOURCE_MAP = {
    "field1": "TEMP",
    "field2": "humidity",
    "field3": "PM2.5",
    "field4": "PM10",
    "field7": "gasValue",
}

http = requests.Session()

# ================= FETCH DATA =================
def fetch_sensor_frame():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=50"
    r = http.get(url)
    r.raise_for_status()

    feeds = r.json()["feeds"]
    df = pd.DataFrame(feeds)

    df = df.rename(columns=SOURCE_MAP)
    df["timestamp"] = pd.to_datetime(df["created_at"])

    for col in ["TEMP", "humidity", "PM2.5", "PM10", "gasValue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df

# ================= BUILD FEATURES =================
def build_features(df):
    latest = df.iloc[-1]

    features = [[
        latest["PM2.5"],
        latest["TEMP"],
        latest["humidity"]
    ]]

    return features

# ================= PREDICT =================
def run_prediction_once_now():
    df = fetch_sensor_frame()

    features = build_features(df)

    pm25_pred = pm25_model.predict(features)[0]
    temp_pred = temp_model.predict(features)[0]
    humidity_pred = humidity_model.predict(features)[0]
    aqi_pred = aqi_model.predict(features)[0]
    caci_pred = caci_model.predict(features)[0]

    return {
        "AQI": float(aqi_pred),
        "CACI": float(caci_pred),
        "PM2.5": float(pm25_pred),
        "TEMP": float(temp_pred),
        "HUMIDITY": float(humidity_pred)
    }