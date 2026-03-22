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
        r = requests.get(url, timeout=60)

        if r.status_code != 200:
            raise Exception(f"Failed to download {filename}")

        if "text/html" in r.headers.get("Content-Type", ""):
            raise Exception(f"Wrong link for {filename}")

        with open(path, "wb") as f:
            f.write(r.content)

    return path


# ================= GOOGLE DRIVE LINKS =================
AQI_URL = "https://drive.google.com/uc?export=download&id=1UKueg89Udbs0ckJGtqyokQKowxUfsWCc"
CACI_URL = "https://drive.google.com/uc?export=download&id=1dLKcHoSG2zgZ9fQjXi2tnOmUOnx4WBwv"
PM25_URL = "https://drive.google.com/uc?export=download&id=1dXErmzkHx_pyO_J85tcaLy7lToGrLBd3"
TEMP_URL = "https://drive.google.com/uc?export=download&id=1GhPG2HzBgrNu4HiP0GJHJwhYgQYAIH3A"
HUMIDITY_URL = "https://drive.google.com/uc?export=download&id=1bUoPT1O1v8TbpcOq9TlaxDUSHRGeP__I"


# ================= LAZY LOAD MODELS =================
aqi_model = None
caci_model = None
pm25_model = None
temp_model = None
humidity_model = None

def load_models():
    global aqi_model, caci_model, pm25_model, temp_model, humidity_model

    if aqi_model is None:
        print("Loading models...")

        aqi_model = joblib.load(download_model(AQI_URL, "aqi_model.pkl"))
        caci_model = joblib.load(download_model(CACI_URL, "caci_model.pkl"))
        pm25_model = joblib.load(download_model(PM25_URL, "pm25_model.pkl"))
        temp_model = joblib.load(download_model(TEMP_URL, "temp_model.pkl"))
        humidity_model = joblib.load(download_model(HUMIDITY_URL, "humidity_model.pkl"))


# ================= THINGSPEAK CONFIG =================
CHANNEL_ID = "3220962"
READ_API_KEY = "TF7VPOAMFV8XK33V"

# 🔥 IMPORTANT: PUT YOUR CHANNEL 2 WRITE KEY HERE
WRITE_API_KEY = "EQR4J8S4J41WU5B8"
WRITE_URL = "https://api.thingspeak.com/update"

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

    r = http.get(url, timeout=15)
    r.raise_for_status()

    feeds = r.json().get("feeds", [])
    if not feeds:
        raise Exception("No data from ThingSpeak")

    df = pd.DataFrame(feeds)

    df = df.rename(columns=SOURCE_MAP)
    df["timestamp"] = pd.to_datetime(df["created_at"])

    for col in ["TEMP", "humidity", "PM2.5", "PM10", "gasValue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    if df.empty:
        raise Exception("No valid sensor data")

    return df


# ================= BUILD FEATURES =================
def build_features(df):
    latest = df.iloc[-1]

    return [[
        float(latest["PM2.5"]),
        float(latest["TEMP"]),
        float(latest["humidity"])
    ]]


# ================= SEND TO THINGSPEAK =================
def send_to_thingspeak(aqi, caci, pm25, temp, humidity):

    if "YOUR_CHANNEL_2_WRITE_KEY" in WRITE_API_KEY:
        raise Exception("WRITE API KEY NOT SET ❌")

    params = {
        "api_key": WRITE_API_KEY,
        "field1": round(float(aqi), 2),
        "field2": round(float(caci), 2),
        "field3": round(float(pm25), 2),
        "field4": round(float(temp), 2),
        "field5": round(float(humidity), 2),
    }

    try:
        response = requests.get(WRITE_URL, params=params, timeout=10)

        print("ThingSpeak Response:", response.text)

        if response.text.strip() == "0":
            raise Exception("ThingSpeak rejected update (maybe too fast or wrong key)")

    except Exception as e:
        print("ThingSpeak ERROR:", e)


# ================= MAIN PREDICTION =================
def run_prediction_once_now():
    load_models()

    df = fetch_sensor_frame()
    features = build_features(df)

    pm25_pred = pm25_model.predict(features)[0]
    temp_pred = temp_model.predict(features)[0]
    humidity_pred = humidity_model.predict(features)[0]
    aqi_pred = aqi_model.predict(features)[0]
    caci_pred = caci_model.predict(features)[0]

    print("Predicted:", aqi_pred, caci_pred)

    # 🔥 SEND TO CHANNEL 2
    send_to_thingspeak(aqi_pred, caci_pred, pm25_pred, temp_pred, humidity_pred)

    return {
        "AQI": float(aqi_pred),
        "CACI": float(caci_pred),
        "PM2.5": float(pm25_pred),
        "TEMP": float(temp_pred),
        "HUMIDITY": float(humidity_pred)
    }