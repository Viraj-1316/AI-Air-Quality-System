# import os
# import requests
# import joblib
# import numpy as np
# import pandas as pd

# # ================= MODEL DOWNLOAD =================
# BASE_DIR = os.path.dirname(__file__)

# def download_model(url, filename):
#     model_dir = os.path.join(BASE_DIR, "models")

#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     path = os.path.join(model_dir, filename)

#     if not os.path.exists(path):
#         print(f"Downloading {filename}...")
#         r = requests.get(url, timeout=60)

#         if r.status_code != 200:
#             raise Exception(f"Failed to download {filename}")

#         if "text/html" in r.headers.get("Content-Type", ""):
#             raise Exception(f"Wrong link for {filename}")

#         with open(path, "wb") as f:
#             f.write(r.content)

#     return path


# # ================= GOOGLE DRIVE LINKS =================
# AQI_URL = "https://drive.google.com/uc?export=download&id=1UKueg89Udbs0ckJGtqyokQKowxUfsWCc"
# CACI_URL = "https://drive.google.com/uc?export=download&id=1dLKcHoSG2zgZ9fQjXi2tnOmUOnx4WBwv"
# PM25_URL = "https://drive.google.com/uc?export=download&id=1dXErmzkHx_pyO_J85tcaLy7lToGrLBd3"
# TEMP_URL = "https://drive.google.com/uc?export=download&id=1GhPG2HzBgrNu4HiP0GJHJwhYgQYAIH3A"
# HUMIDITY_URL = "https://drive.google.com/uc?export=download&id=1bUoPT1O1v8TbpcOq9TlaxDUSHRGeP__I"


# # ================= LAZY LOAD MODELS =================
# aqi_model = None
# caci_model = None
# pm25_model = None
# temp_model = None
# humidity_model = None

# def load_models():
#     global aqi_model, caci_model, pm25_model, temp_model, humidity_model

#     if aqi_model is None:
#         print("Loading models...")

#         aqi_model = joblib.load(download_model(AQI_URL, "aqi_model.pkl"))
#         caci_model = joblib.load(download_model(CACI_URL, "caci_model.pkl"))
#         pm25_model = joblib.load(download_model(PM25_URL, "pm25_model.pkl"))
#         temp_model = joblib.load(download_model(TEMP_URL, "temp_model.pkl"))
#         humidity_model = joblib.load(download_model(HUMIDITY_URL, "humidity_model.pkl"))


# # ================= THINGSPEAK CONFIG =================
# CHANNEL_ID = "3220962"
# READ_API_KEY = "TF7VPOAMFV8XK33V"

# # 🔥 IMPORTANT: PUT YOUR CHANNEL 2 WRITE KEY HERE
# WRITE_API_KEY = "EQR4J8S4J41WU5B8"
# WRITE_URL = "https://api.thingspeak.com/update"

# SOURCE_MAP = {
#     "field1": "TEMP",
#     "field2": "humidity",
#     "field3": "PM2.5",
#     "field4": "PM10",
#     "field7": "gasValue",
# }

# http = requests.Session()


# # ================= FETCH DATA =================
# def fetch_sensor_frame():
#     url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=50"

#     r = http.get(url, timeout=15)
#     r.raise_for_status()

#     feeds = r.json().get("feeds", [])
#     if not feeds:
#         raise Exception("No data from ThingSpeak")

#     df = pd.DataFrame(feeds)

#     df = df.rename(columns=SOURCE_MAP)
#     df["timestamp"] = pd.to_datetime(df["created_at"])

#     for col in ["TEMP", "humidity", "PM2.5", "PM10", "gasValue"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df = df.dropna()

#     if df.empty:
#         raise Exception("No valid sensor data")

#     return df


# # ================= BUILD FEATURES =================
# def build_features(df):
#     latest = df.iloc[-1]

#     return [[
#         float(latest["PM2.5"]),
#         float(latest["TEMP"]),
#         float(latest["humidity"])
#     ]]


# # ================= SEND TO THINGSPEAK =================
# def send_to_thingspeak(aqi, caci, pm25, temp, humidity):

#     if "YOUR_CHANNEL_2_WRITE_KEY" in WRITE_API_KEY:
#         raise Exception("WRITE API KEY NOT SET ❌")

#     params = {
#         "api_key": WRITE_API_KEY,
#         "field1": round(float(aqi), 2),
#         "field2": round(float(caci), 2),
#         "field3": round(float(pm25), 2),
#         "field4": round(float(temp), 2),
#         "field5": round(float(humidity), 2),
#     }

#     try:
#         response = requests.get(WRITE_URL, params=params, timeout=10)

#         print("ThingSpeak Response:", response.text)

#         if response.text.strip() == "0":
#             raise Exception("ThingSpeak rejected update (maybe too fast or wrong key)")

#     except Exception as e:
#         print("ThingSpeak ERROR:", e)


# # ================= MAIN PREDICTION =================
# def run_prediction_once_now():
#     load_models()

#     df = fetch_sensor_frame()
#     features = build_features(df)

#     pm25_pred = pm25_model.predict(features)[0]
#     temp_pred = temp_model.predict(features)[0]
#     humidity_pred = humidity_model.predict(features)[0]
#     aqi_pred = aqi_model.predict(features)[0]
#     caci_pred = caci_model.predict(features)[0]

#     print("Predicted:", aqi_pred, caci_pred)

#     # 🔥 SEND TO CHANNEL 2
#     send_to_thingspeak(aqi_pred, caci_pred, pm25_pred, temp_pred, humidity_pred)

#     return {
#         "AQI": float(aqi_pred),
#         "CACI": float(caci_pred),
#         "PM2.5": float(pm25_pred),
#         "TEMP": float(temp_pred),
#         "HUMIDITY": float(humidity_pred)
#     }

import os
import requests
import joblib
import numpy as np
import pandas as pd

# ================= MODEL DOWNLOAD =================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Google Drive file IDs (kept from your version)
AQI_FILE_ID = "1UKueg89Udbs0ckJGtqyokQKowxUfsWCc"
CACI_FILE_ID = "1dLKcHoSG2zgZ9fQjXi2tnOmUOnx4WBwv"
PM25_FILE_ID = "1dXErmzkHx_pyO_J85tcaLy7lToGrLBd3"
TEMP_FILE_ID = "1GhPG2HzBgrNu4HiP0GJHJwhYgQYAIH3A"
HUMIDITY_FILE_ID = "1bUoPT1O1v8TbpcOq9TlaxDUSHRGeP__I"

DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"

# ================= THINGSPEAK CONFIG =================
CHANNEL_ID = os.getenv("CHANNEL_ID", "3220962")
READ_API_KEY = os.getenv("READ_API_KEY", "TF7VPOAMFV8XK33V")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "EQR4J8S4J41WU5B8")
WRITE_URL = "https://api.thingspeak.com/update"

READ_RESULTS = int(os.getenv("READ_RESULTS", "240"))  # ~8h if 2-min interval

SOURCE_MAP = {
    "field1": "TEMP",
    "field2": "humidity",
    "field3": "PM2.5",
    "field4": "PM10",
    "field7": "gasValue",
}

http = requests.Session()
http.headers.update({"User-Agent": "air-quality-predictor/1.0"})


def _ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


def _download_from_gdrive(file_id: str, out_path: str):
    url = DRIVE_DOWNLOAD_URL.format(file_id=file_id)
    r = http.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").lower()
    content = r.content

    # Basic guard: if HTML is returned, link is likely not a direct downloadable binary
    if "text/html" in ctype or content[:200].lower().find(b"<html") != -1:
        raise RuntimeError(
            f"Download returned HTML for file_id={file_id}. "
            "Use a public direct-download link or different file host."
        )

    with open(out_path, "wb") as f:
        f.write(content)


def download_model(file_id: str, filename: str):
    _ensure_model_dir()
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        print(f"Downloading {filename} ...")
        _download_from_gdrive(file_id, path)
        print(f"Saved: {path}")

    return path


# ================= LAZY LOAD MODELS =================
aqi_model = None
caci_model = None
pm25_model = None
temp_model = None
humidity_model = None

AQI_FEATURES = None
CACI_FEATURES = None
PM25_FEATURES = None
TEMP_FEATURES = None
HUMIDITY_FEATURES = None


def _model_features(model, fallback=None):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if fallback is not None:
        return fallback
    raise ValueError("Loaded model has no feature_names_in_")


def load_models():
    global aqi_model, caci_model, pm25_model, temp_model, humidity_model
    global AQI_FEATURES, CACI_FEATURES, PM25_FEATURES, TEMP_FEATURES, HUMIDITY_FEATURES

    if aqi_model is not None:
        return

    print("Loading models...")

    aqi_model = joblib.load(download_model(AQI_FILE_ID, "aqi_model.pkl"))
    caci_model = joblib.load(download_model(CACI_FILE_ID, "caci_model.pkl"))
    pm25_model = joblib.load(download_model(PM25_FILE_ID, "pm25_model.pkl"))
    temp_model = joblib.load(download_model(TEMP_FILE_ID, "temp_model.pkl"))
    humidity_model = joblib.load(download_model(HUMIDITY_FILE_ID, "humidity_model.pkl"))

    # Fallback allows your simple 3-feature models to still run if needed
    default_fallback = ["PM2.5", "TEMP", "humidity"]

    AQI_FEATURES = _model_features(aqi_model, fallback=default_fallback)
    CACI_FEATURES = _model_features(caci_model, fallback=default_fallback)
    PM25_FEATURES = _model_features(pm25_model, fallback=default_fallback)
    TEMP_FEATURES = _model_features(temp_model, fallback=default_fallback)
    HUMIDITY_FEATURES = _model_features(humidity_model, fallback=default_fallback)

    print("Models loaded.")


# ================= FETCH DATA =================
def fetch_sensor_frame():
    url = (
        f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
        f"?api_key={READ_API_KEY}&results={READ_RESULTS}"
    )

    r = http.get(url, timeout=20)
    r.raise_for_status()

    feeds = r.json().get("feeds", [])
    if not feeds:
        raise ValueError("No data from ThingSpeak")

    raw = pd.DataFrame(feeds)
    keep_cols = ["created_at"] + list(SOURCE_MAP.keys())
    missing = [c for c in keep_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected ThingSpeak fields: {missing}")

    df = raw[keep_cols].copy()
    df = df.rename(columns=SOURCE_MAP)
    df["timestamp"] = pd.to_datetime(df["created_at"], errors="coerce")

    for col in ["TEMP", "humidity", "PM2.5", "PM10", "gasValue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # IMPORTANT: drop NA only for required columns
    required = ["timestamp", "TEMP", "humidity", "PM2.5", "PM10", "gasValue"]
    df = df.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid sensor data after cleaning")

    return df


# ================= FEATURE ENGINEERING =================
def _lag(df, col, steps):
    idx = max(0, len(df) - 1 - steps)
    return float(df[col].iloc[idx])


def _std(df, col, w):
    s = df[col].tail(max(1, w))
    return float(s.std(ddof=0)) if len(s) > 1 else 0.0


def _q(df, col, w, q):
    return float(df[col].tail(max(1, w)).quantile(q))


def _ema(df, col, span):
    return float(df[col].ewm(span=span, adjust=False).mean().iloc[-1])


def _roc(df, col, minutes):
    # Assuming ~2 minute sampling interval from your sensor
    steps = max(1, int(round(minutes / 2)))
    return float(df[col].iloc[-1] - _lag(df, col, steps))


def build_feature_row(df):
    cur = df.iloc[-1]
    lag1 = df.iloc[-2] if len(df) >= 2 else cur
    lag2 = df.iloc[-3] if len(df) >= 3 else lag1

    ts = pd.Timestamp(cur["timestamp"])
    hour = ts.hour
    weekday = ts.weekday()
    day = ts.day

    row = {
        # Basic
        "PM2.5": float(cur["PM2.5"]),
        "PM10": float(cur["PM10"]),
        "gasValue": float(cur["gasValue"]),
        "TEMP": float(cur["TEMP"]),
        "humidity": float(cur["humidity"]),
        "PM2.5_lag1": float(lag1["PM2.5"]),
        "PM2.5_lag2": float(lag2["PM2.5"]),
        "PM10_lag1": float(lag1["PM10"]),
        "PM10_lag2": float(lag2["PM10"]),
        "gasValue_lag1": float(lag1["gasValue"]),
        "gasValue_lag2": float(lag2["gasValue"]),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * weekday / 7)),
        "day": int(day),
        # Engineered (for live models that require these)
        "PM2.5_lag_w7": _lag(df, "PM2.5", 7),
        "PM2.5_std_w7": _std(df, "PM2.5", 7),
        "PM2.5_lag_w15": _lag(df, "PM2.5", 15),
        "PM2.5_std_w15": _std(df, "PM2.5", 15),
        "PM2.5_lag_w22": _lag(df, "PM2.5", 22),
        "PM2.5_std_w22": _std(df, "PM2.5", 22),
        "PM2.5_lag_w30": _lag(df, "PM2.5", 30),
        "PM2.5_std_w30": _std(df, "PM2.5", 30),
        "PM2.5_roc_5min": _roc(df, "PM2.5", 5),
        "PM2.5_roc_10min": _roc(df, "PM2.5", 10),
        "PM2.5_roc_20min": _roc(df, "PM2.5", 20),
        "PM2.5_ema_7": _ema(df, "PM2.5", 7),
        "PM2.5_ema_15": _ema(df, "PM2.5", 15),
        "PM2.5_pct25_20min": _q(df, "PM2.5", 10, 0.25),
        "PM2.5_pct75_20min": _q(df, "PM2.5", 10, 0.75),
        "PM2.5_iqr_20min": _q(df, "PM2.5", 10, 0.75) - _q(df, "PM2.5", 10, 0.25),
        "pm_temp": float(cur["PM2.5"] * cur["TEMP"]),
        "pm_temp_sq": float((cur["PM2.5"] * cur["TEMP"]) ** 2),
        "pm_humidity": float(cur["PM2.5"] * cur["humidity"]),
        "pm_humidity_sq": float((cur["PM2.5"] * cur["humidity"]) ** 2),
        "pm_gasValue": float(cur["PM2.5"] * cur["gasValue"]),
        "PM10_lag_w7": _lag(df, "PM10", 7),
        "gasValue_lag_w7": _lag(df, "gasValue", 7),
        "PM10_lag_w15": _lag(df, "PM10", 15),
        "gasValue_lag_w15": _lag(df, "gasValue", 15),
        "PM10_lag_w30": _lag(df, "PM10", 30),
        "gasValue_lag_w30": _lag(df, "gasValue", 30),
    }

    # Compatibility placeholders if some old model expects AQI/CACI lags
    row["AQI_lag1"] = row["PM2.5_lag1"]
    row["AQI_lag2"] = row["PM2.5_lag2"]
    row["CACI_lag1"] = row["PM10_lag1"]
    row["CACI_lag2"] = row["PM10_lag2"]

    return row


def frame_for_model(row, features):
    missing = [f for f in features if f not in row]
    if missing:
        raise ValueError(f"Missing features for model: {missing}")
    return pd.DataFrame([{f: row[f] for f in features}])


# ================= SEND TO THINGSPEAK =================
def send_to_thingspeak(aqi, caci, pm25, temp, humidity):
    if not WRITE_API_KEY or "YOUR_CHANNEL_2_WRITE_KEY" in WRITE_API_KEY:
        raise ValueError("WRITE_API_KEY not configured")

    params = {
        "api_key": WRITE_API_KEY,
        "field1": round(float(aqi), 2),
        "field2": round(float(caci), 2),
        "field3": round(float(pm25), 2),
        "field4": round(float(temp), 2),
        "field5": round(float(humidity), 2),
    }

    r = http.get(WRITE_URL, params=params, timeout=15)
    r.raise_for_status()
    if r.text.strip() == "0":
        raise RuntimeError("ThingSpeak rejected update (rate limit or wrong key)")


# ================= MAIN PREDICTION =================
def run_prediction_once_now():
    load_models()
    df = fetch_sensor_frame()
    row = build_feature_row(df)

    X_pm25 = frame_for_model(row, PM25_FEATURES)
    X_temp = frame_for_model(row, TEMP_FEATURES)
    X_humidity = frame_for_model(row, HUMIDITY_FEATURES)
    X_aqi = frame_for_model(row, AQI_FEATURES)
    X_caci = frame_for_model(row, CACI_FEATURES)

    pm25_pred = pm25_model.predict(X_pm25)[0]
    temp_pred = temp_model.predict(X_temp)[0]
    humidity_pred = humidity_model.predict(X_humidity)[0]
    aqi_pred = aqi_model.predict(X_aqi)[0]
    caci_pred = caci_model.predict(X_caci)[0]

    print(f"Predicted AQI={aqi_pred:.2f}, CACI={caci_pred:.2f}")

    send_to_thingspeak(aqi_pred, caci_pred, pm25_pred, temp_pred, humidity_pred)

    return {
        "AQI": float(aqi_pred),
        "CACI": float(caci_pred),
        "PM2.5": float(pm25_pred),
        "TEMP": float(temp_pred),
        "HUMIDITY": float(humidity_pred),
    }


if __name__ == "__main__":
    # For local test
    out = run_prediction_once_now()
    print(out)