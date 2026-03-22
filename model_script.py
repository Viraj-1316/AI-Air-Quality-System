import json
import time
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests

def download_model(url, filename):
    import os, requests

    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", filename)

    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

    return path

def load_model_auto(primary_live, fallback):
    try:
        model = joblib.load(primary_live)
        print(f"Using {primary_live}")
        return model
    except Exception:
        model = joblib.load(fallback)
        print(f"Using fallback {fallback}")
        return model


# # 1) LOAD MODELS (live models preferred)
# aqi_model = load_model_auto("aqi_model_live.pkl", "aqi_model_1.pkl")
# caci_model = load_model_auto("caci_model_live.pkl", "caci_model_1.pkl")
# pm25_model = load_model_auto("pm25_model_live.pkl", "pm25_model.pkl")
# temp_model = load_model_auto("temp_model_live.pkl", "temp_model.pkl")
# humidity_model = load_model_auto("humidity_model_live.pkl", "humidity_model.pkl")

AQI_URL = "https://drive.google.com/uc?export=download&id=1UKueg89Udbs0ckJGtqyokQKowxUfsWCc"
CACI_URL = "https://drive.google.com/uc?export=download&id=1dLKcHoSG2zgZ9fQjXi2tnOmUOnx4WBwv"
PM25_URL = "https://drive.google.com/uc?export=download&id=1dXErmzkHx_pyO_J85tcaLy7lToGrLBd3"
TEMP_URL = "https://drive.google.com/uc?export=download&id=1GhPG2HzBgrNu4HiP0GJHJwhYgQYAIH3A"
HUMIDITY_URL = "https://drive.google.com/uc?export=download&id=1bUoPT1O1v8TbpcOq9TlaxDUSHRGeP__I"

aqi_model = joblib.load(download_model(AQI_URL, "aqi_model_1.pkl"))
caci_model = joblib.load(download_model(CACI_URL, "caci_model_1.pkl"))
pm25_model = joblib.load(download_model(PM25_URL, "pm25_model.pkl"))
temp_model = joblib.load(download_model(TEMP_URL, "temp_model.pkl"))
humidity_model = joblib.load(download_model(HUMIDITY_URL, "humidity_model.pkl"))

def _model_features(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError("Loaded model has no feature_names_in_")
    return list(names)


AQI_FEATURES = _model_features(aqi_model)
CACI_FEATURES = _model_features(caci_model)
PM25_FEATURES = _model_features(pm25_model)
TEMP_FEATURES = _model_features(temp_model)
HUMIDITY_FEATURES = _model_features(humidity_model)

ALL_MODEL_FEATURES = set().union(
    AQI_FEATURES, CACI_FEATURES, PM25_FEATURES, TEMP_FEATURES, HUMIDITY_FEATURES
)

NEED_TARGET_LAGS = any(
    f in ALL_MODEL_FEATURES for f in ["AQI_lag1", "AQI_lag2", "CACI_lag1", "CACI_lag2"]
)

if NEED_TARGET_LAGS:
    print("Legacy feature mode: AQI/CACI lag state required")
else:
    print("Deployment-safe feature mode: no AQI/CACI lag dependency")

# 2) THINGSPEAK CONFIG
CHANNEL_ID = "3220962"
READ_API_KEY = "TF7VPOAMFV8XK33V"
PRED_WRITE_API_KEY = "EQR4J8S4J41WU5B8"
WRITE_URL = "https://api.thingspeak.com/update"

# Optional: prediction output channel (used for lag warm-start if legacy models are active)
PRED_CHANNEL_ID = "3124366"
PRED_READ_API_KEY = "L06LTBC1KWFZG75X"

# Persist runtime artifacts
STATE_FILE = Path("prediction_state.json")
LOG_FILE = Path("prediction_log.csv")

# Runtime policy
READ_RESULTS = 240            # ~8 hours when ESP32 pushes every 2 min
PRED_INTERVAL_MIN = 15        # run prediction every 15 minutes
HORIZON_WINDOW_MIN = 60       # predict next 1 hour from last 1-hour context
MIN_POINTS_PER_WINDOW = 12    # at least 24 minutes of samples/window

# Field mapping from source channel
SOURCE_MAP = {
    "field1": "TEMP",
    "field2": "humidity",
    "field3": "PM2.5",
    "field4": "PM10",
    "field7": "gasValue",
}

http = requests.Session()
http.headers.update({"User-Agent": "air-quality-predictor/1.0"})


def fetch_sensor_frame():
    read_url = (
        f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
        f"?api_key={READ_API_KEY}&results={READ_RESULTS}"
    )
    r = http.get(read_url, timeout=15)
    r.raise_for_status()

    payload = r.json()
    feeds = payload.get("feeds", [])
    if not feeds:
        raise ValueError("No feeds returned from ThingSpeak")

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

    df = df.dropna(subset=["timestamp", "TEMP", "humidity", "PM2.5", "PM10", "gasValue"])
    if df.empty:
        raise ValueError("No valid numeric sensor rows after cleaning")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def summarize_window(df, end_ts, minutes):
    start_ts = end_ts - timedelta(minutes=minutes)
    window_df = df[(df["timestamp"] > start_ts) & (df["timestamp"] <= end_ts)].copy()
    if len(window_df) < MIN_POINTS_PER_WINDOW:
        return None

    snapshot = {}
    for col in ["TEMP", "humidity", "PM2.5", "PM10", "gasValue"]:
        low = window_df[col].quantile(0.05)
        high = window_df[col].quantile(0.95)
        clipped = window_df[col].clip(lower=low, upper=high)
        snapshot[col] = float(clipped.mean())

    snapshot["timestamp"] = end_ts
    return snapshot


def build_hourly_context(df):
    latest_ts = df["timestamp"].iloc[-1]

    cur = summarize_window(df, latest_ts, HORIZON_WINDOW_MIN)
    if cur is None:
        if len(df) < MIN_POINTS_PER_WINDOW:
            raise ValueError("Not enough samples yet. Wait for more sensor pushes.")
        cur = summarize_window(df, latest_ts, 24 * 60)
        print("Bootstrap mode: using available history for current window")

    lag1 = summarize_window(df, latest_ts - timedelta(minutes=HORIZON_WINDOW_MIN), HORIZON_WINDOW_MIN)
    lag2 = summarize_window(df, latest_ts - timedelta(minutes=2 * HORIZON_WINDOW_MIN), HORIZON_WINDOW_MIN)

    if lag1 is None:
        lag1 = dict(cur)
        print("Bootstrap mode: lag1 window unavailable, reusing current window")
    if lag2 is None:
        lag2 = dict(lag1)
        print("Bootstrap mode: lag2 window unavailable, reusing lag1 window")

    cur["TEMP"] = float(np.clip(cur["TEMP"], -5, 50))
    cur["humidity"] = float(np.clip(cur["humidity"], 0, 100))
    for key in ["PM2.5", "PM10", "gasValue"]:
        cur[key] = float(max(0.0, cur[key]))

    return cur, lag1, lag2


def build_feature_row(df, cur, lag1, lag2, state):
    ts = cur["timestamp"]
    hour = ts.hour
    weekday = ts.weekday()
    day = ts.day

    def _lag(col, steps):
        if len(df) > steps:
            return float(df[col].iloc[-1 - steps])
        return float(df[col].iloc[0])

    def _std(col, w):
        s = df[col].tail(max(1, w))
        return float(s.std(ddof=0)) if len(s) > 1 else 0.0

    def _q(col, w, q):
        return float(df[col].tail(max(1, w)).quantile(q))

    def _ema(col, span):
        return float(df[col].ewm(span=span, adjust=False).mean().iloc[-1])

    def _roc(col, minutes):
        steps = max(1, int(round(minutes / 2)))  # ~2-min sample interval
        return float(cur[col] - _lag(col, steps))

    row = {
        # base
        "PM2.5": cur["PM2.5"],
        "PM10": cur["PM10"],
        "gasValue": cur["gasValue"],
        "TEMP": cur["TEMP"],
        "humidity": cur["humidity"],
        "PM2.5_lag1": lag1["PM2.5"],
        "PM2.5_lag2": lag2["PM2.5"],
        "PM10_lag1": lag1["PM10"],
        "PM10_lag2": lag2["PM10"],
        "gasValue_lag1": lag1["gasValue"],
        "gasValue_lag2": lag2["gasValue"],
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "day": day,
        # engineered
        "PM2.5_lag_w7": _lag("PM2.5", 7),
        "PM2.5_std_w7": _std("PM2.5", 7),
        "PM2.5_lag_w15": _lag("PM2.5", 15),
        "PM2.5_std_w15": _std("PM2.5", 15),
        "PM2.5_lag_w22": _lag("PM2.5", 22),
        "PM2.5_std_w22": _std("PM2.5", 22),
        "PM2.5_lag_w30": _lag("PM2.5", 30),
        "PM2.5_std_w30": _std("PM2.5", 30),
        "PM2.5_roc_5min": _roc("PM2.5", 5),
        "PM2.5_roc_10min": _roc("PM2.5", 10),
        "PM2.5_roc_20min": _roc("PM2.5", 20),
        "PM2.5_ema_7": _ema("PM2.5", 7),
        "PM2.5_ema_15": _ema("PM2.5", 15),
        "PM2.5_pct25_20min": _q("PM2.5", 10, 0.25),
        "PM2.5_pct75_20min": _q("PM2.5", 10, 0.75),
        "PM2.5_iqr_20min": _q("PM2.5", 10, 0.75) - _q("PM2.5", 10, 0.25),
        "pm_temp": cur["PM2.5"] * cur["TEMP"],
        "pm_temp_sq": (cur["PM2.5"] * cur["TEMP"]) ** 2,
        "pm_humidity": cur["PM2.5"] * cur["humidity"],
        "pm_humidity_sq": (cur["PM2.5"] * cur["humidity"]) ** 2,
        "pm_gasValue": cur["PM2.5"] * cur["gasValue"],
        "PM10_lag_w7": _lag("PM10", 7),
        "gasValue_lag_w7": _lag("gasValue", 7),
        "PM10_lag_w15": _lag("PM10", 15),
        "gasValue_lag_w15": _lag("gasValue", 15),
        "PM10_lag_w30": _lag("PM10", 30),
        "gasValue_lag_w30": _lag("gasValue", 30),
    }

    if NEED_TARGET_LAGS:
        aqi_lag1 = state["aqi_lag"][0] if state["aqi_lag"][0] is not None else lag1["PM2.5"]
        aqi_lag2 = state["aqi_lag"][1] if state["aqi_lag"][1] is not None else lag2["PM2.5"]
        caci_lag1 = state["caci_lag"][0] if state["caci_lag"][0] is not None else lag1["PM10"]
        caci_lag2 = state["caci_lag"][1] if state["caci_lag"][1] is not None else lag2["PM10"]
        row["AQI_lag1"] = aqi_lag1
        row["AQI_lag2"] = aqi_lag2
        row["CACI_lag1"] = caci_lag1
        row["CACI_lag2"] = caci_lag2

    return row


def frame_for_features(row, features):
    missing = [f for f in features if f not in row]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return pd.DataFrame([{f: row[f] for f in features}])


def post_predictions(aqi, caci, pm25_next, temp_next, humidity_next):
    params = {
        "api_key": PRED_WRITE_API_KEY,
        "field1": float(aqi),
        "field2": float(caci),
        "field3": float(pm25_next),
        "field4": float(temp_next),
        "field5": float(humidity_next),
    }
    r = http.get(WRITE_URL, params=params, timeout=15)
    r.raise_for_status()
    if r.text.strip() == "0":
        raise RuntimeError("ThingSpeak rejected update (response=0)")


def append_prediction_log(ts, cur, aqi, caci, pm25_next, temp_next, humidity_next, post_status, post_error=""):
    row = {
        "timestamp": pd.Timestamp(ts).isoformat(),
        "pm25_context": float(cur["PM2.5"]),
        "temp_context": float(cur["TEMP"]),
        "humidity_context": float(cur["humidity"]),
        "gas_context": float(cur["gasValue"]),
        "aqi_pred_next_1h": float(aqi),
        "caci_pred_next_1h": float(caci),
        "pm25_pred_next_1h": float(pm25_next),
        "temp_pred_next_1h": float(temp_next),
        "humidity_pred_next_1h": float(humidity_next),
        "post_status": post_status,
        "post_error": post_error,
    }
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", index=False, header=not LOG_FILE.exists())


def load_state_from_disk(state):
    if not STATE_FILE.exists():
        return
    try:
        content = json.loads(STATE_FILE.read_text())
        aqi_lag = content.get("aqi_lag")
        caci_lag = content.get("caci_lag")
        if isinstance(aqi_lag, list) and len(aqi_lag) == 2:
            state["aqi_lag"] = [float(aqi_lag[0]), float(aqi_lag[1])]
        if isinstance(caci_lag, list) and len(caci_lag) == 2:
            state["caci_lag"] = [float(caci_lag[0]), float(caci_lag[1])]
    except Exception as e:
        print(f"State load warning: {e}")


def save_state_to_disk(state):
    if not NEED_TARGET_LAGS:
        return
    try:
        payload = {
            "aqi_lag": state["aqi_lag"],
            "caci_lag": state["caci_lag"],
            "saved_at": datetime.utcnow().isoformat(),
        }
        STATE_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as e:
        print(f"State save warning: {e}")


def warm_start_lags_from_prediction_channel(state):
    if not NEED_TARGET_LAGS:
        return

    if state["aqi_lag"][0] is not None and state["aqi_lag"][1] is not None:
        if state["caci_lag"][0] is not None and state["caci_lag"][1] is not None:
            return

    if "YOUR_PREDICTION_CHANNEL_ID" in PRED_CHANNEL_ID:
        return

    try:
        read_url = (
            f"https://api.thingspeak.com/channels/{PRED_CHANNEL_ID}/feeds.json"
            f"?api_key={PRED_READ_API_KEY}&results=3"
        )
        r = http.get(read_url, timeout=15)
        r.raise_for_status()
        feeds = r.json().get("feeds", [])
        if len(feeds) < 2:
            return

        pred_df = pd.DataFrame(feeds)
        pred_df["field1"] = pd.to_numeric(pred_df.get("field1"), errors="coerce")
        pred_df["field2"] = pd.to_numeric(pred_df.get("field2"), errors="coerce")
        pred_df = pred_df.dropna(subset=["field1", "field2"]).reset_index(drop=True)
        if len(pred_df) < 2:
            return

        state["aqi_lag"] = [float(pred_df["field1"].iloc[-1]), float(pred_df["field1"].iloc[-2])]
        state["caci_lag"] = [float(pred_df["field2"].iloc[-1]), float(pred_df["field2"].iloc[-2])]
        print("Lag state warm-started from prediction channel")
    except Exception as e:
        print(f"Prediction-channel warm start skipped: {e}")


def predict_once(state):
    df_live = fetch_sensor_frame()
    cur, lag1, lag2 = build_hourly_context(df_live)
    feature_row = build_feature_row(df_live, cur, lag1, lag2, state)

    X_pm25 = frame_for_features(feature_row, PM25_FEATURES)
    X_temp = frame_for_features(feature_row, TEMP_FEATURES)
    X_humidity = frame_for_features(feature_row, HUMIDITY_FEATURES)
    X_aqi = frame_for_features(feature_row, AQI_FEATURES)
    X_caci = frame_for_features(feature_row, CACI_FEATURES)

    pm25_next = pm25_model.predict(X_pm25)[0]
    temp_next = temp_model.predict(X_temp)[0]
    humidity_next = humidity_model.predict(X_humidity)[0]
    aqi_next = aqi_model.predict(X_aqi)[0]
    caci_next = caci_model.predict(X_caci)[0]

    if NEED_TARGET_LAGS:
        prev_aqi = state["aqi_lag"][0] if state["aqi_lag"][0] is not None else float(aqi_next)
        prev_caci = state["caci_lag"][0] if state["caci_lag"][0] is not None else float(caci_next)
        state["aqi_lag"] = [float(aqi_next), float(prev_aqi)]
        state["caci_lag"] = [float(caci_next), float(prev_caci)]
        save_state_to_disk(state)

    print("\n===== FINAL PREDICTION (1 HOUR AHEAD) =====")
    print(f"PM2.5 (1h context): {cur['PM2.5']:.2f} -> {pm25_next:.2f}")
    print(f"TEMP (1h context): {cur['TEMP']:.2f} -> {temp_next:.2f}")
    print(f"humidity (1h context): {cur['humidity']:.2f} -> {humidity_next:.2f}")
    print(f"gasValue (1h context): {cur['gasValue']:.2f}")
    print(f"AQI +1h: {aqi_next:.2f}")
    print(f"CACI +1h: {caci_next:.2f}")

    post_ok = True
    post_error = ""
    try:
        post_predictions(aqi_next, caci_next, pm25_next, temp_next, humidity_next)
    except Exception as e:
        post_ok = False
        post_error = str(e)
        print(f"Channel post warning: {e}")

    append_prediction_log(
        cur["timestamp"],
        cur,
        aqi_next,
        caci_next,
        pm25_next,
        temp_next,
        humidity_next,
        post_status=("ok" if post_ok else "failed"),
        post_error=post_error,
    )
    print(f"Logged prediction to {LOG_FILE}")
    if post_ok:
        print("Sent predictions to Channel 2")


def sleep_until_next_quarter():
    now = datetime.now()
    next_q_minute = ((now.minute // PRED_INTERVAL_MIN) + 1) * PRED_INTERVAL_MIN
    if next_q_minute >= 60:
        target = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        target = now.replace(minute=next_q_minute, second=0, microsecond=0)
    wait_seconds = max(1, int((target - now).total_seconds()))
    time.sleep(wait_seconds)


def run_prediction_service():
    state = {
        "sensor_history": deque(maxlen=3),
        "aqi_lag": [None, None],
        "caci_lag": [None, None],
    }

    if NEED_TARGET_LAGS:
        load_state_from_disk(state)
        warm_start_lags_from_prediction_channel(state)

    while True:
        try:
            predict_once(state)
        except Exception as e:
            print(f"Prediction loop error: {e}")
        sleep_until_next_quarter()


def run_prediction_once_now():
    state = {
        "sensor_history": deque(maxlen=3),
        "aqi_lag": [None, None],
        "caci_lag": [None, None],
    }
    if NEED_TARGET_LAGS:
        load_state_from_disk(state)
        warm_start_lags_from_prediction_channel(state)
    predict_once(state)


# Manual start options:
# 1) run_prediction_once_now()     # quick smoke test
# 2) run_prediction_service()      # continuous 15-minute loop
run_prediction_service()