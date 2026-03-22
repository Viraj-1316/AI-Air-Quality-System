# from flask import Flask, jsonify
# from model_script_new import run_prediction_once_now
# import os

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "Air Quality API Running 🚀"

# @app.route("/predict")
# def predict():
#     try:
#         result = run_prediction_once_now()
#         return jsonify({
#             "status": "success",
#             "data": result
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)

import streamlit as st
import requests
import pandas as pd
import time

# ---------------- CONFIG ----------------
CHANNEL_1_ID = "3220962"
CHANNEL_1_API = "TF7VPOAMFV8XK33V"

CHANNEL_2_ID = "3124366"
CHANNEL_2_API = "L06LTBC1KWFZG75X"

REFRESH_INTERVAL = 15  # seconds

# ---------------- FETCH DATA ----------------
def fetch_thingspeak_data(channel_id, api_key):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {
        "api_key": api_key,
        "results": 100
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data["feeds"])
    df["created_at"] = pd.to_datetime(df["created_at"])

    return df


# ---------------- PROCESS ACTUAL DATA ----------------
def process_actual_data(df):
    df = df.rename(columns={
        "field1": "TEMP",
        "field2": "HUMIDITY",
        "field3": "PM2.5",
        "field4": "PM10",
        "field5": "AQI_actual",
        "field6": "CACI_actual",
        "field7": "GAS"
    })

    cols = ["TEMP", "HUMIDITY", "PM2.5", "PM10", "AQI_actual", "CACI_actual", "GAS"]
    df[cols] = df[cols].astype(float)

    return df


# ---------------- PROCESS PREDICTED DATA ----------------
def process_predicted_data(df):
    df = df.rename(columns={
        "field1": "AQI_pred",
        "field2": "CACI_pred",
        "field3": "PM2.5_pred",
        "field4": "TEMP_pred",
        "field5": "HUMIDITY_pred"
    })

    cols = ["AQI_pred", "CACI_pred", "PM2.5_pred", "TEMP_pred", "HUMIDITY_pred"]
    df[cols] = df[cols].astype(float)

    return df


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

st.title("🌍 AI Air Quality Monitoring Dashboard")

# ---------------- LOAD DATA ----------------
actual_df = process_actual_data(fetch_thingspeak_data(CHANNEL_1_ID, CHANNEL_1_API))
pred_df = process_predicted_data(fetch_thingspeak_data(CHANNEL_2_ID, CHANNEL_2_API))

# ---------------- MERGE DATA ----------------
df = pd.merge_asof(
    actual_df.sort_values("created_at"),
    pred_df.sort_values("created_at"),
    on="created_at",
    direction="nearest",
    tolerance=pd.Timedelta("5min")
)

# Remove mismatched rows
df = df.dropna()

# ---------------- LATEST VALUES ----------------
st.subheader("📌 Latest Values")

latest = df.iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("PM2.5", f"{latest['PM2.5']:.2f}", f"{latest['PM2.5_pred']:.2f}")
col2.metric("AQI", f"{latest['AQI_actual']:.2f}", f"{latest['AQI_pred']:.2f}")
col3.metric("CACI", f"{latest['CACI_actual']:.2f}", f"{latest['CACI_pred']:.2f}")
col4.metric("Temp (°C)", f"{latest['TEMP']:.2f}", f"{latest['TEMP_pred']:.2f}")
col5.metric("Humidity (%)", f"{latest['HUMIDITY']:.2f}", f"{latest['HUMIDITY_pred']:.2f}")

# ---------------- CHARTS ----------------
st.subheader("📊 Actual vs Predicted Graphs")

# PM2.5
st.markdown("### PM2.5")
st.line_chart(df.set_index("created_at")[["PM2.5", "PM2.5_pred"]])

# Temperature
st.markdown("### Temperature")
st.line_chart(df.set_index("created_at")[["TEMP", "TEMP_pred"]])

# Humidity
st.markdown("### Humidity")
st.line_chart(df.set_index("created_at")[["HUMIDITY", "HUMIDITY_pred"]])

# AQI
st.markdown("### AQI")
st.line_chart(df.set_index("created_at")[["AQI_actual", "AQI_pred"]])

# CACI
st.markdown("### CACI")
st.line_chart(df.set_index("created_at")[["CACI_actual", "CACI_pred"]])

# ---------------- ERROR ANALYSIS ----------------
st.subheader("📉 Prediction Error")

df["AQI_error"] = abs(df["AQI_actual"] - df["AQI_pred"])
df["CACI_error"] = abs(df["CACI_actual"] - df["CACI_pred"])
df["PM2.5_error"] = abs(df["PM2.5"] - df["PM2.5_pred"])
df["TEMP_error"] = abs(df["TEMP"] - df["TEMP_pred"])
df["HUMIDITY_error"] = abs(df["HUMIDITY"] - df["HUMIDITY_pred"])

st.line_chart(df.set_index("created_at")[
    ["AQI_error", "CACI_error", "PM2.5_error", "TEMP_error", "HUMIDITY_error"]
])

# ---------------- TABLE ----------------
st.subheader("📋 Recent Data")
st.dataframe(df.tail(10))

# ---------------- AUTO REFRESH ----------------
time.sleep(REFRESH_INTERVAL)
st.rerun()