import pandas as pd
import numpy as np

# ==============================
# 1️⃣ LOAD DATASET
# ==============================
df = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228.csv")

# ==============================
# 2️⃣ CREATE TIMESTAMP
# ==============================
df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

# ==============================
# 3️⃣ CALCULATE HUMIDITY FROM DEW POINT
# ==============================
df["humidity"] = 100 * (
    np.exp((17.625 * df["DEWP"]) / (243.04 + df["DEWP"])) /
    np.exp((17.625 * df["TEMP"]) / (243.04 + df["TEMP"]))
)

# ==============================
# 4️⃣ CREATE gasValue (MATCH MQ135 RANGE)
# ==============================
df["gasValue"] = (
    0.3 * df["CO"] +
    0.2 * df["NO2"] +
    0.2 * df["SO2"] +
    0.3 * df["O3"]
)

# ==============================
# 5️⃣ AQI (ESP32 LOGIC - LINEAR)
# ==============================
def calculate_aqi(pm):
    if pm <= 12.0:
        return (50.0 / 12.0) * pm
    elif pm <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (pm - 12.1) + 51
    elif pm <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm - 35.5) + 101
    elif pm <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (pm - 55.5) + 151
    elif pm <= 250.4:
        return ((300 - 201) / (250.4 - 150.5)) * (pm - 150.5) + 201
    else:
        return 400

df["AQI"] = df["PM2.5"].apply(calculate_aqi)

# ==============================
# 6️⃣ CACI (MATCH ESP32 LOGIC)
# ==============================
def calculate_caci(temp, hum, aqi, gas):

    temp_score = np.clip(100 - abs(temp - 25) * 4, 0, 100)
    hum_score = np.clip(100 - abs(hum - 50) * 2, 0, 100)
    pollution_score = np.clip(100 - (aqi / 3.0), 0, 100)

    # 🔥 Gas thresholds SAME as ESP32
    if gas < 200:
        gas_score = 100
    elif gas < 600:
        gas_score = 70
    elif gas < 1200:
        gas_score = 40
    else:
        gas_score = 10

    caci = (
        0.35 * pollution_score +
        0.25 * temp_score +
        0.25 * hum_score +
        0.15 * gas_score
    )

    return np.clip(caci, 0, 100)


df["CACI"] = df.apply(
    lambda row: calculate_caci(
        row["TEMP"],
        row["humidity"],
        row["AQI"],
        row["gasValue"]
    ),
    axis=1
)

# ==============================
# 7️⃣ KEEP ONLY REQUIRED COLUMNS
# ==============================
df = df[[
    "timestamp",
    "PM2.5",
    "PM10",
    "TEMP",
    "humidity",
    "gasValue",
    "AQI",
    "CACI"
]]

# ==============================
# 8️⃣ REMOVE MISSING VALUES
# ==============================
df = df.dropna()

# ==============================
# 9️⃣ SAVE FINAL DATASET
# ==============================
df.to_csv("final_sensor_dataset.csv", index=False)

# ==============================
# 🔟 DEBUG CHECK
# ==============================
print("✅ FINAL DATASET READY (ESP32 MATCHED)")
print(df.head())

print("\n📊 Gas Value Range:")
print(df["gasValue"].describe())