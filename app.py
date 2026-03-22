import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
CHANNEL_1_ID = "3220962"
CHANNEL_1_API = "TF7VPOAMFV8XK33V"

CHANNEL_2_ID = "3124366"
CHANNEL_2_API = "L06LTBC1KWFZG75X"

REFRESH_INTERVAL = 900  # 10 minutes
FETCH_RESULTS = 100
DISPLAY_TIMEZONE = "Asia/Kolkata"

# ---------------- FETCH DATA ----------------
def fetch_thingspeak_data(channel_id, api_key):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    
    params = {
        "api_key": api_key,
        "results": FETCH_RESULTS,
        "_": int(time.time())  # prevent intermediary caching
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["feeds"])

        if df.empty:
            return df

        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df = (
            df.dropna(subset=["created_at"])
              .drop_duplicates(subset=["created_at"], keep="last")
              .sort_values("created_at")
              .reset_index(drop=True)
        )

        return df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


# ---------------- PROCESS ACTUAL ----------------
def process_actual_data(df):
    if df.empty:
        return df

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

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # ✅ SAFE

    return df


# ---------------- PROCESS PREDICTED ----------------
def process_predicted_data(df):
    if df.empty:
        return df

    df = df.rename(columns={
        "field1": "AQI_pred",
        "field2": "CACI_pred",
        "field3": "PM2.5_pred",
        "field4": "TEMP_pred",
        "field5": "HUMIDITY_pred"
    })

    cols = ["AQI_pred", "CACI_pred", "PM2.5_pred", "TEMP_pred", "HUMIDITY_pred"]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # ✅ SAFE

    return df


# ---------------- MERGE ----------------
def merge_data(actual_df, pred_df):
    if actual_df.empty or pred_df.empty:
        return pd.DataFrame()

    # Ensure merge key is datetime64 for merge_asof
    actual_df = actual_df.copy()
    pred_df = pred_df.copy()
    actual_df["created_at"] = pd.to_datetime(actual_df["created_at"], errors="coerce", utc=True)
    pred_df["created_at"] = pd.to_datetime(pred_df["created_at"], errors="coerce", utc=True)

    actual_df = actual_df.dropna(subset=["created_at"]).sort_values("created_at")
    pred_df = pred_df.dropna(subset=["created_at"]).sort_values("created_at")

    if actual_df.empty or pred_df.empty:
        return pd.DataFrame()

    # Merge on numeric timestamp to avoid dtype edge-cases in merge_asof.
    actual_df["merge_ts"] = actual_df["created_at"].astype("int64")
    pred_df["merge_ts"] = pred_df["created_at"].astype("int64")

    pred_for_merge = pred_df.drop(columns=["created_at"])

    df = pd.merge_asof(
        actual_df,
        pred_for_merge,
        on="merge_ts",
        direction="backward",
        tolerance=int(pd.Timedelta("10min").value)   # 10 minutes in ns for int64 key
    )

    df = df.drop(columns=["merge_ts"])

    # ✅ DO NOT DROP → fill instead
    pred_cols = ["AQI_pred", "CACI_pred", "PM2.5_pred", "TEMP_pred", "HUMIDITY_pred"]
    df[pred_cols] = df[pred_cols].ffill()

    return df


# ---------------- 1 HOUR VALUE ----------------
def get_1hr_ago(df, column):
    latest_time = df["created_at"].iloc[-1]
    past_time = latest_time - pd.Timedelta(hours=1)

    past_row = df[df["created_at"] <= past_time].tail(1)

    if not past_row.empty:
        return past_row[column].values[0]
    return df[column].iloc[0]


# ---------------- PLOT FUNCTION ----------------
def plot_graph(df, actual, pred, title):
    x_axis = "created_at_plot" if "created_at_plot" in df.columns else "created_at"
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_axis],
        y=df[actual],
        mode='lines',
        name='Actual'
    ))

    fig.add_trace(go.Scatter(
        x=df[x_axis],
        y=df[pred],
        mode='lines',
        name='Predicted'
    ))

    fig.update_layout(title=title)

    return fig


# ---------------- UI ----------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌍 AI Air Quality Monitoring Dashboard")

if st.button("Refresh Now"):
    st.rerun()

# ---------------- LOAD ----------------
actual_df = process_actual_data(fetch_thingspeak_data(CHANNEL_1_ID, CHANNEL_1_API))
pred_df = process_predicted_data(fetch_thingspeak_data(CHANNEL_2_ID, CHANNEL_2_API))

df = merge_data(actual_df, pred_df)

if df.empty:
    st.warning("No data available")
    st.stop()

df["created_at_local"] = df["created_at"].dt.tz_convert(DISPLAY_TIMEZONE)
df["created_at_plot"] = df["created_at_local"].dt.tz_localize(None)

# ---------------- METRICS ----------------
latest = df.iloc[-1]

st.subheader("📌 Latest Values (1 Hour Comparison)")

col1, col2, col3, col4, col5 = st.columns(5)

def show_metric(col, label, column):
    past = get_1hr_ago(df, column)
    delta = latest[column] - past

    col.metric(label, f"{latest[column]:.2f}", f"{delta:.2f}")

show_metric(col1, "PM2.5", "PM2.5")
show_metric(col2, "AQI", "AQI_actual")
show_metric(col3, "CACI", "CACI_actual")
show_metric(col4, "Temp (°C)", "TEMP")
show_metric(col5, "Humidity (%)", "HUMIDITY")

# ---------------- CHARTS ----------------
st.subheader("📊 Actual vs Predicted")

st.plotly_chart(plot_graph(df, "PM2.5", "PM2.5_pred", "PM2.5"), width="stretch")
st.plotly_chart(plot_graph(df, "TEMP", "TEMP_pred", "Temperature"), width="stretch")
st.plotly_chart(plot_graph(df, "HUMIDITY", "HUMIDITY_pred", "Humidity"), width="stretch")
st.plotly_chart(plot_graph(df, "AQI_actual", "AQI_pred", "AQI"), width="stretch")
st.plotly_chart(plot_graph(df, "CACI_actual", "CACI_pred", "CACI"), width="stretch")

# ---------------- ERROR ----------------
st.subheader("📉 Prediction Error")

for col in ["AQI", "CACI", "PM2.5", "TEMP", "HUMIDITY"]:
    df[f"{col}_error"] = abs(df[f"{col}_actual" if col in ["AQI", "CACI"] else col] - df[f"{col}_pred"])

st.line_chart(df.set_index("created_at_plot")[
    ["AQI_error", "CACI_error", "PM2.5_error", "TEMP_error", "HUMIDITY_error"]
])

# ---------------- DEBUG ----------------
latest_local = latest["created_at"].tz_convert(DISPLAY_TIMEZONE)
data_age_mins = (pd.Timestamp.now(tz="UTC") - latest["created_at"]).total_seconds() / 60

st.caption(f"Last update ({DISPLAY_TIMEZONE}): {latest_local:%Y-%m-%d %H:%M:%S}")
if data_age_mins > 20:
    st.warning(f"Data appears stale: last point is {int(data_age_mins)} minutes old.")

# ---------------- AUTO REFRESH ----------------
time.sleep(REFRESH_INTERVAL)
st.rerun()