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

REFRESH_INTERVAL = 900
FETCH_RESULTS = 100

st.set_page_config(
    page_title="AI Air Quality Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f4efe7;
            --panel: rgba(255, 255, 255, 0.72);
            --panel-strong: rgba(255, 255, 255, 0.9);
            --text: #10242f;
            --muted: #465765;
            --border: rgba(23, 49, 63, 0.08);
            --shadow: 0 18px 50px rgba(20, 40, 52, 0.10);
            --teal: #2d7c7a;
            --green: #6d8b3d;
            --gold: #c58a25;
            --blue: #3a6ea5;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 224, 179, 0.55), transparent 30%),
                radial-gradient(circle at top right, rgba(140, 196, 255, 0.18), transparent 28%),
                linear-gradient(180deg, #fbf7f0 0%, #f4efe7 45%, #eef3f5 100%);
            color: var(--text);
            font-weight: 400;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.64));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.4rem 1.6rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
            backdrop-filter: blur(14px);
        }

        .hero-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.05;
            letter-spacing: -0.03em;
        }

        .hero p {
            margin: 0.45rem 0 0;
            color: #31424e;
            max-width: 60rem;
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .pill-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .pill {
            border: 1px solid rgba(45, 124, 122, 0.18);
            background: rgba(45, 124, 122, 0.12);
            color: #1d5f5d;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .pill.alt {
            border-color: rgba(58, 110, 165, 0.18);
            background: rgba(58, 110, 165, 0.12);
            color: #234f7f;
        }

        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,248,239,0.96));
            border: 1px solid rgba(23, 49, 63, 0.10);
            border-radius: 24px;
            padding: 1rem 1rem 0.9rem;
            box-shadow: 0 14px 34px rgba(20, 40, 52, 0.12);
            height: 100%;
            backdrop-filter: blur(14px);
        }

        .card-label {
            color: #29414d;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.48rem;
            font-weight: 700;
        }

        .card-value {
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.35rem;
            color: #0f2430;
            background: rgba(58, 110, 165, 0.08);
            display: inline-block;
            padding: 0.18rem 0.45rem;
            border-radius: 12px;
        }

        .card-meta {
            color: #31424e;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .card-meta .actual-line {
            display: inline-block;
            margin-bottom: 0.18rem;
            padding: 0.15rem 0.42rem;
            border-radius: 999px;
            background: rgba(45, 124, 122, 0.10);
            color: #1d5f5d;
            font-weight: 700;
        }

        .prediction-value {
            display: inline-block;
            margin-top: 0.24rem;
            padding: 0.2rem 0.45rem;
            border-radius: 999px;
            background: rgba(193, 122, 34, 0.10);
            color: #9c560e;
            font-weight: 700;
        }

        .card.good { border-left: 5px solid #4b8f5a; }
        .card.warning { border-left: 5px solid #d18a22; }
        .card.info { border-left: 5px solid #3a6ea5; }
        .card.soft { border-left: 5px solid #2d7c7a; }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 1.25rem 0 0.2rem;
            letter-spacing: -0.02em;
            color: #10242f;
        }

        .section-subtitle {
            color: #4d5d69;
            margin-bottom: 0.9rem;
            font-size: 0.94rem;
            line-height: 1.5;
        }

        .stMetric {
            background: rgba(255,255,255,0.55);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.8rem 0.9rem;
            box-shadow: var(--shadow);
        }

        div[data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.58);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 0.25rem 0.4rem 0.2rem;
            overflow: hidden;
        }

        .status-line {
            color: #526270;
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_stat_card(label, value, meta, tone="soft"):
    st.markdown(
        f"""
        <div class="card {tone}">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def format_timestamp(ts):
    if pd.isna(ts):
        return "Unavailable"
    return pd.Timestamp(ts).tz_convert("Asia/Kolkata").strftime("%d %b %Y, %I:%M %p IST")


def format_age_minutes(ts):
    if pd.isna(ts):
        return None
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)).total_seconds() / 60
    return int(age)

# ---------------- FETCH ----------------
def fetch_thingspeak_data(channel_id, api_key):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {
        "api_key": api_key,
        "results": FETCH_RESULTS,
        "_": int(time.time()),
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data.get("feeds", []))

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
        st.error(f"Error fetching ThingSpeak data: {e}")
        return pd.DataFrame()

# ---------------- PROCESS ----------------
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

    for col in ["TEMP","HUMIDITY","PM2.5","PM10","AQI_actual","CACI_actual"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def process_predicted_data(df):
    df = df.rename(columns={
        "field1": "AQI_pred",
        "field2": "CACI_pred",
        "field3": "PM2.5_pred",
        "field4": "TEMP_pred",
        "field5": "HUMIDITY_pred"
    })

    for col in ["AQI_pred","CACI_pred","PM2.5_pred","TEMP_pred","HUMIDITY_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ---------------- STATUS ----------------
def display_value(val):
    return "N/A" if pd.isna(val) else f"{float(val):.2f}"


def get_aqi_status(aqi):
    if pd.isna(aqi): return "N/A"
    if aqi <= 50: return "GOOD"
    elif aqi <= 100: return "SATISFACT"
    elif aqi <= 200: return "MODERATE"
    elif aqi <= 300: return "POOR"
    elif aqi <= 400: return "VERY POOR"
    else: return "SEVERE"

def get_caci_status(caci):
    if pd.isna(caci): return "N/A"
    if caci <= 25: return "OPTIMAL"
    elif caci <= 50: return "GOOD"
    elif caci <= 75: return "MODERATE"
    else: return "WARNING"

def get_temp_status(temp):
    if pd.isna(temp): return "N/A"
    if temp < 10: return "VERY COLD"
    elif temp < 18: return "COLD"
    elif temp <= 26: return "COMFORT"
    elif temp <= 30: return "WARM"
    elif temp <= 35: return "HOT"
    else: return "VERY HOT"

def get_hum_status(hum):
    if pd.isna(hum): return "N/A"
    if hum < 20: return "VERY DRY"
    elif hum < 40: return "DRY"
    elif hum <= 60: return "COMFORT"
    elif hum <= 70: return "HUMID"
    else: return "VERY HUMID"

def get_pm_status(pm):
    if pd.isna(pm): return "N/A"
    if pm <= 12: return "GOOD"
    elif pm <= 35: return "MODERATE"
    elif pm <= 55: return "UNHLTHY-SG"
    elif pm <= 150: return "UNHEALTHY"
    else: return "HAZARDOUS"

# ---------------- SAFE ----------------
def safe(val):
    return None if pd.isna(val) else float(val)

# ---------------- MERGE (ONLY FOR GRAPHS) ----------------
def merge_for_graph(actual_df, pred_df):
    if actual_df.empty or pred_df.empty:
        return pd.DataFrame()

    actual_df = actual_df.sort_values("created_at")
    pred_df = pred_df.sort_values("created_at")

    df = pd.merge_asof(
        actual_df,
        pred_df,
        on="created_at",
        direction="backward",
        tolerance=pd.Timedelta("20min")
    )

    pred_cols = ["AQI_pred","CACI_pred","PM2.5_pred","TEMP_pred","HUMIDITY_pred"]
    df[pred_cols] = df[pred_cols].ffill()

    return df

# ---------------- PLOT ----------------
def plot_graph(df, actual, pred, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["created_at"],
            y=df[actual],
            name="Actual",
            mode="lines",
            line=dict(color="#1f6f8b", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["created_at"],
            y=df[pred],
            name="Predicted",
            mode="lines",
            line=dict(color="#c77d36", width=3, dash="dot"),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=360,
        margin=dict(l=24, r=24, t=52, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17313f"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(23,49,63,0.08)", zeroline=False)
    return fig

# ---------------- UI ----------------
if st.button("Refresh"):
    st.rerun()

# ---------------- LOAD ----------------
actual_df = process_actual_data(fetch_thingspeak_data(CHANNEL_1_ID, CHANNEL_1_API))
pred_df = process_predicted_data(fetch_thingspeak_data(CHANNEL_2_ID, CHANNEL_2_API))

if actual_df.empty or pred_df.empty:
    st.warning("No data available")
    st.stop()

# 🔥 IMPORTANT: latest values separately
latest_actual = actual_df.iloc[-1]
latest_pred = pred_df.iloc[-1]
latest_time = max(actual_df["created_at"].iloc[-1], pred_df["created_at"].iloc[-1])
last_updated = format_timestamp(latest_time)
age_minutes = format_age_minutes(latest_time)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-top">
            <div>
                <h1>AI Air Quality Dashboard</h1>
                <p>Live monitoring of actual sensor readings and 1-hour-ahead predictions from ThingSpeak. The dashboard highlights current conditions, forecast values, and trend comparisons in a single view.</p>
            </div>
        </div>
        <div class="pill-row">
            <span class="pill">Live feed</span>
            <span class="pill alt">Predictions synced from channel 2</span>
            <span class="pill">Last update: {last_updated}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- METRICS ----------------
section_header("Latest Values", "Current readings are shown with their paired predicted values beneath each card.")

col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

with col1:
    render_stat_card(
        "PM2.5",
        f"{display_value(latest_actual['PM2.5'])}",
        f"<span class='actual-line'>Actual: {get_pm_status(latest_actual['PM2.5'])}</span><br><span class='prediction-value'>Pred: {display_value(latest_pred['PM2.5_pred'])} ({get_pm_status(latest_pred['PM2.5_pred'])})</span>",
        "warning" if pd.notna(latest_actual['PM2.5']) and latest_actual['PM2.5'] > 35 else "good",
    )

with col2:
    render_stat_card(
        "AQI",
        f"{display_value(latest_actual['AQI_actual'])}",
        f"<span class='actual-line'>Actual: {get_aqi_status(latest_actual['AQI_actual'])}</span><br><span class='prediction-value'>Pred: {display_value(latest_pred['AQI_pred'])} ({get_aqi_status(latest_pred['AQI_pred'])})</span>",
        "warning" if pd.notna(latest_actual['AQI_actual']) and latest_actual['AQI_actual'] > 100 else "info",
    )

with col3:
    render_stat_card(
        "CACI",
        f"{display_value(latest_actual['CACI_actual'])}",
        f"<span class='actual-line'>Actual: {get_caci_status(latest_actual['CACI_actual'])}</span><br><span class='prediction-value'>Pred: {display_value(latest_pred['CACI_pred'])} ({get_caci_status(latest_pred['CACI_pred'])})</span>",
        "good" if pd.notna(latest_actual['CACI_actual']) and latest_actual['CACI_actual'] <= 25 else "soft",
    )

with col4:
    render_stat_card(
        "Temp",
        f"{display_value(latest_actual['TEMP'])} °C",
        f"<span class='actual-line'>Actual: {get_temp_status(latest_actual['TEMP'])}</span><br><span class='prediction-value'>Pred: {display_value(latest_pred['TEMP_pred'])} °C ({get_temp_status(latest_pred['TEMP_pred'])})</span>",
        "info",
    )

with col5:
    render_stat_card(
        "Humidity",
        f"{display_value(latest_actual['HUMIDITY'])} %",
        f"<span class='actual-line'>Actual: {get_hum_status(latest_actual['HUMIDITY'])}</span><br><span class='prediction-value'>Pred: {display_value(latest_pred['HUMIDITY_pred'])} % ({get_hum_status(latest_pred['HUMIDITY_pred'])})</span>",
        "soft",
    )

if age_minutes is not None:
    st.markdown(
        f'<div class="status-line">Data freshness: {age_minutes} minutes old</div>',
        unsafe_allow_html=True,
    )

# ---------------- GRAPHS ----------------
df = merge_for_graph(actual_df, pred_df)

section_header("Trends", "Historical comparisons of actual versus predicted readings.")

st.plotly_chart(plot_graph(df, "PM2.5", "PM2.5_pred", "PM2.5"))
st.plotly_chart(plot_graph(df, "AQI_actual", "AQI_pred", "AQI"))
st.plotly_chart(plot_graph(df, "CACI_actual", "CACI_pred", "CACI"))
st.plotly_chart(plot_graph(df, "TEMP", "TEMP_pred", "Temperature"))
st.plotly_chart(plot_graph(df, "HUMIDITY", "HUMIDITY_pred", "Humidity"))

# ---------------- ERROR ----------------
section_header("Prediction Error", "Absolute error for the quality indices.")

df["AQI_error"] = abs(df["AQI_actual"] - df["AQI_pred"])
df["CACI_error"] = abs(df["CACI_actual"] - df["CACI_pred"])

st.line_chart(df[["AQI_error", "CACI_error"]])

# ---------------- AUTO REFRESH ----------------
time.sleep(REFRESH_INTERVAL)
st.rerun()