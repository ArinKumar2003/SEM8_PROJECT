import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import datetime
from prophet import Prophet
import plotly.graph_objects as go

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- THEME TOGGLE ----
theme = st.sidebar.radio("🌗 Theme", ["Light Mode", "Dark Mode"])
if theme == "Dark Mode":
    st.markdown("""
        <style>
            body, .stApp { background-color: #1E1E1E; color: white; }
            hr { border-color: white; }
        </style>
    """, unsafe_allow_html=True)

# ---- DASHBOARD HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🌍 AI Climate Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📡 Real-Time Weather, AI Forecasts & Advanced Analysis</h3>
    <hr>
""", unsafe_allow_html=True)

# ---- WEATHER API CONFIG ----
API_KEY = st.secrets.get("WEATHERSTACK_API_KEY")

def get_live_weather(city):
    """Fetch real-time weather data."""
    if not API_KEY:
        return None
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    return response.json().get("current")

# ---- TABS ----
tabs = st.tabs(["🌦 Live Weather", "📈 AI Forecasts", "🔮 Trends", "📊 Climate Score", "⚠️ Extreme Weather", "🛰️ Satellite View"])

# ---- FILE UPLOAD ----
uploaded_file = st.sidebar.file_uploader("📂 Upload Climate CSV", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.sidebar.error("⚠️ The uploaded CSV is empty.")
            df = None
        elif not all(col in df.columns for col in ["Years", "Temperature"]):
            st.sidebar.error("⚠️ Invalid CSV format. Required: Years, Temperature.")
            df = None
        else:
            # Ensure proper datetime format
            if all(col in df.columns for col in ["Years", "Month", "Day"]):
                df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
            else:
                df["ds"] = pd.to_datetime(df["Years"], format="%Y")

            df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})

    except pd.errors.EmptyDataError:
        st.sidebar.error("⚠️ Uploaded file is empty or corrupted.")
        df = None
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- TAB 1: LIVE WEATHER ----
with tabs[0]:
    st.subheader("🌦 Live Weather")
    cities = st.text_input("Enter Cities (comma-separated)", "New York, London, Tokyo")
    city_list = [city.strip() for city in cities.split(",")]

    if st.button("🔍 Get Live Weather"):
        for city in city_list:
            weather = get_live_weather(city)
            if weather:
                st.write(f"### {city}")
                st.metric("Temperature", f"{weather['temperature']}°C")
                st.write(f"**☁️ {weather['weather_descriptions'][0]}**")
                st.write(f"💧 Humidity: {weather['humidity']}%  |  🌬 Wind: {weather['wind_speed']} km/h")
            else:
                st.error(f"❌ No data for {city}")

# ---- TAB 2: AI FORECASTS ----
with tabs[1]:
    st.subheader("📈 AI Climate Forecasts")

    if df is not None and len(df) > 1:  # Ensure enough data for training
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends")
        st.plotly_chart(fig)
    elif df is not None:
        st.error("⚠️ Not enough data to train AI model.")

# ---- TAB 3: INTERACTIVE TRENDS ----
with tabs[2]:
    st.subheader("🔮 Interactive Climate Trends")
    if df is not None:
        fig1 = px.scatter(df, x="ds", y="y", title="Temperature Trends Over Time")
        st.plotly_chart(fig1)
        
        fig2 = px.histogram(df, x="y", title="Temperature Distribution")
        st.plotly_chart(fig2)

# ---- TAB 4: CLIMATE SCORE ----
with tabs[3]:
    st.subheader("📊 Climate Impact Score")
    if df is not None:
        df["climate_score"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min()) * 100
        fig = px.line(df, x="ds", y="climate_score", title="Climate Impact Score")
        st.plotly_chart(fig)

# ---- TAB 5: EXTREME WEATHER ----
with tabs[4]:
    st.subheader("⚠️ Extreme Weather Alerts")
    if df is not None:
        threshold = st.slider("Set Temperature Alert Threshold", int(df["y"].min()), int(df["y"].max()), 35)
        alerts = df[df["y"] > threshold]

        st.write("### 🔥 Heatwave Alerts", alerts)
        fig = px.line(df, x="ds", y="y", title="Extreme Temperature Trends", markers=True)
        fig.add_trace(go.Scatter(x=alerts["ds"], y=alerts["y"], mode="markers", marker=dict(color="red", size=10), name="Extreme Heat"))
        st.plotly_chart(fig)

# ---- TAB 6: SATELLITE VIEW ----
with tabs[5]:
    st.subheader("🛰️ Live Climate Satellite View")
    st.image("https://earthobservatory.nasa.gov/blogs/earthmatters/wp-content/uploads/sites/9/2019/05/earthmap.png")
