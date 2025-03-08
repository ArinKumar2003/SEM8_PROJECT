import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import datetime
from prophet import Prophet
import plotly.graph_objects as go
import time

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- THEME TOGGLE ----
theme = st.sidebar.radio("🌗 Theme", ["Light Mode", "Dark Mode"])
if theme == "Dark Mode":
    st.markdown("""
        <style>
            body { background-color: #1E1E1E; color: white; }
            .stApp { background-color: #1E1E1E; }
        </style>
    """, unsafe_allow_html=True)

# ---- DASHBOARD HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🌍 AI Climate Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📡 Real-Time Weather, AI Forecasts & Advanced Analysis</h3>
    <hr>
""", unsafe_allow_html=True)

# ---- WEATHER API CONFIG ----
try:
    API_KEY = st.secrets["WEATHERSTACK_API_KEY"]
except KeyError:
    API_KEY = None

def get_live_weather(city):
    """Fetch real-time weather data."""
    if not API_KEY:
        return None
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    data = response.json()
    return data.get("current")

# ---- TABS ----
tabs = st.tabs([
    "🌦 Live Weather", "📈 AI Forecasts", "🔮 Trends", "📊 Climate Score", "⚠️ Extreme Weather", "🛰️ Satellite View"
])

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
    uploaded_file = st.file_uploader("Upload Climate CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("⚠️ Uploaded CSV is empty. Please upload a valid file.")
            elif "Years" in df.columns and "Temperature" in df.columns:
                df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
                df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})
                model = Prophet()
                model.fit(df)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends")
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Invalid CSV format. Required columns: Years, Month, Day, Temperature.")
        except pd.errors.EmptyDataError:
            st.error("⚠️ Error: The uploaded file is empty or corrupted.")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

# ---- TAB 3: INTERACTIVE TRENDS ----
with tabs[2]:
    st.subheader("🔮 Interactive Climate Trends")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
        df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})
        fig = px.scatter(df, x="ds", y="y", title="Temperature Trends Over Time")
        st.plotly_chart(fig)
        fig2 = px.histogram(df, x="y", title="Temperature Distribution")
        st.plotly_chart(fig2)

# ---- TAB 4: CLIMATE SCORE ----
with tabs[3]:
    st.subheader("📊 Climate Impact Score")
    if uploaded_file:
        df["climate_score"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min()) * 100
        fig = px.line(df, x="ds", y="climate_score", title="Climate Impact Score")
        st.plotly_chart(fig)

# ---- TAB 5: EXTREME WEATHER ----
with tabs[4]:
    st.subheader("⚠️ Extreme Weather Alerts")
    if uploaded_file:
        threshold = st.slider("Set Temperature Alert Threshold", int(df["y"].min()), int(df["y"].max()), 35)
        alerts = df[df["y"] > threshold]
        st.write("### 🔥 Heatwave Alerts", alerts)
        fig = px.line(df, x="ds", y="y", title="Extreme Temperature Trends", markers=True)
        fig.add_trace(go.Scatter(x=alerts["ds"], y=alerts["y"], mode="markers", marker=dict(color="red", size=10), name="Extreme Heat"))
        st.plotly_chart(fig)

# ---- TAB 6: SATELLITE VIEW ----
with tabs[5]:
    st.subheader("🛰️ Live Climate Satellite View")
    st.markdown("🚀 Integrate with OpenWeatherMap's Satellite API or Google Maps.")
    st.image("https://earthobservatory.nasa.gov/blogs/earthmatters/wp-content/uploads/sites/9/2019/05/earthmap.png")
