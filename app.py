import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import datetime

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- THEME TOGGLE ----
theme = st.sidebar.radio("🌗 Theme", ["Light Mode", "Dark Mode"])
if theme == "Dark Mode":
    st.markdown("""
        <style>
            body, .stApp { background-color: #1E1E1E; color: white; }
            hr { border-color: white; }
            .sidebar .sidebar-content { background-color: #2C2F33; }
        </style>
    """, unsafe_allow_html=True)

# ---- API CONFIG ----
API_KEY = st.secrets.get("WEATHERSTACK_API_KEY")

@st.cache_data
def get_live_weather(city):
    """Fetch real-time weather data."""
    if not API_KEY:
        return None
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    return response.json().get("current")

@st.cache_data
def get_satellite_image(city):
    """Fetch satellite weather image (placeholder if API unavailable)."""
    if not API_KEY:
        return "https://earthobservatory.nasa.gov/blogs/earthmatters/wp-content/uploads/sites/9/2019/05/earthmap.png"
    return f"http://api.weatherstack.com/satellite?access_key={API_KEY}&query={city}&layer=infrared"

# ---- FILE UPLOAD ----
uploaded_file = st.sidebar.file_uploader("📂 Upload Climate CSV", type=["csv"])
df = None

@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
        if data.empty:
            return None, "⚠️ Uploaded file is empty."
        if not all(col in data.columns for col in ["Years", "Temperature"]):
            return None, "⚠️ Invalid CSV format. Required: Years, Temperature."

        # Handle datetime conversion
        if all(col in data.columns for col in ["Years", "Month", "Day"]):
            data["ds"] = pd.to_datetime(data[["Years", "Month", "Day"]])
        else:
            data["ds"] = pd.to_datetime(data["Years"], format="%Y")

        data = data[["ds", "Temperature"]].rename(columns={"Temperature": "y"})
        return data, None
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

if uploaded_file:
    df, error_message = load_data(uploaded_file)
    if error_message:
        st.sidebar.error(error_message)

# ---- NAVIGATION FIX ----
menu = st.sidebar.radio("📌 Select a Section", [
    "🌦 Live Weather", "📈 AI Forecasts", "🔮 Trends", "📊 Climate Score",
    "⚠️ Extreme Weather", "🛰️ Satellite View", "🌍 Air Quality", "🌱 Carbon Footprint", "📰 Climate News"
])

# ---- TAB 1: LIVE WEATHER ----
if menu == "🌦 Live Weather":
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
if menu == "📈 AI Forecasts":
    st.subheader("📈 AI Climate Forecasts")
    if df is not None and len(df) > 2:
        with st.spinner("🔄 Training AI Model..."):
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends")
        st.plotly_chart(fig)
    elif df is not None:
        st.error("⚠️ Not enough data to train AI model.")

# ---- TAB 3: INTERACTIVE TRENDS ----
if menu == "🔮 Trends":
    st.subheader("🔮 Interactive Climate Trends")
    if df is not None:
        fig1 = px.scatter(df, x="ds", y="y", title="Temperature Trends Over Time", trendline="lowess")
        st.plotly_chart(fig1)

# ---- TAB 4: CLIMATE SCORE ----
if menu == "📊 Climate Score":
    st.subheader("📊 Climate Impact Score")
    if df is not None:
        df["climate_score"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min()) * 100
        fig = px.line(df, x="ds", y="climate_score", title="Climate Impact Score")
        st.plotly_chart(fig)

# ---- TAB 5: EXTREME WEATHER ----
if menu == "⚠️ Extreme Weather":
    st.subheader("⚠️ Extreme Weather Alerts")
    if df is not None:
        threshold = st.slider("Set Temperature Alert Threshold", int(df["y"].min()), int(df["y"].max()), 35)
        alerts = df[df["y"] > threshold]
        st.write("### 🔥 Heatwave Alerts", alerts)

# ---- TAB 6: SATELLITE VIEW ----
if menu == "🛰️ Satellite View":
    st.subheader("🛰️ Live Climate Satellite View")
    city = st.text_input("Enter City for Satellite View", "New York")
    if st.button("🛰️ Get Satellite View"):
        satellite_url = get_satellite_image(city)
        st.image(satellite_url, caption=f"Satellite View of {city}")

# ---- TAB 7: AIR QUALITY ----
if menu == "🌍 Air Quality":
    st.subheader("🌍 Air Quality Index (AQI)")
    st.write("🚀 Integrate with OpenWeather API for real-time AQI data.")
    st.image("https://www.iqair.com/assets/img/aqi-us-en.png")

# ---- TAB 8: CARBON FOOTPRINT ----
if menu == "🌱 Carbon Footprint":
    st.subheader("🌱 Carbon Footprint Tracker")
    st.write("⚡ Estimate your personal carbon footprint based on travel, energy use, and consumption.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5f/Carbon_Footprint.png")

# ---- TAB 9: CLIMATE NEWS ----
if menu == "📰 Climate News":
    st.subheader("📰 Latest Climate News")
    st.write("🌎 Stay updated with real-time climate news from global sources.")
    st.image("https://www.un.org/sites/un2.un.org/files/2021/08/ipcc_sixth_assessment_report.jpg")
