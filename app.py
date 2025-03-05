import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- DARK / LIGHT MODE ----
mode = st.sidebar.radio("Choose Mode:", ["🌙 Dark Mode", "☀️ Light Mode"])
theme = "dark" if mode == "🌙 Dark Mode" else "light"

# ---- WEATHERSTACK API CONFIG ----
API_KEY = "YOUR_WEATHERSTACK_API_KEY"

def get_live_weather(city):
    """Fetches real-time weather data from Weatherstack API."""
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    data = response.json()

    if "current" in data:
        return {
            "temperature": data["current"]["temperature"],
            "description": data["current"]["weather_descriptions"][0],
            "humidity": data["current"]["humidity"],
            "wind_speed": data["current"]["wind_speed"]
        }
    else:
        return None

# ---- MAIN APP ----
st.title("🌍 AI Climate Change Prediction Dashboard")
tabs = st.tabs(["🌦 Live Weather", "📂 Upload Data", "📈 Predictions", "📊 Analytics"])

# ---- TAB 1: LIVE WEATHER ----
with tabs[0]:
    st.subheader("🌦 Live Weather Conditions")
    city = st.text_input("Enter City", value="New York")

    if st.button("🔍 Get Live Weather"):
        weather_data = get_live_weather(city)
        if weather_data:
            temp, desc, humidity, wind_speed = weather_data.values()

            # Display weather details
            st.markdown(f"""
            <div st
