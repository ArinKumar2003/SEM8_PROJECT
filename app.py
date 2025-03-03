import streamlit as st
import requests
import pandas as pd

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- DASHBOARD HEADER ----
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🌍 AI Climate Change Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>📊 Live Weather & Future Climate Predictions</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---- WEATHERSTACK API CONFIG ----
API_KEY = st.secrets["WEATHERSTACK_API_KEY"]

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

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📂 Upload Climate Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.subheader("🤖 Choose Prediction Model")
    model_choice = st.radio("", ["Gradient Boosting", "LSTM", "Prophet"])

    st.subheader("🌦 Live Weather Data")
    city = st.text_input("Enter City", value="New York")
    if st.button("🔍 Get Live Weather"):
        st.session_state.weather_city = city  # Save city in session

# ---- LIVE WEATHER DISPLAY ----
st.subheader("🌦 Current Weather Conditions")
city = st.session_state.get("weather_city", "New York")
weather_data = get_live_weather(city)

if weather_data:
    temp = weather_data["temperature"]
    desc = weather_data["description"]
    humidity = weather_data["humidity"]
    wind_speed = weather_data["wind_speed"]

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; background: #ecf0f1; padding: 20px; border-radius: 12px;">
            <h2>🌆 {city}</h2>
            <h1>🌡 {temp}°C</h1>
            <h3>☁️ {desc}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="text-align: center; background: #ecf0f1; padding: 20px; border-radius: 12px;">
            <p>💧 Humidity: <b>{humidity}%</b></p>
            <p>🌬 Wind Speed: <b>{wind_speed} km/h</b></p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.error("❌ Unable to fetch weather data. Check city name or API key.")

st.markdown("---")

# ---- CLIMATE DATA PREDICTIONS ----
st.subheader("📈 Future Climate Predictions")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())  # Display first few rows
    st.success("✅ Data uploaded successfully! Choose a model to proceed.")
else:
    st.info("📂 Upload a CSV file to generate predictions.")

st.markdown("---")
st.markdown("<h4 style='text-align: center;'>🔍 AI-Driven Climate Insights for a Sustainable Future</h4>", unsafe_allow_html=True)
