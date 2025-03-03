import streamlit as st
import requests
import pandas as pd

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- DASHBOARD HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50; font-size: 36px;'>🌍 AI Climate Change Prediction Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📊 Live Weather & Future Climate Predictions</h3>
    <hr style="border:1px solid #ddd;">
""", unsafe_allow_html=True)

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

# ---- MAIN PAGE CONTENT ----
st.subheader("🌦 Live Weather Conditions")

# Input for city on the main page
city = st.text_input("Enter City", value="New York")
if st.button("🔍 Get Live Weather"):
    weather_data = get_live_weather(city)

    if weather_data:
        temp = weather_data["temperature"]
        desc = weather_data["description"]
        humidity = weather_data["humidity"]
        wind_speed = weather_data["wind_speed"]

        # Display live weather details in a visually appealing layout
        st.markdown(f"""
        <div style="text-align: center; background: #ecf0f1; padding: 20px; border-radius: 12px;">
            <h2>🌆 {city}</h2>
            <h1 style="color:#e74c3c;">🌡 {temp}°C</h1>
            <h3>☁️ {desc}</h3>
            <p>💧 Humidity: <b>{humidity}%</b></p>
            <p>🌬 Wind Speed: <b>{wind_speed} km/h</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Unable to fetch weather data. Check city name or API key.")
else:
    st.info("Enter a city name and click 'Get Live Weather' to see the conditions.")

st.markdown("<hr>", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📂 Upload Climate Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.subheader("🤖 Choose Prediction Model")
    model_choice = st.radio("", ["Gradient Boosting", "LSTM", "Prophet"])

# ---- CLIMATE DATA PREDICTIONS ----
st.subheader("📈 Future Climate Predictions")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())  # Display first few rows
    st.success("✅ Data uploaded successfully! Choose a model to proceed.")
else:
    st.info("📂 Upload a CSV file to generate predictions.")
