import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from prophet import Prophet

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- CUSTOM STYLING ----
st.markdown("""
    <style>
    .main-title { text-align: center; color: #2c3e50; font-size: 48px; font-weight: bold; margin-bottom: 10px; }
    .sub-title { text-align: center; color: #7f8c8d; font-size: 22px; margin-bottom: 30px; }
    .weather-card { background: #ecf0f1; padding: 15px; border-radius: 12px; text-align: center; margin-top: 20px; }
    .hot { background: linear-gradient(to right, #ff416c, #ff4b2b); color: white; }
    .mild { background: linear-gradient(to right, #6dd5ed, #2193b0); color: white; }
    .cold { background: linear-gradient(to right, #3498db, #2c3e50); color: white; }
    </style>
""", unsafe_allow_html=True)

# ---- DASHBOARD HEADER ----
st.markdown("<h1 class='main-title'>🌍 AI Climate Change Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-title'>📊 Analyze global climate trends, visualize insights, and predict future conditions with AI.</h3>", unsafe_allow_html=True)
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

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    try:
        gb_model = joblib.load("climate_gb_model.pkl")
        lstm_model = load_model("climate_lstm_model.keras")
        return gb_model, lstm_model
    except Exception as e:
        st.error(f"🚨 Error loading models: {e}")
        return None, None

gb_model, lstm_model = load_models()

# ---- SIDEBAR ----
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# ---- LIVE WEATHER DISPLAY ----
st.sidebar.markdown("### 🌦 Live Weather Data")
city = st.sidebar.text_input("Enter City", value="New York")

if st.sidebar.button("🔍 Get Live Weather"):
    weather_data = get_live_weather(city)
    
    if weather_data:
        temp = weather_data["temperature"]
        desc = weather_data["description"]
        humidity = weather_data["humidity"]
        wind_speed = weather_data["wind_speed"]

        # Dynamic Background Based on Temperature
        bg_class = "hot" if temp > 30 else "mild" if temp > 15 else "cold"

        st.markdown(f"""
        <div class="weather-card {bg_class}">
            <h2>🌆 {city}</h2>
            <h1>🌡 {temp}°C</h1>
            <h3>☁️ {desc}</h3>
            <p>💧 Humidity: {humidity}% | 🌬 Wind Speed: {wind_speed} km/h</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("❌ Unable to fetch weather data.")

# ---- MODEL SELECTION ----
model_choice = st.sidebar.radio("🤖 Choose Prediction Model", ["Gradient Boosting", "LSTM", "Prophet"])

# ---- MANUAL PREDICTION INPUT ----
st.sidebar.markdown("### 🔢 Manual Input for Prediction")
year_input = st.sidebar.slider("Year", 1900, 2100, 2025)
month_input = st.sidebar.slider("Month", 1, 12, 6)
day_input = st.sidebar.slider("Day", 1, 31, 15)
co2_input = st.sidebar.number_input("CO2 Level (ppm)", min_value=200, max_value=600, value=400)
humidity_input = st.sidebar.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
sealevel_input = st.sidebar.number_input("Sea Level Rise (mm)", min_value=0, max_value=500, value=100)

manual_input = pd.DataFrame({
    "Years": [year_input],
    "Month": [month_input],
    "Day": [day_input],
    "CO2": [co2_input],
    "Humidity": [humidity_input],
    "SeaLevel": [sealevel_input]
})

# ---- DATA PROCESSING ----
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predictions", "🛠️ Manual Prediction"])

    with tab1:
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)
        st.write("### 📊 Data Summary")
        st.write(df.describe())

    with tab2:
        feature = st.selectbox("Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])
        fig = px.line(df, x="Years", y=feature, title=f"{feature} Trends Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        required_features = ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel"]

        if all(col in df.columns for col in required_features):
            X_new = df[required_features]
            predictions = gb_model.predict(X_new) if model_choice == "Gradient Boosting" else lstm_model.predict(np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))).flatten()
            df["Predicted Temperature"] = predictions
            df["Weather Description"] = np.where(df["Predicted Temperature"] > 30, "☀️ Hot", "🌥 Mild" if df["Predicted Temperature"] > 15 else "❄️ Cold")

            st.write("### 🔥 Predictions")
            st.dataframe(df[["Years", "Predicted Temperature", "Weather Description"]])

            fig_pred = px.line(df, x="Years", y="Predicted Temperature", title="Predicted Temperature Trends")
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning("🚨 The dataset is missing required columns!")

    with tab4:
        manual_prediction = gb_model.predict(manual_input)[0] if model_choice == "Gradient Boosting" else lstm_model.predict(np.array(manual_input).reshape((1, manual_input.shape[1], 1))).flatten()[0]
        weather_desc = "☀️ Hot" if manual_prediction > 30 else "🌥 Mild" if manual_prediction > 15 else "❄️ Cold"
        st.metric(label="🌡️ Predicted Temperature (°C)", value=f"{manual_prediction:.2f}")
        st.success(f"🌦 Expected Weather: {weather_desc}")

df.to_csv("predictions.csv", index=False)
st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"), file_name="predictions.csv", mime="text/csv")
