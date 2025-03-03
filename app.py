import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from prophet import Prophet
import requests

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 Climate Change Dashboard", layout="wide")

# ---- DASHBOARD HEADER ----
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌍 Climate Change Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>📊 Analyze trends, visualize data, and predict future climate conditions.</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---- LOAD API KEY FROM SECRETS ----
API_KEY = st.secrets["fd6db116aab81dbc975b89c502692ac0"]

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    try:
        gb_model = joblib.load("climate_gb_model.pkl")  # Gradient Boosting Model
        lstm_model = load_model("climate_lstm_model.keras")  # LSTM Model
        return gb_model, lstm_model
    except Exception as e:
        st.error(f"🚨 Error loading models: {e}")
        return None, None

gb_model, lstm_model = load_models()

# ---- LIVE WEATHER FUNCTION ----
def get_live_weather(city):
    """Fetches real-time weather data from Weatherstack API."""
    url = f"http://api.weatherstack.com/current?access_key=fd6db116aab81dbc975b89c502692ac0&query=New York"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "current" in data:
            return {
                "temperature": data["current"]["temperature"],
                "weather_desc": data["current"]["weather_descriptions"][0],
                "humidity": data["current"]["humidity"],
                "wind_speed": data["current"]["wind_speed"]
            }
    return None

# ---- SIDEBAR ----
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Sidebar: Model Selection
model_choice = st.sidebar.radio("🤖 Choose Prediction Model", ["Gradient Boosting", "LSTM", "Prophet"])

# Sidebar: City for Live Weather
city = st.sidebar.text_input("🌍 Enter City for Live Weather", "New York")

# Sidebar: Manual Prediction Input
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

# ---- MAIN CONTENT ----
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---- TABS FOR NAVIGATION ----
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predictions", "🌍 Live Weather"])

    # 📊 ---- DATA OVERVIEW ----
    with tab1:
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)

        st.write("### 📊 Data Summary")
        st.write(df.describe())

    # 📈 ---- VISUALIZATION ----
    with tab2:
        st.write("### 📊 Climate Trends Over Time")

        feature = st.selectbox("Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])
        fig = px.line(df, x="Years", y=feature, title=f"{feature} Trends Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # 🔮 ---- PREDICTIONS ----
    with tab3:
        st.write("### 🔮 Predict Future Climate Conditions")

        required_features = ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel"]
        if all(col in df.columns for col in required_features):
            X_new = df[required_features]

            if model_choice == "Gradient Boosting":
                predictions = gb_model.predict(X_new)
            elif model_choice == "LSTM":
                X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                predictions = lstm_model.predict(X_new_lstm).flatten()
            elif model_choice == "Prophet":
                df_prophet = df.rename(columns={"Years": "ds", "Temperature": "y"})
                prophet_model = Prophet()
                prophet_model.fit(df_prophet)
                future = prophet_model.make_future_dataframe(periods=365)
                forecast = prophet_model.predict(future)
                predictions = forecast["yhat"]

            df["Predicted Temperature"] = predictions
            df["Weather Description"] = df["Predicted Temperature"].apply(lambda x: "Very Hot 🔥" if x > 35 else "Warm ☀️" if x > 20 else "Cool 🌬️" if x > 10 else "Cold ❄️")

            st.write("### 🔥 Predictions with Weather Description")
            st.dataframe(df[["Years", "Predicted Temperature", "Weather Description"]])

            fig_pred = px.line(df, x="Years", y="Predicted Temperature", title="Predicted Temperature Trends")
            st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.warning("🚨 The dataset is missing required columns!")

    # 🌍 ---- LIVE WEATHER ----
    with tab4:
        st.write(f"### 🌍 Live Weather in {city}")

        weather_data = get_live_weather(city)
        if weather_data:
            st.metric(label="🌡️ Current Temperature (°C)", value=f"{weather_data['temperature']}°C")
            st.metric(label="💨 Wind Speed", value=f"{weather_data['wind_speed']} km/h")
            st.metric(label="💧 Humidity", value=f"{weather_data['humidity']}%")
            st.write(f"**Weather Description:** {weather_data['weather_desc']}")
        else:
            st.error("⚠️ Could not fetch weather data. Check API Key or city name!")

    # 📥 ---- DOWNLOAD PREDICTIONS ----
    df.to_csv("predictions.csv", index=False)
    st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
