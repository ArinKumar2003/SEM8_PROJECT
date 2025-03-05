import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from fbprophet import Prophet

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- THEME TOGGLE (DARK/LIGHT MODE) ----
theme = st.sidebar.radio("🌗 Theme", ["Light Mode", "Dark Mode"])
if theme == "Dark Mode":
    st.markdown(
        """
        <style>
            body { background-color: #1E1E1E; color: white; }
            .stApp { background-color: #1E1E1E; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---- DASHBOARD HEADER ----
st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>🌍 AI Climate Change Prediction Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📊 Live Weather, AI Predictions & Interactive Analysis</h3>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---- WEATHERSTACK API CONFIG ----
try:
    API_KEY = st.secrets["WEATHERSTACK_API_KEY"]
except KeyError:
    st.error("❌ API key is missing. Set `WEATHERSTACK_API_KEY` in `secrets.toml`.")
    API_KEY = None

def get_live_weather(city):
    """Fetch real-time weather data from Weatherstack API."""
    if not API_KEY:
        return None
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    data = response.json()

    if "current" in data:
        return {
            "temperature": data["current"]["temperature"],
            "description": data["current"]["weather_descriptions"][0],
            "humidity": data["current"]["humidity"],
            "wind_speed": data["current"]["wind_speed"],
            "feels_like": data["current"]["feelslike"],
            "pressure": data["current"]["pressure"],
        }
    return None

# ---- TABS FOR NAVIGATION ----
tab1, tab2, tab3, tab4 = st.tabs(["🌦 Live Weather", "📈 AI Predictions", "🔮 Interactive Analysis", "📊 Climate Trends"])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌦 Live Weather Conditions")

    cities = st.text_input("Enter Cities (comma-separated)", value="New York, London, Tokyo")
    city_list = [city.strip() for city in cities.split(",")]

    if st.button("🔍 Get Live Weather"):
        for city in city_list:
            weather_data = get_live_weather(city)
            if weather_data:
                st.markdown(f"""
                <div style="text-align: center; background: #ecf0f1; padding: 20px; border-radius: 12px; margin-bottom: 10px;">
                    <h2>🌆 {city}</h2>
                    <h1 style="color:#e74c3c;">🌡 {weather_data["temperature"]}°C</h1>
                    <h3>☁️ {weather_data["description"]}</h3>
                    <p>💧 Humidity: <b>{weather_data["humidity"]}%</b></p>
                    <p>🌬 Wind Speed: <b>{weather_data["wind_speed"]} km/h</b></p>
                    <p>🌡 Feels Like: <b>{weather_data["feels_like"]}°C</b></p>
                    <p>🛠 Pressure: <b>{weather_data["pressure"]} hPa</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Unable to fetch weather data for {city}.")

# ---- TAB 2: AI CLIMATE PREDICTIONS ----
with tab2:
    st.subheader("📈 AI-Powered Climate Predictions")

    with st.sidebar:
        st.header("📂 Upload Climate Data")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        if len(df.columns) < 2:
            st.error("⚠️ The uploaded CSV must have at least 2 columns: Date, Temperature.")
        else:
            date_col = df.columns[0]
            temp_col = df.columns[1]

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: "ds", temp_col: "y"})

            # AI Forecasting with Prophet
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            st.write("### 🔮 AI-Powered Temperature Prediction")
            fig_forecast = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends")
            st.plotly_chart(fig_forecast)

# ---- TAB 3: INTERACTIVE ANALYSIS ----
with tab3:
    st.subheader("🔮 Interactive Climate Data Analysis")

    if uploaded_file:
        st.write("### 🌡 Temperature vs Humidity")
        fig_scatter = px.scatter(df, x="y", y=df.columns[2], title="Temperature vs Humidity", opacity=0.7)
        st.plotly_chart(fig_scatter)

        st.write("### 🔥 Heatmap of Climate Data")
        fig_heatmap = px.imshow(df.corr(), title="Correlation Heatmap")
        st.plotly_chart(fig_heatmap)

        st.write("### 📊 Histogram of Temperatures")
        fig_hist = px.histogram(df, x="y", title="Temperature Distribution")
        st.plotly_chart(fig_hist)

# ---- TAB 4: CLIMATE TRENDS ----
with tab4:
    st.subheader("📊 Climate Trends & Comparisons")

    if uploaded_file:
        st.write("### 🌍 Compare Historical vs Future Trends")
        combined_df = pd.concat([df, forecast.rename(columns={"yhat": "Predicted Temperature"})], ignore_index=True)

        fig_compare = px.line(combined_df, x="ds", y=["y", "Predicted Temperature"], title="Historical vs Future Temperature Trends")
        st.plotly_chart(fig_compare)
