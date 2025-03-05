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

# ---- THEME TOGGLE (DARK/LIGHT MODE) ----
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

# ---- API CONFIG ----
try:
    API_KEY = st.secrets["WEATHERSTACK_API_KEY"]
except KeyError:
    st.error("❌ API key is missing. Set `WEATHERSTACK_API_KEY` in `secrets.toml`.")
    API_KEY = None

def get_live_weather(city):
    """Fetch real-time weather data."""
    if not API_KEY:
        return None
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    data = response.json()
    if "current" in data:
        return data["current"]
    return None

# ---- TABS FOR NAVIGATION ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌦 Live Weather", "📈 AI Forecasts", "🔮 Interactive Trends", "📊 Climate Score", "⚠️ Extreme Weather", "🛰️ Satellite View"
])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌦 Live Weather")

    cities = st.text_input("Enter Cities (comma-separated)", value="New York, London, Tokyo")
    city_list = [city.strip() for city in cities.split(",")]

    if st.button("🔍 Get Live Weather"):
        for city in city_list:
            weather = get_live_weather(city)
            if weather:
                st.markdown(f"""
                <div style="text-align: center; background: #ecf0f1; padding: 20px; border-radius: 12px;">
                    <h2>🌆 {city}</h2>
                    <h1 style="color:#e74c3c;">🌡 {weather['temperature']}°C</h1>
                    <h3>☁️ {weather['weather_descriptions'][0]}</h3>
                    <p>💧 Humidity: <b>{weather['humidity']}%</b></p>
                    <p>🌬 Wind Speed: <b>{weather['wind_speed']} km/h</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Unable to fetch weather data for {city}.")

# ---- TAB 2: AI FORECASTS ----
with tab2:
    st.subheader("📈 AI Forecasts using Prophet")

    uploaded_file = st.file_uploader("Upload Climate CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        if len(df.columns) < 2:
            st.error("⚠️ CSV must have Date and Temperature columns.")
        else:
            df.columns = ["ds", "y"]
            df["ds"] = pd.to_datetime(df["ds"])

            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends")
            st.plotly_chart(fig)

# ---- TAB 3: INTERACTIVE CLIMATE TRENDS ----
with tab3:
    st.subheader("🔮 Interactive Climate Trends")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["ds"] = pd.to_datetime(df.iloc[:, 0])
        df["y"] = df.iloc[:, 1]

        fig = px.scatter(df, x="ds", y="y", title="Temperature Trends Over Time", opacity=0.7)
        st.plotly_chart(fig)

        fig2 = px.histogram(df, x="y", title="Temperature Distribution")
        st.plotly_chart(fig2)

# ---- TAB 4: CLIMATE SCORE ----
with tab4:
    st.subheader("📊 Climate Impact Score")

    if uploaded_file:
        df["climate_score"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min()) * 100

        fig = px.line(df, x="ds", y="climate_score", title="Climate Impact Score")
        st.plotly_chart(fig)

# ---- TAB 5: EXTREME WEATHER ALERTS ----
with tab5:
    st.subheader("⚠️ Extreme Weather Alerts")

    if uploaded_file:
        threshold = st.slider("Set Temperature Alert Threshold", min_value=int(df["y"].min()), max_value=int(df["y"].max()), value=35)

        alerts = df[df["y"] > threshold]
        st.write("### 🔥 Heatwave Alerts", alerts)

        fig = px.line(df, x="ds", y="y", title="Extreme Temperature Trends", markers=True)
        fig.add_trace(go.Scatter(x=alerts["ds"], y=alerts["y"], mode="markers", marker=dict(color="red", size=10), name="Extreme Heat"))
        st.plotly_chart(fig)

# ---- TAB 6: SATELLITE VIEW ----
with tab6:
    st.subheader("🛰️ Live Climate Satellite View")
    st.markdown("""
        🚀 Integrate with **OpenWeatherMap's Satellite API** or **Google Maps**
        for live climate conditions.
    """)
    st.image("https://earthobservatory.nasa.gov/blogs/earthmatters/wp-content/uploads/sites/9/2019/05/earthmap.png")

---

## **🚀 Final Upgrades Recap**
✔ **Multi-City Weather** 🌍  
✔ **AI-Powered Forecasting (Prophet, LSTMs Coming Soon!)** 🤖  
✔ **Interactive Trends & Climate Score Analysis** 📊  
✔ **Extreme Weather Alerts & Heatwave Detection** ⚠️  
✔ **Live Satellite View (Future API Integration)** 🛰️  

🔥 **Want more AI models, real-time maps, or other features?** Let’s build it! 🚀
