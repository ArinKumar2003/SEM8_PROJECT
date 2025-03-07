import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
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
            .sidebar .sidebar-content { background-color: #2C2F33; }
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

@st.cache_data
def get_live_weather(city):
    """Fetch real-time weather data."""
    if not API_KEY:
        return None
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    return response.json().get("current")

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

# ---- TABS ----
tabs = st.tabs(["🌦 Live Weather", "📈 AI Forecasts", "🔮 Trends", "📊 Climate Score", "⚠️ Extreme Weather", "🛰️ Satellite View"])

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

    if df is not None and len(df) > 2:
        with st.spinner("🔄 Training AI Model..."):
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends")
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot", color="gray")))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot", color="gray")))
        st.plotly_chart(fig)

    elif df is not None:
        st.error("⚠️ Not enough data to train AI model.")

# ---- TAB 3: INTERACTIVE TRENDS ----
with tabs[2]:
    st.subheader("🔮 Interactive Climate Trends")
    if df is not None:
        fig1 = px.scatter(df, x="ds", y="y", title="Temperature Trends Over Time", trendline="lowess")
        st.plotly_chart(fig1)

        fig2 = px.histogram(df, x="y", title="Temperature Distribution", nbins=20)
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
    st.markdown("🚀 Integrate with OpenWeatherMap's Satellite API or Google Maps.")
    st.image("https://earthobservatory.nasa.gov/blogs/earthmatters/wp-content/uploads/sites/9/2019/05/earthmap.png")
