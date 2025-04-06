import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("WEATHERSTACK_API_KEY") or "YOUR_API_KEY_HERE"

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("GlobalWeatherRepository.csv")
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    return df

df = load_data()

# Streamlit config
st.set_page_config(page_title="🌎 Weather Dashboard", layout="wide")

st.sidebar.title("🌐 Weather Dashboard")
tab = st.sidebar.radio("Select Tab", ["Live Weather", "Historical Trends", "Forecast", "Map View", "Raw Data"])

# Sidebar Filters
country = st.sidebar.selectbox("🌍 Select Country", sorted(df['country'].dropna().unique()))
city_list = sorted(df[df['country'] == country]['location_name'].dropna().unique())
city = st.sidebar.selectbox("🏙️ Select City", city_list)

filtered_df = df[(df['country'] == country) & (df['location_name'] == city)].copy().sort_values('last_updated')

# LIVE WEATHER (Weatherstack)
if tab == "Live Weather":
    st.title(f"🌤️ Live Weather for {city}, {country}")

    def get_live_weather(city):
        url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
        return None

    data = get_live_weather(city)

    if data and "current" in data:
        current = data["current"]
        location = data["location"]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Temperature", f"{current['temperature']} °C")
        col2.metric("💧 Humidity", f"{current['humidity']} %")
        col3.metric("🌬️ Wind Speed", f"{current['wind_speed']} km/h")

        col4, col5, col6 = st.columns(3)
        col4.metric("🌥️ Condition", current['weather_descriptions'][0])
        col5.metric("📊 Pressure", f"{current['pressure']} hPa")
        col6.metric("📍 Coordinates", f"{location['lat']}, {location['lon']}")

    else:
        st.error("❌ Could not fetch data from Weatherstack. Check your API key or city name.")

# HISTORICAL TRENDS
elif tab == "Historical Trends":
    st.title(f"📈 Historical Trends in {city}")

    metric = st.selectbox("Metric", ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'uv_index'])
    fig = px.line(filtered_df, x='last_updated', y=metric, title=f"{metric.replace('_', ' ').title()} Over Time")
    st.plotly_chart(fig, use_container_width=True)

# FORECAST TAB
elif tab == "Forecast":
    st.title(f"🔮 Forecasting for {city}")

    metric = st.selectbox("Select Metric to Forecast", ['temperature_celsius', 'humidity', 'pressure_mb'])

    if len(filtered_df) >= 20:
        prophet_df = filtered_df[['last_updated', metric]].rename(columns={"last_updated": "ds", metric: "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=24, freq='H')
        forecast = model.predict(future)
        forecast_fig = plot_plotly(model, forecast)
        st.plotly_chart(forecast_fig, use_container_width=True)
    else:
        st.warning("Not enough data to generate a forecast.")

# MAP VIEW
elif tab == "Map View":
    st.title("🗺️ Weather Map")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        map_df = df[['location_name', 'latitude', 'longitude', 'temperature_celsius']].dropna()
        st.map(map_df)
    else:
        st.warning("No latitude/longitude data available.")

# RAW DATA
elif tab == "Raw Data":
    st.title("📄 Raw Weather Data")
    st.dataframe(filtered_df, use_container_width=True)
