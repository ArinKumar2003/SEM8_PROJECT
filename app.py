import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import requests
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(page_title="🌍 Climate Dashboard", layout="wide")

# Dashboard title
st.markdown("<h1 style='text-align: center;'>🌎 Global Climate Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# WeatherAPI key from secrets
WEATHER_API_KEY = st.secrets["weatherapi"]["api_key"]
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

def get_live_weather(city):
    params = {"key": WEATHER_API_KEY, "q": city}
    try:
        response = requests.get(WEATHER_API_URL, params=params)
        data = response.json()
        if "current" in data:
            return {
                "location": data["location"]["name"],
                "country": data["location"]["country"],
                "temp_c": data["current"]["temp_c"],
                "condition": data["current"]["condition"]["text"],
                "humidity": data["current"]["humidity"],
                "wind_kph": data["current"]["wind_kph"],
                "icon": data["current"]["condition"]["icon"]
            }
    except Exception as e:
        return None

# Sidebar: Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload Cleaned Weather Data CSV", type="csv")

# Navigation Tabs
selected = option_menu(
    menu_title=None,
    options=["Live Weather", "Forecast", "Visualization", "Climate Awareness"],
    icons=["cloud-sun", "calendar3", "bar-chart", "info-circle"],
    orientation="horizontal"
)

# LIVE WEATHER TAB
if selected == "Live Weather":
    st.subheader("☁️ Live Weather Information")
    city = st.text_input("Enter City", "New York")

    if city:
        weather = get_live_weather(city)
        if weather:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("🌡 Temperature (°C)", weather["temp_c"])
                st.metric("💧 Humidity", f"{weather['humidity']}%")
                st.metric("🌬 Wind (kph)", weather["wind_kph"])
                st.write(f"📍 {weather['location']}, {weather['country']}")
                st.success(weather["condition"])
            with col2:
                st.image("http:" + weather["icon"], width=100)
        else:
            st.warning("Could not fetch weather data. Check the city name or API key.")

# FORECAST TAB
elif selected == "Forecast":
    st.subheader("📈 Forecast Temperature using Prophet")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "ds" not in df.columns or "y" not in df.columns:
            st.warning("Uploaded CSV must contain 'ds' (datetime) and 'y' (temperature) columns.")
        else:
            df['ds'] = pd.to_datetime(df['ds'])
            st.write("📊 Sample Data", df.head())

            periods = st.slider("Select number of days to forecast", min_value=7, max_value=365, value=30)
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            fig = px.line(forecast, x='ds', y='yhat', title="Forecasted Temperature")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a dataset to view forecast.")

# VISUALIZATION TAB
elif selected == "Visualization":
    st.subheader("📊 Weather Data Visualization")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['ds'] = pd.to_datetime(df['ds'])

        col = st.selectbox("Select a column to visualize", options=df.columns[1:])
        fig = px.line(df, x='ds', y=col, title=f"{col} Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a dataset to visualize.")

# CLIMATE AWARENESS TAB
elif selected == "Climate Awareness":
    st.subheader("🌱 Climate Summary & Awareness")

    st.markdown("""
    ### 🌍 Climate Facts
    - Global temperature has increased by **~1.1°C** since the 1800s.
    - Arctic sea ice is shrinking by **13% per decade**.
    - Sea levels have risen by **~8 inches** in the past century.
    - More frequent **droughts, floods, and wildfires**.

    ### 💡 How You Can Help
    - Reduce emissions: use public transport 🚲 or carpool 🚗.
    - Save energy: turn off appliances 💡 when not in use.
    - Eat sustainably: reduce food waste and support local 🌾.
    - Plant trees 🌳 and support reforestation projects.

    > 🧠 *"We do not inherit the Earth from our ancestors, we borrow it from our children."*
    """)
