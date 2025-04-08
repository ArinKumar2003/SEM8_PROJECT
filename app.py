import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
import plotly.express as px
from datetime import datetime
from io import StringIO

# ---- Weather API Config ----
API_KEY = st.secrets["weatherapi"]["api_key"]
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# ---- App Layout ----
st.set_page_config(page_title="Climate Forecast Dashboard", layout="wide")

st.title("🌦️ Climate Forecast Dashboard")
st.markdown("Get real-time weather updates and future climate predictions.")

# ---- Upload Dataset ----
uploaded_file = st.sidebar.file_uploader("📁 Upload your climate dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    if "date" in df.columns:
        df.rename(columns={"date": "ds"}, inplace=True)
    elif "datetime" in df.columns:
        df.rename(columns={"datetime": "ds"}, inplace=True)

    target_col = "y"
    for col in df.columns:
        if col != "ds":
            df.rename(columns={col: target_col}, inplace=True)
            break

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df.dropna(subset=["ds", "y"], inplace=True)
    return df

# ---- Live Weather ----
def get_live_weather(city="London"):
    params = {"key": API_KEY, "q": city}
    res = requests.get(WEATHER_API_URL, params=params)
    data = res.json()

    if "error" in data:
        return {"error": data["error"]["message"]}

    current = data["current"]
    location = data["location"]
    weather = {
        "city": location["name"],
        "region": location["region"],
        "country": location["country"],
        "temp_c": current["temp_c"],
        "condition": current["condition"]["text"],
        "icon": "https:" + current["condition"]["icon"],
        "last_updated": current["last_updated"]
    }
    return weather

# ---- Tabs ----
tabs = st.tabs(["🌍 Live Weather", "📈 Forecast", "📊 Visualizations", "📚 Climate Summary", "🌱 Awareness"])

# ==== Tab 1: Live Weather ====
with tabs[0]:
    st.subheader("Live Weather Info")
    city = st.text_input("Enter City for Live Weather", "London")
    weather = get_live_weather(city)

    if "error" in weather:
        st.error(f"⚠️ {weather['error']}")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(weather["icon"], width=100)
        with col2:
            st.markdown(f"### {weather['city']}, {weather['country']}")
            st.write(f"**Temperature**: {weather['temp_c']}°C")
            st.write(f"**Condition**: {weather['condition']}")
            st.write(f"**Last Updated**: {weather['last_updated']}")

# ==== Tab 2: Forecast ====
with tabs[1]:
    st.subheader("Forecasting Climate Trends")
    if uploaded_file:
        df = load_data(uploaded_file)

        st.write("Sample Data:")
        st.dataframe(df.head())

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        st.plotly_chart(px.line(forecast, x="ds", y="yhat", title="Forecasted Temperature / Target"))

        st.write("Forecast Data:")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))
    else:
        st.info("Please upload a dataset in the sidebar to generate a forecast.")

# ==== Tab 3: Visualizations ====
with tabs[2]:
    st.subheader("Climate Data Visualizations")
    if uploaded_file:
        df = load_data(uploaded_file)

        st.plotly_chart(px.line(df, x="ds", y="y", title="Historical Climate Trends"))
        st.plotly_chart(px.scatter(df, x="ds", y="y", title="Scatter View"))
    else:
        st.info("Upload a dataset to see visualizations.")

# ==== Tab 4: Climate Summary ====
with tabs[3]:
    st.subheader("Climate Summary")
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("📊 **Summary Statistics:**")
        st.dataframe(df.describe())
    else:
        st.info("Upload a dataset to display summary statistics.")

# ==== Tab 5: Awareness ====
with tabs[4]:
    st.subheader("Climate Change Awareness 🌍")
    st.markdown("""
    Climate change is one of the most pressing issues of our time.  
    Here's what you can do to contribute:

    - Reduce, reuse, and recycle
    - Switch to renewable energy
    - Use public transportation or cycle
    - Plant trees and support afforestation
    - Support environmental organizations

    Learn more at [UN Climate Action](https://www.un.org/en/climatechange).
    """)

