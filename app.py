import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from datetime import datetime

# Load API key from secrets
API_KEY = st.secrets["weatherapi"]["api_key"]

st.set_page_config(page_title="🌦️ Climate Insights Dashboard", layout="wide")
st.title("🌍 Climate Insights Dashboard")
st.markdown("Upload your climate dataset and explore trends, forecasts, and real-time weather.")

# === File Uploader ===
uploaded_file = st.file_uploader("📂 Upload your climate CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()

    # Try to find datetime column
    datetime_col = None
    for col in df.columns:
        if "date" in col or "time" in col:
            datetime_col = col
            break
    if not datetime_col:
        datetime_col = st.selectbox("📅 Select datetime column", df.columns)
    df.rename(columns={datetime_col: "ds"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # Detect numeric column
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    target_col = "temperature" if "temperature" in df.columns else None
    if not target_col:
        target_col = st.selectbox("📈 Select target column", numeric_cols)
    df.rename(columns={target_col: "y"}, inplace=True)
    df.dropna(subset=["ds", "y"], inplace=True)
    return df

# === Live Weather ===
@st.cache_data(ttl=600)
def get_live_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    res = requests.get(url)
    data = res.json()
    current = data["current"]
    location = data["location"]
    return {
        "City": location["name"],
        "Country": location["country"],
        "Temperature (°C)": current["temp_c"],
        "Humidity (%)": current["humidity"],
        "Condition": current["condition"]["text"],
        "Icon": "https:" + current["condition"]["icon"],
        "Last Updated": current["last_updated"]
    }

# === Tabs ===
tabs = st.tabs(["🌤️ Live Weather", "📊 Forecast", "📁 Data Summary", "🌱 Climate Awareness"])

# === Tab 1: Live Weather ===
with tabs[0]:
    st.header("🌤️ Live Weather")
    city = st.text_input("Enter a city name", value="New York")
    if city:
        try:
            weather = get_live_weather(city)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("Temperature (°C)", weather["Temperature (°C)"])
                st.metric("Humidity (%)", weather["Humidity (%)"])
                st.write("**Condition:**", weather["Condition"])
                st.write("**Location:**", f"{weather['City']}, {weather['Country']}")
                st.write("**Last Updated:**", weather["Last Updated"])
            with col2:
                st.image(weather["Icon"])
        except:
            st.error("Could not retrieve weather data. Please check city name or API.")

# === Other tabs only work if a file is uploaded ===
if uploaded_file:
    df = load_data(uploaded_file)

    # === Tab 2: Forecast ===
    with tabs[1]:
        st.header("📊 Forecast with Prophet")
        period = st.slider("Select forecast period (days)", 7, 60, 30)
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)

    # === Tab 3: Data Summary ===
    with tabs[2]:
        st.header("📁 Data Summary")
        st.dataframe(df.head(100), use_container_width=True)
        st.subheader("📈 Data Distribution")
        fig = px.histogram(df, x="y", nbins=30, title="Distribution of Values")
        st.plotly_chart(fig, use_container_width=True)

# === Tab 4: Climate Awareness ===
with tabs[3]:
    st.header("🌱 Climate Awareness")
    st.markdown("""
    Climate change is real and happening now. Here are some ways you can help:
    - ♻️ Reduce, reuse, and recycle
    - 🚲 Walk, bike, or use public transportation
    - 💡 Conserve energy
    - 🌳 Plant trees and support reforestation
    - 🧠 Educate others and raise awareness
    """)
    st.image("https://climate.nasa.gov/system/internal_resources/details/original/3099_1-global-temp.jpg", caption="NASA Climate Change Data")
