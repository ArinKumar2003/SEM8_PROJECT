import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pydeck as pdk
from streamlit_echarts import st_echarts

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- CUSTOM THEME ----
def set_theme():
    theme = st.toggle("🌗 Toggle Dark Mode")
    if theme:
        st.markdown(
            """
            <style>
                body { background-color: #1e1e1e; color: white; }
                .stApp { background-color: #1e1e1e; }
            </style>
            """,
            unsafe_allow_html=True,
        )
set_theme()

# ---- WEATHERSTACK API CONFIG ----
WEATHER_API_KEY = st.secrets["WEATHERSTACK_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]

def get_live_weather(city):
    """Fetches real-time weather data from Weatherstack API."""
    url = f"http://api.weatherstack.com/current?access_key={WEATHER_API_KEY}&query={city}"
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

def get_city_coordinates(city):
    """Fetches city latitude & longitude for map visualization."""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data:
        return data[0]["lat"], data[0]["lon"]
    return None, None

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📂 Upload Climate Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.subheader("🤖 Choose Prediction Model")
    model_choice = st.radio("", ["Gradient Boosting", "LSTM", "Prophet", "AutoML AI"])

# ---- MULTIPLE TABS ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌦 Live Weather", "📊 Weather Trends", "📈 Climate Predictions", "🗺️ Weather Map", "🤖 AI Models"
])

# ---- LIVE WEATHER TAB ----
with tab1:
    st.subheader("🌦 Live Weather Conditions")

    col1, col2 = st.columns([2, 3])
    with col1:
        city = st.text_input("Enter City", value="New York")
        if st.button("🔍 Get Live Weather"):
            st.session_state.weather_city = city  

    city = st.session_state.get("weather_city", "New York")
    weather_data = get_live_weather(city)

    if weather_data:
        temp = weather_data["temperature"]
        desc = weather_data["description"]
        humidity = weather_data["humidity"]
        wind_speed = weather_data["wind_speed"]

        weather_icon = "☀️" if "Sunny" in desc else "☁️" if "Cloud" in desc else "🌧️" if "Rain" in desc else "🌪️"

        with col2:
            st.markdown(f"""
            <div style="text-align: center; background: #ecf0f1; padding: 20px; border-radius: 12px;">
                <h2>🌆 {city}</h2>
                <h1 style="color:#e74c3c;">{weather_icon} {temp}°C</h1>
                <h3>☁️ {desc}</h3>
                <p>💧 Humidity: <b>{humidity}%</b></p>
                <p>🌬 Wind Speed: <b>{wind_speed} km/h</b></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("❌ Unable to fetch weather data. Check city name or API key.")

# ---- WEATHER TRENDS TAB ----
with tab2:
    st.subheader("📊 Temperature Trend Visualization")
    weather_history = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Temperature": [23, 25, 22, 26, 27, 24, 25]
    })
    fig = px.line(weather_history, x="Day", y="Temperature", markers=True, title="Past Week Temperature Trend")
    st.plotly_chart(fig, use_container_width=True)

# ---- CLIMATE PREDICTIONS TAB ----
with tab3:
    st.subheader("📈 Future Climate Predictions")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())  

        st.subheader("🌡 Temperature & Humidity Trends")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        options = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": df["Date"].astype(str).tolist() if "Date" in df.columns else df.index.tolist()},
            "yAxis": {"type": "value"},
            "series": [
                {"name": "Temperature", "type": "line", "data": df["Temperature"].tolist() if "Temperature" in df.columns else []},
                {"name": "Humidity", "type": "line", "data": df["Humidity"].tolist() if "Humidity" in df.columns else []}
            ]
        }
        st_echarts(options=options, height="400px")
    else:
        st.info("📂 Upload a CSV file to generate predictions.")

# ---- WEATHER MAP TAB ----
with tab4:
    st.subheader("🗺️ Real-Time Weather Map")

    city = st.text_input("Enter City for Map", value="New York")
    lat, lon = get_city_coordinates(city)

    if lat and lon:
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=8, pitch=50),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
                    get_position=["lon", "lat"],
                    get_color=[255, 0, 0, 160],
                    get_radius=10000,
                )
            ]
        ))
    else:
        st.error("❌ City not found.")

# ---- AI MODELS TAB ----
with tab5:
    st.subheader("🤖 AI-Based Climate Prediction Models")

    if model_choice == "Gradient Boosting":
        st.info("📌 **Gradient Boosting Model:** Best for structured climate data.")
    elif model_choice == "LSTM":
        st.info("📌 **LSTM:** Best for time-series climate predictions.")
    elif model_choice == "Prophet":
        st.info("📌 **Prophet Model:** Best for long-term forecasting.")
    elif model_choice == "AutoML AI":
        st.info("🚀 **AutoML AI:** Fully automated climate predictions with deep learning.")

    st.success("✅ Select a model and upload CSV for AI-driven predictions.")

# ---- FOOTER ----
st.markdown("<h4 style='text-align: center;'>🔍 AI-Driven Climate Insights for a Sustainable Future</h4>", unsafe_allow_html=True)
