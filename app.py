import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_echarts import st_echarts

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- DASHBOARD HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50; font-size: 36px;'>🌍 AI Climate Change Prediction Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📊 Live Weather & Future Climate Predictions</h3>
    <hr style="border:1px solid #ddd;">
""", unsafe_allow_html=True)

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

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📂 Upload Climate Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.subheader("🤖 Choose Prediction Model")
    model_choice = st.radio("", ["Gradient Boosting", "LSTM", "Prophet"])

# ---- MULTIPLE TABS ----
tab1, tab2, tab3, tab4 = st.tabs(["🌦 Live Weather", "📊 Weather Trends", "📈 Climate Predictions", "🤖 AI Models"])

# ---- LIVE WEATHER TAB ----
with tab1:
    st.subheader("🌦 Live Weather Conditions")

    # Use columns for better layout
    col1, col2 = st.columns([2, 3])

    with col1:
        city = st.text_input("Enter City", value="New York")
        if st.button("🔍 Get Live Weather"):
            st.session_state.weather_city = city  # Store city in session

    # Fetch weather data
    city = st.session_state.get("weather_city", "New York")
    weather_data = get_live_weather(city)

    if weather_data:
        temp = weather_data["temperature"]
        desc = weather_data["description"]
        humidity = weather_data["humidity"]
        wind_speed = weather_data["wind_speed"]

        # Dynamic icons based on weather conditions
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

    # Simulated historical weather data for visualization
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
        st.write(df.head())  # Display first few rows
        st.success("✅ Data uploaded successfully! Generating Insights...")

        # ---- INTERACTIVE CLIMATE INSIGHTS ----
        st.subheader("🌡 Temperature & Humidity Trends")

        # Convert Date column to datetime if available
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # Plot Temperature and Humidity Trends using ECharts
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

# ---- AI MODELS TAB ----
with tab4:
    st.subheader("🤖 AI-Based Climate Prediction Models")

    if model_choice == "Gradient Boosting":
        st.info("📌 **Gradient Boosting Model:** Best for structured climate data and historical patterns.")

    elif model_choice == "LSTM":
        st.info("📌 **LSTM (Long Short-Term Memory):** Best for time-series analysis and detecting climate trends.")

    elif model_choice == "Prophet":
        st.info("📌 **Prophet Model:** Best for long-term climate forecasting with trend detection.")

    st.success("✅ Select a model and upload CSV for AI-driven predictions.")

# ---- FOOTER ----
st.markdown("<h4 style='text-align: center;'>🔍 AI-Driven Climate Insights for a Sustainable Future</h4>", unsafe_allow_html=True)
