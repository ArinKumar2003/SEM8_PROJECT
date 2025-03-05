import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- DARK / LIGHT MODE ----
mode = st.sidebar.radio("Choose Mode:", ["🌙 Dark Mode", "☀️ Light Mode"])
theme = "dark" if mode == "🌙 Dark Mode" else "light"

# ---- WEATHERSTACK API CONFIG ----
API_KEY = "YOUR_WEATHERSTACK_API_KEY"

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

# ---- MAIN APP ----
st.title("🌍 AI Climate Change Prediction Dashboard")
tabs = st.tabs(["🌦 Live Weather", "📂 Upload Data", "📈 Predictions", "📊 Analytics"])

# ---- TAB 1: LIVE WEATHER ----
with tabs[0]:
    st.subheader("🌦 Live Weather Conditions")
    city = st.text_input("Enter City", value="New York")

    if st.button("🔍 Get Live Weather"):
        weather_data = get_live_weather(city)
        if weather_data:
            temp, desc, humidity, wind_speed = weather_data.values()

            # Display weather details
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 12px; background: {'#2c3e50' if theme == 'dark' else '#ecf0f1'}; color: {'white' if theme == 'dark' else 'black'};">
                <h2>🌆 {city}</h2>
                <h1 style="color:#e74c3c;">🌡 {temp}°C</h1>
                <h3>☁️ {desc}</h3>
                <p>💧 Humidity: <b>{humidity}%</b></p>
                <p>🌬 Wind Speed: <b>{wind_speed} km/h</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Unable to fetch weather data. Check city name or API key.")

# ---- TAB 2: UPLOAD DATA ----
with tabs[1]:
    st.subheader("📂 Upload Climate Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Fixing PyArrow timestamp issue
        date_col = df.columns[0]  # Assuming first column is date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])  # Drop invalid dates

        st.write("### 📋 Data Preview", df.head())

# ---- TAB 3: PREDICTIONS ----
with tabs[2]:
    st.subheader("📈 Future Climate Predictions")

    # Model selection
    model_choice = st.radio("Choose Prediction Model:", ["Gradient Boosting", "LSTM", "Prophet"])

    if uploaded_file:
        st.success("✅ Data uploaded! Choose a model to proceed.")

        # Dummy prediction visualization (Replace with actual model implementation)
        fig = px.line(df, x=date_col, y=df.columns[1], title="📊 Future Climate Predictions")
        st.plotly_chart(fig)

# ---- TAB 4: ANALYTICS ----
with tabs[3]:
    st.subheader("📊 Climate Data Analytics")

    if uploaded_file:
        st.write("## 🌡 Temperature Trend")
        fig_temp = px.line(df, x=date_col, y=df.columns[1], title="Temperature Trends Over Time")
        st.plotly_chart(fig_temp)

        st.write("## 💧 Humidity Levels")
        fig_hum = px.bar(df, x=date_col, y=df.columns[2], title="Humidity Levels Over Time")
        st.plotly_chart(fig_hum)

        st.write("## 🌬 Wind Speed Trends")
        fig_wind = px.scatter(df, x=date_col, y=df.columns[3], title="Wind Speed Variations")
        st.plotly_chart(fig_wind)

