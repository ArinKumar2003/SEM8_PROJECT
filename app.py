import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- THEME TOGGLE (DARK/LIGHT MODE) ----
theme = st.sidebar.radio("🌗 Theme", ["Light Mode", "Dark Mode"])
if theme == "Dark Mode":
    st.markdown(
        """
        <style>
            body {
                background-color: #1E1E1E;
                color: white;
            }
            .stApp {
                background-color: #1E1E1E;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---- DASHBOARD HEADER ----
st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>🌍 AI Climate Change Prediction Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📊 Live Weather & Future Climate Predictions</h3>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---- WEATHERSTACK API CONFIG ----
API_KEY = st.secrets["WEATHERSTACK_API_KEY"]  # Ensure this is set in secrets.toml

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
            "wind_speed": data["current"]["wind_speed"],
            "feels_like": data["current"]["feelslike"],
            "pressure": data["current"]["pressure"],
        }
    else:
        return None

# ---- TABS FOR BETTER NAVIGATION ----
tab1, tab2 = st.tabs(["🌦 Live Weather", "📈 Climate Predictions"])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌦 Live Weather Conditions")

    city = st.text_input("Enter City", value="New York")
    if st.button("🔍 Get Live Weather"):
        weather_data = get_live_weather(city)

        if weather_data:
            temp = weather_data["temperature"]
            desc = weather_data["description"]
            humidity = weather_data["humidity"]
            wind_speed = weather_data["wind_speed"]
            feels_like = weather_data["feels_like"]
            pressure = weather_data["pressure"]

            # Display live weather details
            st.markdown(f"""
            <div style="text-align: center; background: #ecf0f1; padding: 20px; border-radius: 12px;">
                <h2>🌆 {city}</h2>
                <h1 style="color:#e74c3c;">🌡 {temp}°C</h1>
                <h3>☁️ {desc}</h3>
                <p>💧 Humidity: <b>{humidity}%</b></p>
                <p>🌬 Wind Speed: <b>{wind_speed} km/h</b></p>
                <p>🌡 Feels Like: <b>{feels_like}°C</b></p>
                <p>🛠 Pressure: <b>{pressure} hPa</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Unable to fetch weather data. Check city name or API key.")
    else:
        st.info("Enter a city name and click 'Get Live Weather' to see the conditions.")

# ---- TAB 2: CLIMATE PREDICTIONS ----
with tab2:
    st.subheader("📈 Future Climate Predictions")

    # Sidebar for uploading CSV
    with st.sidebar:
        st.header("📂 Upload Climate Data")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        # Ensure the dataset has required columns
        if len(df.columns) < 3:
            st.error("⚠️ The uploaded CSV must have at least 3 columns: Date, Temperature, Humidity, Wind Speed.")
        else:
            date_col = df.columns[0]
            temp_col = df.columns[1]
            humidity_col = df.columns[2]
            wind_col = df.columns[3] if len(df.columns) > 3 else None

            # Convert date column to datetime format
            df[date_col] = pd.to_datetime(df[date_col])

            # Temperature Trend
            st.write("### 🌡 Temperature Trends")
            fig_temp = px.line(df, x=date_col, y=temp_col, title="Temperature Trend Over Time", markers=True)
            st.plotly_chart(fig_temp, use_container_width=True)

            # Humidity Trend
            st.write("### 💧 Humidity Trends")
            fig_humidity = px.line(df, x=date_col, y=humidity_col, title="Humidity Trend Over Time", markers=True)
            st.plotly_chart(fig_humidity, use_container_width=True)

            # Wind Speed Trend (if available)
            if wind_col:
                st.write("### 🌬 Wind Speed Trends")
                fig_wind = px.line(df, x=date_col, y=wind_col, title="Wind Speed Trend Over Time", markers=True)
                st.plotly_chart(fig_wind, use_container_width=True)

            # Show statistical summary
            st.write("### 📊 Summary Statistics")
            st.write(df.describe())

            st.success("✅ Data uploaded successfully! View trends above.")
    else:
        st.info("📂 Upload a CSV file to generate predictions.")

st.markdown("<hr>", unsafe_allow_html=True)
