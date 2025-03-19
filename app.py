import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
from prophet import Prophet

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Forecast", layout="wide")

# ---- TITLE ----
st.markdown("<h1 style='text-align: center;'>🌍 AI Climate Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>📊 Live & Historical Climate Analysis (1971 - 2030)</h3>", unsafe_allow_html=True)

# ---- WEATHER API CONFIG ----
API_KEY = st.secrets["WEATHERAPI_KEY"] if "WEATHERAPI_KEY" in st.secrets else None  

@st.cache_data()
def get_live_weather(city):
    """Fetch real-time weather data from WeatherAPI.com."""
    if not API_KEY:
        st.error("❌ API Key is missing! Please check your Streamlit secrets.")
        return None
    
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    try:
        with st.spinner("Fetching live weather data..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        
        if "error" in data:
            st.error(f"⚠️ {data['error']['message']}")
            return None
        return {
            "ds": datetime.datetime.now(), 
            "y": data["current"]["temp_c"],
            "Humidity": data["current"]["humidity"],
            "CO2": data["current"].get("air_quality", {}).get("co", "Unavailable"),
            "Condition": data["current"]["condition"]["text"],
            "Wind Speed (km/h)": data["current"]["wind_kph"],
            "Pressure (hPa)": data["current"]["pressure_mb"],
            "Visibility (km)": data["current"]["vis_km"],
            "Icon": data["current"]["condition"]["icon"]
        }
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request failed: {e}")
        return None

# ---- SIDEBAR FILE UPLOAD ----
uploaded_file = st.sidebar.file_uploader("📂 Upload Climate CSV", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, usecols=["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel", "Temperature"], dtype={"Years": int, "Month": int, "Day": int})
        df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
        df = df[["ds", "CO2", "Humidity", "SeaLevel", "Temperature"]].rename(columns={"Temperature": "y"})
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Live Weather", "Historical Data", "Monthly Forecast", "Yearly Forecast", "Extreme Conditions", "Summary"])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌍 Live Weather Data")
    city = st.selectbox("Select City", ["New York", "London", "Tokyo", "Sydney", "Mumbai"], index=0)
    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.write(f"### {city}: {live_weather['Condition']}")
            st.image(f"https:{live_weather['Icon']}", width=80)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Temperature (°C)", live_weather["y"])
            col2.metric("Humidity (%)", live_weather["Humidity"])
            col3.metric("CO₂ Level (ppm)", live_weather["CO2"])
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Wind Speed (km/h)", live_weather["Wind Speed (km/h)"])
            col5.metric("Pressure (hPa)", live_weather["Pressure (hPa)"])
            col6.metric("Visibility (km)", live_weather["Visibility (km)"])

# ---- TAB 2: HISTORICAL DATA ----
with tab2:
    st.subheader("📜 Historical Climate Data (1971-Present)")
    if df is not None:
        fig_hist = px.line(df, x="ds", y="y", title="📊 Temperature Trends (1971-Present)", labels={"y": "Temperature (°C)"})
        st.plotly_chart(fig_hist)
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TAB 5: EXTREME CONDITIONS ----
with tab5:
    st.subheader("🚨 Extreme Climate Alerts & Visualizations")
    if df is not None:
        st.write("Extreme weather conditions will be displayed here.")

# ---- TAB 6: SUMMARY ----
with tab6:
    st.subheader("📖 Climate Summary & Predictions")
    st.markdown("""
        ### 🔹 **Climate Trends & Insights**
        - 📜 Historical data analysis since **1971**
        - 📅 AI-driven forecasts for **2025–2030**
        - 🚨 Alerts for **extreme temperature fluctuations**

        **Key Observations:**
        - Rising global temperatures predicted.
        - Potential increase in extreme weather events.
        - Need for sustainable actions to mitigate climate change.

        ✅ Stay informed, stay prepared!
    """)

# ---- FOOTER ----
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**")
