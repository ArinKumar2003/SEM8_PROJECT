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

@st.cache_data(ttl=600)
def get_live_weather(city):
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
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.sidebar.error("⚠️ The uploaded CSV is empty.")
            df = None
        elif not all(col in df.columns for col in ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel", "Temperature"]):
            st.sidebar.error("⚠️ Invalid CSV format. Required columns: Years, Month, Day, CO2, Humidity, SeaLevel, Temperature.")
            df = None
        else:
            df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
            df = df[["ds", "CO2", "Humidity", "SeaLevel", "Temperature"]].rename(columns={"Temperature": "y"})
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- TRAIN MODEL & FORECAST ----
if df is not None:
    model = Prophet(seasonality_mode="multiplicative")
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df)
    future = model.make_future_dataframe(periods=365 * 5)
    forecast = model.predict(future)
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    forecast.set_index("ds", inplace=True)
    future_monthly = forecast.resample("M").mean(numeric_only=True).reset_index()
    future_yearly = forecast.resample("Y").mean(numeric_only=True).reset_index()

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Live Weather", "Historical Data", "Monthly Forecast", 
    "Yearly Forecast", "Extreme Conditions", "Climate Trends", "Climate Summary & Actions"
])

# ---- TAB 7: CLIMATE SUMMARY & ACTIONS ----
with tab7:
    st.subheader("🌏 Climate Summary & Recommended Actions")
    if df is not None:
        avg_temp = df["y"].mean()
        avg_co2 = df["CO2"].mean()
        avg_humidity = df["Humidity"].mean()
        
        st.write(f"### 🌡️ Average Temperature: {avg_temp:.2f}°C")
        st.write(f"### 🌿 Average CO₂ Levels: {avg_co2:.2f} ppm")
        st.write(f"### 💧 Average Humidity: {avg_humidity:.2f}%")
        
        if avg_temp > 30:
            st.error("⚠️ High temperatures detected. Consider reducing carbon emissions and increasing green cover.")
        if avg_co2 > 400:
            st.warning("⚠️ Elevated CO₂ levels. Encourage the use of renewable energy and carbon capture technologies.")
        if avg_humidity < 30:
            st.info("💡 Dry conditions detected. Promote water conservation and afforestation efforts.")
    else:
        st.warning("📂 Please upload a CSV file for climate insights.")

# ---- FOOTER ----
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**")
