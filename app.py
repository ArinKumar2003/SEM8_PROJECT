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

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌍 Live Weather Data")
    city = st.text_input("Enter City", "New York")
    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.write(f"### {city}: {live_weather['Condition']}")
            st.image(f"https:{live_weather['Icon']}", width=80)
            
            # Display weather metrics
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

# ---- TAB 3: MONTHLY FORECAST ----
with tab3:
    st.subheader("📅 Monthly Climate Forecast")
    if df is not None:
        fig_monthly = px.line(future_monthly, x="ds", y="yhat", title="📈 Predicted Monthly Temperature", labels={"yhat": "Temperature (°C)"})
        st.plotly_chart(fig_monthly)
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TAB 4: YEARLY FORECAST ----
with tab4:
    st.subheader("📆 Yearly Climate Predictions (Next 5 Years)")
    if df is not None:
        fig_yearly = px.line(future_yearly, x="ds", y="yhat", title="📊 Predicted Yearly Temperature", labels={"yhat": "Temperature (°C)"})
        st.plotly_chart(fig_yearly)
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TAB 5: EXTREME CONDITIONS ----
with tab5:
    st.subheader("🚨 Extreme Climate Alerts & Visualizations")
    if df is not None:
        extreme_temps = future_monthly[future_monthly["yhat"] > future_monthly["yhat"].quantile(0.95)]
        if not extreme_temps.empty:
            st.error("⚠️ High-Temperature Alert! Unusual spikes detected.")
            fig_extreme_hot = px.bar(extreme_temps, x="ds", y="yhat", title="🔥 Extreme Heat Predictions", labels={"yhat": "Temperature (°C)"})
            st.plotly_chart(fig_extreme_hot)
        extreme_cold = future_monthly[future_monthly["yhat"] < future_monthly["yhat"].quantile(0.05)]
        if not extreme_cold.empty:
            st.warning("⚠️ Cold Spell Alert! Sudden drops detected.")
            fig_extreme_cold = px.bar(extreme_cold, x="ds", y="yhat", title="❄️ Extreme Cold Predictions", labels={"yhat": "Temperature (°C)"})
            st.plotly_chart(fig_extreme_cold)

# ---- TAB 6: CLIMATE TRENDS ----
with tab6:
    st.subheader("📈 Climate Change & Global Warming Trends")
    if df is not None:
        fig_trend = px.line(forecast, x=forecast.index, y="yhat", title="🌍 Climate Change Trends", labels={"yhat": "Temperature (°C)"})
        st.plotly_chart(fig_trend)
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TAB 7: CLIMATE SUMMARY & ACTIONS ----
with tab7:
    st.subheader("📢 Climate Summary & Recommended Actions")
    if df is not None:
        avg_temp = df["y"].mean()
        avg_co2 = df["CO2"].mean()
        avg_humidity = df["Humidity"].mean()

        st.write(f"### 🌡️ Average Temperature: {avg_temp:.2f}°C")
        st.write(f"### 💨 Average CO₂ Levels: {avg_co2:.2f} ppm")
        st.write(f"### 💧 Average Humidity: {avg_humidity:.2f}%")
        
        st.markdown("### 🔎 Key Insights:")
        if avg_temp > 30:
            st.error("🔥 High temperatures detected! Possible heat waves and drought risks.")
        if avg_co2 > 400:
            st.warning("🌍 Elevated CO₂ levels detected! Could contribute to global warming.")
        if avg_humidity < 30:
            st.warning("💧 Low humidity levels! Risk of dry conditions and wildfires.")

        st.markdown("### ✅ Recommended Actions:")
        st.write("- Reduce carbon emissions by promoting renewable energy and sustainable practices.")
        st.write("- Implement urban green spaces to combat rising temperatures.")
        st.write("- Monitor air quality and take preventive measures against pollution.")
        st.write("- Stay prepared for extreme weather conditions by following climate alerts.")
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- FOOTER ----
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**")
