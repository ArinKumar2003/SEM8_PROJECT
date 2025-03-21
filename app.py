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
forecast, future_monthly, future_yearly = None, None, None

if df is not None:
    try:
        with st.spinner("Training AI Climate Model... ⏳"):
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=365 * 5)
            forecast = model.predict(future)
            
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            forecast.set_index("ds", inplace=True)
            
            future_monthly = forecast.resample("M").mean(numeric_only=True).reset_index()
            future_yearly = forecast.resample("Y").mean(numeric_only=True).reset_index()
    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        df = None  

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Live Weather", "Historical Data", "Monthly Forecast", "Yearly Forecast", 
    "Extreme Conditions", "Climate Trends", "AQI Monitoring", 
    "Heatwaves & Coldwaves", "Global Climate Comparisons", "Summary"
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
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Temperature (°C)", live_weather["y"])
            col2.metric("Humidity (%)", live_weather["Humidity"])
            col3.metric("CO₂ Level (ppm)", live_weather["CO2"])

# ---- TAB 2: HISTORICAL DATA ----
with tab2:
    st.subheader("📜 Historical Climate Data (1971-Present)")
    if df is not None:
        fig_hist = px.line(df, x="ds", y="y", title="📊 Temperature Trends (1971-Present)", labels={"y": "Temperature (°C)"})
        st.plotly_chart(fig_hist)

# ---- TAB 6: CLIMATE TRENDS ----
with tab6:
    st.subheader("📈 Climate Trends Analysis")
    if df is not None:
        fig_trends = px.scatter(df, x="ds", y="y", color="CO2", title="🌡️ Climate Change Impact")
        st.plotly_chart(fig_trends)

# ---- TAB 7: AQI MONITORING ----
with tab7:
    st.subheader("🌫️ Air Quality Index (AQI) Monitoring")
    if df is not None:
        fig_aqi = px.line(df, x="ds", y="CO2", title="📊 CO2 Emissions Over Time")
        st.plotly_chart(fig_aqi)

# ---- TAB 8: HEATWAVES & COLDWAVES ----
with tab8:
    st.subheader("🔥 Heatwave & ❄️ Coldwave Detection")
    if future_monthly is not None:
        high_temps = future_monthly[future_monthly["yhat"] > future_monthly["yhat"].quantile(0.95)]
        low_temps = future_monthly[future_monthly["yhat"] < future_monthly["yhat"].quantile(0.05)]
        
        if not high_temps.empty:
            st.error("⚠️ Heatwave Alert! Extreme temperatures detected.")
            fig_heatwave = px.line(high_temps, x="ds", y="yhat", title="🔥 Heatwave Trends")
            st.plotly_chart(fig_heatwave)

# ---- TAB 9: GLOBAL CLIMATE COMPARISON ----
with tab9:
    st.subheader("🌍 Compare Climate Across Cities")
    city1, city2 = st.columns(2)
    city1_name = city1.text_input("City 1", "New York")
    city2_name = city2.text_input("City 2", "London")

    if st.button("Compare Climate"):
        weather1 = get_live_weather(city1_name)
        weather2 = get_live_weather(city2_name)
        if weather1 and weather2:
            data = pd.DataFrame([
                {"City": city1_name, "Temp (°C)": weather1["y"]},
                {"City": city2_name, "Temp (°C)": weather2["y"]}
            ])
            st.bar_chart(data.set_index("City"))

# ---- FOOTER ----
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**")
