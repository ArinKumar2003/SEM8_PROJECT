import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
from prophet import Prophet
import plotly.graph_objects as go

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- WEATHER API CONFIG ----
API_KEY = st.secrets["WEATHERAPI_KEY"] if "WEATHERAPI_KEY" in st.secrets else None  

def get_live_weather(city):
    """Fetch real-time weather data from WeatherAPI.com."""
    if not API_KEY:
        st.error("❌ API Key is missing! Please check your Streamlit secrets.")
        return None
    
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
    try:
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

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Live Weather", "Historical Data", "Monthly Forecast", "Yearly Forecast", "Extreme Conditions", "Help"])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌍 Live Weather Data")
    city = st.text_input("Enter City", "New York")
    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.write(f"### {city}: {live_weather['Condition']}")
            st.image(f"https:{live_weather['Icon']}", width=50)
            st.metric("Temperature (°C)", live_weather["y"])
            st.metric("Humidity (%)", live_weather["Humidity"])
            st.metric("CO₂ Level (ppm)", live_weather["CO2"])

# ---- TAB 2: HISTORICAL DATA ----
with tab2:
    st.subheader("📜 Historical Climate Data (1971-Present)")
    if df is not None:
        fig_hist = px.line(df, x="ds", y="y", title="📊 Temperature Trends (1971-Present)")
        st.plotly_chart(fig_hist)
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TRAIN MODEL & FORECAST ----
if df is not None:
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365 * 5)
    forecast = model.predict(future)

    forecast["ds"] = pd.to_datetime(forecast["ds"])
    forecast.set_index("ds", inplace=True)

    future_monthly = forecast.resample("M").mean(numeric_only=True).reset_index()
    future_yearly = forecast.resample("Y").mean(numeric_only=True).reset_index()

    forecast.reset_index(inplace=True)

# ---- TAB 3: MONTHLY FORECAST ----
with tab3:
    st.subheader("📅 Monthly Climate Forecast (2025–2030)")
    if df is not None:
        fig_monthly = px.line(future_monthly, x="ds", y="yhat", title="📊 Predicted Monthly Temperature Trends")
        st.plotly_chart(fig_monthly)

# ---- TAB 4: YEARLY FORECAST ----
with tab4:
    st.subheader("📆 Yearly Climate Forecast (2025–2030)")
    if df is not None:
        fig_yearly = px.bar(future_yearly, x="ds", y="yhat", title="📊 Predicted Yearly Temperature Trends")
        st.plotly_chart(fig_yearly)

# ---- TAB 5: EXTREME CONDITIONS & ALERTS ----
with tab5:
    st.subheader("🚨 Extreme Climate Alerts")
    if df is not None:
        extreme_temps = future_monthly[future_monthly["yhat"] > future_monthly["yhat"].quantile(0.95)]
        if not extreme_temps.empty:
            st.error("⚠️ High-Temperature Alert! Unusual spikes detected in future months.")
            st.table(extreme_temps[["ds", "yhat"]])
        else:
            st.success("✅ No extreme temperature conditions detected.")

        extreme_humidity = future_monthly[future_monthly["yhat"] < future_monthly["yhat"].quantile(0.05)]
        if not extreme_humidity.empty:
            st.warning("⚠️ Low-Temperature Alert! Potential cold periods detected.")
            st.table(extreme_humidity[["ds", "yhat"]])

# ---- TAB 6: HELP & INFORMATION ----
with tab6:
    st.subheader("ℹ️ Help & Information")
    st.markdown("""
        - **Live Weather:** Fetch real-time weather updates for any city.
        - **Historical Data:** View climate data from 1971 onwards.
        - **Monthly & Yearly Forecasts:** AI-powered predictions for upcoming months & years.
        - **Extreme Conditions:** Alerts for extreme climate changes.
        - **How Predictions Work:** Data is trained using historical records and real-time updates.
    """)

# ---- FOOTER ----
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**")
