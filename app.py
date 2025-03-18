import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
from prophet import Prophet
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- TITLE ----
st.title("🌍 AI Climate Dashboard - Live & Historical Forecasts")

# ---- WEATHER API CONFIG ----
API_KEY = st.secrets.get("WEATHERAPI_KEY")

def get_live_weather(city):
    """Fetch real-time weather data from WeatherAPI.com and estimate CO₂ levels."""
    if not API_KEY:
        st.error("❌ API Key is missing! Please check your Streamlit secrets.")
        return None

    url_weather = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    try:
        response_weather = requests.get(url_weather, timeout=10)
        response_weather.raise_for_status()
        data_weather = response_weather.json()
        if "error" in data_weather:
            st.error(f"⚠️ {data_weather['error']['message']}")
            return None

        # Fetch weather icon
        icon_url = f"https:{data_weather['current']['condition']['icon']}"
        icon_img = None
        try:
            response_icon = requests.get(icon_url, timeout=5)
            if response_icon.status_code == 200:
                icon_img = Image.open(BytesIO(response_icon.content))
        except Exception:
            st.warning("⚠️ Weather icon could not be loaded.")

        co2_level = 410  # Placeholder CO₂ level (Replace with API if available)

        return {
            "ds": datetime.datetime.now(),
            "Temperature": float(data_weather["current"]["temp_c"]),
            "Humidity": data_weather["current"]["humidity"],
            "CO2": co2_level,
            "SeaLevel": None,
            "Condition": data_weather["current"]["condition"]["text"],
            "Icon": icon_img  # Save image object
        }
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request failed: {e}")
        return None

# ---- SIDEBAR FILE UPLOAD ----
st.sidebar.subheader("📂 Upload Climate CSV (1971+)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.sidebar.error("⚠️ The uploaded CSV is empty.")
            df = None
        elif not all(col in df.columns for col in ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel", "Temperature"]):
            st.sidebar.error("⚠️ Invalid CSV format. Required: Years, Month, Day, CO2, Humidity, SeaLevel, Temperature.")
            df = None
        else:
            df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
            df = df[["ds", "CO2", "Humidity", "SeaLevel", "Temperature"]]
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- Initialize Variables ----
live_weather = None  # ✅ Ensure `live_weather` is always initialized

# ---- TAB LAYOUT ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Live Weather", "📜 Historical Data (1971)", "🔮 AI Predictions (2025+)", "📈 Monthly & Yearly Forecasts", "🆘 Help"])

# ---- TAB 1: LIVE WEATHER DASHBOARD ----
with tab1:
    st.subheader("🌍 Live Weather Dashboard")
    city = st.text_input("Enter City", "New York")

    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.success(f"✔️ Live weather for {city} fetched successfully!")

            if live_weather["Icon"]:
                st.image(live_weather["Icon"], width=50)  # Display image correctly

            st.write(f"**Condition:** {live_weather['Condition']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("🌡 Temperature", f"{live_weather['Temperature']}°C")
            col2.metric("💧 Humidity", f"{live_weather['Humidity']}%")
            col3.metric("🌎 CO₂ Levels", f"{live_weather['CO2']} ppm")

# ---- TAB 2: HISTORICAL CLIMATE DATA ----
with tab2:
    st.subheader("📜 Historical Climate Data (Since 1971)")
    
    if df is not None:
        st.write("📅 **Displaying first few rows of historical data:**")
        st.dataframe(df.head())

        fig_hist = px.line(df, x="ds", y="Temperature", title="📊 Historical Temperature Trends (1971+)")
        st.plotly_chart(fig_hist)
    else:
        st.warning("📌 Upload historical climate data to view insights.")

# ---- TAB 3: PREDICTIVE CLIMATE CONDITIONS (2025+) ----
with tab3:
    st.subheader("🔮 AI Climate Predictions (2025+)")

    if df is not None and len(df) > 1:
        model = Prophet()
        model.fit(df.rename(columns={"Temperature": "y"}))

        future_5y = model.make_future_dataframe(periods=365 * 5)  # 5 years ahead
        forecast_5y = model.predict(future_5y)

        fig_forecast = px.line(forecast_5y, x="ds", y="yhat", title="🌡 Future Temperature Predictions (2025-2030)")
        st.plotly_chart(fig_forecast)

    else:
        st.warning("📌 Upload historical data to generate predictions.")

# ---- TAB 4: MONTHLY & YEARLY CLIMATE FORECASTS ----
with tab4:
    st.subheader("📈 Monthly & Yearly Climate Forecasts")

    if df is not None:
        future_monthly = forecast_5y.resample("M", on="ds").mean().reset_index()
        future_yearly = forecast_5y.resample("Y", on="ds").mean().reset_index()

        st.write("📅 **Monthly Predictions:**")
        fig_monthly = px.line(future_monthly, x="ds", y="yhat", title="📊 Monthly Predicted Climate Trends (2025–2030)")
        st.plotly_chart(fig_monthly)

        st.write("📅 **Yearly Predictions:**")
        fig_yearly = px.line(future_yearly, x="ds", y="yhat", title="📊 Yearly Predicted Climate Trends (2025–2030)")
        st.plotly_chart(fig_yearly)
    else:
        st.warning("📌 Upload historical data to view forecasts.")

# ---- TAB 5: HELP & ALERTS ----
with tab5:
    st.subheader("🆘 Help & Alerts")

    st.markdown("""
    - **Live Weather Data**: Enter a city and fetch real-time temperature, humidity, and estimated CO₂ levels.
    - **Historical Data**: Upload historical climate data (1971+) to train AI models.
    - **Predictions (2025+)**: AI-based climate forecasting using Prophet.
    - **Extreme Weather Alerts**: Automated alerts for dangerous temperatures.
    """)

    if live_weather:  # ✅ Check if `live_weather` exists before accessing it
        if live_weather["Temperature"] > 40:
            st.error("🚨 Heatwave Alert! High temperatures detected.")
        elif live_weather["Temperature"] < 0:
            st.warning("❄️ Cold Weather Alert! Freezing conditions expected.")
        elif live_weather["CO2"] > 450:
            st.warning("🌎 High CO₂ Levels Detected! Consider environmental precautions.")
    else:
        st.info("ℹ️ Fetch live weather data to enable alerts.")
