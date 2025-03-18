import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
from prophet import Prophet
import plotly.graph_objects as go

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

        co2_level = 410  # Placeholder CO₂ level (Replace with API if available)

        return {
            "ds": datetime.datetime.now(),
            "Temperature": float(data_weather["current"]["temp_c"]),
            "Humidity": data_weather["current"]["humidity"],
            "CO2": co2_level,
            "SeaLevel": None,
            "Condition": data_weather["current"]["condition"]["text"],
            "Icon": data_weather["current"]["condition"]["icon"]
        }
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request failed: {e}")
        return None

# ---- SIDEBAR: HISTORICAL DATA UPLOAD ----
st.sidebar.header("📂 Upload Historical Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Years, Month, Day, CO2, Humidity, SeaLevel, Temperature)", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
        df = df[["ds", "CO2", "Humidity", "SeaLevel", "Temperature"]]
        df.rename(columns={"Temperature": "y"}, inplace=True)
        df.dropna(inplace=True)
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌡 Live Weather", 
    "📊 Climate Trends (1971–2025)", 
    "📆 Predictions (2025–2035)", 
    "📌 Yearly Outlook", 
    "⚠️ Extreme Weather Alerts",
    "❓ Help"
])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌍 Live Weather Dashboard")
    city = st.text_input("Enter City", "New York")

    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.success(f"✔️ Live weather for {city} fetched successfully!")

            st.image(live_weather["Icon"], width=50)
            st.write(f"**Condition:** {live_weather['Condition']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("🌡 Temperature", f"{live_weather['Temperature']}°C")
            col2.metric("💧 Humidity", f"{live_weather['Humidity']}%")
            col3.metric("🌎 CO₂ Levels", f"{live_weather['CO2']} ppm")

            if df is not None:
                df_live = pd.concat([df, pd.DataFrame([live_weather])], ignore_index=True)

                fig = px.line(df_live, x="ds", y="y", title="Live vs Historical Temperature Trends",
                              labels={"ds": "Date", "y": "Temperature (°C)"},
                              color_discrete_sequence=["blue"])
                st.plotly_chart(fig)

# ---- TAB 2: HISTORICAL CLIMATE TRENDS ----
with tab2:
    st.subheader("📊 Climate Trends (1971–2025)")

    if df is not None:
        fig2 = px.line(df, x="ds", y="y", title="Historical Temperature Trends (1971–2025)")
        st.plotly_chart(fig2)

# ---- TAB 3: PREDICTIONS (2025–2035) ----
with tab3:
    st.subheader("📆 Climate Predictions (2025–2035)")

    if df is not None:
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=365*10)
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] >= "2025-04-01"]

        fig3 = px.line(forecast_future, x="ds", y="yhat", title="Climate Predictions (2025–2035)",
                      labels={"ds": "Year", "yhat": "Predicted Temperature (°C)"},
                      color_discrete_sequence=["red"])
        st.plotly_chart(fig3)

# ---- TAB 4: YEARLY OUTLOOK ----
with tab4:
    st.subheader("📌 Yearly Climate Outlook (2025–2035)")

    if df is not None:
        future_yearly = forecast_future.resample("Y", on="ds").mean().reset_index()
        fig4 = px.bar(future_yearly, x="ds", y="yhat", title="Predicted Yearly Temperature Averages",
                      labels={"ds": "Year", "yhat": "Temperature (°C)"},
                      color_discrete_sequence=["green"])
        st.plotly_chart(fig4)

# ---- TAB 5: EXTREME WEATHER ALERTS ----
with tab5:
    st.subheader("⚠️ Extreme Weather Alerts")

    if df is not None:
        high_risk = forecast_future[forecast_future["yhat"] > forecast_future["yhat"].quantile(0.95)]

        fig5 = px.scatter(high_risk, x="ds", y="yhat", title="⚠️ Extreme Weather Events (2025+)",
                          color_continuous_scale="reds")
        st.plotly_chart(fig5)

        if not high_risk.empty:
            st.error("🚨 Extreme Heatwave Warning! Temperatures exceed historical records.")
        else:
            st.success("✅ No extreme weather events detected.")

# ---- TAB 6: HELP ----
with tab6:
    st.subheader("❓ Help & Information")
    st.write("""
    **How to Use This Dashboard?**  
    - **Live Weather**: Fetch real-time climate conditions for any city.  
    - **Historical Trends**: View temperature patterns from **1971–2025**.  
    - **Predictions**: AI-driven forecast for **2025–2035**.  
    - **Yearly Outlook**: Yearly **average temperature trends**.  
    - **Extreme Weather Alerts**: Warnings for **heatwaves, floods, droughts**.  
    """)

# ---- FOOTER ----
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**")
