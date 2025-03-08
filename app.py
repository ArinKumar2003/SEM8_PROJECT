import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
from prophet import Prophet
import plotly.graph_objects as go

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- DASHBOARD HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🌍 AI Climate Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📡 Real-Time Weather, AI Forecasts & Advanced Analysis</h3>
    <hr>
""", unsafe_allow_html=True)

# ---- WEATHER API CONFIG ----
API_KEY = st.secrets["WEATHERSTACK_API_KEY"] if "WEATHERSTACK_API_KEY" in st.secrets else None

def get_live_weather(city):
    """Fetch real-time weather data from WeatherStack API."""
    if not API_KEY:
        return None
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("current")
    return None

# ---- TABS ----
tabs = st.tabs([
    "🌦 Live Weather", "📈 AI Forecasts", "🔮 Trends", "📊 Climate Score", "⚠️ Extreme Weather"
])

# ---- TAB 1: LIVE WEATHER ----
with tabs[0]:
    st.subheader("🌦 Live Weather")
    cities = st.text_input("Enter Cities (comma-separated)", "New York, London, Tokyo")
    city_list = [city.strip() for city in cities.split(",")]

    if st.button("🔍 Get Live Weather"):
        for city in city_list:
            weather = get_live_weather(city)
            if weather:
                st.write(f"### {city}")
                st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                st.write(f"**☁️ {weather.get('weather_descriptions', ['N/A'])[0]}**")
                st.write(f"💧 Humidity: {weather.get('humidity', 'N/A')}%  |  🌬 Wind: {weather.get('wind_speed', 'N/A')} km/h")
            else:
                st.error(f"❌ No weather data available for {city}")

# ---- TAB 2: AI FORECASTS ----
with tabs[1]:
    st.subheader("📈 AI Climate Forecasts")
    uploaded_file = st.file_uploader("📂 Upload Climate CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("⚠️ Uploaded CSV is empty. Please upload a valid file.")
            elif {"Years", "Month", "Day", "Temperature"}.issubset(df.columns):
                df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
                df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})
                
                model = Prophet()
                model.fit(df)
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends", 
                              labels={"ds": "Date", "yhat": "Predicted Temperature (°C)"})
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Invalid CSV format. Required columns: Years, Month, Day, Temperature.")
        except pd.errors.EmptyDataError:
            st.error("⚠️ Error: The uploaded file is empty or corrupted.")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

# ---- TAB 3: INTERACTIVE TRENDS ----
with tabs[2]:
    st.subheader("🔮 Interactive Climate Trends")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if {"Years", "Month", "Day", "Temperature"}.issubset(df.columns):
            df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
            df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})

            fig = px.scatter(df, x="ds", y="y", title="Temperature Trends Over Time",
                             labels={"ds": "Date", "y": "Temperature (°C)"},
                             color_discrete_sequence=["#3498db"])
            st.plotly_chart(fig)

            fig2 = px.histogram(df, x="y", title="Temperature Distribution",
                                labels={"y": "Temperature (°C)"},
                                color_discrete_sequence=["#e74c3c"])
            st.plotly_chart(fig2)
        else:
            st.error("⚠️ Invalid CSV format.")

# ---- TAB 4: CLIMATE SCORE ----
with tabs[3]:
    st.subheader("📊 Climate Impact Score")

    if uploaded_file:
        df["climate_score"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min()) * 100

        fig = px.line(df, x="ds", y="climate_score", title="Climate Impact Score",
                      labels={"ds": "Date", "climate_score": "Score (%)"},
                      color_discrete_sequence=["#2ecc71"])
        st.plotly_chart(fig)

# ---- TAB 5: EXTREME WEATHER ----
with tabs[4]:
    st.subheader("⚠️ Extreme Weather Alerts")

    if uploaded_file:
        threshold = st.slider("Set Temperature Alert Threshold", int(df["y"].min()), int(df["y"].max()), 35)
        alerts = df[df["y"] > threshold]

        if not alerts.empty:
            st.write("### 🔥 Heatwave Alerts")
            st.dataframe(alerts)

            fig = px.line(df, x="ds", y="y", title="Extreme Temperature Trends", markers=True,
                          labels={"ds": "Date", "y": "Temperature (°C)"},
                          color_discrete_sequence=["#f39c12"])
            fig.add_trace(go.Scatter(
                x=alerts["ds"], y=alerts["y"], 
                mode="markers", marker=dict(color="red", size=10), name="Extreme Heat"
            ))
            st.plotly_chart(fig)
        else:
            st.success("✅ No extreme weather alerts detected.")

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherStack & Streamlit**", unsafe_allow_html=True)
