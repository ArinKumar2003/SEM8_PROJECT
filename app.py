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
API_KEY = st.secrets.get("WEATHERAPI_KEY")  # Add your API key in Streamlit secrets

def get_live_weather(city):
    """Fetch real-time weather data from WeatherAPI.com."""
    if not API_KEY:
        st.error("❌ API Key is missing! Please check your Streamlit secrets.")
        return None

    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"❌ API Error: {response.status_code}")
        return None

    data = response.json()
    if "error" in data:
        st.error(f"⚠️ {data['error']['message']}")
        return None

    return data["current"]

# ---- DASHBOARD HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🌍 AI Climate Dashboard</h1>
    <h3 style='text-align: center; color: #7f8c8d;'>📡 Real-Time Weather, AI Forecasts & Advanced Analysis</h3>
    <hr>
""", unsafe_allow_html=True)

# ---- SIDEBAR FILE UPLOAD ----
uploaded_file = st.sidebar.file_uploader("📂 Upload Climate CSV", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.sidebar.error("⚠️ The uploaded CSV is empty.")
            df = None
        elif not all(col in df.columns for col in ["Years", "Temperature"]):
            st.sidebar.error("⚠️ Invalid CSV format. Required: Years, Temperature.")
            df = None
        else:
            # Ensure proper datetime format
            if all(col in df.columns for col in ["Years", "Month", "Day"]):
                df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
            else:
                df["ds"] = pd.to_datetime(df["Years"], format="%Y")

            df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})

    except pd.errors.EmptyDataError:
        st.sidebar.error("⚠️ Uploaded file is empty or corrupted.")
        df = None
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- TABS ----
tabs = st.tabs(["🌦 Live Weather", "📈 AI Forecasts", "🔮 Trends", "📊 Climate Score", "⚠️ Extreme Weather"])

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
                st.metric("Temperature", f"{weather['temp_c']}°C")
                st.write(f"**☁️ {weather['condition']['text']}**")
                st.write(f"💧 Humidity: {weather['humidity']}%  |  🌬 Wind: {weather['wind_kph']} km/h")
            else:
                st.error(f"❌ No data for {city}")

# ---- TAB 2: AI FORECASTS ----
with tabs[1]:
    st.subheader("📈 AI Climate Forecasts")

    if df is not None and len(df) > 1:  # Ensure enough data for training
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends")
        st.plotly_chart(fig)
    elif df is not None:
        st.error("⚠️ Not enough data to train AI model.")

# ---- TAB 3: INTERACTIVE TRENDS ----
with tabs[2]:
    st.subheader("🔮 Interactive Climate Trends")
    if df is not None:
        fig1 = px.scatter(df, x="ds", y="y", title="Temperature Trends Over Time")
        st.plotly_chart(fig1)
        
        fig2 = px.histogram(df, x="y", title="Temperature Distribution")
        st.plotly_chart(fig2)

# ---- TAB 4: CLIMATE SCORE ----
with tabs[3]:
    st.subheader("📊 Climate Impact Score")
    if df is not None:
        df["climate_score"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min()) * 100
        fig = px.line(df, x="ds", y="climate_score", title="Climate Impact Score")
        st.plotly_chart(fig)

# ---- TAB 5: EXTREME WEATHER ----
with tabs[4]:
    st.subheader("⚠️ Extreme Weather Alerts")
    if df is not None:
        threshold = st.slider("Set Temperature Alert Threshold", int(df["y"].min()), int(df["y"].max()), 35)
        alerts = df[df["y"] > threshold]

        st.write("### 🔥 Heatwave Alerts", alerts)
        fig = px.line(df, x="ds", y="y", title="Extreme Temperature Trends", markers=True)
        fig.add_trace(go.Scatter(x=alerts["ds"], y=alerts["y"], mode="markers", marker=dict(color="red", size=10), name="Extreme Heat"))
        st.plotly_chart(fig)

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**", unsafe_allow_html=True)
