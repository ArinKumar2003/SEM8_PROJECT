import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(page_title="🌦️ Weather Insights Dashboard", layout="wide")

# --- Secrets for API key ---
def get_live_weather(city):
    api_key = st.secrets["weatherstack"]["api_key"]
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'current' in data:
            return {
                "temperature": data["current"]["temperature"],
                "weather_descriptions": data["current"]["weather_descriptions"][0],
                "humidity": data["current"]["humidity"],
                "wind_speed": data["current"]["wind_speed"],
                "icon": data["current"]["weather_icons"][0]
            }
    return None

# --- Forecast Function ---
def forecast_weather(df, periods=30):
    df_prophet = df[['ds', 'y']].dropna()
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🌍 Global Weather Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Live weather, forecasts, and climate insights — all in one place</h4>", unsafe_allow_html=True)

# --- Upload Data in Sidebar ---
st.sidebar.header("📁 Upload Your Weather Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'ds' (date) and 'y' (temp)", type=["csv"])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["ds"])
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.sidebar.error("CSV must contain 'ds' and 'y' columns.")
            df = None
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = None

# --- Tab Layout (no upload tab) ---
tabs = st.tabs([
    "🌍 Live Weather",
    "📊 Dataset Overview",
    "📈 Forecast",
    "📉 Trend Analysis",
    "📤 Export Forecast",
    "🌱 Climate Awareness"
])

# --- Tab 1: Live Weather ---
with tabs[0]:
    st.header("🌍 Real-Time Weather Conditions")
    city = st.text_input("Enter a city", "New York")
    if city:
        weather = get_live_weather(city)
        if weather:
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
            with col1:
                st.image(weather["icon"], width=80)
            with col2:
                st.metric("Temperature (°C)", weather["temperature"])
            with col3:
                st.metric("Humidity", f"{weather['humidity']}%")
            with col4:
                st.metric("Wind Speed", f"{weather['wind_speed']} km/h")
            st.caption(f"Description: {weather['weather_descriptions']}")
        else:
            st.warning("⚠️ Weather data not found for this city.")

# --- Tab 2: Dataset Overview ---
with tabs[1]:
    st.header("📊 Dataset Overview")
    if df is not None:
        st.write(df.head(10))
        st.success(f"✅ {len(df)} records from {df['ds'].min().date()} to {df['ds'].max().date()}")
        fig = px.line(df, x='ds', y='y', title="Temperature Over Time", labels={'ds': 'Date', 'y': 'Temperature (°C)'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a dataset from the sidebar to view data.")

# --- Tab 3: Forecast ---
with tabs[2]:
    st.header("📈 Temperature Forecast")
    if df is not None:
        days = st.slider("Days to Forecast", 7, 90, 30)
        with st.spinner("Generating forecast..."):
            model, forecast = forecast_weather(df, days)
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days))
    else:
        st.info("Upload data to generate a forecast.")

# --- Tab 4: Trend Analysis ---
with tabs[3]:
    st.header("📉 Trend Analysis")
    if df is not None:
        df["month"] = df["ds"].dt.to_period("M")
        monthly_avg = df.groupby("month")["y"].mean().reset_index()
        monthly_avg["month"] = monthly_avg["month"].astype(str)

        st.subheader("📆 Monthly Average Temperature")
        fig = px.line(monthly_avg, x='month', y='y', title="Monthly Average", labels={"y": "Temp (°C)"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 7-Day Moving Average")
        df["rolling"] = df["y"].rolling(window=7).mean()
        fig = px.line(df, x="ds", y="rolling", title="7-Day Moving Average")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend analysis available without data.")

# --- Tab 5: Export Forecast ---
with tabs[4]:
    st.header("📤 Export Forecast")
    if df is not None:
        _, forecast = forecast_weather(df)
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
    else:
        st.info("Generate a forecast before exporting.")

# --- Tab 6: Climate Awareness ---
with tabs[5]:
    st.header("🌱 Climate Summary & Awareness")
    if df is not None:
        st.subheader("🧾 Data Summary")
        st.write("📅 Date Range:", df["ds"].min().date(), "→", df["ds"].max().date())
        st.write("🌡 Avg Temp:", round(df["y"].mean(), 2), "°C")
        st.write("🔺 Max Temp:", df["y"].max(), "°C")
        st.write("🔻 Min Temp:", df["y"].min(), "°C")
    else:
        st.info("Upload data to view summary.")

    st.subheader("🌍 Climate Tips")
    st.markdown("""
    - 🚶 Walk or bike for short trips  
    - 🌲 Plant trees and support reforestation  
    - 💡 Use energy-efficient lighting  
    - ♻️ Practice the 3Rs: Reduce, Reuse, Recycle  
    - 🌐 Spread awareness on climate change
    """)

    st.subheader("📚 Learn More")
    st.markdown("""
    - [UN Climate Action](https://www.un.org/en/climatechange)  
    - [NASA Climate Change](https://climate.nasa.gov/)
    """)

