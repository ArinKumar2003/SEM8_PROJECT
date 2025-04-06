import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(page_title="🌤 Weather Insights Dashboard", layout="wide")

# --- API: Get live weather ---
def get_live_weather(city):
    api_key = st.secrets["weatherstack"]["api_key"]
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'current' in data:
            return {
                "temperature": data["current"]["temperature"],
                "weather_descriptions": data["current"]["weather_descriptions"],
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

# --- Title & Header ---
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🌤 Weather Insights & Forecast Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Track live weather, explore trends, forecast the future, and promote climate awareness")

# --- Live Weather Display ---
with st.expander("🌍 Live Weather Overview", expanded=True):
    city = st.text_input("Enter city name", "New York")
    weather = get_live_weather(city)
    if weather:
        col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
        with col1: st.image(weather["icon"], width=80)
        with col2: st.metric("Temp (°C)", weather["temperature"])
        with col3: st.metric("Humidity", f"{weather['humidity']}%")
        with col4: st.metric("Wind Speed", f"{weather['wind_speed']} km/h")
    else:
        st.warning("⚠️ Could not fetch live weather.")

# --- Upload Data ---
st.sidebar.header("📁 Upload Your Weather CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with 'ds' (datetime) and 'y' (temperature)", type=["csv"])

if not uploaded_file:
    st.warning("Please upload a file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, parse_dates=["ds"])
    if 'ds' not in df.columns or 'y' not in df.columns:
        st.error("CSV must contain 'ds' and 'y' columns.")
        st.stop()
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# --- Create Tabs ---
tabs = st.tabs([
    "📊 Dataset Overview",
    "📈 Forecast",
    "📉 Trend Analysis",
    "📍 Location Insights",
    "📤 Export Forecast",
    "🌱 Climate Awareness"
])

# --- Tab 1: Dataset Overview ---
with tabs[0]:
    st.header("📊 Dataset Overview")
    st.write(df.head(10))
    st.write("### Temperature Trend")
    fig = px.line(df, x='ds', y='y', title="Temperature over Time", labels={'ds': 'Date', 'y': 'Temperature (°C)'})
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Forecast ---
with tabs[1]:
    st.header("📈 Forecast Temperature")
    days = st.slider("Forecast Days", 7, 60, 30)
    with st.spinner("Building forecast..."):
        model, forecast = forecast_weather(df, days)
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)
    st.write("### Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days))

# --- Tab 3: Trend Analysis ---
with tabs[2]:
    st.header("📉 Trend Analysis")
    df["month"] = df["ds"].dt.to_period("M")
    monthly_avg = df.groupby("month")["y"].mean().reset_index()
    monthly_avg["month"] = monthly_avg["month"].astype(str)
    st.subheader("Monthly Average Temperature")
    fig = px.line(monthly_avg, x='month', y='y', title="Monthly Average", labels={"y": "Temp (°C)"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Moving Average")
    df["rolling"] = df["y"].rolling(window=7).mean()
    fig = px.line(df, x="ds", y="rolling", title="7-Day Moving Average")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Location Insights ---
with tabs[3]:
    st.header("📍 Location-Based Insights")
    if "location" in df.columns:
        st.write("Average Temp by Location")
        loc_avg = df.groupby("location")["y"].mean().reset_index()
        fig = px.bar(loc_avg, x="location", y="y", color="y", labels={"y": "Temp (°C)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Your dataset does not contain a 'location' column.")

# --- Tab 5: Export Forecast ---
with tabs[4]:
    st.header("📤 Export Forecast")
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")

# --- Tab 6: Climate Awareness ---
with tabs[5]:
    st.header("🌱 Climate Awareness & Summary")
    st.subheader("📌 Data Summary")
    st.write("Date Range:", df["ds"].min().date(), "to", df["ds"].max().date())
    st.write("Avg Temp:", round(df["y"].mean(), 2), "°C")
    st.write("Max Temp:", df["y"].max(), "°C")
    st.write("Min Temp:", df["y"].min(), "°C")

    st.subheader("🌍 Sustainability Tips")
    st.markdown("""
    - 💡 Turn off lights when not in use  
    - 🚲 Walk, bike, or use public transport  
    - 🌳 Plant trees and green your space  
    - ♻️ Reuse, recycle, reduce waste  
    - 🔋 Use energy-efficient appliances
    """)

    st.subheader("📚 Learn More")
    st.markdown("""
    - [UN Climate Action](https://www.un.org/en/climatechange)
    - [NASA Climate Change](https://climate.nasa.gov/)
    """)

