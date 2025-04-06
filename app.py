import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import requests

# ----------------------------
# CONFIGURATION
# ----------------------------
WEATHERSTACK_API_KEY = "YOUR_API_KEY_HERE"  # ⬅️ Replace with your actual API key

st.set_page_config(page_title="🌍 Weather Dashboard", layout="wide")

# ----------------------------
# LOAD CLEANED DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_weather.csv")
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[['ds', 'y']].dropna()
    return df

# ----------------------------
# GET LIVE WEATHER FROM API
# ----------------------------
def get_live_weather(city):
    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"
    try:
        response = requests.get(url)
        data = response.json()
        if "current" in data:
            return {
                "temperature": data["current"]["temperature"],
                "weather": data["current"]["weather_descriptions"][0],
                "humidity": data["current"]["humidity"],
                "wind_speed": data["current"]["wind_speed"],
                "icon": data["current"]["weather_icons"][0]
            }
    except Exception as e:
        return None
    return None

# ----------------------------
# UI LAYOUT WITH TABS
# ----------------------------
st.title("🌤️ Global Weather Intelligence Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Forecasting", "🌍 Live Weather"])

# ----------------------------
# TAB 1: Dashboard
# ----------------------------
with tab1:
    st.subheader("Historical Weather Overview")
    df = load_data()

    with st.expander("📁 Raw Data Preview"):
        st.dataframe(df.head(20))

    st.markdown("### 🌡️ Temperature Trend Over Time")
    fig = px.line(df, x='ds', y='y', labels={'ds': 'Date', 'y': 'Temperature (°C)'})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# TAB 2: Forecasting with Prophet
# ----------------------------
with tab2:
    st.subheader("📈 Forecast Temperature")
    df = load_data()

    # Slider for forecast duration
    periods_input = st.slider("Select forecast days", min_value=7, max_value=90, value=30)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods_input)
    forecast = model.predict(future)

    st.markdown("### 🔮 Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("📊 Forecast Data"):
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input))

    st.markdown("### 🧩 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)

# ----------------------------
# TAB 3: Live Weather
# ----------------------------
with tab3:
    st.subheader("🌍 Live Weather Information")
    city = st.text_input("Enter a city name:", "New York")

    if st.button("Get Weather"):
        weather = get_live_weather(city)
        if weather:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(weather['icon'], width=80)
            with col2:
                st.markdown(f"### 🌡️ {weather['temperature']}°C — {weather['weather']}")
                st.markdown(f"💧 **Humidity**: {weather['humidity']}%")
                st.markdown(f"🌬️ **Wind Speed**: {weather['wind_speed']} km/h")
        else:
            st.error("Failed to fetch weather data. Please check the city name or your API key.")
