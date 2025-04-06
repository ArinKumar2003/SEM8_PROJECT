import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(page_title="Weather Dashboard", layout="wide")

# Live Weather API
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

# Forecasting
def forecast_weather(df, periods=30):
    df_prophet = df[['ds', 'y']].dropna()
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# Upload section
st.sidebar.header("📁 Upload Weather Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["ds"])
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("CSV must contain 'ds' (datetime) and 'y' (temperature/target) columns.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file with 'ds' and 'y' columns to begin.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌤 Live Weather", "📊 Dataset", "📈 Forecast", "🌱 Climate Awareness"])

# Tab 1: Live Weather
with tab1:
    st.header("🌤 Live Weather")
    city = st.text_input("Enter a city name", "New York")
    if st.button("Get Weather"):
        weather = get_live_weather(city)
        if weather:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(weather["icon"], width=80)
            with col2:
                st.metric("Temperature (°C)", weather["temperature"])
                st.metric("Humidity (%)", weather["humidity"])
                st.metric("Wind Speed (km/h)", weather["wind_speed"])
                st.write("**Condition:**", weather["weather_descriptions"][0])
        else:
            st.error("Unable to fetch weather data.")

# Tab 2: Dataset View
with tab2:
    st.header("📊 Weather Dataset")
    st.write("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("### Temperature Over Time")
    fig = px.line(df, x="ds", y="y", labels={"ds": "Date", "y": "Temperature (°C)"})
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Forecast
with tab3:
    st.header("📈 Weather Forecast")
    n_days = st.slider("Select forecast days", 7, 60, 30)
    with st.spinner("Forecasting..."):
        model, forecast = forecast_weather(df, periods=n_days)
        st.subheader("Forecasted Temperature")
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Forecast Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days), use_container_width=True)

# Tab 4: Climate Awareness
with tab4:
    st.header("🌱 Climate Awareness & Summary")

    st.subheader("🔎 Summary of Uploaded Data")
    st.write("Data Range:", df["ds"].min(), "to", df["ds"].max())
    st.write("Average Temperature:", round(df["y"].mean(), 2), "°C")
    st.write("Max Temperature:", df["y"].max(), "°C")
    st.write("Min Temperature:", df["y"].min(), "°C")

    st.subheader("🌍 Climate Awareness Tips")
    st.markdown("""
    - 🌿 **Plant trees** to absorb CO₂.
    - 🔌 **Unplug electronics** when not in use.
    - 🚲 **Use sustainable transport**.
    - 🌞 **Adopt solar and clean energy**.
    - ♻️ **Recycle** and minimize waste.
    """)

    st.subheader("📚 Learn More")
    st.markdown("""
    - [NASA Climate Change](https://climate.nasa.gov/)
    - [UN Climate Action](https://www.un.org/en/climatechange)
    """)

