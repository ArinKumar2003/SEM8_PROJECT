import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from io import StringIO
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Climate Forecast App", layout="wide")

st.title("🌍 Climate Forecasting and Weather Insights")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("climate_large_data_sorted.csv")
    df['Date'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
    return df

df = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Forecasting", "📈 Historical Trends", "☀️ Live Weather"])

# --- TAB 1: Forecasting ---
with tab1:
    st.header("📊 Forecast Future Climate Data")

    metric = st.selectbox("Select metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'])

    # Prepare data
    data = df[['Date', metric]].rename(columns={"Date": "ds", metric: "y"})

    # Forecast
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    st.subheader("Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Download
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Forecast", data=csv, file_name='forecast.csv', mime='text/csv')

# --- TAB 2: Historical Trends ---
with tab2:
    st.header("📈 Visualize Historical Trends")

    metric = st.selectbox("Select metric for trend analysis", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='trend')

    fig2 = px.line(df, x='Date', y=metric, title=f"Historical {metric} Trends")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📉 Seasonal Decomposition")
    result = seasonal_decompose(df[metric], period=12, model='additive')
    st.line_chart(result.trend)
    st.line_chart(result.seasonal)
    st.line_chart(result.resid)

# --- TAB 3: Live Weather ---
with tab3:
    st.header("☀️ Live Weather Data")

    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your real key
    city = st.text_input("Enter city name", "Mohali")

    def get_weather(city):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        return response.json()

    if city:
        weather = get_weather(city)
        if weather.get("main"):
            st.metric("Temperature (°C)", weather['main']['temp'])
            st.metric("Humidity (%)", weather['main']['humidity'])
            st.write(f"🌤️ **Condition**: {weather['weather'][0]['description'].title()}")
        else:
            st.error("City not found or API issue.")
