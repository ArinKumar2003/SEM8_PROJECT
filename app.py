import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from datetime import datetime, timedelta
import requests

# Set page configuration
st.set_page_config(page_title="Climate Dashboard", layout="wide")

# Load data
df = pd.read_csv("climate.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)

# Sidebar navigation
st.sidebar.title("Navigation")
tabs = ["Live Weather", "Overview", "Forecasting"]
choice = st.sidebar.radio("Go to", tabs)

# Constants for weather API
API_KEY = "e12e93484a0645f2802141629250803"
WEATHER_URL = "http://api.weatherapi.com/v1/current.json"

if choice == "Live Weather":
    st.title("🌍 Live Weather")
    city = st.text_input("Enter city for live weather:", value="New York")
    if st.button("Get Weather"):
        params = {"key": API_KEY, "q": city}
        response = requests.get(WEATHER_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            st.subheader(f"Weather in {data['location']['name']}, {data['location']['country']}")
            st.write(f"**Temperature:** {data['current']['temp_c']} °C")
            st.write(f"**Humidity:** {data['current']['humidity']}%")
            st.write(f"**Wind Speed:** {data['current']['wind_kph']} kph")
            st.write(f"**Condition:** {data['current']['condition']['text']}")
        else:
            st.error("Failed to retrieve weather data.")

elif choice == "Overview":
    st.title("🌍 Climate Data Overview")
    st.write("### Raw Data")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Temperature Trend")
    fig, ax = plt.subplots()
    df.groupby('Date')['Temperature'].mean().plot(ax=ax, title="Average Temperature Over Time")
    st.pyplot(fig)

elif choice == "Forecasting":
    st.title("🌎 Forecasting Temperature")

    df_forecast = df[['Date', 'Temperature']].dropna()
    df_forecast = df_forecast.rename(columns={'Date': 'ds', 'Temperature': 'y'})

    model = Prophet()
    model.fit(df_forecast)

    future = model.make_future_dataframe(periods=365, freq='D')
    forecast = model.predict(future)

    st.write("### Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.write("### Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    def get_prediction(date):
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        closest = forecast.iloc[(forecast['ds'] - date).abs().argsort()[:1]]
        if not closest.empty:
            return round(closest['yhat'].values[0], 2)
        return "No data"

    tomorrow = datetime.today() + timedelta(days=1)
    next_month = datetime.today() + timedelta(days=30)

    st.write("### 📆 Key Date Predictions")
    st.write(f"\U0001F4CD Tomorrow ({tomorrow.date()}): {get_prediction(tomorrow)} °C")
    st.write(f"\U0001F4CD Next Month ({next_month.date()}): {get_prediction(next_month)} °C")
