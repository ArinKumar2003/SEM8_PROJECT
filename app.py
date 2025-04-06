import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import requests
from io import StringIO

# -----------------------------
# Set up the Streamlit layout
st.set_page_config(page_title="📊 Weather Dashboard", layout="wide")
st.title("🌦️ Enhanced Weather Dashboard")

# -----------------------------
# Sidebar navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["📈 Upload & Forecast", "🌍 Live Weather", "📊 Raw Data"])

# -----------------------------
# Weatherstack API Key (replace with your actual key)
API_KEY = st.secrets["weatherstack"]["api_key"]

# -----------------------------
# Weatherstack live weather fetcher
def get_live_weather(city):
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "current" in data:
            return {
                "Temperature (°C)": data["current"]["temperature"],
                "Weather Description": data["current"]["weather_descriptions"][0],
                "Humidity (%)": data["current"]["humidity"],
                "Wind Speed (km/h)": data["current"]["wind_speed"],
                "Observation Time": data["current"]["observation_time"]
            }
        else:
            return {"Error": "City not found or data missing."}
    else:
        return {"Error": "Failed to connect to API."}

# -----------------------------
# Forecast Tab
if tab == "📈 Upload & Forecast":
    st.subheader("Upload a Weather Dataset for Forecasting")

    uploaded_file = st.file_uploader("Upload CSV (must contain 'ds' for date and 'y' for temperature)", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if 'ds' not in df.columns or 'y' not in df.columns:
                st.error("CSV must contain 'ds' and 'y' columns.")
            else:
                df['ds'] = pd.to_datetime(df['ds'])
                df = df[['ds', 'y']].dropna()

                st.success("File successfully uploaded and processed.")
                st.write("Preview of Data:", df.head())

                forecast_days = st.slider("Select forecast horizon (days)", 7, 90, 30)

                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

                st.subheader("📉 Forecast Plot")
                fig1 = plot_plotly(model, forecast)
                st.plotly_chart(fig1, use_container_width=True)

                st.subheader("📋 Forecast Data (tail)")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a CSV file to continue.")

# -----------------------------
# Live Weather Tab
elif tab == "🌍 Live Weather":
    st.subheader("Live Weather via Weatherstack API")
    city = st.text_input("Enter a city name", "New York")

    if st.button("Get Live Weather"):
        weather = get_live_weather(city)
        if "Error" in weather:
            st.error(weather["Error"])
        else:
            st.success(f"Current Weather in {city}")
            st.json(weather)

# -----------------------------
# Raw Data Viewer Tab
elif tab == "📊 Raw Data":
    st.subheader("View Uploaded CSV Data")
    raw_file = st.file_uploader("Upload CSV to preview its content", type="csv", key="raw")

    if raw_file:
        try:
            raw_df = pd.read_csv(raw_file)
            st.write("Preview of Raw Data", raw_df.head())
            st.dataframe(raw_df)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    else:
        st.info("Upload a CSV file to view it here.")
