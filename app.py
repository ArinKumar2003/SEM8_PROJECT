import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import requests
import os

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(layout="wide", page_title="🌍 Climate Forecast Dashboard")
st.title("🌍 Climate Forecast Dashboard")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data(path="GlobalWeatherRepository.csv"):
    df = pd.read_csv(path)
    # Auto-detect date and value columns
    date_col = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
    value_col = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]

    if not date_col or not value_col:
        st.error("CSV must contain a datetime column (renamed to 'ds') and a temperature or target column ('y').")
        return None

    df = df[[date_col[0], value_col[0]]]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna()

    return df

# Allow file upload or use default
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV with Climate Data", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)
else:
    if os.path.exists("GlobalWeatherRepository.csv"):
        df = load_data("GlobalWeatherRepository.csv")
    else:
        st.warning("No file uploaded and default dataset not found.")
        df = None

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(["🌦️ Live Weather", "📈 Historical Data", "📅 Monthly Forecast", "📆 Yearly Forecast", "📊 Climate Summary"])

# ----------------------------
# 🌦️ Tab 1: Live Weather
# ----------------------------
with tabs[0]:
    st.subheader("🌦️ Live Weather via Weatherstack API")
    city = st.text_input("Enter City", value="New York")
    if st.button("Get Weather"):
        try:
            api_key = st.secrets["weatherstack"]["api_key"]
            url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
            res = requests.get(url)
            weather_data = res.json()

            if "error" in weather_data:
                st.error(weather_data["error"]["info"])
            else:
                current = weather_data["current"]
                st.metric("🌡️ Temperature (°C)", current["temperature"])
                st.metric("💧 Humidity", f"{current['humidity']}%")
                st.metric("🌬️ Wind Speed", f"{current['wind_speed']} km/h")
                st.write("🔍", current["weather_descriptions"][0])

        except Exception as e:
            st.error(f"Failed to retrieve weather: {e}")

# ----------------------------
# 📈 Tab 2: Historical Data
# ----------------------------
with tabs[1]:
    st.subheader("📈 Historical Climate Data")
    if df is not None:
        fig = px.line(df, x="ds", y="y", title="Temperature Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail())
    else:
        st.warning("Please upload a dataset.")

# ----------------------------
# 📅 Tab 3: Monthly Forecast
# ----------------------------
with tabs[2]:
    st.subheader("📅 Monthly Forecast (Prophet)")
    if df is not None:
        months = st.slider("Months to Forecast", 1, 24, 6)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=months * 30)
        forecast = model.predict(future)
        fig = model.plot(forecast)
        st.pyplot(fig)
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    else:
        st.warning("Please upload a dataset.")

# ----------------------------
# 📆 Tab 4: Yearly Forecast
# ----------------------------
with tabs[3]:
    st.subheader("📆 Yearly Forecast (Prophet)")
    if df is not None:
        years = st.slider("Years to Forecast", 1, 10, 3)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=years * 365)
        forecast = model.predict(future)
        fig = model.plot(forecast)
        st.pyplot(fig)
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    else:
        st.warning("Please upload a dataset.")

# ----------------------------
# 📊 Tab 5: Climate Summary
# ----------------------------
with tabs[4]:
    st.subheader("📊 Climate Summary")
    if df is not None:
        st.markdown("### 📌 Key Statistics")
        stats = df["y"].describe().rename({
            "count": "Data Points",
            "mean": "Mean Temp (°C)",
            "std": "Standard Dev",
            "min": "Min Temp (°C)",
            "25%": "25th Percentile",
            "50%": "Median (°C)",
            "75%": "75th Percentile",
            "max": "Max Temp (°C)"
        })
        st.table(stats)

        st.markdown("### 🌡️ Temperature Distribution")
        fig = px.histogram(df, x="y", nbins=30, title="Temperature Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📆 Monthly Average Temperature")
        df_month = df.copy()
        df_month["Month"] = df_month["ds"].dt.to_period("M").astype(str)
        monthly_avg = df_month.groupby("Month")["y"].mean().reset_index()
        fig = px.line(monthly_avg, x="Month", y="y", title="Monthly Average Temperature", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload a dataset.")
