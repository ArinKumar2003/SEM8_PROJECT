import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import requests

st.set_page_config(layout="wide")
st.title("🌍 Climate Forecast Dashboard")

# Upload CSV file
st.sidebar.header("📂 Upload Historical Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "ds" not in df.columns or "y" not in df.columns:
        st.sidebar.error("CSV must contain 'ds' and 'y' columns.")
        df = None
    else:
        df["ds"] = pd.to_datetime(df["ds"])  # Ensure datetime format
        st.sidebar.success("✅ File uploaded successfully")
else:
    st.sidebar.info("Please upload a dataset.")

# Tabs

TAB_NAMES = ["Live Weather", "Historical Data", "Monthly Forecast", "Yearly Forecast", "Climate Summary"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_NAMES)

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌦️ Live Weather Report")
    city = st.text_input("Enter a city name", "New York")
    api_key = st.secrets["weatherstack"]["api_key"]  # Replace with your API key
    if st.button("Get Weather"):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        try:
            res = requests.get(url)
            res.raise_for_status()
            weather_data = res.json()
            temp = weather_data['main']['temp']
            desc = weather_data['weather'][0]['description'].title()
            humidity = weather_data['main']['humidity']
            wind = weather_data['wind']['speed']

            st.metric(label="🌡️ Temperature (°C)", value=temp)
            st.metric(label="🌤️ Description", value=desc)
            st.metric(label="💧 Humidity", value=f"{humidity}%")
            st.metric(label="🌬️ Wind Speed", value=f"{wind} m/s")
        except:
            st.error("Could not retrieve weather data. Check your API key or city name.")

# ---- TAB 2: HISTORICAL DATA ----
with tab2:
    st.subheader("📊 Historical Climate Data")
    if df is not None:
        st.line_chart(df.set_index("ds"))
        st.write(df.describe())
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TAB 3: MONTHLY FORECAST ----
with tab3:
    st.subheader("🗓️ Monthly Forecast")
    if df is not None:
        periods_input = st.slider("Select number of months to forecast:", 1, 24, 6)

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=periods_input * 30)
        forecast = m.predict(future)

        fig1 = m.plot(forecast)
        st.write(fig1)
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TAB 4: YEARLY FORECAST ----
with tab4:
    st.subheader("📅 Yearly Forecast")
    if df is not None:
        years = st.slider("Select number of years to forecast:", 1, 10, 3)

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=years * 365)
        forecast = m.predict(future)

        fig2 = m.plot(forecast)
        st.write(fig2)
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    else:
        st.warning("📂 Please upload a CSV file.")

# ---- TAB 5: CLIMATE SUMMARY ----
with tab5:
    st.subheader("📌 Climate Summary")
    if df is not None:
        st.markdown("### 🔹 Key Statistics")
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
        fig_dist = px.histogram(df, x="y", nbins=30, title="Temperature Distribution", labels={"y": "Temperature (°C)"})
        st.plotly_chart(fig_dist)

        st.markdown("### 📆 Monthly Average Temperature")
        df_month = df.copy()
        df_month["Month"] = df_month["ds"].dt.to_period("M").astype(str)
        monthly_avg = df_month.groupby("Month")["y"].mean().reset_index()
        fig_month_avg = px.line(monthly_avg, x="Month", y="y", title="Monthly Avg Temperature", labels={"y": "Temperature (°C)"})
        st.plotly_chart(fig_month_avg)
    else:
        st.warning("📂 Please upload a CSV file.")
