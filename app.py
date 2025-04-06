import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import requests

st.set_page_config(page_title="Climate Forecast Dashboard", layout="wide")
st.title("🌍 Climate Forecast Dashboard")

# ----------------------------
# Sidebar: Upload file & config
# ----------------------------
st.sidebar.header("📂 Upload Historical Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Automatically detect time/date column and rename to 'ds' if needed
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        if date_col and 'ds' not in df.columns:
            df = df.rename(columns={date_col: "ds"})
        if 'ds' not in df.columns or 'temperature' not in df.columns and 'y' not in df.columns:
            st.error("CSV must contain a datetime column (renamed to 'ds') and a temperature or target column ('y').")
            return None
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'] if 'y' in df.columns else df['temperature']
        return df[['ds', 'y']]
    return None

df = load_data(uploaded_file)

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(["🌦️ Live Weather", "📊 Historical Data", "📅 Monthly Forecast", "📈 Yearly Forecast", "📌 Climate Summary"])

# ----------------------------
# Tab 1: Live Weather (Weatherstack API)
# ----------------------------
with tabs[0]:
    st.subheader("🌦️ Live Weather Report")
    city = st.text_input("Enter city name", "New York")
    weatherstack_key = st.secrets["weatherstack"]["api_key"]

    if st.button("Get Weather"):
        url = f"http://api.weatherstack.com/current?access_key={weatherstack_key}&query={city}"
        try:
            response = requests.get(url)
            data = response.json()
            if 'current' in data:
                current = data['current']
                st.metric("🌡️ Temperature (°C)", current['temperature'])
                st.metric("🌤️ Description", current['weather_descriptions'][0])
                st.metric("💧 Humidity", f"{current['humidity']}%")
                st.metric("🌬️ Wind Speed", f"{current['wind_speed']} km/h")
            else:
                st.error("❌ Unable to fetch weather. Check city name or API key.")
        except Exception as e:
            st.error("Failed to fetch weather data.")

# ----------------------------
# Tab 2: Historical Data
# ----------------------------
with tabs[1]:
    st.subheader("📊 Historical Climate Data")
    if df is not None:
        st.line_chart(df.set_index("ds"))
        st.dataframe(df.describe())
    else:
        st.warning("📂 Upload data in the sidebar.")

# ----------------------------
# Tab 3: Monthly Forecast
# ----------------------------
with tabs[2]:
    st.subheader("🗓️ Monthly Forecast")
    if df is not None:
        months = st.slider("Select number of months to forecast", 1, 24, 6)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=months * 30)
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.plotly_chart(fig1)
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    else:
        st.warning("📂 Upload data to generate forecast.")

# ----------------------------
# Tab 4: Yearly Forecast
# ----------------------------
with tabs[3]:
    st.subheader("📅 Yearly Forecast")
    if df is not None:
        years = st.slider("Select number of years to forecast", 1, 10, 3)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=years * 365)
        forecast = model.predict(future)

        fig2 = model.plot(forecast)
        st.plotly_chart(fig2)
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    else:
        st.warning("📂 Upload data to generate forecast.")

# ----------------------------
# Tab 5: Climate Summary
# ----------------------------
with tabs[4]:
    st.subheader("📌 Climate Summary")
    if df is not None:
        st.markdown("### 📈 Key Statistics")
        stats = df['y'].describe().rename({
            "count": "Data Points",
            "mean": "Mean Temp (°C)",
            "std": "Standard Deviation",
            "min": "Min Temp (°C)",
            "25%": "25th Percentile",
            "50%": "Median Temp (°C)",
            "75%": "75th Percentile",
            "max": "Max Temp (°C)",
        })
        st.table(stats)

        st.markdown("### 🌡️ Temperature Distribution")
        fig_hist = px.histogram(df, x="y", nbins=30, title="Temperature Distribution")
        st.plotly_chart(fig_hist)

        st.markdown("### 📆 Monthly Averages")
        df['Month'] = df['ds'].dt.to_period("M").astype(str)
        monthly_avg = df.groupby("Month")['y'].mean().reset_index()
        fig_month = px.line(monthly_avg, x="Month", y="y", title="Monthly Average Temperature")
        st.plotly_chart(fig_month)
    else:
        st.warning("📂 Upload data to view climate summary.")
