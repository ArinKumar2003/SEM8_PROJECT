import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="🌍 Climate Forecast Dashboard", layout="centered")

# --- Sidebar: Upload CSV ---
st.sidebar.title("📁 Upload Your Climate Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

# --- Global Vars ---
API_KEY = "e12e93484a0645f2802141629250803"
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# --- Load Data Function ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
    return df

# --- App Title ---
st.title("🌿 Climate Forecasting & Live Weather Insights")

# --- Data Load Check ---
if uploaded_file is not None:
    df = load_data(uploaded_file)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "☀️ Live Weather", "📊 Forecasting", "📈 Historical Trends",
        "📉 Seasonal Decomposition", "🧮 Summary Stats", "🔍 Correlation Map"
    ])

    # --- Tab 1: Live Weather ---
    with tab1:
        st.header("☀️ Real-Time Weather Info")
        city = st.text_input("Enter city name", "Mohali")

        if city:
            params = {"key": API_KEY, "q": city}
            res = requests.get(WEATHER_API_URL, params=params)
            weather = res.json()

            if "current" in weather:
                st.subheader(f"Weather in {city}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Temperature (°C)", weather["current"]["temp_c"])
                col2.metric("Humidity (%)", weather["current"]["humidity"])
                col3.metric("Condition", weather["current"]["condition"]["text"])
                st.image("https:" + weather["current"]["condition"]["icon"], width=100)
            else:
                st.error("Unable to fetch weather data. Check API or city name.")

    # --- Tab 2: Forecasting ---
    with tab2:
        st.header("📊 Climate Metric Forecasting")
        metric = st.selectbox("Select metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'])

        data = df[['Date', metric]].rename(columns={"Date": "ds", metric: "y"})

        model = Prophet()
        model.fit(data)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.subheader("📈 Forecast Plot")
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)

        # Show next date + next month
        st.subheader("🔮 Forecast Summary")
        next_day = forecast.iloc[-30]
        next_month = forecast.iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric("Next Day Prediction", round(next_day['yhat'], 2))
        col2.metric("Next Month Prediction", round(next_month['yhat'], 2))

        st.download_button("⬇️ Download Forecast CSV", data=forecast.to_csv(index=False), file_name="forecast.csv")

    # --- Tab 3: Historical Trends ---
    with tab3:
        st.header("📈 Historical Climate Trends")
        metric = st.selectbox("Choose metric to view trend", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key="hist")
        fig2 = px.line(df, x='Date', y=metric, title=f"{metric} over Time")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 4: Seasonal Decomposition ---
    with tab4:
        st.header("📉 Seasonal Pattern Decomposition")
        metric = st.selectbox("Select metric for decomposition", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key="decomp")
        result = seasonal_decompose(df[metric], period=12, model='additive')
        st.line_chart(result.trend.dropna(), use_container_width=True)
        st.line_chart(result.seasonal.dropna(), use_container_width=True)
        st.line_chart(result.resid.dropna(), use_container_width=True)

    # --- Tab 5: Summary Stats ---
    with tab5:
        st.header("🧮 Dataset Summary")
        st.dataframe(df.describe().round(2))

    # --- Tab 6: Correlation Heatmap ---
    with tab6:
        st.header("🔍 Correlation Between Metrics")
        numeric_df = df[['CO2', 'Humidity', 'SeaLevel', 'Temperature']]
        corr = numeric_df.corr()

        fig_corr, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)

else:
    st.warning("📌 Please upload your dataset from the left sidebar to get started.")
