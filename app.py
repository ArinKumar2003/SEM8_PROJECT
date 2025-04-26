import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------ CONFIG ------------------------
st.set_page_config(page_title="🌦️ Climate Dashboard", layout="centered")
st.title("🌎 Advanced Climate Forecasting Dashboard")

# ------------------------ UPLOAD ------------------------
data_file = st.sidebar.file_uploader("📤 Upload your climate CSV file", type=["csv"])
forecast_days = st.sidebar.slider("📅 Forecast Period (days)", 7, 365, 30)

# ------------------------ LIVE WEATHER FIRST TAB ------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "☀️ Live Weather", 
    "📊 Forecasting", 
    "📈 Historical Trends", 
    "📉 Seasonal Analysis", 
    "⚠️ Anomaly Detection", 
    "🌍 Geo Visualization"
])

with tab1:
    st.header("☀️ Live Weather")

    city = st.text_input("Enter city name", "Mohali")
    API_KEY = "e12e93484a0645f2802141629250803"
    WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

    if city:
        try:
            response = requests.get(f"{WEATHER_API_URL}?key={API_KEY}&q={city}")
            data = response.json()

            if "current" in data:
                st.metric("Temperature (°C)", data['current']['temp_c'])
                st.metric("Humidity (%)", data['current']['humidity'])
                st.metric("Condition", data['current']['condition']['text'])
                st.image(f"https:{data['current']['condition']['icon']}", width=64)
            else:
                st.error("Weather data not found.")
        except Exception as e:
            st.error("API request failed.")

# ------------------------ MAIN DASHBOARD ------------------------
if data_file is not None:
    df = pd.read_csv(data_file)

    # Fix and parse date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)

    # Rename columns for easier reference
    df.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)

    climate_metrics = ['Temperature', 'CO2_Emissions', 'Sea_Level_Rise', 'Precipitation', 'Humidity', 'Wind_Speed']

    # ---------------- FORECASTING ----------------
    with tab2:
        st.header("📊 Forecast Future Climate Data")

        metric = st.selectbox("Select metric to forecast", climate_metrics)
        df_forecast = df[['Date', metric]].rename(columns={"Date": "ds", metric: "y"})

        model = Prophet()
        model.fit(df_forecast)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        st.subheader("📈 Forecast Plot")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        # Tomorrow and next month predictions
        tomorrow = datetime.now() + timedelta(days=1)
        next_month = datetime.now() + timedelta(days=30)

        def get_prediction(date):
            pred_row = forecast[forecast['ds'] == date.strftime('%Y-%m-%d')]
            if not pred_row.empty:
                return round(pred_row['yhat'].values[0], 2)
            return "No data"

        st.subheader("📆 Key Date Predictions")
        st.write(f"📍 **Tomorrow ({tomorrow.strftime('%d %b %Y')}):** {get_prediction(tomorrow)} {metric}")
        st.write(f"📍 **Next Month ({next_month.strftime('%d %b %Y')}):** {get_prediction(next_month)} {metric}")

    # ---------------- HISTORICAL TRENDS ----------------
    with tab3:
        st.header("📈 Historical Trends")
        metric = st.selectbox("Select metric", climate_metrics, key='trend')
        fig2 = px.line(df, x='Date', y=metric, title=f"Historical {metric} Trends")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- SEASONAL ANALYSIS ----------------
    with tab4:
        st.header("📉 Seasonal Decomposition")
        metric = st.selectbox("Select metric for decomposition", climate_metrics, key='decomp')
        try:
            result = seasonal_decompose(df[metric].dropna(), period=12, model='additive')
            st.subheader("📊 Trend")
            st.line_chart(result.trend)
            st.subheader("📊 Seasonal")
            st.line_chart(result.seasonal)
            st.subheader("📊 Residual")
            st.line_chart(result.resid)
        except Exception as e:
            st.warning("Decomposition failed. Try a different metric.")

    # ---------------- ANOMALY DETECTION ----------------
    with tab5:
        st.header("⚠️ Anomaly Detection")
        metric = st.selectbox("Select metric to analyze anomalies", climate_metrics, key='anomaly')
        q1 = df[metric].quantile(0.25)
        q3 = df[metric].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        anomalies = df[(df[metric] < lower) | (df[metric] > upper)]

        st.warning(f"Detected {len(anomalies)} anomalies in {metric}.")
        st.dataframe(anomalies[['Date', metric]])
        fig3 = px.scatter(df, x='Date', y=metric, title="Anomalies Highlighted")
        fig3.add_scatter(x=anomalies['Date'], y=anomalies[metric],
                         mode='markers', marker=dict(color='red', size=8), name="Anomaly")
        st.plotly_chart(fig3, use_container_width=True)

    # ---------------- GEO VISUALIZATION ----------------
    with tab6:
        st.header("🌍 Climate Geo Map (Experimental)")
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            metric = st.selectbox("Select metric to map", climate_metrics, key='geo')
            fig4 = px.scatter_geo(df,
                                  lat='Latitude', lon='Longitude',
                                  color=metric,
                                  size=metric,
                                  projection="natural earth",
                                  title=f"Global {metric} Distribution")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Latitude and Longitude columns not found in uploaded data.")
else:
    st.info("📁 Please upload a climate dataset CSV to unlock full features.")
