import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

# Load fixed dataset
DATA_PATH = "C:/Users/arink/Documents/climate.csv"
df = pd.read_csv(DATA_PATH)

# Clean & prepare date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df.sort_values('Date', inplace=True)

# Set page config
st.set_page_config(page_title="🌦️ Climate Dashboard", layout="wide")
st.title("🌎 Advanced Climate Forecasting Dashboard")

# Tabs setup
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "☀️ Live Weather", "📊 Forecasting", "📈 Historical Trends",
    "📉 Seasonal Analysis", "⚠️ Anomaly Detection", "🌍 Geo Visualization"
])

# ================= LIVE WEATHER TAB =================
with tab1:
    st.header("☀️ Live Weather")
    city = st.text_input("Enter city name", "Mohali")
    API_KEY = "e12e93484a0645f2802141629250803"
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"

    try:
        res = requests.get(url)
        data = res.json()
        if "current" in data:
            st.metric("Temperature (°C)", data["current"]["temp_c"])
            st.metric("Humidity (%)", data["current"]["humidity"])
            st.metric("Condition", data["current"]["condition"]["text"])
            st.image(f"https:{data['current']['condition']['icon']}", width=64)
        else:
            st.warning("City not found or API error.")
    except Exception:
        st.error("Failed to retrieve live weather data.")

# ================= FORECASTING TAB =================
with tab2:
    st.header("📊 Forecast Future Climate Data")

    metric = st.selectbox("Select metric to forecast", ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Humidity'])

    # Prepare data for Prophet
    forecast_df = df[['Date', metric]].rename(columns={"Date": "ds", metric: "y"})
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=60)  # forecast 60 days ahead
    forecast = model.predict(future)

    st.subheader("Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    # Key date forecasts
    st.subheader("📆 Key Date Predictions")
    target_dates = {
        "Tomorrow": datetime(2025, 4, 27),
        "Next Month": datetime(2025, 5, 26)
    }

    for label, date in target_dates.items():
        row = forecast[forecast['ds'] == date]
        if not row.empty:
            temp = row['yhat'].values[0]
            st.markdown(f"📍 **{label} ({date.strftime('%d %b %Y')}):** {temp:.2f} {metric}")
        else:
            st.markdown(f"📍 **{label} ({date.strftime('%d %b %Y')}):** No data")

# ================= HISTORICAL TRENDS TAB =================
with tab3:
    st.header("📈 Historical Trends")
    metric = st.selectbox("Select metric", ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Humidity'], key='trend')
    fig2 = px.line(df, x='Date', y=metric, title=f"Historical {metric} Trend")
    st.plotly_chart(fig2, use_container_width=True)

# ================= SEASONAL DECOMPOSITION TAB =================
with tab4:
    st.header("📉 Seasonal Decomposition")
    metric = st.selectbox("Select metric", ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Humidity'], key='decomp')
    try:
        result = seasonal_decompose(df[metric].dropna(), period=12, model='additive', extrapolate_trend='freq')
        st.subheader("Trend")
        st.line_chart(result.trend)
        st.subheader("Seasonality")
        st.line_chart(result.seasonal)
        st.subheader("Residuals")
        st.line_chart(result.resid)
    except Exception as e:
        st.error("Not enough data points or failed to decompose.")

# ================= ANOMALY DETECTION TAB =================
with tab5:
    st.header("⚠️ Anomaly Detection")
    metric = st.selectbox("Select metric to analyze", ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Humidity'], key='anomaly')
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    anomalies = df[(df[metric] < lower) | (df[metric] > upper)]

    st.warning(f"Detected {len(anomalies)} anomalies.")
    st.dataframe(anomalies[['Date', metric]])
    fig3 = px.scatter(df, x='Date', y=metric, title="Anomalies Highlighted")
    fig3.add_scatter(x=anomalies['Date'], y=anomalies[metric], mode='markers', marker=dict(color='red', size=8), name="Anomaly")
    st.plotly_chart(fig3, use_container_width=True)

# ================= GEO VISUALIZATION TAB =================
with tab6:
    st.header("🌍 Climate Geo Visualization")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        metric = st.selectbox("Select metric to map", ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Humidity'], key='geo')
        fig4 = px.scatter_geo(df,
                              lat='Latitude', lon='Longitude',
                              color=metric, size=metric,
                              projection="natural earth",
                              title=f"{metric} Distribution on Globe")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Latitude and Longitude columns not found in the dataset.")
