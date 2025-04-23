import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="🌦️ Climate Dashboard", layout="centered")
st.title("🌎 Advanced Climate Forecasting Dashboard")

# Sidebar - Upload CSV
data_file = st.sidebar.file_uploader("📤 Upload your climate CSV file", type=["csv"])
forecast_days = st.sidebar.slider("📅 Forecast Period (days)", min_value=7, max_value=365, value=60)

# Live Weather Tab
st.subheader("☀️ Live Weather")
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
            st.error("Weather data not found for this city.")
    except Exception as e:
        st.error("API request failed.")

# Proceed only if data file is uploaded
if data_file is not None:
    df = pd.read_csv(data_file)

    # Parse 'Date' column
    df['Date'] = pd.to_datetime(df['Date'])

    # Standardize column names
    df.rename(columns={
        'CO2 Emissions': 'CO2',
        'Sea Level Rise': 'SeaLevel',
        'Wind Speed': 'WindSpeed',
        'Temperature': 'Temperature',
        'Humidity': 'Humidity'
    }, inplace=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Forecasting", "📈 Historical Trends", "📉 Seasonal Analysis", "⚠️ Anomaly Detection", "🌍 Geo Visualization"])

    # Forecasting Tab
    with tab1:
        st.header("📊 Forecast Future Climate Data")
        metric = st.selectbox("Select metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'])

        data = df[['Date', metric]].dropna().rename(columns={"Date": "ds", metric: "y"})
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        st.subheader("Forecast Plot")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        # Show predictions for specific dates
        target_dates = pd.to_datetime(["2025-04-24", "2025-05-24"])
        predictions = forecast[forecast['ds'].isin(target_dates)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        st.subheader("📅 Forecast for Specific Future Dates")
        if not predictions.empty:
            st.write(predictions)
        else:
            st.info("Selected dates are outside the forecast range. Try increasing the forecast period.")

    # Historical Trends Tab
    with tab2:
        st.header("📈 Visualize Historical Trends")
        metric = st.selectbox("Select metric", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='trend')
        fig2 = px.line(df, x='Date', y=metric, title=f"Historical {metric} Trends")
        st.plotly_chart(fig2, use_container_width=True)

    # Seasonal Analysis Tab
    with tab3:
        st.header("📉 Seasonal Decomposition")
        metric = st.selectbox("Select metric for decomposition", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='decomp')
        data = df.set_index('Date')[metric].dropna()
        result = seasonal_decompose(data, model='additive', period=12)
        st.line_chart(result.trend.rename("Trend"))
        st.line_chart(result.seasonal.rename("Seasonality"))
        st.line_chart(result.resid.rename("Residuals"))

    # Anomaly Detection Tab
    with tab4:
        st.header("⚠️ Anomaly Detection")
        metric = st.selectbox("Select metric to analyze anomalies", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='anomaly')
        q1 = df[metric].quantile(0.25)
        q3 = df[metric].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        anomalies = df[(df[metric] < lower) | (df[metric] > upper)]

        st.warning(f"Detected {len(anomalies)} anomalies in {metric}.")
        st.dataframe(anomalies[['Date', metric]])
        fig3 = px.scatter(df, x='Date', y=metric, title="Anomalies Highlighted")
        fig3.add_scatter(x=anomalies['Date'], y=anomalies[metric], mode='markers', marker=dict(color='red', size=8), name="Anomaly")
        st.plotly_chart(fig3, use_container_width=True)

    # Geo Visualization Tab
    with tab5:
        st.header("🌍 Climate Geo Map (Experimental)")
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            metric = st.selectbox("Select metric to map", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='geo')
            fig4 = px.scatter_geo(df,
                                  lat='Latitude', lon='Longitude',
                                  color=metric,
                                  size=metric,
                                  projection="natural earth",
                                  title=f"Global {metric} Distribution")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Latitude and Longitude columns not found in the uploaded data.")
else:
    st.info("📁 Please upload a climate dataset CSV to unlock full features.")
