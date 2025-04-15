import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from io import StringIO
from statsmodels.tsa.seasonal import seasonal_decompose

# Page configuration
st.set_page_config(page_title="Climate Forecast App", layout="centered")

st.title("🌍 Climate Forecasting & Live Weather Dashboard")

# Sidebar: Upload dataset
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
    df = df.sort_values('Date')

    # Weather API config
    API_KEY = st.secrets["weatherapi"]["api_key"]
    WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

    # Tabs setup
    tab1, tab2, tab3, tab4 = st.tabs([
        "☀️ Live Weather",
        "📊 Forecasting",
        "📈 Historical Trends",
        "🔍 Raw Data"
    ])

    # --- TAB 1: LIVE WEATHER ---
    with tab1:
        st.header("☀️ Live Weather Data")
        city = st.text_input("Enter city name", "Mohali")

        def get_weather(city):
            url = f"{WEATHER_API_URL}?key={API_KEY}&q={city}"
            response = requests.get(url)
            return response.json()

        if city:
            weather = get_weather(city)
            if weather.get("current"):
                st.metric("Temperature (°C)", weather['current']['temp_c'])
                st.metric("Humidity (%)", weather['current']['humidity'])
                st.metric("Wind (kph)", weather['current']['wind_kph'])
                st.write(f"**Condition**: {weather['current']['condition']['text']}")
            else:
                st.error("City not found or API issue.")

    # --- TAB 2: FORECASTING ---
    with tab2:
        st.header("📊 Forecast Future Climate Data")
        metric = st.selectbox("Select metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'])

        if metric in df.columns:
            data = df[['Date', metric]].dropna()
            data = data.rename(columns={"Date": "ds", metric: "y"})

            model = Prophet()
            model.fit(data)

            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            # Plot forecast
            st.subheader("Forecast Plot")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            # Predictions
            last_date = data['ds'].max()
            next_day = last_date + pd.Timedelta(days=1)
            next_month = last_date + pd.DateOffset(months=1)

            next_day_pred = forecast.loc[forecast['ds'] == next_day]
            next_month_pred = forecast.loc[forecast['ds'] == next_month]

            if next_day_pred.empty:
                next_day_pred = forecast[forecast['ds'] > next_day].head(1)
            if next_month_pred.empty:
                next_month_pred = forecast[forecast['ds'] > next_month].head(1)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label=f"🗓️ Next Day ({next_day_pred['ds'].dt.date.values[0]})",
                    value=f"{next_day_pred['yhat'].values[0]:.2f} {metric}"
                )
            with col2:
                st.metric(
                    label=f"📅 Next Month ({next_month_pred['ds'].dt.date.values[0]})",
                    value=f"{next_month_pred['yhat'].values[0]:.2f} {metric}"
                )

            # Table and download
            st.subheader("Forecast Data Preview")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Forecast", data=csv, file_name='forecast.csv', mime='text/csv')
        else:
            st.warning("Selected metric is not in dataset.")

    # --- TAB 3: HISTORICAL TRENDS ---
    with tab3:
        st.header("📈 Historical Data Trends")
        metric = st.selectbox("Select metric for historical view", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='hist')

        fig2 = px.line(df, x='Date', y=metric, title=f"Historical Trend of {metric}")
        st.plotly_chart(fig2)

        st.subheader("📉 Seasonal Decomposition")
        result = seasonal_decompose(df[metric], period=12, model='additive', extrapolate_trend='freq')
        st.line_chart(result.trend.rename("Trend"))
        st.line_chart(result.seasonal.rename("Seasonality"))
        st.line_chart(result.resid.rename("Residual"))

    # --- TAB 4: RAW DATA ---
    with tab4:
        st.header("🔍 Uploaded Dataset")
        st.dataframe(df.head(100))
        st.download_button("⬇️ Download Original Data", data=uploaded_file, file_name="climate_data.csv")
else:
    st.warning("Please upload a climate dataset (CSV format) from the sidebar to begin.")
