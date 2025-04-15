import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests
from io import StringIO
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Climate Forecast App", layout="wide")

st.title("🌍 Climate Forecasting and Weather Insights")

# File uploader for custom data
uploaded_file = st.file_uploader("Upload your climate data CSV", type=["csv"])

if uploaded_file is not None:
    # Load dataset from the uploaded file
    @st.cache_data
    def load_data():
        df = pd.read_csv(uploaded_file)
        # Ensure correct date parsing (adjust based on your data columns)
        df['Date'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
        return df

    df = load_data()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Forecasting", "📈 Historical Trends", "☀️ Live Weather"])

    # --- TAB 1: Forecasting ---
    with tab1:
        st.header("📊 Forecast Future Climate Data")

        metric = st.selectbox("Select metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'])

        # Prepare data for Prophet model
        data = df[['Date', metric]].rename(columns={"Date": "ds", metric: "y"})

        # Forecast with Prophet
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        st.subheader("Forecast Plot")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Download forecast data
        csv = forecast.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Forecast", data=csv, file_name='forecast.csv', mime='text/csv')

    # --- TAB 2: Historical Trends ---
    with tab2:
        st.header("📈 Visualize Historical Trends")

        metric = st.selectbox("Select metric for trend analysis", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='trend')

        # Plot historical trend using Plotly
        fig2 = px.line(df, x='Date', y=metric, title=f"Historical {metric} Trends", labels={"Date": "Year", metric: metric})
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📉 Seasonal Decomposition")
        result = seasonal_decompose(df[metric], period=12, model='additive')
        
        # Display seasonal decomposition plots
        st.write("**Trend Component**")
        st.line_chart(result.trend)
        
        st.write("**Seasonal Component**")
        st.line_chart(result.seasonal)
        
        st.write("**Residual Component**")
        st.line_chart(result.resid)

    # --- TAB 3: Live Weather ---
    with tab3:
        st.header("☀️ Live Weather Data")

        # Retrieve API key from Streamlit secrets
        API_KEY = st.secrets["weatherapi"]["api_key"]
        
        city = st.text_input("Enter city name", "Mohali")

        # Function to get weather data
        def get_weather(city):
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
                response = requests.get(url)
                response.raise_for_status()  # Will raise an HTTPError if the response code is 4xx or 5xx
                return response.json()
            except requests.exceptions.HTTPError as errh:
                st.error(f"HTTP Error: {errh}")
            except requests.exceptions.ConnectionError as errc:
                st.error(f"Error Connecting: {errc}")
            except requests.exceptions.Timeout as errt:
                st.error(f"Timeout Error: {errt}")
            except requests.exceptions.RequestException as err:
                st.error(f"Something went wrong: {err}")

        # Get weather information for the city
        if city:
            weather = get_weather(city)
            if weather and weather.get("main"):
                st.metric("Temperature (°C)", weather['main']['temp'])
                st.metric("Humidity (%)", weather['main']['humidity'])
                st.write(f"🌤️ **Condition**: {weather['weather'][0]['description'].title()}")
            elif weather:
                st.error("City not found or there was an issue with the API response.")
            else:
                st.error("Failed to retrieve weather data.")
else:
    st.info("Please upload a CSV file to begin.")

