import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(page_title="Weather App", layout="wide")

st.title("🌤️ Weather Forecast Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Live Weather", "Forecast", "About"])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌦️ Live Weather Report")
    city = st.text_input("Enter a city name", "New York")
    api_key = st.secrets["weatherstack"]["api_key"]  # Store this in .streamlit/secrets.toml

    if st.button("Get Weather"):
        url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
        try:
            res = requests.get(url)
            res.raise_for_status()
            weather_data = res.json()

            if 'current' not in weather_data:
                raise Exception("Invalid response")

            temp = weather_data['current']['temperature']
            desc = weather_data['current']['weather_descriptions'][0]
            humidity = weather_data['current']['humidity']
            wind = weather_data['current']['wind_speed']

            st.metric(label="🌡️ Temperature (°C)", value=temp)
            st.metric(label="🌤️ Description", value=desc)
            st.metric(label="💧 Humidity", value=f"{humidity}%")
            st.metric(label="🌬️ Wind Speed", value=f"{wind} km/h")
        except:
            st.error("Could not retrieve weather data. Check your API key or city name.")

# ---- TAB 2: FORECASTING ----
with tab2:
    st.subheader("📈 Forecasting Weather Data")

    uploaded_file = st.file_uploader("Upload your CSV file (must contain 'ds' and 'y' columns)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure 'ds' and 'y' columns exist
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("CSV must contain 'ds' and 'y' columns (date and value).")
        else:
            df['ds'] = pd.to_datetime(df['ds'])

            st.write("📊 Uploaded Data", df.tail())

            period = st.slider("Select forecast period (days)", min_value=1, max_value=365, value=30)

            # Prophet model
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)

            st.write("📈 Forecast Data", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            fig2 = px.line(forecast, x='ds', y='yhat', title='Forecasted Values Over Time')
            st.plotly_chart(fig2)

# ---- TAB 3: ABOUT ----
with tab3:
    st.subheader("ℹ️ About This App")
    st.markdown("""
        This app lets you:
        - 🌦️ View **real-time weather** using Weatherstack API  
        - 📈 Upload historical weather data and generate **forecast** using Facebook Prophet  
        - 🔍 Built using **Streamlit**, **Plotly**, and **Prophet**

        **Note**: The uploaded CSV must contain:
        - `ds`: Date column  
        - `y`: Value to forecast (e.g., temperature)

        ---
        Created with ❤️ by You!
    """)

