import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
import requests

# Set up the page configuration
st.set_page_config(page_title="Climate Forecast App", layout="wide")

# Title of the app
st.title("🌍 Climate Forecasting and Weather Insights")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your climate dataset (climate_large_data_sorted.csv)", type=["csv"])

# Sidebar for live weather input
st.sidebar.header("Live Weather")
API_KEY = "e12e93484a0645f2802141629250803"  # Replace with your own API key from WeatherAPI
city = st.sidebar.text_input("Enter city name for live weather", "Mohali")

# Function to get live weather data
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?q={city}&key={API_KEY}&aqi=no"
    response = requests.get(url)
    return response.json()

# If a file is uploaded, load and process the data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset successfully loaded!")

    # Check for the necessary columns in the uploaded data
    if "Years" in df.columns and "Month" in df.columns and "Day" in df.columns:
        # Combine Year, Month, Day to create a Date column
        df['Date'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
    else:
        st.error("Uploaded data must contain 'Years', 'Month', and 'Day' columns to create a Date.")

    # Tabs for different features
    option = st.radio("Select Section", ["🌦️ Live Weather", "📊 Forecasting", "📈 Historical Trends"])

    # --- LIVE WEATHER TAB ---
    if option == "🌦️ Live Weather":
        st.header("☀️ Live Weather Data")
        st.write("Get live weather updates from your city!")
        
        if city:
            weather = get_weather(city)
            if "current" in weather:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Temperature (°C)", weather['current']['temp_c'])
                    st.metric("Humidity (%)", weather['current']['humidity'])
                with col2:
                    st.write(f"🌤️ **Condition**: {weather['current']['condition']['text']}")
                    st.write(f"📍 Location: {weather['location']['name']}, {weather['location']['country']}")
            else:
                st.error("City not found or there was an issue with the weather API.")

    # --- FORECASTING TAB ---
    elif option == "📊 Forecasting":
        st.header("📊 Forecast Future Climate Data")
        st.write("Use this section to forecast future climate data trends using Prophet.")

        # Select metric for forecasting
        metric = st.selectbox("Select metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], help="Select the climate metric you want to forecast.")

        # Prepare data for forecasting
        if metric in df.columns:
            data = df[['Date', metric]].rename(columns={"Date": "ds", metric: "y"})

            # Create and fit the Prophet model
            model = Prophet()
            model.fit(data)
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            st.subheader("Forecast Plot")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            st.subheader("Forecast Data")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Download button for forecast data
            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Forecast", data=csv, file_name='forecast.csv', mime='text/csv')
        else:
            st.error(f"'{metric}' not found in uploaded data.")

    # --- HISTORICAL TRENDS TAB ---
    elif option == "📈 Historical Trends":
        st.header("📈 Visualize Historical Trends")
        st.write("Explore historical trends of climate metrics over time.")

        if 'Date' in df.columns:
            # Select metric for historical trend visualization
            metric = st.selectbox("Select metric for trend analysis", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='trend')

            if metric in df.columns:
                # Plot historical trends using Plotly
                fig2 = px.line(df, x='Date', y=metric, title=f"Historical {metric} Trends")
                st.plotly_chart(fig2)

                # Seasonal decomposition of the selected metric
                st.subheader("📉 Seasonal Decomposition")
                result = seasonal_decompose(df[metric], period=12, model='additive')
                
                st.write("Trend Component")
                st.line_chart(result.trend)
                
                st.write("Seasonal Component")
                st.line_chart(result.seasonal)
                
                st.write("Residual Component")
                st.line_chart(result.resid)
            else:
                st.error(f"'{metric}' not found in uploaded data.")
        else:
            st.error("Uploaded data does not contain a valid 'Date' column.")

else:
    st.sidebar.write("Please upload your `climate_large_data_sorted.csv` to get started.")
    st.write("Upload a valid CSV file for analysis.")
