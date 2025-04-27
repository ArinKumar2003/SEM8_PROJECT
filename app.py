import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to fetch WeatherAPI forecast
def fetch_forecast(city, days=7):
    API_KEY = "e12e93484a0645f2802141629250803"
    url = f"http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": API_KEY,
        "q": city,
        "days": days,
        "aqi": "no",
        "alerts": "no"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Streamlit app setup
st.set_page_config(page_title="🌦️ Hybrid Weather Forecast", layout="wide")
st.title("🌦️ Hybrid Weather Forecast & Analysis Dashboard")

# Tabs for various sections
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Live Weather", "📊 Climate Dataset", "📆 Predictions", "📊 Data Insights"])

# Shared df
df = None

# TAB 1: Live Weather
with tab1:
    st.subheader("🌍 Live Weather Forecast")
    city = st.text_input("Enter city for forecast:", "Mohali")
    if city:
        forecast_data = fetch_forecast(city)
        if forecast_data:
            today = forecast_data['forecast']['forecastday'][0]['day']
            st.metric("🌡️ Today's Avg Temp", f"{today['avgtemp_c']} °C")
            st.metric("🌧️ Total Precipitation", f"{today['totalprecip_mm']} mm")
            st.metric("🌬️ Max Wind", f"{today['maxwind_kph']} kph")
        else:
            st.error("Failed to fetch weather data.")

# TAB 2: Upload and Clean Dataset
with tab2:
    st.header("📊 Upload Climate Dataset")

    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Parse 'Date.Full'
            st.subheader("📅 Cleaning 'Date.Full' column...")
            invalid_dates = []
            parsed_dates = []

            for i, val in enumerate(df["Date.Full"]):
                try:
                    parsed_dates.append(pd.to_datetime(val))
                except Exception as e:
                    invalid_dates.append((i, val))
                    parsed_dates.append(pd.NaT)

            df["Date"] = parsed_dates

            if invalid_dates:
                st.warning("⚠️ Some rows had invalid date formats and were set to NaT. Here are a few examples:")
                st.code("\n".join([f"Row {i}: '{val}'" for i, val in invalid_dates[:5]]))
            else:
                st.success("✅ All dates parsed successfully!")

            df.dropna(subset=["Date"], inplace=True)

            st.success("✅ Dataset successfully loaded and cleaned!")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

# TAB 3: Predictions (Hybrid Model)
with tab3:
    st.header("📆 Predict Temperature from Today")

    if df is not None:
        if df.empty:
            st.error("❌ DataFrame is empty after cleaning. Cannot generate predictions.")
        elif 'Data.Temperature.Avg Temp' not in df.columns:
            st.error("❌ 'Data.Temperature.Avg Temp' column not found in the dataset.")
        else:
            df = df.dropna(subset=['Date', 'Data.Temperature.Avg Temp'])
            df = df.sort_values("Date")

            # Linear Regression Model for Temperature Prediction
            df['DayOfYear'] = df['Date'].dt.dayofyear  # Use day of year as a feature
            X = df['DayOfYear'].values.reshape(-1, 1)  # Feature: Day of Year
            y = df['Data.Temperature.Avg Temp'].values  # Target: Temperature

            # Train a linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict future temperatures (e.g., next 7 days)
            last_date = df['Date'].iloc[-1]
            predicted_temps = model.predict(np.array([last_date.dayofyear + i for i in range(1, 8)]).reshape(-1, 1))

            # Create DataFrame with predictions
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Temp (°C)': predicted_temps
            })

            st.markdown("### 🔮 7-Day Hybrid Temperature Forecast")
            st.dataframe(forecast_df)

            # Fetch live forecast data
            city = st.text_input("City for live forecast", "Mohali")
            forecast_data = fetch_forecast(city)

            if forecast_data:
                forecast_days = forecast_data['forecast']['forecastday']
                forecast_live_df = pd.DataFrame([
                    {
                        "Date": pd.to_datetime(day['date']),
                        "Forecast Avg Temp (°C)": day['day']['avgtemp_c']
                    } for day in forecast_days
                ])

                combined = pd.merge(forecast_live_df, forecast_df, on="Date", how="outer").sort_values("Date")
                st.markdown("### 🔮 Combined 7-Day Forecast (Live + Prediction)")
                st.dataframe(combined)

                # Plotting
                fig, ax = plt.subplots()
                ax.plot(combined['Date'], combined['Forecast Avg Temp (°C)'], label='Live Forecast', marker='o')
                ax.plot(combined['Date'], combined['Predicted Temp (°C)'], label='Predicted Temp', marker='x')
                ax.set_xlabel('Date')
                ax.set_ylabel('Temperature (°C)')
                ax.set_title(f'Hybrid Temperature Forecast for {city}')
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Could not fetch live forecast. Showing only historical prediction.")
                st.dataframe(forecast_df)
    else:
        st.warning("📂 Please upload the dataset first in the previous tab.")

# TAB 4: Insights (Visualizations)
with tab4:
    st.header("📊 Data Insights")
    if df is not None:
        with st.expander("📈 Avg Temperature Over Time"):
            fig, ax = plt.subplots()
            df.plot(x="Date", y="Data.Temperature.Avg Temp", ax=ax)
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("Average Temperature Over Time")
            st.pyplot(fig)

        with st.expander("🌧️ Precipitation Over Time"):
            fig, ax = plt.subplots()
            df.plot(x="Date", y="Data.Precipitation", ax=ax, color="green")
            ax.set_ylabel("Precipitation (mm)")
            ax.set_title("Precipitation Over Time")
            st.pyplot(fig)
    else:
        st.warning("📂 Please upload the dataset to view insights.")
