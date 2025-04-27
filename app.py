import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# 🌐 WEATHER API CONFIG
API_KEY = "e12e93484a0645f2802141629250803"
WEATHER_URL = "http://api.weatherapi.com/v1/current.json"

# 🔧 Streamlit Page Config
st.set_page_config(page_title="🌦️ Climate Forecast & Analysis", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["🌍 Live Weather", "📊 Climate Dataset", "📆 Predictions", "📊 Data Insights"])

# Shared dataframe
df = None

# 🔹 TAB 1: Live Weather
with tab1:
    st.subheader("🌍 Live Weather")
    city = st.text_input("Enter a city name (e.g., New York, Delhi, London)", "New York")

    if st.button("Get Live Weather"):
        params = {"key": API_KEY, "q": city}
        response = requests.get(WEATHER_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            current = data["current"]

            st.success(f"✅ Weather in {data['location']['name']}, {data['location']['country']}")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("🌡️ Temperature (°C)", current["temp_c"])
                st.metric("💧 Humidity (%)", current["humidity"])
                st.metric("💨 Wind (kph)", current["wind_kph"])

            with col2:
                st.image("https:" + current["condition"]["icon"], caption=current["condition"]["text"])
                st.write("Last updated:", current["last_updated"])

        else:
            st.error("❌ Failed to fetch weather data. Please check the city name or API key.")

# 🔹 TAB 2: Upload and Clean Dataset
with tab2:
    st.header("📊 Upload Climate Dataset")
    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

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

# 🔹 TAB 3: Predictions
with tab3:
    st.header("📆 Predict Temperature for Today and Next Week (2025)")

    if df is not None:
        if df.empty:
            st.error("❌ DataFrame is empty after cleaning. Cannot generate predictions.")
        elif 'Data.Temperature.Avg Temp' not in df.columns:
            st.error("❌ 'Data.Temperature.Avg Temp' column not found in the dataset.")
        else:
            df = df.dropna(subset=['Date', 'Data.Temperature.Avg Temp'])
            df = df.sort_values("Date")

            if len(df) < 2:
                st.warning("⚠️ Not enough data to calculate trends. Need at least 2 rows.")
            else:
                # Calculate average daily change
                df['Temp_Change'] = df['Data.Temperature.Avg Temp'].diff()
                avg_daily_change = df['Temp_Change'].mean()

                # Predict from last known temp
                last_date = df["Date"].iloc[-1]
                today_2025 = datetime(2025, 4, 27)
                days_since_last = (today_2025 - last_date).days

                today_temp = df["Data.Temperature.Avg Temp"].iloc[-1]
                pred_today = today_temp + (avg_daily_change * days_since_last)
                pred_week = pred_today + (avg_daily_change * 7)

                st.metric("📍 Predicted Temperature Today (April 27, 2025)", f"{pred_today:.2f} °C")
                st.markdown(f"📍 **Next Week (May 4, 2025)**: `{pred_week:.2f} °C`")

                forecast_df = pd.DataFrame({
                    "Date": [today_2025 + timedelta(days=i) for i in range(1, 8)],
                    "Predicted Avg Temp (°C)": [pred_today + (avg_daily_change * i) for i in range(1, 8)]
                })
                st.markdown("### 🔮 7-Day Forecast")
                st.dataframe(forecast_df)
    else:
        st.warning("📂 Please upload the dataset in the previous tab first.")

# 🔹 TAB 4: Insights
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
            ax.set_ylabel("Precipitation")
            ax.set_title("Precipitation Over Time")
            st.pyplot(fig)
    else:
        st.warning("📂 Please upload the dataset to view insights.")
