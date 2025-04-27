import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import requests

# Page config
st.set_page_config(page_title="🌦️ Climate Forecast & Analysis", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Live Weather", "📊 Climate Dataset", "📆 Predictions", "📊 Data Insights"])

# Shared dataframe
df = None

# Weather API Function
def fetch_live_weather(city_name):
    api_key = "e12e93484a0645f2802141629250803"
    base_url = "http://api.weatherapi.com/v1"
    endpoint = f"{base_url}/current.json"
    params = {
        "key": api_key,
        "q": city_name
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "location": data["location"]["name"],
            "region": data["location"]["region"],
            "country": data["location"]["country"],
            "temperature": data["current"]["temp_c"],
            "condition": data["current"]["condition"]["text"]
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# TAB 1: Live Weather
with tab1:
    st.subheader("🌍 Live Weather")
    city = st.text_input("Enter city name:", "Mohali")
    if city:
        weather = fetch_live_weather(city)
        if weather:
            st.metric("📍 Location", f"{weather['location']}, {weather['region']}")
            st.metric("🌡️ Temperature", f"{weather['temperature']} °C")
            st.write(f"**Condition:** {weather['condition']}")
        else:
            st.error("❌ Unable to fetch live weather. Check the city name or try again later.")

# TAB 2: Upload and Clean Dataset
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

# TAB 3: Predictions
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

            if len(df) < 2:
                st.warning("⚠️ Not enough data to calculate trends. Need at least 2 rows.")
            else:
                df['Temp_Change'] = df['Data.Temperature.Avg Temp'].diff()
                avg_daily_change = df['Temp_Change'].mean()

                today_temp = df['Data.Temperature.Avg Temp'].iloc[-1]
                pred_tomorrow = today_temp + avg_daily_change
                pred_next_week = today_temp + (avg_daily_change * 7)

                st.metric("📌 Last Recorded Temp", f"{today_temp:.2f} °C")
                st.markdown(f"📍 **Tomorrow (2025)**: `{pred_tomorrow:.2f} °C`")
                st.markdown(f"📍 **Next Week (2025)**: `{pred_next_week:.2f} °C`")

                forecast_df = pd.DataFrame({
                    "Date": [pd.to_datetime("2025-04-28") + timedelta(days=i) for i in range(7)],
                    "Predicted Avg Temp (°C)": [today_temp + (avg_daily_change * (i + 1)) for i in range(7)]
                })

                st.markdown("### 🔮 7-Day Forecast (from April 28, 2025)")
                st.dataframe(forecast_df)
    else:
        st.warning("📂 Please upload the dataset first in the previous tab.")

# TAB 4: Insights
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
