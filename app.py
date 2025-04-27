import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import timedelta

st.set_page_config(page_title="🌦️ Climate Forecast & Analysis", layout="wide")

st.title("🌦️ Climate Forecast & Analysis Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Live Weather", 
    "📊 Climate Dataset", 
    "🗖️ Predictions", 
    "📊 Data Insights", 
    "🔮 Hybrid Forecast"
])

# Shared df
df = None

# TAB 1: Live Weather
with tab1:
    st.subheader("🌍 Live Weather")
    city = st.text_input("Enter city for current weather:", "New York")
    if st.button("Get Live Weather"):
        API_KEY = "e12e93484a0645f2802141629250803"
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            st.metric("🌡️ Temperature", f"{data['current']['temp_c']} °C")
            st.write(f"**Condition**: {data['current']['condition']['text']}")
            st.write(f"**Humidity**: {data['current']['humidity']}%")
            st.write(f"**Wind**: {data['current']['wind_kph']} kph")
        else:
            st.error("Failed to fetch live weather data.")

# TAB 2: Upload and Clean Dataset
with tab2:
    st.header("📊 Upload Climate Dataset")
    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("🗓️ Cleaning 'Date.Full' column...")
            invalid_dates = []
            parsed_dates = []
            for i, val in enumerate(df["Date.Full"]):
                try:
                    parsed_dates.append(pd.to_datetime(val))
                except:
                    invalid_dates.append((i, val))
                    parsed_dates.append(pd.NaT)
            df["Date"] = parsed_dates
            if invalid_dates:
                st.warning("Some rows had invalid date formats and were set to NaT:")
                st.code("\n".join([f"Row {i}: '{val}'" for i, val in invalid_dates[:5]]))
            else:
                st.success("All dates parsed successfully!")
            df.dropna(subset=["Date"], inplace=True)
            st.success("Dataset loaded and cleaned!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

# TAB 3: Historical Predictions
with tab3:
    st.header("🗖️ Predict Temperature from Today")
    if df is not None:
        if df.empty or 'Data.Temperature.Avg Temp' not in df.columns:
            st.error("Dataset must have valid 'Data.Temperature.Avg Temp' values.")
        else:
            df = df.dropna(subset=['Date', 'Data.Temperature.Avg Temp']).sort_values("Date")
            if len(df) < 2:
                st.warning("Not enough data to calculate trends.")
            else:
                df['Temp_Change'] = df['Data.Temperature.Avg Temp'].diff()
                avg_daily_change = df['Temp_Change'].mean()
                today_temp = df['Data.Temperature.Avg Temp'].iloc[-1]
                pred_tomorrow = today_temp + avg_daily_change
                pred_next_week = today_temp + (avg_daily_change * 7)
                st.metric("Today's Temp", f"{today_temp:.2f} °C")
                st.markdown(f"**Tomorrow**: `{pred_tomorrow:.2f} °C`")
                st.markdown(f"**Next Week**: `{pred_next_week:.2f} °C`")
                forecast_df = pd.DataFrame({
                    "Date": [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 8)],
                    "Predicted Avg Temp (°C)": [today_temp + (avg_daily_change * i) for i in range(1, 8)]
                })
                st.markdown("### 🔮 7-Day Forecast")
                st.dataframe(forecast_df)
    else:
        st.warning("Upload dataset first.")

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
            if 'Data.Precipitation' in df.columns:
                fig, ax = plt.subplots()
                df.plot(x="Date", y="Data.Precipitation", ax=ax, color="green")
                ax.set_ylabel("Precipitation")
                ax.set_title("Precipitation Over Time")
                st.pyplot(fig)
            else:
                st.warning("No 'Data.Precipitation' column found.")
    else:
        st.warning("Upload dataset to view insights.")

# TAB 5: Hybrid Forecast
with tab5:
    st.header("🔮 Hybrid Forecast (Live + Historical)")
    city = st.text_input("City for forecast (7 days)", "New York")
    if st.button("Get Hybrid Forecast"):
        forecast_data = None
        try:
            API_KEY = "e12e93484a0645f2802141629250803"
            url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=7&aqi=no&alerts=no"
            res = requests.get(url)
            forecast_data = res.json()['forecast']['forecastday']
            forecast_df = pd.DataFrame([
                {"Date": day['date'], "Forecast Avg Temp (°C)": day['day']['avgtemp_c']} for day in forecast_data
            ])
            if df is not None:
                df = df.dropna(subset=['Date', 'Data.Temperature.Avg Temp'])
                df['Temp_Change'] = df['Data.Temperature.Avg Temp'].diff()
                avg_daily_change = df['Temp_Change'].mean()
                last_temp = df['Data.Temperature.Avg Temp'].iloc[-1]
                historical_pred = pd.DataFrame({
                    "Date": [(df['Date'].iloc[-1] + timedelta(days=i)).date() for i in range(1, 8)],
                    "Historical Predicted Temp (°C)": [last_temp + avg_daily_change * i for i in range(1, 8)]
                })
                combined = pd.merge(forecast_df, historical_pred, on="Date", how="outer")
                st.dataframe(combined)
                fig, ax = plt.subplots()
                if 'Forecast Avg Temp (°C)' in combined:
                    ax.plot(combined['Date'], combined['Forecast Avg Temp (°C)'], label='Live Forecast', marker='o')
                if 'Historical Predicted Temp (°C)' in combined:
                    ax.plot(combined['Date'], combined['Historical Predicted Temp (°C)'], label='Historical Trend', marker='x')
                ax.set_title(f"Hybrid Forecast for {city}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Temperature (°C)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.dataframe(forecast_df)
        except Exception as e:
            st.error(f"Error fetching forecast: {e}")
