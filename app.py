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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 Live Weather", 
    "📊 Climate Dataset", 
    "📆 Predictions", 
    "📊 Data Insights", 
    "📖 App Overview", 
    "❓ Help/FAQ"
])

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

# TAB 4: Enhanced Insights (Visualizations and Data Analysis)
with tab4:
    st.header("📊 Data Insights")

    if df is not None:
        # Summary Statistics
        st.subheader("📈 Statistical Summary")
        st.write("This section provides the basic statistics for the climate data, such as mean, median, standard deviation.")
        summary_stats = df[['Data.Temperature.Avg Temp', 'Data.Precipitation']].describe().transpose()
        st.dataframe(summary_stats)

        # Trend Analysis: Plotting temperature trend over time
        with st.expander("📈 Temperature Trend Over Time"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(x="Date", y="Data.Temperature.Avg Temp", ax=ax, color='tab:red')
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("Temperature Trend Over Time")
            ax.grid(True)
            st.pyplot(fig)

        # Moving Average: Smoothing temperature data
        with st.expander("📉 7-Day Moving Average of Temperature"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Temp_MA7'] = df['Data.Temperature.Avg Temp'].rolling(window=7).mean()  # 7-day moving average
            df.plot(x="Date", y="Temp_MA7", ax=ax, color='tab:blue')
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("7-Day Moving Average of Temperature")
            ax.grid(True)
            st.pyplot(fig)

        # Precipitation Trend
        with st.expander("🌧️ Precipitation Trend Over Time"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(x="Date", y="Data.Precipitation", ax=ax, color='tab:green')
            ax.set_ylabel("Precipitation (mm)")
            ax.set_title("Precipitation Trend Over Time")
            ax.grid(True)
            st.pyplot(fig)

        # Correlation Analysis: Heatmap to understand relationships between variables
        import seaborn as sns

        with st.expander("🔍 Correlation Analysis"):
            corr_matrix = df[['Data.Temperature.Avg Temp', 'Data.Precipitation']].corr()
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation between Temperature and Precipitation")
            st.pyplot(fig)

        # Anomaly Detection (Outliers)
        with st.expander("🚨 Anomaly Detection (Outliers)"):
            # Calculate z-scores to detect outliers
            from scipy.stats import zscore
            df['Temp_zscore'] = zscore(df['Data.Temperature.Avg Temp'])
            outliers = df[df['Temp_zscore'].abs() > 3]  # Z-score > 3 indicates an outlier

            if not outliers.empty:
                st.warning(f"⚠️ Found {len(outliers)} temperature outliers!")
                st.dataframe(outliers[['Date', 'Data.Temperature.Avg Temp', 'Temp_zscore']])
            else:
                st.success("✅ No significant temperature anomalies found.")

        # Seasonal Patterns: Grouping by Month and Year
        with st.expander("📅 Seasonal Patterns"):
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            seasonal_avg = df.groupby(['Year', 'Month'])['Data.Temperature.Avg Temp'].mean().unstack()

            fig, ax = plt.subplots(figsize=(10, 6))
            seasonal_avg.plot(ax=ax, marker='o', linestyle='-', color=['blue', 'green', 'red', 'purple', 'orange'])
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("Seasonal Temperature Patterns (Yearly Comparison)")
            ax.grid(True)
            st.pyplot(fig)

        # Rolling Averages: Moving averages over different windows
        with st.expander("📊 Rolling Averages (30-Day and 60-Day)"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Temp_MA30'] = df['Data.Temperature.Avg Temp'].rolling(window=30).mean()
            df['Temp_MA60'] = df['Data.Temperature.Avg Temp'].rolling(window=60).mean()
            df.plot(x="Date", y=["Temp_MA30", "Temp_MA60"], ax=ax)
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("Rolling 30-Day and 60-Day Moving Averages of Temperature")
            ax.grid(True)
            st.pyplot(fig)

    else:
        st.warning("📂 Please upload the dataset to view insights.")


# TAB 5: App Overview
with tab5:
    st.subheader("📖 App Overview")
    st.write("""
    This app provides a hybrid weather forecast by combining live weather data from WeatherAPI with temperature predictions generated using historical climate data uploaded by the user. 
    It allows users to:
    - View the current weather forecast for any city.
    - Upload their climate dataset for analysis.
    - Generate temperature predictions for the next 7 days.
    - Visualize climate trends over time.
    """)

# TAB 6: Help/FAQ
with tab6:
    st.subheader("❓ Help/FAQ")
    st.write("""
    **Q: How do I upload a dataset?**
    - Click on the "📊 Climate Dataset" tab and upload a CSV file with temperature data.

    **Q: What type of data should my dataset contain?**
    - Your dataset should include columns like 'Date.Full' and 'Data.Temperature.Avg Temp'. Ensure that the 'Date.Full' column is in a valid date format.

    **Q: How accurate are the temperature predictions?**
    - The temperature predictions are based on historical data and a simple linear regression model. For better accuracy, ensure that your dataset is large and well-structured.

    **Q: How can I change the city for the weather forecast?**
    - Enter the desired city in the input box under the "🌍 Live Weather" tab.

    **Q: Why is there a discrepancy between the live forecast and predictions?**
    - The live forecast is provided by WeatherAPI based on up-to-date meteorological models, while predictions are based on historical data trends.

    **Q: What if my dataset contains invalid dates?**
    - The app will automatically handle invalid dates by setting them to NaT (Not a Time) and showing a warning message.
    """)
