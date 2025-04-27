import streamlit as st
import pandas as pd
import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Set up API key for weather
WEATHER_API_KEY = "e12e93484a0645f2802141629250803"

# Page setup
st.set_page_config(page_title="🌦️ Climate Forecast & Analysis Dashboard", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Live Weather", "📊 Climate Dataset", "📆 Predictions", "📈 Data Insights"])

# --- TAB 1: LIVE WEATHER ---
with tab1:
    st.header("Live Weather")

    city = st.text_input("Enter City", "New York")

    if st.button("Get Live Weather"):
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            location = data['location']
            current = data['current']

            st.subheader(f"Current Weather in {location['name']}, {location['country']}")
            st.metric("🌡️ Temperature (°C)", current['temp_c'])
            st.metric("💧 Humidity (%)", current['humidity'])
            st.metric("🌬️ Wind Speed (kph)", current['wind_kph'])
            st.metric("🌤️ Condition", current['condition']['text'])
        else:
            st.error("Failed to retrieve weather data. Please check the city name or API key.")

# --- TAB 2: CLIMATE DATASET ---
with tab2:
    st.header("Upload Climate Dataset")

    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Attempt to clean/convert the 'Date' column (which is in HH:MM.S format)
            def convert_time_to_datetime(x):
                try:
                    base_date = datetime.datetime.combine(datetime.date.today(), datetime.time(0))
                    minutes, seconds = map(float, x.split(":"))
                    return base_date + datetime.timedelta(minutes=minutes, seconds=seconds)
                except:
                    return pd.NaT

            df['Date'] = df['Date'].astype(str).apply(convert_time_to_datetime)

            # Track invalid rows
            invalid_dates = df['Date'].isna().sum()

            if invalid_dates > 0:
                st.warning(f"⚠️ {invalid_dates} rows had invalid time formats and have been set to NaT.")

            st.success("✅ Dataset successfully loaded and 'Date' column cleaned!")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            st.stop()

# --- TAB 3: PREDICTIONS ---
with tab3:
    st.header("📆 Predictions")

    if 'df' in locals():
        today = pd.Timestamp.today().normalize()
        df_sorted = df.sort_values('Date')
        df_sorted = df_sorted[df_sorted['Date'].notna()]
        df_sorted = df_sorted[df_sorted['Date'] <= today]

        if not df_sorted.empty:
            try:
                df_sorted['Temp_Change'] = df_sorted['Data.Temperature.Avg Temp'].diff()
                avg_change = df_sorted['Temp_Change'].mean()
                last_temp = df_sorted['Data.Temperature.Avg Temp'].iloc[-1]

                pred_tomorrow = last_temp + avg_change
                pred_next_month = last_temp + (avg_change * 30)

                st.markdown(f"📍 **Today ({today.strftime('%d %b %Y')}):** `{last_temp:.2f} °F`")
                st.markdown(f"📍 **Tomorrow ({(today + pd.Timedelta(days=1)).strftime('%d %b %Y')}):** `{pred_tomorrow:.2f} °F`")
                st.markdown(f"📍 **Next Month ({(today + pd.DateOffset(months=1)).strftime('%d %b %Y')}):** `{pred_next_month:.2f} °F`")
            except:
                st.error("❌ Prediction failed. Ensure 'Data.Temperature.Avg Temp' column exists and is numeric.")
        else:
            st.warning("⚠️ No valid data found before or on today's date.")
    else:
        st.info("📂 Please upload the dataset in the 'Climate Dataset' tab.")

# --- TAB 4: DATA INSIGHTS ---
with tab4:
    st.header("📈 Data Insights")

    if 'df' in locals():
        df = df[df['Date'].notna()]
        st.subheader("Visualize Climate Variables Over Time")

        available_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != 'Date']
        if not available_columns:
            st.warning("No numeric columns found to visualize.")
        else:
            selected_col = st.selectbox("Select variable to visualize", available_columns)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=df, x='Date', y=selected_col, ax=ax)
            ax.set_title(f"{selected_col} Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.info("Upload a dataset to explore insights.")
