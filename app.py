import streamlit as st
import pandas as pd
import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Set up API key
WEATHER_API_KEY = "e12e93484a0645f2802141629250803"

# Page configuration
st.set_page_config(page_title="Climate Forecast App", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["🌍 Live Weather", "📊 Climate Analysis", "📆 Key Date Predictions"])

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

# --- TAB 2: CLIMATE ANALYSIS ---
with tab2:
    st.header("Upload Climate Dataset")

    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            st.error("❌ Error parsing 'Date' column. Please check your dataset format.")
            st.stop()

        st.success("✅ Dataset successfully loaded!")
        st.write(df.head())

        # Show statistics
        st.subheader("📈 Climate Trends")
        columns = ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']
        selected_col = st.selectbox("Select variable to visualize", columns)

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Date', y=selected_col, ax=ax)
        ax.set_title(f"{selected_col} Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- TAB 3: KEY DATE PREDICTIONS ---
with tab3:
    st.header("📆 Key Date Predictions")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            st.error("❌ Could not parse 'Date' column.")
            st.stop()

        latest_date = df['Date'].max()
        tomorrow = latest_date + pd.Timedelta(days=1)
        next_month = latest_date + pd.DateOffset(months=1)

        if 'Temperature' in df.columns:
            # Calculate average daily change
            df_sorted = df.sort_values('Date')
            df_sorted['Temp_Change'] = df_sorted['Temperature'].diff()
            temp_change = df_sorted['Temp_Change'].mean()

            last_temp = df_sorted['Temperature'].iloc[-1]
            pred_tomorrow = last_temp + temp_change
            pred_next_month = last_temp + (temp_change * 30)

            st.markdown(f"📍 **Tomorrow ({tomorrow.strftime('%d %b %Y')}):** `{pred_tomorrow:.2f} °C`")
            st.markdown(f"📍 **Next Month ({next_month.strftime('%d %b %Y')}):** `{pred_next_month:.2f} °C`")
        else:
            st.warning("⚠️ 'Temperature' column not found in dataset.")
    else:
        st.warning("📂 Please upload the dataset in the 'Climate Analysis' tab to enable predictions.")
