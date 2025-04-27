import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Constants
WEATHER_API_KEY = "e12e93484a0645f2802141629250803"

# Page settings
st.set_page_config(page_title="Climate Forecast & Analysis Dashboard", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Live Weather", 
    "📊 Climate Dataset", 
    "📆 Predictions", 
    "📊 Data Insights"
])

# --- TAB 1: LIVE WEATHER ---
with tab1:
    st.header("🌍 Live Weather")
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
    st.header("📊 Climate Dataset")

    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Parse date
            df['Date'] = pd.to_datetime(df['Date.Full'], errors='coerce')
            invalid_rows = df[df['Date'].isna()]

            if not invalid_rows.empty:
                st.warning(f"⚠️ Some rows have invalid dates and have been set to NaT. Rows: {invalid_rows.index.tolist()}")

            df.dropna(subset=['Date'], inplace=True)
            st.success("✅ Dataset successfully loaded and 'Date' column cleaned!")
            st.write(df.head())
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

# --- TAB 3: PREDICTIONS ---
with tab3:
    st.header("📆 Predictions")

    if 'df' in locals():
        latest_date = df['Date'].max()
        next_day = latest_date + pd.Timedelta(days=1)
        next_month = latest_date + pd.DateOffset(months=1)

        df_sorted = df.sort_values('Date')
        df_sorted['Temp_Change'] = df_sorted['Data.Temperature.Avg Temp'].diff()
        avg_change = df_sorted['Temp_Change'].mean()
        last_temp = df_sorted['Data.Temperature.Avg Temp'].iloc[-1]

        pred_tomorrow = last_temp + avg_change
        pred_next_month = last_temp + (avg_change * 30)

        st.markdown(f"📍 **Next Day ({next_day.strftime('%d %b %Y')}):** `{pred_tomorrow:.2f} °F`")
        st.markdown(f"📍 **Next Month ({next_month.strftime('%d %b %Y')}):** `{pred_next_month:.2f} °F`")
    else:
        st.info("Please upload the dataset in the previous tab.")

# --- TAB 4: INSIGHTS ---
with tab4:
    st.header("📊 Data Insights")

    if 'df' in locals():
        variable_options = [
            'Data.Temperature.Avg Temp',
            'Data.Temperature.Max Temp',
            'Data.Temperature.Min Temp',
            'Data.Wind.Speed',
            'Data.Precipitation'
        ]
        selected_var = st.selectbox("Select variable to visualize:", variable_options)

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Date', y=selected_var, ax=ax)
        ax.set_title(f"{selected_var} Over Time")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        st.info("Please upload the dataset in the 'Climate Dataset' tab.")
