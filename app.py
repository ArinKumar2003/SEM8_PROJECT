import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up API key
WEATHER_API_KEY = "e12e93484a0645f2802141629250803"

# Page configuration
st.set_page_config(page_title="Climate Forecast App", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Live Weather", "📊 Climate Dataset", "📆 Predictions", "📊 Data Insights"])

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
        df = pd.read_csv(uploaded_file)

        # Attempt to parse 'Date' column
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='raise')
            st.success("✅ Dataset successfully loaded!")
        except Exception as e:
            st.error(f"❌ Error parsing 'Date' column: {e}")
            st.stop()

        # Show a sample of the dataset
        st.write("### Dataset Preview:")
        st.write(df.head())

# --- TAB 3: PREDICTIONS ---
with tab3:
    st.header("📆 Key Date Predictions")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='raise')
        except Exception as e:
            st.error(f"❌ Could not parse 'Date' column: {e}")
            st.stop()

        latest_date = df['Date'].max()
        tomorrow = latest_date + pd.Timedelta(days=1)
        next_month = latest_date + pd.DateOffset(months=1)

        st.write(f"### Latest Data: {latest_date.strftime('%d %b %Y')}")

        if 'Temperature' in df.columns:
            # Calculate average daily change for Temperature
            df_sorted = df.sort_values('Date')
            df_sorted['Temp_Change'] = df_sorted['Temperature'].diff()
            temp_change = df_sorted['Temp_Change'].mean()

            last_temp = df_sorted['Temperature'].iloc[-1]
            pred_tomorrow = last_temp + temp_change
            pred_next_month = last_temp + (temp_change * 30)

            st.markdown(f"📍 **Tomorrow ({tomorrow.strftime('%d %b %Y')}):** `{pred_tomorrow:.2f} °C`")
            st.markdown(f"📍 **Next Month ({next_month.strftime('%d %b %Y')}):** `{pred_next_month:.2f} °C`")

        if 'CO2 Emissions' in df.columns:
            # Predict CO2 Emissions using similar logic
            df_sorted['CO2_Change'] = df_sorted['CO2 Emissions'].diff()
            co2_change = df_sorted['CO2_Change'].mean()

            last_co2 = df_sorted['CO2 Emissions'].iloc[-1]
            pred_tomorrow_co2 = last_co2 + co2_change
            pred_next_month_co2 = last_co2 + (co2_change * 30)

            st.markdown(f"📍 **Tomorrow CO2 Emissions:** `{pred_tomorrow_co2:.2f} ppm`")
            st.markdown(f"📍 **Next Month CO2 Emissions:** `{pred_next_month_co2:.2f} ppm`")

        else:
            st.warning("⚠️ 'Temperature' or 'CO2 Emissions' column not found in dataset.")
    else:
        st.warning("📂 Please upload the dataset in the 'Climate Dataset' tab to enable predictions.")

# --- TAB 4: DATA INSIGHTS ---
with tab4:
    st.header("📊 Data Insights")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='raise')
        except Exception as e:
            st.error(f"❌ Error parsing 'Date' column: {e}")
            st.stop()

        # Show basic statistics for numeric columns
        st.write("### Basic Statistical Summary:")
        st.write(df.describe())

        # Correlation heatmap of numeric features
        st.write("### Correlation Heatmap:")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Optionally show distributions of specific variables
        st.subheader("Visualize Variable Distributions")
        columns = ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']
        selected_col = st.selectbox("Select variable to visualize", columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

    else:
        st.warning("📂 Please upload the dataset in the 'Climate Dataset' tab to enable insights.")
