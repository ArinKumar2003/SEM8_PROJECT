import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import requests

# Constants
WEATHER_API_KEY = "e12e93484a0645f2802141629250803"

# Page settings
st.set_page_config(page_title="🌦️ Climate Forecast Dashboard", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Live Weather", 
    "📊 Climate Dataset", 
    "📆 Predictions", 
    "📈 Data Insights"
])

# --- TAB 1: LIVE WEATHER ---
with tab1:
    st.header("Live Weather")
    city = st.text_input("Enter City", "New York")

    if st.button("Get Live Weather"):
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            location = data['location']

            st.subheader(f"Current Weather in {location['name']}, {location['country']}")
            st.metric("🌡️ Temperature (°C)", current['temp_c'])
            st.metric("💧 Humidity (%)", current['humidity'])
            st.metric("🌬️ Wind Speed (kph)", current['wind_kph'])
            st.metric("🌤️ Condition", current['condition']['text'])
        else:
            st.error("❌ Failed to retrieve weather data.")

# Shared Dataset Variable
df = None

# --- TAB 2: DATA UPLOAD ---
with tab2:
    st.header("📊 Upload Climate Dataset")
    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Columns in your dataset:", df.columns.tolist())

            # Convert Date.Full into a datetime column
            def convert_time_string(t):
                try:
                    minutes, seconds = map(float, str(t).split(":"))
                    base_date = datetime.datetime.combine(datetime.date.today(), datetime.time(0))
                    return base_date + datetime.timedelta(minutes=minutes, seconds=seconds)
                except:
                    return pd.NaT

            df['Date'] = df['Date.Full'].apply(convert_time_string)
            invalid_count = df['Date'].isna().sum()

            if invalid_count > 0:
                st.warning(f"⚠️ {invalid_count} rows had invalid time formats and were set to NaT.")

            df.dropna(subset=['Date'], inplace=True)
            st.success("✅ 'Date' column successfully converted!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

# --- TAB 3: PREDICTIONS ---
with tab3:
    st.header("📆 Predict Temperature from Today")

    if df is not None:
        df = df.sort_values("Date")
        df['Temp_Change'] = df['Data.Temperature.Avg Temp'].diff()
        avg_daily_change = df['Temp_Change'].mean()

        today_temp = df['Data.Temperature.Avg Temp'].iloc[-1]
        pred_tomorrow = today_temp + avg_daily_change
        pred_next_week = today_temp + (avg_daily_change * 7)

        st.metric("Today’s Temp", f"{today_temp:.2f} °C")
        st.markdown(f"📍 **Tomorrow**: `{pred_tomorrow:.2f} °C`")
        st.markdown(f"📍 **Next Week**: `{pred_next_week:.2f} °C`")
    else:
        st.warning("📂 Please upload the dataset first in the previous tab.")

# --- TAB 4: DATA INSIGHTS ---
with tab4:
    st.header("📈 Data Visualizations")

    if df is not None:
        columns = [
            'Data.Temperature.Avg Temp', 
            'Data.Temperature.Max Temp', 
            'Data.Temperature.Min Temp',
            'Data.Precipitation', 
            'Data.Wind.Speed'
        ]

        selected_col = st.selectbox("Select variable to visualize over time", columns)
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Date', y=selected_col, ax=ax)
        ax.set_title(f"{selected_col} Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("📂 Please upload the dataset first in the 'Climate Dataset' tab.")
