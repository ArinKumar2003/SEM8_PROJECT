import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import requests
from streamlit_option_menu import option_menu

# Page config
st.set_page_config("🌎 Global Climate Dashboard", layout="wide")

# Weather API
WEATHER_API_KEY = st.secrets["weatherapi"]["api_key"]
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload Climate Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df.rename(columns={"date": "ds"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors='coerce')
        return df.dropna()

    df = load_data(uploaded_file)

    # Check for city column
    city_filter_enabled = 'city' in df.columns
    if city_filter_enabled:
        cities = sorted(df['city'].dropna().unique().tolist())

    # Weather API call
    def get_live_weather(city):
        try:
            response = requests.get(WEATHER_API_URL, params={"key": WEATHER_API_KEY, "q": city})
            data = response.json()
            return {
                "location": data["location"]["name"],
                "country": data["location"]["country"],
                "temp_c": data["current"]["temp_c"],
                "condition": data["current"]["condition"]["text"],
                "humidity": data["current"]["humidity"],
                "wind_kph": data["current"]["wind_kph"],
                "icon": data["current"]["condition"]["icon"]
            }
        except:
            return None

    # Tabs
    selected = option_menu(
        None,
        ["Live Weather", "Forecast", "Visualization", "Map View", "Monthly Summary", "Climate Awareness"],
        icons=["cloud-sun", "calendar3", "bar-chart", "map", "calendar-event", "info-circle"],
        orientation="horizontal"
    )

    # Tab: Live Weather
    if selected == "Live Weather":
        st.subheader("☁️ Live Weather")
        city = st.text_input("Enter City", "New York")
        if city:
            weather = get_live_weather(city)
            if weather:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric("🌡 Temperature", f"{weather['temp_c']} °C")
                    st.metric("💧 Humidity", f"{weather['humidity']}%")
                    st.metric("🌬 Wind", f"{weather['wind_kph']} kph")
                    st.success(f"{weather['location']}, {weather['country']} - {weather['condition']}")
                with col2:
                    st.image("http:" + weather["icon"], width=100)

    # Tab: Forecast
    elif selected == "Forecast":
        st.subheader("📈 Climate Forecast (Temperature)")
        if city_filter_enabled:
            selected_city = st.selectbox("Select City for Forecast", cities)
            df_city = df[df['city'] == selected_city]
        else:
            df_city = df

        df_prophet = df_city[["ds", "temperature"]].rename(columns={"temperature": "y"})
        periods = st.slider("Days to Forecast", 7, 180, 30)

        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        fig = px.line(forecast, x='ds', y='yhat', title=f"Temperature Forecast")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download Forecast CSV", forecast.to_csv(index=False), "forecast.csv")

    # Tab: Visualization
    elif selected == "Visualization":
        st.subheader("📊 Explore Climate Data")
        selected_column = st.selectbox("Choose Variable", df.select_dtypes('number').columns)
        fig = px.line(df, x="ds", y=selected_column, title=f"{selected_column} over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download Dataset", df.to_csv(index=False), "climate_data.csv")

    # Tab: Map View
    elif selected == "Map View":
        st.subheader("🗺 Climate Variables on Map")
        if "latitude" in df.columns and "longitude" in df.columns:
            variable = st.selectbox("Choose Variable", ["temperature", "humidity", "wind_speed"])
            latest = df.sort_values("ds").dropna(subset=[variable])
            latest = latest.groupby("city", as_index=False).last()

            fig = px.scatter_mapbox(
                latest, lat="latitude", lon="longitude", color=variable, size=variable,
                hover_name="city", zoom=1, mapbox_style="carto-positron"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Latitude and Longitude columns are required for map view.")

    # Tab: Monthly Summary
    elif selected == "Monthly Summary":
        st.subheader("📅 Monthly Climate Summary")
        df['month'] = df['ds'].dt.to_period('M').astype(str)
        metric = st.selectbox("Select Metric", ["temperature", "humidity", "wind_speed"])
        summary = df.groupby('month')[metric].mean().reset_index()
        fig = px.bar(summary, x='month', y=metric, title=f"Monthly Avg {metric.title()}")
        st.plotly_chart(fig, use_container_width=True)

    # Tab: Awareness
    elif selected == "Climate Awareness":
        st.subheader("🌱 Climate Summary & Awareness")
        st.markdown("""
        ### 🔍 Key Climate Stats
        - 🌡 **Global temps rising ~1.1°C**
        - ❄ Arctic sea ice shrinking **13%/decade**
        - 🌊 Sea levels have risen ~8 inches

        ### 🌍 What You Can Do
        - 🚴 Use public transport
        - 💡 Reduce energy use
        - 🥕 Eat less meat
        - 🌳 Plant trees

        > _“There is no Planet B.”_
        """)
else:
    st.info("📂 Please upload a climate dataset to get started.")
