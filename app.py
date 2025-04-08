import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import requests
from streamlit_option_menu import option_menu

# Set page config
st.set_page_config("🌎 Global Climate Dashboard", layout="wide")

# Weather API
WEATHER_API_KEY = st.secrets["weatherapi"]["api_key"]
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# Header
st.markdown("<h1 style='text-align: center; color: #4e8cff;'>🌍 Global Climate Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Visualize weather patterns, forecast climate trends, and raise awareness 🌡🌱</h4><hr>", unsafe_allow_html=True)

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

    # Navigation Tabs
    selected = option_menu(
        None,
        ["Live Weather", "Forecast", "Visualization", "Map View", "Monthly Summary", "Climate Awareness"],
        icons=["cloud-sun", "calendar3", "bar-chart", "map", "calendar-event", "info-circle"],
        orientation="horizontal"
    )

    # ===== LIVE WEATHER TAB =====
    if selected == "Live Weather":
        st.markdown("## ☁️ Live Weather Conditions")
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

    # ===== FORECAST TAB =====
    elif selected == "Forecast":
        st.markdown("## 📈 Forecast Temperature Using Prophet")
        if city_filter_enabled:
            selected_city = st.selectbox("Select City", cities)
            df_city = df[df['city'] == selected_city]
        else:
            df_city = df

        df_prophet = df_city[["ds", "temperature"]].rename(columns={"temperature": "y"})
        periods = st.slider("Forecast Days", 7, 180, 30)

        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        fig = px.line(forecast, x='ds', y='yhat', title=f"{selected_city if city_filter_enabled else ''} Temperature Forecast")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("📩 Download Forecast CSV", forecast.to_csv(index=False), "forecast.csv")

    # ===== VISUALIZATION TAB =====
    elif selected == "Visualization":
        st.markdown("## 📊 Climate Variable Over Time")
        selected_column = st.selectbox("Choose Variable", df.select_dtypes('number').columns)
        fig = px.line(df, x="ds", y=selected_column, title=f"{selected_column.title()} Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("📩 Download Full Dataset", df.to_csv(index=False), "climate_data.csv")

    # ===== MAP TAB =====
    elif selected == "Map View":
        st.markdown("## 🗺 Climate Map Visualization")
        if "latitude" in df.columns and "longitude" in df.columns:
            variable = st.selectbox("Select Variable", ["temperature", "humidity", "wind_speed"])
            latest = df.sort_values("ds").dropna(subset=[variable])
            latest = latest.groupby("city", as_index=False).last()

            fig = px.scatter_mapbox(
                latest, lat="latitude", lon="longitude", color=variable, size=variable,
                hover_name="city", zoom=1, mapbox_style="carto-positron"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("🌍 Your data must include 'latitude' and 'longitude' columns.")

    # ===== MONTHLY SUMMARY TAB =====
    elif selected == "Monthly Summary":
        st.markdown("## 📅 Monthly Climate Averages")
        df['month'] = df['ds'].dt.to_period('M').astype(str)
        metric = st.selectbox("Select Metric", ["temperature", "humidity", "wind_speed"])
        summary = df.groupby('month')[metric].mean().reset_index()
        fig = px.bar(summary, x='month', y=metric, title=f"Monthly Average {metric.title()}")
        st.plotly_chart(fig, use_container_width=True)

    # ===== CLIMATE AWARENESS TAB =====
    elif selected == "Climate Awareness":
        st.markdown("## 🌱 Climate Summary & Awareness")
        st.markdown("""
        ### 🌡 Global Climate Quick Facts
        - 🌍 Earth's temp has risen ~1.1°C since 1880.
        - 🌊 Sea levels have risen ~20 cm since 1900.
        - ❄ Arctic ice is shrinking ~13% per decade.

        ### 💡 Simple Actions That Help
        - 🚲 Use bicycles or public transport.
        - 💡 Switch to energy-saving bulbs.
        - 🥬 Choose local & plant-based foods.
        - 🌳 Support reforestation projects.

        > _"The climate is changing. So should we."_ – United Nations
        """)

else:
    st.info("📂 Please upload a valid climate dataset to begin using the dashboard.")

