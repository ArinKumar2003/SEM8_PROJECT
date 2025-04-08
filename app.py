import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
import plotly.express as px

st.set_page_config(page_title="Smart Weather Dashboard 🌦️", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🌍 Smart Weather Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Live Weather • Forecast • Insights • Awareness</h4>", unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_weather.csv", parse_dates=['datetime'])
    df.rename(columns={'datetime': 'ds', 'temperature': 'y'}, inplace=True)
    df = df[['ds', 'y']].dropna()
    return df

df = load_data()

# WeatherAPI function
def get_live_weather(city):
    api_key = st.secrets["weatherapi"]["api_key"]
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current = data["current"]
        return {
            "temperature": current["temp_c"],
            "description": current["condition"]["text"],
            "humidity": current["humidity"],
            "wind": current["wind_kph"],
            "icon": f"https:{current['condition']['icon']}"
        }
    return None

# Main tab layout
with st.sidebar:
    selected = option_menu("Main Menu", [
        "Live Weather", 
        "Forecast", 
        "Climate Summary", 
        "Awareness & Tips"
    ], icons=["cloud-sun", "graph-up", "bar-chart", "lightbulb"], menu_icon="cast", default_index=0)

# --- Tab 1: Live Weather ---
if selected == "Live Weather":
    st.subheader("📍 Real-Time Weather")
    city = st.text_input("Enter a city", value="Delhi")

    if city:
        weather = get_live_weather(city)
        if weather:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric(label="🌡️ Temperature (°C)", value=weather["temperature"])
                st.metric(label="💧 Humidity (%)", value=weather["humidity"])
                st.metric(label="🌬️ Wind Speed (kph)", value=weather["wind"])
                st.write(f"**Condition:** {weather['description']}")
            with col2:
                st.image(weather["icon"], width=100)
        else:
            st.warning("Could not retrieve weather data. Please check the city name or try again.")

# --- Tab 2: Forecasting ---
elif selected == "Forecast":
    st.subheader("📈 Weather Forecast")
    st.write("Using historical data from your uploaded dataset.")

    period = st.slider("Forecast period (days)", 1, 30, 7)
    
    # Prophet model
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.write("Forecast data:")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period))

# --- Tab 3: Climate Summary ---
elif selected == "Climate Summary":
    st.subheader("📊 Climate Summary from Dataset")

    st.write("This section provides basic analytics and visualizations from the uploaded historical weather data.")
    col1, col2 = st.columns(2)
    col1.metric("Records", df.shape[0])
    col2.metric("Avg Temperature (°C)", round(df['y'].mean(), 2))

    # Line chart of temperature trends
    fig = px.line(df, x='ds', y='y', title='Temperature Over Time', labels={'ds': 'Date', 'y': 'Temperature (°C)'})
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Awareness ---
elif selected == "Awareness & Tips":
    st.subheader("💡 Climate Change Awareness & Weather Safety Tips")

    st.markdown("""
    ### 🌎 Climate Awareness
    - The Earth's temperature has increased by **1.1°C** since the late 1800s.
    - Frequent extreme weather events are direct consequences of climate change.
    - Small actions like reducing emissions and conserving energy can have a global impact.

    ### 🛡️ Weather Safety Tips
    - **Extreme Heat:** Stay hydrated, avoid outdoor activity during peak hours.
    - **Floods:** Avoid flood-prone areas, listen to local alerts.
    - **Storms:** Secure loose items, stay indoors.

    > Stay informed. Be prepared. Act sustainably.
    """)

    st.image("https://climate.nasa.gov/system/content_pages/main_images/1051_graphic-global-temp-update-2023-1200px.jpg", caption="Source: NASA", use_column_width=True)
