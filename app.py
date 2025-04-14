import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from datetime import datetime
from io import StringIO

st.set_page_config(page_title="🌦️ Climate Forecasting", layout="wide")

st.title("🌤️ Climate Forecasting Dashboard (Prophet)")
st.caption("Forecast variables like temperature, humidity, wind speed, and more.")

# API key (optional for weather info)
API_KEY = st.secrets["weatherapi"]["api_key"]

@st.cache_data
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    res = requests.get(url).json()
    return {
        "location": res["location"]["name"],
        "country": res["location"]["country"],
        "temp_c": res["current"]["temp_c"],
        "condition": res["current"]["condition"]["text"],
        "icon": res["current"]["condition"]["icon"]
    }

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df

# Upload
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload climate CSV", type=["csv"])

# Weather (optional)
st.sidebar.header("🌍 City Weather")
city = st.sidebar.text_input("City", "Delhi")
if city:
    try:
        weather = get_weather(city)
        st.sidebar.markdown(f"""
        **{weather['location']}, {weather['country']}**
        - 🌡️ {weather['temp_c']}°C  
        - 🌤️ {weather['condition']}
        """)
        st.sidebar.image(f"https:{weather['icon']}")
    except:
        st.sidebar.warning("Couldn't fetch weather info")

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📊 Raw Data Overview")
    with st.expander("Show Raw Data"):
        st.dataframe(df, use_container_width=True)

    st.markdown("### ⚙️ Forecast Settings")

    # Target column
    numeric_cols = ["temperature", "humidity", "wind_speed", "pressure", "rain"]
    target = st.selectbox("📈 Choose a variable to forecast", numeric_cols)

    # Resampling frequency
    freq_label = st.radio("⏱️ Time Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)
    freq = "D" if freq_label == "Daily" else "W" if freq_label == "Weekly" else "M"

    # Filter + Rename
    data = df[["datetime", target]].rename(columns={"datetime": "ds", target: "y"})
    data = data.dropna()

    # Resample
    data = data.set_index("ds").resample(freq).mean().reset_index()

    # Stats
    latest = data.dropna().tail(30)
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Mean", f"{latest['y'].mean():.2f}")
    col2.metric("⬆️ Max", f"{latest['y'].max():.2f}")
    col3.metric("⬇️ Min", f"{latest['y'].min():.2f}")

    # Forecast range
    periods = st.slider("📆 Forecast length", 30, 730, 90, step=30)

    # Prophet model
    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    st.subheader("📈 Forecast Chart")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    with st.expander("🧾 Forecast Data Table"):
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30))

    # Download forecast
    csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
    st.download_button("📥 Download Forecast CSV", csv, file_name=f"{target}_forecast.csv", mime="text/csv")

else:
    st.info("👆 Upload your climate dataset to begin.")
