import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="🌤️ Climate Forecasting App", layout="wide")
st.title("🌦️ Climate Forecasting with Prophet")

# Load API key from secrets
API_KEY = st.secrets["weatherapi"]["api_key"]

@st.cache_data(show_spinner=False)
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

# Upload dataset
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your climate CSV file", type=["csv"])

# Weather section
st.sidebar.header("☁️ Current Weather")
city = st.sidebar.text_input("City", value="Delhi")

if city:
    try:
        weather = get_weather(city)
        st.sidebar.markdown(f"""
        **{weather['location']}, {weather['country']}**
        - 🌡️ Temp: {weather['temp_c']}°C  
        - 🌤️ {weather['condition']}
        """)
        st.sidebar.image(f"https:{weather['icon']}")
    except:
        st.sidebar.error("⚠️ Could not fetch weather info")

# Load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    with st.expander("Show Raw Data"):
        st.dataframe(df)

    st.markdown("### 📌 Forecast Settings")

    # Target variable selector
    options = df.columns.drop("datetime")
    target_col = st.selectbox("🎯 Select a column to forecast", options, index=0)

    # Granularity
    granularity = st.radio("⏱️ Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)

    # Resample data
    df = df[["datetime", target_col]].dropna()
    df = df.rename(columns={"datetime": "ds", target_col: "y"})

    if granularity == "Weekly":
        df = df.resample("W-MON", on="ds").mean().reset_index()
    elif granularity == "Monthly":
        df = df.resample("M", on="ds").mean().reset_index()

    # Forecast period
    periods = st.slider("📆 Forecast how many future periods?", min_value=30, max_value=730, step=30, value=90)

    # Train model
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="D" if granularity == "Daily" else "W" if granularity == "Weekly" else "M")
    forecast = model.predict(future)

    # Forecast plot
    st.subheader("📈 Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    # Components
    st.subheader("🧩 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Summary metrics
    with st.expander("📌 Forecast Summary Table"):
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20))

    # Download forecast
    csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
    st.download_button("📥 Download Forecast CSV", csv, file_name="forecast.csv", mime="text/csv")

else:
    st.info("👆 Upload your dataset to begin forecasting.")
