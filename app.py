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

# Sidebar Upload & Weather
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your climate CSV file", type=["csv"])

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

# Load and process data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(dict(year=df["Years"], month=df["Month"], day=df["Day"]), errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📊 Data Overview")
    st.dataframe(df.head(100))

    # Select column for forecasting
    forecast_cols = ["CO2", "Humidity", "SeaLevel", "Temperature"]
    target_col = st.selectbox("🎯 Select a column to forecast", forecast_cols)

    # Filter date range
    min_date, max_date = df["datetime"].min(), df["datetime"].max()
    date_range = st.slider("📅 Select date range", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date))
    df = df[(df["datetime"] >= date_range[0]) & (df["datetime"] <= date_range[1])]

    # Plot preview
    st.markdown("### 📈 Historical Trend")
    fig = px.line(df, x="datetime", y=target_col, title=f"{target_col} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Show stats
    with st.expander("📌 Descriptive Statistics"):
        st.write(df[forecast_cols].describe())

    st.markdown("### ⚙️ Forecast Settings")

    # Granularity
    granularity = st.radio("⏱️ Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)

    df_prophet = df[["datetime", target_col]].dropna().rename(columns={"datetime": "ds", target_col: "y"})

    if granularity == "Weekly":
        df_prophet = df_prophet.resample("W-MON", on="ds").mean().reset_index()
    elif granularity == "Monthly":
        df_prophet = df_prophet.resample("M", on="ds").mean().reset_index()

    # Forecast period
    periods = st.slider("📆 Forecast how many future periods?", min_value=30, max_value=730, step=30, value=90)

    # Fit model
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods, freq="D" if granularity == "Daily" else "W" if granularity == "Weekly" else "M")
    forecast = model.predict(future)

    st.subheader("📈 Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🧩 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    with st.expander("📌 Forecast Summary Table"):
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20))

    # Download
    csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
    st.download_button("📥 Download Forecast CSV",
                       csv,
                       file_name=f"{target_col}_forecast.csv",
                       mime="text/csv")
else:
    st.info("👆 Upload your dataset to begin forecasting.")
