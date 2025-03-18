import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
from prophet import Prophet
import plotly.graph_objects as go

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- WEATHER API CONFIG ----
API_KEY = st.secrets.get("WEATHERAPI_KEY")  # Fetch API key safely

def get_live_weather(city):
    """Fetch real-time weather data from WeatherAPI.com."""
    if not API_KEY:
        st.error("❌ API Key is missing! Please check your Streamlit secrets.")
        return None

    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            st.error(f"⚠️ {data['error']['message']}")
            return None

        return {"ds": datetime.datetime.now(), "y": float(data["current"]["temp_c"])}  # Convert to float

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request failed: {e}")
        return None

# ---- SIDEBAR FILE UPLOAD ----
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Years, Temperature)", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate CSV format
        required_columns = {"Years", "Temperature"}
        if not required_columns.issubset(df.columns):
            st.sidebar.error("⚠️ Invalid CSV format. Required columns: Years, Temperature.")
            df = None
        else:
            df["ds"] = pd.to_datetime(df["Years"], errors="coerce")  # Convert to datetime
            df.dropna(subset=["ds"], inplace=True)  # Remove invalid dates
            df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})  # Rename for Prophet
            df["y"] = pd.to_numeric(df["y"], errors="coerce")  # Ensure temperature is numeric
            df.dropna(inplace=True)  # Remove any rows with NaN values

    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- FETCH LIVE WEATHER AND MERGE ----
st.sidebar.subheader("🌍 Live Weather Data")
cities = st.sidebar.text_input("Enter City for Live Data", "New York")
live_weather = None

if st.sidebar.button("Fetch Live Weather"):
    live_weather = get_live_weather(cities)
    if live_weather:
        if df is not None:
            df = pd.concat([df, pd.DataFrame([live_weather])], ignore_index=True)
        else:
            df = pd.DataFrame([live_weather])
        st.sidebar.success(f"✔️ Live weather for {cities} added to dataset!")

# ---- AI FORECASTS ----
st.header("📈 AI Climate Forecasts with Live Data")

if df is not None and len(df) > 1:
    try:
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=365)  # Predict next 365 days
        forecast = model.predict(future)

        # Plot Actual Data + Forecast
        fig = go.Figure()

        # Actual Data
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="markers", name="Actual Data"))

        # Forecasted Trend
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecasted Trend"))

        fig.update_layout(title="Predicted Temperature Trends (Including Live Data)", xaxis_title="Year", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Forecasting error: {e}")
elif df is not None:
    st.error("⚠️ Not enough data to train AI model.")

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**", unsafe_allow_html=True)
