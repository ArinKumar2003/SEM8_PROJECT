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
API_KEY = st.secrets.get("WEATHERAPI_KEY")

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

        return {
            "ds": datetime.datetime.now(),
            "y": float(data["current"]["temp_c"]),
            "humidity": data["current"]["humidity"],
            "wind_kph": data["current"]["wind_kph"],
            "precip_mm": data["current"]["precip_mm"],
            "condition": data["current"]["condition"]["text"],
            "icon": data["current"]["condition"]["icon"]
        }

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
        df["ds"] = pd.to_datetime(df["Years"], errors="coerce")
        df = df[["ds", "Temperature"]].rename(columns={"Temperature": "y"})  
        df.dropna(inplace=True)
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡 Live Weather", 
    "📊 Climate Forecast", 
    "📆 Monthly Predictions", 
    "📌 Yearly Predictions", 
    "⚠️ Extreme Weather Events"
])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌍 Live Weather Dashboard")
    city = st.text_input("Enter City for Live Data", "New York")

    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.success(f"✔️ Live weather for {city} fetched successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric("🌡 Temperature", f"{live_weather['y']}°C")
            col2.metric("💨 Wind Speed", f"{live_weather['wind_kph']} km/h")
            col3.metric("💧 Humidity", f"{live_weather['humidity']}%")

            st.image(f"https:{live_weather['icon']}", width=80)
            st.markdown(f"### {live_weather['condition']}")

            if df is not None:
                df = pd.concat([df, pd.DataFrame([live_weather])], ignore_index=True)

# ---- TAB 2: CLIMATE FORECAST (2025+) ----
with tab2:
    st.subheader("📈 AI Climate Forecast (April 2025–2030)")

    if df is not None and len(df) > 1:
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=2000)  # Extending beyond 2025
        forecast = model.predict(future)

        # 🔥 Show only predictions from April 2025 onward
        forecast_future = forecast[forecast["ds"] >= "2025-04-01"]

        fig = px.line(forecast_future, x="ds", y="yhat", title="Predicted Temperature Trends (2025+)")
        fig.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot"))
        fig.add_scatter(x=forecast_future["ds"], y=forecast_future["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot"))
        st.plotly_chart(fig)

    else:
        st.info("📂 Upload a CSV file to enable forecasting.")

# ---- TAB 3: MONTHLY PREDICTIONS ----
with tab3:
    st.subheader("📆 Monthly Climate Predictions (April 2025–2030)")

    if df is not None:
        model = Prophet()
        model.fit(df)

        future_5y = model.make_future_dataframe(periods=2000)
        forecast_5y = model.predict(future_5y)

        # 🔥 Ensure predictions start from April 2025
        forecast_future = forecast_5y[forecast_5y["ds"] >= "2025-04-01"]

        # Resample to Monthly Predictions
        future_monthly = forecast_future[["ds", "yhat"]].set_index("ds").resample("M").mean().reset_index()

        fig = px.line(future_monthly, x="ds", y="yhat", title="📊 Monthly Predicted Climate Trends (2025–2030)")
        st.plotly_chart(fig)

# ---- TAB 4: YEARLY PREDICTIONS ----
with tab4:
    st.subheader("📌 Yearly Climate Predictions (2025–2030)")

    if df is not None:
        # 🔥 Filter only future data (April 2025+)
        future_yearly = forecast_future[["ds", "yhat"]].set_index("ds").resample("Y").mean().reset_index()

        fig2 = px.bar(
            future_yearly, x="ds", y="yhat",
            title="🌍 Yearly Temperature Averages (2025–2030)",
            color="yhat", color_continuous_scale="thermal"
        )
        st.plotly_chart(fig2)

# ---- TAB 5: EXTREME WEATHER PREDICTIONS ----
with tab5:
    st.subheader("⚠️ Predicting Extreme Weather Events (2025+)")

    if df is not None:
        model = Prophet()
        model.add_seasonality(name="yearly", period=365, fourier_order=10)
        model.fit(df)

        future_extreme = model.make_future_dataframe(periods=2000)
        forecast_extreme = model.predict(future_extreme)

        # 🔥 Filter for extreme events
        high_risk = forecast_extreme[(forecast_extreme["yhat"] > forecast_extreme["yhat"].quantile(0.95)) & (forecast_extreme["ds"] >= "2025-04-01")]

        fig3 = px.scatter(high_risk, x="ds", y="yhat", color="yhat", title="⚠️ Predicted Extreme Weather Events (2025+)", color_continuous_scale="reds")
        st.plotly_chart(fig3)

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**", unsafe_allow_html=True)
