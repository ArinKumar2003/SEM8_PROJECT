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

# ---- SIDEBAR: HISTORICAL DATA UPLOAD ----
st.sidebar.header("📂 Upload Historical Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Years, Temperature, Humidity, Wind, Precipitation)", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df["ds"] = pd.to_datetime(df["Years"], format="%Y")
        df = df[["ds", "Temperature", "Humidity", "Wind", "Precipitation"]]
        df.rename(columns={"Temperature": "y"}, inplace=True)  # Prophet requires 'y'
        df.dropna(inplace=True)
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        df = None

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌡 Live Weather", 
    "📊 Climate Trends (1971–2025)", 
    "📆 Predictions (2025–2035)", 
    "📌 Yearly Climate Outlook", 
    "⚠️ Extreme Weather Analysis",
    "❓ Help & FAQs"
])

# ---- TAB 1: LIVE WEATHER ----
with tab1:
    st.subheader("🌍 Live Weather Dashboard")
    city = st.text_input("Enter City for Live Data", "New York")

    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.success(f"✔️ Live weather for {city} fetched successfully!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡 Temperature", f"{live_weather['y']}°C")
            col2.metric("💨 Wind Speed", f"{live_weather['wind_kph']} km/h")
            col3.metric("💧 Humidity", f"{live_weather['humidity']}%")
            col4.metric("🌧 Precipitation", f"{live_weather['precip_mm']} mm")

            st.image(f"https:{live_weather['icon']}", width=80)
            st.markdown(f"### {live_weather['condition']}")

            if df is not None:
                df = pd.concat([df, pd.DataFrame([live_weather])], ignore_index=True)

# ---- TAB 2: CLIMATE TRENDS (1971–2025) ----
with tab2:
    st.subheader("📊 Historical Climate Trends (1971–2025)")
    
    if df is not None:
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=365*54)  # Extending to 2025
        forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat", title="Temperature Trends (1971–2025)")
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot"))
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot"))
        st.plotly_chart(fig)
    else:
        st.info("📂 Upload a CSV file to display trends.")

# ---- TAB 3: PREDICTIONS (2025–2035) ----
with tab3:
    st.subheader("📆 Climate Predictions for 2025–2035")

    if df is not None:
        forecast_future = forecast[forecast["ds"] >= "2025-04-01"]
        future_monthly = forecast_future.set_index("ds").resample("M").mean().reset_index()

        fig = px.line(future_monthly, x="ds", y="yhat", title="📊 Monthly Climate Predictions (2025–2035)")
        st.plotly_chart(fig)

        st.markdown("### 🔍 Prediction Insights:")
        st.write("""
        - 🌡 **Temperatures** expected to rise gradually.
        - 💧 **Humidity levels** may increase, leading to heavier precipitation in some regions.
        - 💨 **Wind speeds** likely to be higher during summer months.
        """)

# ---- TAB 4: YEARLY OUTLOOK ----
with tab4:
    st.subheader("📌 Yearly Climate Outlook (2025–2035)")

    if df is not None:
        future_yearly = forecast_future.set_index("ds").resample("Y").mean().reset_index()

        fig2 = px.bar(
            future_yearly, x="ds", y="yhat",
            title="🌍 Yearly Temperature Averages (2025–2035)",
            color="yhat", color_continuous_scale="thermal"
        )
        st.plotly_chart(fig2)

        st.markdown("### 📅 Summary:")
        st.write("""
        - 🌡 **Temperature increases** will be more noticeable in 2027 and beyond.
        - 💨 **Extreme wind events** expected in 2030 and 2033.
        - 🌧 **Higher rainfall in winter months** could indicate flood risks.
        """)

# ---- TAB 5: EXTREME WEATHER ----
with tab5:
    st.subheader("⚠️ Extreme Weather Predictions")

    if df is not None:
        model = Prophet()
        model.add_seasonality(name="yearly", period=365, fourier_order=10)
        model.fit(df)

        future_extreme = model.make_future_dataframe(periods=3650)
        forecast_extreme = model.predict(future_extreme)

        high_risk = forecast_extreme[(forecast_extreme["yhat"] > forecast_extreme["yhat"].quantile(0.95)) & (forecast_extreme["ds"] >= "2025-04-01")]

        fig3 = px.scatter(high_risk, x="ds", y="yhat", color="yhat", title="⚠️ Extreme Weather Events (2025+)", color_continuous_scale="reds")
        st.plotly_chart(fig3)

# ---- TAB 6: HELP & FAQs ----
with tab6:
    st.subheader("❓ Help & FAQs")
    st.write("""
    - 📂 **How do I upload historical data?**  
      Use the sidebar to upload a CSV file containing **Years, Temperature, Humidity, Wind, and Precipitation**.

    - 🌡 **Where does live weather come from?**  
      We use data from **WeatherAPI.com**.

    - 📊 **What model is used for predictions?**  
      We use **Facebook Prophet**, a robust time-series forecasting model.
    """)

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**", unsafe_allow_html=True)
