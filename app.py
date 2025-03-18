import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
from prophet import Prophet
import plotly.graph_objects as go

# ---- STREAMLIT CONFIG ----
st.set_page_config(page_title="🌍 AI Climate Dashboard", layout="wide")

# ---- TITLE & DESCRIPTION ----
st.title("🌍 AI Climate Dashboard - Live & Historical Forecasts")
st.markdown("Welcome to the **AI-Powered Climate Dashboard**. This tool integrates **historical climate data** with **real-time weather** to forecast future trends. 🚀")

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
            "Temperature": float(data["current"]["temp_c"]),
            "Humidity": data["current"]["humidity"],
            "CO2": None,  # Live CO₂ data not available, but can be manually integrated
            "SeaLevel": None,  # Live sea level data unavailable
            "Condition": data["current"]["condition"]["text"],
            "Icon": data["current"]["condition"]["icon"]
        }

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request failed: {e}")
        return None

# ---- SIDEBAR: HISTORICAL DATA UPLOAD ----
st.sidebar.header("📂 Upload Historical Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Years, Month, Day, CO2, Humidity, SeaLevel, Temperature)", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Ensure correct datetime format
        df["ds"] = pd.to_datetime(df[["Years", "Month", "Day"]])
        df = df[["ds", "CO2", "Humidity", "SeaLevel", "Temperature"]]
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

            # Display weather information with icons
            st.markdown(f"""
            <div style="display: flex; align-items: center;">
                <img src="{live_weather['Icon']}" width="50">
                <h3 style="margin-left: 10px;">{live_weather['Condition']}</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("🌡 Temperature", f"{live_weather['Temperature']}°C")
            col2.metric("💧 Humidity", f"{live_weather['Humidity']}%")
            col3.metric("🌎 CO₂ Levels", "Unavailable")  # Placeholder

            # Display interactive historical vs. live temperature trends
            if df is not None:
                df_live = pd.concat([df, pd.DataFrame([live_weather])], ignore_index=True)

                fig = px.line(df_live, x="ds", y="y", title="Live vs Historical Temperature Trends",
                              labels={"ds": "Date", "y": "Temperature (°C)"},
                              color_discrete_sequence=["blue"])
                fig.add_trace(go.Scatter(x=[live_weather["ds"]], y=[live_weather["Temperature"]],
                                         mode='markers+text', text=["Live Data"],
                                         marker=dict(color="red", size=10)))

                st.plotly_chart(fig)

# ---- TAB 2: CLIMATE TRENDS (1971–2025) ----
with tab2:
    st.subheader("📊 Historical Climate Trends (1971–2025)")

    if df is not None:
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=365*54)  # Extending to 2025
        forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat", title="Temperature Trends (1971–2025)", 
                      labels={"ds": "Year", "yhat": "Predicted Temperature (°C)"})
        fig.update_traces(mode='lines+markers', marker=dict(size=5))
        fig.update_layout(
            xaxis_rangeslider_visible=True, 
            hovermode="x unified"
        )
        st.plotly_chart(fig)

# ---- TAB 3: PREDICTIONS (2025–2035) ----
with tab3:
    st.subheader("📆 Climate Predictions for 2025–2035")

    if df is not None:
        future = model.make_future_dataframe(periods=365*10)
        forecast = model.predict(future)
        forecast_future = forecast[forecast["ds"] >= "2025-04-01"]

        fig = px.line(forecast_future, x="ds", y="yhat", title="📊 Climate Predictions (2025–2035)",
                      labels={"ds": "Year", "yhat": "Predicted Temperature (°C)"},
                      color_discrete_sequence=["red"])
        st.plotly_chart(fig)

# ---- TAB 4: YEARLY OUTLOOK ----
with tab4:
    st.subheader("📌 Yearly Climate Outlook (2025–2035)")
    if df is not None:
        future_yearly = forecast_future.set_index("ds").resample("Y").mean().reset_index()

        fig2 = px.bar(future_yearly, x="ds", y="yhat",
                      title="🌍 Yearly Temperature Averages (2025–2035)",
                      color="yhat", color_continuous_scale="thermal",
                      labels={"ds": "Year", "yhat": "Predicted Temperature (°C)"})
        st.plotly_chart(fig2)

# ---- TAB 5: EXTREME WEATHER ----
with tab5:
    st.subheader("⚠️ Extreme Weather Predictions")
    if df is not None:
        high_risk = forecast_future[forecast_future["yhat"] > forecast_future["yhat"].quantile(0.95)]
        fig3 = px.scatter(high_risk, x="ds", y="yhat", title="⚠️ Extreme Weather Events (2025+)",
                          color_continuous_scale="reds")
        st.plotly_chart(fig3)

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**", unsafe_allow_html=True)
