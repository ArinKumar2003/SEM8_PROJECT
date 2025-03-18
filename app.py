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

        return {
            "city": city,
            "ds": datetime.datetime.now(),
            "y": float(data["current"]["temp_c"]),  # Temperature in Celsius
            "humidity": data["current"]["humidity"],
            "wind_kph": data["current"]["wind_kph"],
            "precip_mm": data["current"]["precip_mm"]
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

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["🌡 Live Weather Data", "📊 Climate Data & Forecast", "📌 About"])

with tab1:
    st.subheader("🌍 Live Weather Conditions")
    
    city = st.text_input("Enter City for Live Data", "New York")
    live_weather = None

    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.success(f"✔️ Live weather for {city} fetched successfully!")
            
            # Display Data
            st.write(f"📍 Location: **{live_weather['city']}**")
            st.write(f"🌡 Temperature: **{live_weather['y']}°C**")
            st.write(f"💧 Humidity: **{live_weather['humidity']}%**")
            st.write(f"💨 Wind Speed: **{live_weather['wind_kph']} km/h**")
            st.write(f"🌧 Precipitation: **{live_weather['precip_mm']} mm**")
            
            # Visualizing Live Weather Data
            weather_df = pd.DataFrame({
                "Condition": ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (mm)"],
                "Value": [live_weather["y"], live_weather["humidity"], live_weather["wind_kph"], live_weather["precip_mm"]]
            })
            
            fig = px.bar(weather_df, x="Condition", y="Value", title="Live Weather Conditions", color="Condition")
            st.plotly_chart(fig)

        if df is not None:
            df = pd.concat([df, pd.DataFrame([live_weather])], ignore_index=True)

with tab2:
    st.subheader("📈 AI Climate Forecast")

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

            # Confidence Interval
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot")))

            fig.update_layout(title="Predicted Temperature Trends (Including Live Data)", xaxis_title="Year", yaxis_title="Temperature (°C)")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"❌ Forecasting error: {e}")
    elif df is not None:
        st.error("⚠️ Not enough data to train AI model.")
    else:
        st.info("📂 Upload a CSV file with climate data to enable forecasting.")

with tab3:
    st.subheader("📌 About This App")
    st.markdown("""
        Welcome to the **AI Climate Dashboard**! 🌍  
        This tool provides **real-time weather insights** and **AI-powered climate forecasts**.

        ### Features:
        - **Live Weather Data** 🌡  
          Fetch real-time **temperature, humidity, wind speed, and precipitation** for any city.
        - **Climate Forecasting** 📊  
          Upload historical **temperature data** and generate AI-based predictions.
        - **Interactive Visuals** 📈  
          View **weather charts** and **forecast trends** dynamically.
          
        ### How to Use:
        1. Go to the **Live Weather** tab and enter a city.
        2. Click **Fetch Live Weather** to view real-time data.
        3. Upload a **CSV file** (Years, Temperature) to get AI-powered forecasts.
        4. Check **graphs & trends** to understand climate patterns.
        
        🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**
    """)

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**", unsafe_allow_html=True)
