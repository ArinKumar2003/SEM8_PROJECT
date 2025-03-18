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
            "precip_mm": data["current"]["precip_mm"],
            "feels_like": data["current"]["feelslike_c"],
            "uv_index": data["current"]["uv"],
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
tab1, tab2, tab3 = st.tabs(["🌡 Live Weather", "📊 Climate Forecast", "📌 Future Climate Insights"])

with tab1:
    st.subheader("🌍 Live Weather Dashboard")
    
    city = st.text_input("Enter City for Live Data", "New York")
    live_weather = None

    if st.button("Fetch Live Weather"):
        live_weather = get_live_weather(city)
        if live_weather:
            st.success(f"✔️ Live weather for {city} fetched successfully!")

            # Display Live Weather Conditions in a Card Layout
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                st.metric(label="🌡 Temperature", value=f"{live_weather['y']}°C", delta=f"Feels Like {live_weather['feels_like']}°C")

            with col2:
                st.metric(label="💨 Wind Speed", value=f"{live_weather['wind_kph']} km/h")

            with col3:
                st.metric(label="💧 Humidity", value=f"{live_weather['humidity']}%")

            col4, col5 = st.columns([2, 2])

            with col4:
                st.metric(label="🌧 Precipitation", value=f"{live_weather['precip_mm']} mm")

            with col5:
                st.metric(label="🌞 UV Index", value=live_weather["uv_index"])

            # Display Weather Condition Icon & Text
            st.image(f"https:{live_weather['icon']}", width=100)
            st.markdown(f"### {live_weather['condition']}")

            if df is not None:
                df = pd.concat([df, pd.DataFrame([live_weather])], ignore_index=True)

with tab2:
    st.subheader("📈 AI Climate Forecast")

    if df is not None and len(df) > 1:
        try:
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=1825)  # Predict next 5 years
            forecast = model.predict(future)

            # Forecast Visualization
            fig = px.line(forecast, x="ds", y="yhat", title="Predicted Temperature Trends (Including Live Data)")
            fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound")
            fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound")

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"❌ Forecasting error: {e}")
    else:
        st.info("📂 Upload a CSV file with climate data to enable forecasting.")

with tab3:
    st.subheader("📌 Future Climate Insights (2025–2030)")
    
    if df is not None:
        # Forecasting for next 5 years (until 2030)
        future_5y = model.make_future_dataframe(periods=1825)  # 5 years * 365 days
        forecast_5y = model.predict(future_5y)

        # Filtering only future dates after March 2025
        future_5y = forecast_5y[forecast_5y["ds"] > "2025-03-01"]
        future_5y["ds"] = pd.to_datetime(future_5y["ds"])
        future_5y.set_index("ds", inplace=True)

        # Select only numeric columns for resampling
        numeric_cols = future_5y.select_dtypes(include=["number"]).columns
        future_monthly = future_5y[numeric_cols].resample("M").mean().reset_index()

        st.write("### 🔮 Climate Predictions from April 2025 Onwards:")

        for _, row in future_monthly.iterrows():
            temp = round(row["yhat"], 2)
            date = row["ds"].strftime("%B %Y")  # Format: "April 2025"

            # AI-based climate condition descriptions
            description = "🌡 Moderate Climate" if 10 <= temp <= 30 else "🔥 Extreme Heat" if temp > 30 else "❄️ Cold Weather"

            st.markdown(f"**{date}**: {temp}°C - {description}")

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 **Developed by AI Climate Team | Powered by WeatherAPI & Streamlit**", unsafe_allow_html=True)
