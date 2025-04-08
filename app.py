import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="Climate Insights Dashboard", layout="wide")

# --- Secrets ---
API_KEY = st.secrets["WEATHERAPI_KEY"]

# --- Title ---
st.markdown("# 🌍 Climate Insights Dashboard")
st.markdown("Real-time weather, forecast visualization, and climate awareness.")

# --- WeatherAPI Live Weather Fetch ---
def get_live_weather(city="New York"):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        if "location" not in data or "current" not in data:
            st.error("⚠️ Unexpected response format from WeatherAPI.")
            st.json(data)
            return None

        current = data["current"]
        location = data["location"]

        return {
            "city": location.get("name", city),
            "region": location.get("region", ""),
            "country": location.get("country", ""),
            "temperature": current.get("temp_c", "N/A"),
            "condition": current.get("condition", {}).get("text", "Unknown"),
            "icon": current.get("condition", {}).get("icon", ""),
            "humidity": current.get("humidity", "N/A"),
            "wind_kph": current.get("wind_kph", "N/A")
        }

    except Exception as e:
        st.error(f"⚠️ Could not fetch weather data: {e}")
        return None

# --- Load Climate Dataset from Upload ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# --- File Upload ---
st.sidebar.header("📁 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("📂 Please upload a dataset to continue.")
    st.stop()

# --- Tabs ---
tabs = st.tabs(["🌤️ Live Weather", "📈 Forecast", "📊 Visualizations", "📚 Climate Summary & Awareness"])

# --- Live Weather Tab ---
with tabs[0]:
    st.subheader("Real-Time Weather")
    city = st.text_input("Enter a city", value="New York")
    if city:
        weather = get_live_weather(city)
        if weather:
            col1, col2 = st.columns([1, 2])
            with col1:
                if weather["icon"]:
                    st.image(f"https:{weather['icon']}", width=80)
            with col2:
                st.markdown(f"### {weather['city']}, {weather['region']}, {weather['country']}")
                st.metric("🌡️ Temperature (°C)", weather["temperature"])
                st.metric("☁️ Condition", weather["condition"])
                st.metric("💧 Humidity", f"{weather['humidity']}%")
                st.metric("💨 Wind Speed", f"{weather['wind_kph']} kph")

# --- Forecast Tab ---
with tabs[1]:
    st.subheader("📈 Forecast Trends")
    if 'date' in df.columns:
        metric = st.selectbox("Select metric to visualize", options=[col for col in df.columns if col != 'date'])
        st.line_chart(df.set_index("date")[metric])
    else:
        st.error("❌ The dataset must contain a 'date' column.")

# --- Visualizations Tab ---
with tabs[2]:
    st.subheader("📊 Climate Visualizations")
    st.markdown("Explore your dataset with automatic charts.")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", options=numeric_cols)
        y = st.selectbox("Y-axis", options=numeric_cols, index=1)
        st.scatter_chart(df[[x, y]])
    else:
        st.warning("📉 Not enough numeric data for scatter plot.")

# --- Climate Summary & Awareness Tab ---
with tabs[3]:
    st.subheader("📚 Climate Summary & Awareness")
    st.markdown("""
    Climate change is impacting weather patterns globally. This dashboard helps visualize trends and raise awareness.

    #### Things to Consider:
    - 🔥 Rising global temperatures
    - 🌊 Sea level rise
    - 🌪️ Increased frequency of extreme events
    - 🌾 Impact on agriculture and biodiversity

    #### What You Can Do:
    - Reduce your carbon footprint 🌱
    - Support sustainable practices ♻️
    - Stay informed and educate others 📢
    """)

    st.success("🌟 Knowledge is power. Let's use data to inspire climate action!")

