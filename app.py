import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set Streamlit config
st.set_page_config(page_title="🌦️ Climate Insight Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌎 Climate Insight Dashboard</h1>", unsafe_allow_html=True)

# Upload data
uploaded_file = st.sidebar.file_uploader("📤 Upload Climate Data CSV", type=["csv"])

# WeatherAPI Key
API_KEY = st.secrets["weatherapi"]["api_key"]

# ------------ WEATHERAPI -------------
def get_live_weather(city="New York"):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    res = requests.get(url).json()
    if "error" in res:
        return None
    current = res["current"]
    return {
        "city": res["location"]["name"],
        "region": res["location"]["region"],
        "country": res["location"]["country"],
        "temperature": current["temp_c"],
        "condition": current["condition"]["text"],
        "icon": current["condition"]["icon"],
        "humidity": current["humidity"],
        "wind_kph": current["wind_kph"]
    }

# -------------- LOAD DATA --------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    for col in df.columns:
        if 'date' in col.lower():
            df.rename(columns={col: 'ds'}, inplace=True)
            break
    else:
        st.error("❌ Could not find a datetime column (e.g. 'date').")
        st.stop()

    target_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['temp', 'target', 'value']):
            target_col = col
            break
    if not target_col:
        st.error("❌ Could not find a target value column (e.g. 'temperature').")
        st.stop()
    df.rename(columns={target_col: 'y'}, inplace=True)

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    return df

# -------------- TABS --------------
tabs = st.tabs(["🌤️ Live Weather", "📈 Forecast", "📊 Visualization", "🧠 Climate Summary & Awareness"])

# ------------ TAB 1: LIVE WEATHER ------------
with tabs[0]:
    st.subheader("🌤️ Real-Time Weather")
    city = st.text_input("Enter City", value="New York")
    if city:
        weather = get_live_weather(city)
        if weather:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(f"https:{weather['icon']}", width=80)
            with col2:
                st.markdown(f"### {weather['city']}, {weather['region']}, {weather['country']}")
                st.metric("Temperature (°C)", weather["temperature"])
                st.metric("Condition", weather["condition"])
                st.metric("Humidity (%)", weather["humidity"])
                st.metric("Wind Speed (kph)", weather["wind_kph"])
        else:
            st.warning("⚠️ Could not fetch weather data. Try another city.")

# ------------ LOAD & CLEAN DATA IF FILE EXISTS ------------
if uploaded_file:
    df = load_data(uploaded_file)

    # ------------ TAB 2: FORECAST ------------
    with tabs[1]:
        st.subheader("📈 Temperature Forecasting")
        period = st.slider("Forecast Days", 7, 90)
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

    # ------------ TAB 3: VISUALIZATION ------------
    with tabs[2]:
        st.subheader("📊 Data Visualization")
        st.line_chart(df.set_index("ds")["y"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name='Temperature', mode='lines'))
        st.plotly_chart(fig, use_container_width=True)

    # ------------ TAB 4: CLIMATE SUMMARY ------------
    with tabs[3]:
        st.subheader("🧠 Climate Summary & Awareness")
        st.markdown(f"""
        ### 🌍 Summary
        - Records: **{len(df)}**
        - Date Range: **{df["ds"].min().date()} to {df["ds"].max().date()}**
        - Average Value: **{df["y"].mean():.2f}**

        ### 🌿 Awareness Tips
        - 🌱 Reduce energy usage and switch to renewables.
        - 🚴‍♂️ Choose biking or public transport.
        - ♻️ Recycle and reduce plastic waste.
        - 🌲 Support tree planting efforts.
        - 📢 Educate others and advocate climate action.
        """)

else:
    with tabs[1]:
        st.info("📤 Please upload a dataset to see forecast.")
    with tabs[2]:
        st.info("📤 Please upload a dataset for visualizations.")
    with tabs[3]:
        st.info("📤 Please upload a dataset for summary insights.")
