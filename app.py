import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Climate Forecast App", layout="wide")

st.title("🌦️ Climate Forecasting Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your climate dataset", type=["csv"])
if uploaded_file:
    # Load and prepare data
    df = pd.read_csv(uploaded_file)
    df['ds'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
    df = df.sort_values("ds")

    st.success("✅ Data loaded successfully!")
    st.write(df.head())

    metric = st.selectbox(
        "📊 Choose metric to forecast",
        ["CO2", "Humidity", "SeaLevel", "Temperature"]
    )

    # Prophet expects "ds" and "y" columns
    df_prophet = df[["ds", metric]].rename(columns={metric: "y"})

    # Forecast
    period = st.slider("📅 Months to forecast", 1, 60, 24)
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=period, freq="M")
    forecast = model.predict(future)

    # Forecast plot
    st.subheader("📈 Forecast")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    # Forecast table
    st.subheader("📋 Forecast Data")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

    # Download forecast CSV
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Forecast Data",
        data=csv,
        file_name=f"{metric}_forecast.csv",
        mime='text/csv'
    )

    # Multi-metric trend plot
    st.subheader("📊 Historical Trends (All Metrics)")
    fig2 = px.line(df, x="ds", y=["CO2", "Humidity", "SeaLevel", "Temperature"])
    st.plotly_chart(fig2, use_container_width=True)

    # Seasonal decomposition
    st.subheader("🔍 Seasonal Decomposition")
    try:
        result = seasonal_decompose(df[metric], model='additive', period=12)
        st.line_chart(result.trend, height=150)
        st.line_chart(result.seasonal, height=150)
        st.line_chart(result.resid, height=150)
    except:
        st.warning("📉 Not enough data for seasonal decomposition.")
else:
    st.info("📤 Please upload your dataset to get started.")
