import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(page_title="📈 Climate Forecast App", layout="wide")
st.title("🌍 Climate Forecasting using Time Series (Prophet)")
st.markdown("Upload your dataset and forecast climate indicators like CO₂, humidity, sea level, or temperature.")

# File upload
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Combine date columns into datetime
    try:
        df['ds'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
    except Exception as e:
        st.error(f"❌ Error creating datetime column: {e}")
        st.stop()

    # Rename target columns
    df = df.rename(columns={
        'CO2': 'CO2',
        'Humidity': 'Humidity',
        'SeaLevel': 'SeaLevel',
        'Temperature': 'Temperature'
    })

    st.success("✅ Dataset successfully loaded and datetime created!")
    st.write(df.head())

    # Let user select a metric to forecast
    metric = st.selectbox("📊 Select a metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'])

    periods = st.slider("📆 Days to forecast into the future", min_value=7, max_value=365, value=30)

    # Prepare data for Prophet
    prophet_df = df[['ds', metric]].rename(columns={metric: 'y'})

    # Build and fit model
    model = Prophet()
    model.fit(prophet_df)

    # Make future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Show forecast plot
    st.subheader("📈 Forecast Plot")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)

    # Show forecast table
    st.subheader("📋 Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

else:
    st.info("👆 Upload a dataset with columns: Years, Month, Day, CO2, Humidity, SeaLevel, Temperature to begin.")
