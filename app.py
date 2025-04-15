import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from statsmodels.tsa.seasonal import seasonal_decompose

# Set up the page configuration
st.set_page_config(page_title="Climate Forecast App", layout="centered")

# Title of the app
st.title("🌍 Climate Forecasting and Weather Insights")

# File uploader for user to upload the 'climate_large_data_sorted.csv'
uploaded_file = st.file_uploader("Upload your climate dataset (climate_large_data_sorted.csv)", type=["csv"])

# If a file is uploaded, load and process the data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset successfully loaded!")

    # Check for the necessary columns in the uploaded data
    if "Years" in df.columns and "Month" in df.columns and "Day" in df.columns:
        # Combine Year, Month, Day to create a Date column
        df['Date'] = pd.to_datetime(df[['Years', 'Month', 'Day']])
    else:
        st.error("Uploaded data must contain 'Years', 'Month', and 'Day' columns to create a Date.")

    # Use radio buttons to navigate between tabs
    option = st.radio("Select Section", ["📊 Forecasting", "📈 Historical Trends"])

    # --- Forecasting Section ---
    if option == "📊 Forecasting":
        st.header("📊 Forecast Future Climate Data")

        # Select metric for forecasting
        metric = st.selectbox("Select metric to forecast", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'])

        # Prepare data for forecasting
        if metric in df.columns:
            data = df[['Date', metric]].rename(columns={"Date": "ds", metric: "y"})

            # Create and fit the Prophet model
            model = Prophet()
            model.fit(data)
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            st.subheader("Forecast Plot")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            st.subheader("Forecast Data")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Download button for forecast data
            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Forecast", data=csv, file_name='forecast.csv', mime='text/csv')
        else:
            st.error(f"'{metric}' not found in uploaded data.")

    # --- Historical Trends Section ---
    elif option == "📈 Historical Trends":
        st.header("📈 Visualize Historical Trends")

        if 'Date' in df.columns:
            # Select metric for historical trend visualization
            metric = st.selectbox("Select metric for trend analysis", ['CO2', 'Humidity', 'SeaLevel', 'Temperature'], key='trend')

            if metric in df.columns:
                # Plot historical trends using Plotly
                fig2 = px.line(df, x='Date', y=metric, title=f"Historical {metric} Trends")
                st.plotly_chart(fig2)

                # Seasonal decomposition of the selected metric
                st.subheader("📉 Seasonal Decomposition")
                result = seasonal_decompose(df[metric], period=12, model='additive')
                st.line_chart(result.trend)
                st.line_chart(result.seasonal)
                st.line_chart(result.resid)
            else:
                st.error(f"'{metric}' not found in uploaded data.")
        else:
            st.error("Uploaded data does not contain a valid 'Date' column.")
else:
    st.write("Please upload your `climate_large_data_sorted.csv` to get started.")
