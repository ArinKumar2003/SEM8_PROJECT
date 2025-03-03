import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from prophet import Prophet

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(page_title="🌍 Climate Change Dashboard", layout="wide")

# ---- DASHBOARD HEADER ----
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌍 Climate Change Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>📊 Analyze trends, visualize data, and predict future climate conditions.</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---- CACHING MODEL LOAD ----
@st.cache_resource
def load_models():
    try:
        gb_model = joblib.load("climate_gb_model.pkl")  # Gradient Boosting Model
        lstm_model = load_model("climate_lstm_model.keras")  # LSTM Model
        return gb_model, lstm_model
    except Exception as e:
        st.error(f"🚨 Error loading models: {e}")
        return None, None

gb_model, lstm_model = load_models()

# ---- SIDEBAR ----
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Sidebar: Model Selection
model_choice = st.sidebar.radio("🤖 Choose Prediction Model", ["Gradient Boosting", "LSTM", "Prophet"])

# Sidebar: Filter Data
st.sidebar.subheader("📅 Filter Data")
selected_year = st.sidebar.slider("Select Year", 1900, 2100, 2020)

# Sidebar: Manual Prediction Input (only for GB & LSTM)
if model_choice != "Prophet":
    st.sidebar.markdown("### 🔢 Manual Input for Prediction")
    year_input = st.sidebar.slider("Year", 1900, 2100, 2025)
    month_input = st.sidebar.slider("Month", 1, 12, 6)
    day_input = st.sidebar.slider("Day", 1, 31, 15)
    co2_input = st.sidebar.number_input("CO2 Level (ppm)", min_value=200, max_value=600, value=400)
    humidity_input = st.sidebar.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    sealevel_input = st.sidebar.number_input("Sea Level Rise (mm)", min_value=0, max_value=500, value=100)

    manual_input = pd.DataFrame({
        "Years": [year_input],
        "Month": [month_input],
        "Day": [day_input],
        "CO2": [co2_input],
        "Humidity": [humidity_input],
        "SeaLevel": [sealevel_input]
    })

# Sidebar: Help Section
st.sidebar.markdown("### ℹ️ How to Use:")
st.sidebar.info("Upload a CSV file with `Years`, `Month`, `Day`, `CO2`, `Humidity`, and `SeaLevel` columns. Select a model to predict temperature.")

# ---- MAIN CONTENT ----
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---- TABS FOR NAVIGATION ----
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predictions", "🛠️ Manual Prediction"])

    # 📊 ---- DATA OVERVIEW ----
    with tab1:
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)

        st.write("### 📊 Data Summary")
        st.write(df.describe())

    # 📈 ---- VISUALIZATION ----
    with tab2:
        st.write("### 📊 Climate Trends Over Time")

        # Select Feature to Visualize
        feature = st.selectbox("Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])
        
        # Filter Data
        df_filtered = df[df["Years"] == selected_year]

        # Interactive Line Chart
        fig = px.line(df, x="Years", y=feature, title=f"{feature} Trends Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # Histogram of Selected Feature
        fig_hist = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
        st.plotly_chart(fig_hist, use_container_width=True)

    # 🔮 ---- PREDICTIONS ----
    with tab3:
        st.write("### 🔮 Predict Future Climate Conditions")
        required_features = ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel"]

        if all(col in df.columns for col in required_features):
            X_new = df[required_features]

            predictions = None  # Default value

            if model_choice == "Gradient Boosting":
                predictions = gb_model.predict(X_new)

            elif model_choice == "LSTM":
                X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                predictions = lstm_model.predict(X_new_lstm).flatten()

            elif model_choice == "Prophet":
                st.write("### 📈 Prophet Forecasting")
                df_prophet = df.rename(columns={"Years": "ds", "Temperature": "y"})
                prophet_model = Prophet()
                prophet_model.fit(df_prophet)
                future = prophet_model.make_future_dataframe(periods=365)
                forecast = prophet_model.predict(future)
                predictions = forecast["yhat"]

                # Prophet Forecast Visualization
                fig_forecast = px.line(forecast, x="ds", y="yhat", title="Prophet Forecasted Temperature")
                st.plotly_chart(fig_forecast, use_container_width=True)

            if predictions is not None and model_choice != "Prophet":
                df["Predicted Temperature"] = predictions
                st.write("### 🔥 Predictions")
                st.dataframe(df[["Years", "Predicted Temperature"]])

                # 📉 Prediction Visualization
                fig_pred = px.line(df, x="Years", y="Predicted Temperature", title="Predicted Temperature Trends")
                st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.warning("🚨 The dataset is missing required columns!")

    # 🛠️ ---- MANUAL PREDICTION ----
    with tab4:
        st.write("### 🎛️ Predict Temperature from Manual Inputs")

        if model_choice == "Prophet":
            st.warning("⚠️ Prophet model does not support manual input predictions. Please use Gradient Boosting or LSTM.")
        else:
            try:
                manual_prediction = None

                if model_choice == "Gradient Boosting":
                    manual_prediction = gb_model.predict(manual_input)[0]
                elif model_choice == "LSTM":
                    manual_input_lstm = np.array(manual_input).reshape((1, manual_input.shape[1], 1))
                    manual_prediction = lstm_model.predict(manual_input_lstm).flatten()[0]

                if isinstance(manual_prediction, (int, float)):
                    st.metric(label="🌡️ Predicted Temperature (°C)", value=f"{manual_prediction:.2f}")
                else:
                    st.warning("⚠️ No valid prediction was generated.")

            except Exception as e:
                st.error(f"🚨 Error in manual prediction: {e}")

    # 📥 ---- DOWNLOAD PREDICTIONS ----
    df.to_csv("predictions.csv", index=False)
    st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
