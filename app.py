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
st.title("🌍 Climate Change Prediction Dashboard")
st.markdown("##### 📈 Analyze climate trends, visualize changes, and predict future conditions.")

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

# Sidebar: Help Section
st.sidebar.markdown("### ℹ️ How to Use:")
st.sidebar.info("Upload a CSV file with `Years`, `Month`, `Day`, `CO2`, `Humidity`, and `SeaLevel` columns. Select a model to get temperature predictions.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---- TABS FOR NAVIGATION ----
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predictions"])

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

            # Gradient Boosting Model
            if model_choice == "Gradient Boosting":
                predictions = gb_model.predict(X_new)

            # LSTM Model
            elif model_choice == "LSTM":
                X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                predictions = lstm_model.predict(X_new_lstm).flatten()

            # Prophet Model
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

            # Store predictions in DataFrame
            df["Predicted Temperature"] = predictions

            # Show Predictions
            st.write("### 🔥 Predictions")
            st.dataframe(df[["Years", "Predicted Temperature"]])

            # 📉 Prediction Visualization
            fig_pred = px.line(df, x="Years", y="Predicted Temperature", title="Predicted Temperature Trends")
            st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.warning("🚨 The dataset is missing required columns!")

    # 📥 ---- DOWNLOAD PREDICTIONS ----
    df.to_csv("predictions.csv", index=False)
    st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
