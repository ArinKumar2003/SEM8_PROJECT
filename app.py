import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ✅ Move `st.set_page_config()` to the very top
st.set_page_config(page_title="🌍 Climate Dashboard", layout="wide")

# Load ML Models with error handling
gb_model = None
lstm_model = None

try:
    gb_model = joblib.load("climate_gb_model.pkl")  # Gradient Boosting Model
    st.success("✅ Gradient Boosting Model Loaded Successfully!")
except FileNotFoundError:
    st.warning("🚨 Gradient Boosting Model file is missing!")

try:
    lstm_model = load_model("climate_lstm_model.keras")  # LSTM Model
    st.success("✅ LSTM Model Loaded Successfully!")
except (FileNotFoundError, OSError):
    st.warning("🚨 LSTM Model file is missing or corrupted!")

# Sidebar for Upload & Filters
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Model Selection
model_choice = st.sidebar.radio("🤖 Choose Prediction Model", ["Gradient Boosting", "LSTM"])

# Data Filters
st.sidebar.subheader("📅 Filter Data")
selected_year = st.sidebar.slider("Select Year", 1900, 2100, 2020)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # **TABS for Navigation**
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predictions"])

    # **📊 Data Overview Tab**
    with tab1:
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)

        st.write("### 📊 Data Summary")
        st.write(df.describe())

    # **📈 Interactive Visualization Tab**
    with tab2:
        st.write("### 📊 Climate Trends Over Time")

        # Select Feature to Visualize
        feature = st.selectbox("Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])

        # Filter Data
        df_filtered = df[df["Years"] == selected_year]

        # Plotly Line Chart
        fig = px.line(df, x="Years", y=feature, title=f"{feature} Trends Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # Histogram of Selected Feature
        fig_hist = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
        st.plotly_chart(fig_hist, use_container_width=True)

    # **🔮 Prediction Tab**
    with tab3:
        st.write("### 🔮 Predict Future Climate Conditions")
        required_features = ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel"]

        if all(col in df.columns for col in required_features):
            X_new = df[required_features]

            # Make Predictions (only if the selected model is loaded)
            if model_choice == "Gradient Boosting":
                if gb_model:
                    predictions = gb_model.predict(X_new)
                    df["Predicted Temperature"] = predictions
                else:
                    st.error("🚨 Gradient Boosting Model is missing! Upload the model file.")
            else:
                if lstm_model:
                    X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                    predictions = lstm_model.predict(X_new_lstm).flatten()
                    df["Predicted Temperature"] = predictions
                else:
                    st.error("🚨 LSTM Model is missing! Upload the model file.")

            # Show Predictions (only if predictions exist)
            if "Predicted Temperature" in df.columns:
                st.write("### 🔥 Predictions")
                st.dataframe(df[["Years", "Predicted Temperature"]])

                # **📉 Prediction Visualization**
                fig_pred = px.line(df, x="Years", y="Predicted Temperature", title="Predicted Temperature Trends")
                st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning("🚨 The dataset is missing required columns!")

    # **📥 Download Predictions**
    df.to_csv("predictions.csv", index=False)
    st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
