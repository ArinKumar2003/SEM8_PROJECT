import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# **Load Models with Error Handling**
try:
    gb_model = joblib.load("climate_gb_model.pkl")  # Gradient Boosting
except Exception as e:
    st.error(f"🚨 Error loading Gradient Boosting model: {e}")
    gb_model = None

try:
    lstm_model = load_model("climate_lstm_model.keras")  # LSTM
except Exception as e:
    st.error(f"🚨 Error loading LSTM model: {e}")
    lstm_model = None

# **Set up Streamlit UI**
st.set_page_config(page_title="🌍 Climate Dashboard", layout="wide")

# **Sidebar for File Upload**
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# **Sidebar Model Selection**
model_choice = st.sidebar.radio("🤖 Choose Prediction Model", ["Gradient Boosting", "LSTM"])

# **Sidebar Filters**
st.sidebar.subheader("📅 Filter Data")
selected_year = st.sidebar.slider("Select Year", 1900, 2100, 2020)

# **If a file is uploaded, process it**
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # **Ensure required columns exist**
    required_columns = ["Year", "Month", "Day", "CO2", "Humidity", "SeaLevel", "Temperature"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"🚨 Missing columns in dataset: {', '.join(missing_columns)}")
    else:
        # **Tabs for Navigation**
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Interactive Visualizations", "🔮 Predictions"])

        # **📊 Data Overview Tab**
        with tab1:
            st.write("### 📄 Uploaded Data")
            st.dataframe(df)

            st.write("### 📊 Data Summary")
            st.write(df.describe())

        # **📈 Interactive Visualization Tab**
        with tab2:
            st.write("### 📊 Climate Trends Over Time")

            # **Select Feature to Visualize**
            feature = st.selectbox("Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])
            
            # **Filter Data by Selected Year**
            df_filtered = df[df["Year"] == selected_year]

            # **Line Chart**
            fig = px.line(df, x="Year", y=feature, title=f"{feature} Trends Over Time", markers=True)
            st.plotly_chart(fig, use_container_width=True)

            # **Histogram**
            fig_hist = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
            st.plotly_chart(fig_hist, use_container_width=True)

        # **🔮 Prediction Tab**
        with tab3:
            st.write("### 🔮 Predict Future Climate Conditions")

            if gb_model and lstm_model:  # Ensure models are loaded
                X_new = df[["Year", "Month", "Day", "CO2", "Humidity", "SeaLevel"]]

                # **Make Predictions**
                if model_choice == "Gradient Boosting":
                    predictions = gb_model.predict(X_new)
                else:
                    # **Reshape data for LSTM**
                    X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                    predictions = lstm_model.predict(X_new_lstm).flatten()

                # **Add Predictions to DataFrame**
                df["Predicted Temperature"] = predictions

                # **Show Predictions**
                st.write("### 🔥 Predictions")
                st.dataframe(df[["Year", "Predicted Temperature"]])

                # **📉 Prediction Visualization**
                fig_pred = px.line(df, x="Year", y="Predicted Temperature", title="Predicted Temperature Trends")
                st.plotly_chart(fig_pred, use_container_width=True)

                # **📥 Download Predictions**
                df.to_csv("predictions.csv", index=False)
                st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"),
                                           file_name="predictions.csv", mime="text/csv")
            else:
                st.warning("🚨 Models are not loaded correctly. Check for missing files.")
