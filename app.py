import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="🌍 Climate Change Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌍 Climate Change Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>📊 Analyze trends, visualize data, and predict future climate conditions.</h3>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_models():
    try:
        gb_model = joblib.load("climate_gb_model.pkl")
        lstm_model = load_model("climate_lstm_model.keras")
        return gb_model, lstm_model
    except Exception as e:
        st.error(f"🚨 Error loading models: {e}")
        return None, None

gb_model, lstm_model = load_models()

st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
model_choice = st.sidebar.radio("🤖 Choose Model", ["Gradient Boosting", "LSTM", "Prophet"])
selected_year = st.sidebar.slider("Select Year", 1900, 2100, 2020)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🔮 Predictions", "🛠️ Manual Input"])

    with tab1:
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)
        st.write("### 📊 Summary")
        st.write(df.describe())

    with tab2:
        st.write("### 📊 Climate Trends")
        feature = st.selectbox("Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])
        fig = px.line(df, x="Years", y=feature, title=f"{feature} Trends Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 🌡️ Climate Feature Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with tab3:
        st.write("### 🔮 Predict Climate Conditions")
        required_features = ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel"]

        if all(col in df.columns for col in required_features):
            X_new = df[required_features]

            if model_choice == "Gradient Boosting":
                predictions = gb_model.predict(X_new)
            elif model_choice == "LSTM":
                X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                predictions = lstm_model.predict(X_new_lstm).flatten()
            elif model_choice == "Prophet":
                df_prophet = df.rename(columns={"Years": "ds", "Temperature": "y"})
                prophet_model = Prophet()
                prophet_model.fit(df_prophet)
                future = prophet_model.make_future_dataframe(periods=365)
                forecast = prophet_model.predict(future)
                fig_forecast = px.line(forecast, x="ds", y="yhat", title="Prophet Forecasted Temperature")
                st.plotly_chart(fig_forecast, use_container_width=True)
                predictions = forecast["yhat"]

            df["Predicted Temperature"] = predictions
            st.write("### 🔥 Predictions")
            st.dataframe(df[["Years", "Predicted Temperature"]])
            fig_pred = px.line(df, x="Years", y="Predicted Temperature", title="Predicted Temperature Trends")
            st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.warning("🚨 Dataset missing required columns!")

    with tab4:
        st.write("### 🎛️ Manual Input Prediction")
        year_input = st.slider("Year", 1900, 2100, 2025)
        co2_input = st.number_input("CO2 Level (ppm)", min_value=200, max_value=600, value=400)
        humidity_input = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        sealevel_input = st.number_input("Sea Level Rise (mm)", min_value=0, max_value=500, value=100)

        manual_input = pd.DataFrame({
            "Years": [year_input], "CO2": [co2_input],
            "Humidity": [humidity_input], "SeaLevel": [sealevel_input]
        })

        if model_choice == "Gradient Boosting":
            manual_prediction = gb_model.predict(manual_input)[0]
        elif model_choice == "LSTM":
            manual_input_lstm = np.array(manual_input).reshape((1, manual_input.shape[1], 1))
            manual_prediction = lstm_model.predict(manual_input_lstm).flatten()[0]
        else:
            manual_prediction = None  # Prophet does not support manual input

        if manual_prediction is not None:
            st.metric(label="🌡️ Predicted Temperature (°C)", value=f"{manual_prediction:.2f}")
        else:
            st.warning("⚠️ Prophet does not support manual input predictions.")

    df.to_csv("predictions.csv", index=False)
    st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"), file_name="predictions.csv", mime="text/csv")
