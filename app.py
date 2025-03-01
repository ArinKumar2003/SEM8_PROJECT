%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests

# =======================
# 🚀 Load Models
# =======================
gb_model = joblib.load("climate_gb_model.pkl")  # Gradient Boosting
lstm_model = load_model("climate_lstm_model.keras")  # LSTM
xgb_model = joblib.load("climate_xgb_model.pkl")  # XGBoost

# =======================
# 🌍 Streamlit Page Config
# =======================
st.set_page_config(page_title="🌍 Climate Prediction Dashboard", layout="wide")

# =======================
# 📂 Sidebar: File Upload & Filters
# =======================
st.sidebar.header("📂 Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
model_choice = st.sidebar.multiselect("🤖 Select Models", ["Gradient Boosting", "LSTM", "XGBoost"])
selected_year = st.sidebar.slider("📅 Select Year", 1900, 2100, 2020)

# =======================
# 🌡️ Live Climate Data API (OpenWeather)
# =======================
st.sidebar.header("🌡️ Live Climate Data")
city = st.sidebar.text_input("Enter City", "New York")
API_KEY = "your_api_key"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
response = requests.get(url).json()

if "main" in response:
    temperature = response["main"]["temp"]
    humidity = response["main"]["humidity"]
    st.sidebar.metric(f"🌡️ Temp in {city}", f"{temperature}°C")
    st.sidebar.metric(f"💧 Humidity", f"{humidity}%")

# =======================
# 🏠 Page Navigation
# =======================
page = st.sidebar.radio("📍 Navigation", ["Home", "Data Overview", "Visualizations", "Predictions"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # =======================
    # 📄 Data Overview
    # =======================
    if page == "Data Overview":
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)
        st.write("### 📊 Data Summary")
        st.write(df.describe())

    # =======================
    # 📊 Interactive Visualizations
    # =======================
    elif page == "Visualizations":
        st.write("### 📈 Climate Trends Over Time")
        
        feature = st.selectbox("📊 Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])
        df_filtered = df[df["Years"] == selected_year]

        # 📉 Line Chart
        fig = px.line(df, x="Years", y=feature, title=f"{feature} Trends", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # 🔥 Histogram
        fig_hist = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
        st.plotly_chart(fig_hist, use_container_width=True)

        # 📍 Scatter Plot
        fig_scatter = px.scatter(df, x="CO2", y="Temperature", color="Years", title="CO2 vs Temperature")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 🔥 Correlation Heatmap
        st.write("### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # =======================
    # 🔮 Predictions
    # =======================
    elif page == "Predictions":
        st.write("### 🔮 Predict Future Climate Conditions")
        required_features = ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel"]

        if all(col in df.columns for col in required_features):
            X_new = df[required_features]

            # Store predictions in a dictionary
            predictions_dict = {}

            # ✅ Gradient Boosting
            if "Gradient Boosting" in model_choice:
                predictions_dict["Gradient Boosting"] = gb_model.predict(X_new)

            # ✅ XGBoost
            if "XGBoost" in model_choice:
                predictions_dict["XGBoost"] = xgb_model.predict(X_new)

            # ✅ LSTM (Reshape for RNN)
            if "LSTM" in model_choice:
                X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                predictions_dict["LSTM"] = lstm_model.predict(X_new_lstm).flatten()

            # Add Predictions to DataFrame
            for model, preds in predictions_dict.items():
                df[f"{model} Predicted Temp"] = preds

            # 📊 Show Predictions
            st.write("### 🔥 Predictions")
            st.dataframe(df[["Years"] + [f"{model} Predicted Temp" for model in model_choice]])

            # 📈 Prediction Visualization
            fig_pred = px.line(df, x="Years", y=[f"{model} Predicted Temp" for model in model_choice], title="Predicted Temperature Trends")
            st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.warning("🚨 The dataset is missing required columns!")

        # 📥 Download Predictions
        df.to_csv("predictions.csv", index=False)
        st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"), file_name="climate_predictions.csv", mime="text/csv")

# =======================
# 🏠 Home Page
# =======================
elif page == "Home":
    st.title("🌍 Climate Change Prediction Dashboard")
    st.write("Upload climate data to generate predictions!")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Climate_Change_Graph.png", use_column_width=True)

# =======================
# 🚀 Deployment Notes
# =======================
st.sidebar.info("**Next Steps:**\n✅ Add Deep Learning Models\n✅ Enhance UI/UX\n✅ Deploy to Production")

