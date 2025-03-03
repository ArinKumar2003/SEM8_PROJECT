import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# 🔹 Set Streamlit Page Config (Must be First)
st.set_page_config(page_title="🌍 Climate Dashboard", layout="wide")

# 🔹 Add Dashboard Title & Description
st.title("🌍 Climate Change Prediction Dashboard")
st.markdown("""
#### 📊 Explore climate data trends and predict future conditions using Machine Learning models.  
✅ Upload your dataset 📂, select a prediction model 🤖, and visualize climate patterns over time!  
""")

# 🔹 Cache Model Loading to Improve Performance
@st.cache_resource
def load_models():
    """Loads ML models and caches them for performance."""
    gb_model, lstm_model = None, None

    try:
        gb_model = joblib.load("climate_gb_model.pkl")
      
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Gradient Boosting Model not found!")

    try:
        lstm_model = load_model("climate_lstm_model.keras")
        
    except (FileNotFoundError, OSError):
        st.sidebar.warning("⚠️ LSTM Model file is missing or corrupted!")

    return gb_model, lstm_model

gb_model, lstm_model = load_models()

# 🔹 Sidebar: File Upload & Filters
st.sidebar.header("📂 Upload Climate Data")
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file) if uploaded_file else None

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
df = load_data(uploaded_file)

# 🔹 Model Selection & Filters
model_choice = st.sidebar.radio("🤖 Choose Prediction Model", ["Gradient Boosting", "LSTM"])
st.sidebar.subheader("📅 Filter Data")
selected_year = st.sidebar.slider("Select Year", 1900, 2100, 2020)

# 🔹 Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predictions"])

# 📊 **Tab 1: Data Overview**
with tab1:
    if df is not None:
        st.write("### 📄 Uploaded Data")
        st.dataframe(df)
        st.write("### 📊 Data Summary")
        st.write(df.describe())
    else:
        st.info("📥 Upload a CSV file to see the data.")

# 📈 **Tab 2: Visualizations**
with tab2:
    if df is not None:
        st.write("### 📊 Climate Trends Over Time")
        
        feature = st.selectbox("📊 Select Feature", ["Temperature", "CO2", "Humidity", "SeaLevel"])
        df_filtered = df[df["Years"] == selected_year]

        # 📈 Line Chart
        fig = px.line(df, x="Years", y=feature, title=f"{feature} Trends Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # 📊 Histogram
        fig_hist = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("📥 Upload a CSV file to see visualizations.")

# 🔮 **Tab 3: Predictions**
with tab3:
    if df is not None:
        st.write("### 🔮 Predict Future Climate Conditions")

        required_features = ["Years", "Month", "Day", "CO2", "Humidity", "SeaLevel"]

        if all(col in df.columns for col in required_features):
            X_new = df[required_features]

            # Make Predictions
            predictions = None
            if model_choice == "Gradient Boosting" and gb_model:
                predictions = gb_model.predict(X_new)
            elif model_choice == "LSTM" and lstm_model:
                X_new_lstm = np.array(X_new).reshape((X_new.shape[0], X_new.shape[1], 1))
                predictions = lstm_model.predict(X_new_lstm).flatten()

            if predictions is not None:
                df["Predicted Temperature"] = predictions

                # 🔥 Display Predictions
                st.write("### 🔥 Predictions")
                st.dataframe(df[["Years", "Predicted Temperature"]])

                # 📈 Prediction Visualization
                fig_pred = px.line(df, x="Years", y="Predicted Temperature", title="Predicted Temperature Trends")
                st.plotly_chart(fig_pred, use_container_width=True)

                # 📥 Download Predictions
                st.sidebar.download_button("📥 Download Predictions", data=df.to_csv().encode("utf-8"),
                                           file_name="predictions.csv", mime="text/csv")
            else:
                st.warning("⚠️ Model not loaded. Ensure the model file is available.")
        else:
            st.warning("⚠️ The dataset is missing required columns!")
    else:
        st.info("📥 Upload a CSV file to generate predictions.")
