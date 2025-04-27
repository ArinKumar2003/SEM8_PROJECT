import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns
from scipy.stats import zscore

# Page config
st.set_page_config(page_title="🌦️ Climate Forecast & Analysis", layout="wide")
st.title("🌦️ Climate Forecast & Analysis Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Live Weather", "📊 Climate Dataset", "📆 Predictions", "📊 Data Insights"])

# Shared df
df = None

# TAB 1: Live Weather (Placeholder)
with tab1:
    st.subheader("🌍 Live Weather")
    st.write("Coming soon...")

# TAB 2: Upload and Clean Dataset
with tab2:
    st.header("📊 Upload Climate Dataset")

    uploaded_file = st.file_uploader("Upload your climate dataset (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Try to parse 'Date.Full'
            st.subheader("📅 Cleaning 'Date.Full' column...")
            invalid_dates = []
            parsed_dates = []

            for i, val in enumerate(df["Date.Full"]):
                try:
                    parsed_dates.append(pd.to_datetime(val))
                except Exception as e:
                    invalid_dates.append((i, val))
                    parsed_dates.append(pd.NaT)

            df["Date"] = parsed_dates

            if invalid_dates:
                st.warning("⚠️ Some rows had invalid date formats and were set to NaT. Here are a few examples:") 
                st.code("\n".join([f"Row {i}: '{val}'" for i, val in invalid_dates[:5]]))
            else:
                st.success("✅ All dates parsed successfully!")

            df.dropna(subset=["Date"], inplace=True)

            st.success("✅ Dataset successfully loaded and cleaned!")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

# TAB 3: Predictions
with tab3:
    st.header("📆 Predict Temperature from Today")

    if df is not None:
        if df.empty:
            st.error("❌ DataFrame is empty after cleaning. Cannot generate predictions.")
        elif 'Data.Temperature.Avg Temp' not in df.columns:
            st.error("❌ 'Data.Temperature.Avg Temp' column not found in the dataset.")
        else:
            df = df.dropna(subset=['Date', 'Data.Temperature.Avg Temp'])
            df = df.sort_values("Date")

            if len(df) < 2:
                st.warning("⚠️ Not enough data to calculate trends. Need at least 2 rows.")
            else:
                df['Temp_Change'] = df['Data.Temperature.Avg Temp'].diff()
                avg_daily_change = df['Temp_Change'].mean()

                today_temp = df['Data.Temperature.Avg Temp'].iloc[-1]
                pred_tomorrow = today_temp + avg_daily_change
                pred_next_week = today_temp + (avg_daily_change * 7)

                st.metric("📌 Today's Temp", f"{today_temp:.2f} °C")
                st.markdown(f"📍 **Tomorrow**: `{pred_tomorrow:.2f} °C`")
                st.markdown(f"📍 **Next Week**: `{pred_next_week:.2f} °C`")

                # Forecast Table
                forecast_df = pd.DataFrame({
                    "Date": [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 8)],
                    "Predicted Avg Temp (°C)": [today_temp + (avg_daily_change * i) for i in range(1, 8)]
                })
                st.markdown("### 🔮 7-Day Forecast")
                st.dataframe(forecast_df)
    else:
        st.warning("📂 Please upload the dataset first in the previous tab.")

# TAB 4: Enhanced Insights (Visualizations and Data Analysis)
with tab4:
    st.header("📊 Data Insights")

    if df is not None:
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        st.write("This section provides the basic statistics for the climate data, such as mean, median, standard deviation.")
        summary_stats = df[['Data.Temperature.Avg Temp', 'Data.Precipitation']].describe().transpose()
        st.dataframe(summary_stats)

        # Trend Analysis: Plotting temperature trend over time
        with st.expander("📈 Temperature Trend Over Time"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(x="Date", y="Data.Temperature.Avg Temp", ax=ax, color='tab:red')
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("Temperature Trend Over Time")
            ax.grid(True)
            st.pyplot(fig)

        # Moving Average: Smoothing temperature data
        with st.expander("📉 7-Day Moving Average of Temperature"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Temp_MA7'] = df['Data.Temperature.Avg Temp'].rolling(window=7).mean()  # 7-day moving average
            df.plot(x="Date", y="Temp_MA7", ax=ax, color='tab:blue')
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("7-Day Moving Average of Temperature")
            ax.grid(True)
            st.pyplot(fig)

        # Precipitation Trend
        with st.expander("🌧️ Precipitation Trend Over Time"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(x="Date", y="Data.Precipitation", ax=ax, color='tab:green')
            ax.set_ylabel("Precipitation (mm)")
            ax.set_title("Precipitation Trend Over Time")
            ax.grid(True)
            st.pyplot(fig)

        # Correlation Analysis: Heatmap to understand relationships between variables
        with st.expander("🔍 Correlation Analysis"):
            corr_matrix = df[['Data.Temperature.Avg Temp', 'Data.Precipitation']].corr()
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation between Temperature and Precipitation")
            st.pyplot(fig)

        # Anomaly Detection (Outliers)
        with st.expander("🚨 Anomaly Detection (Outliers)"):
            # Calculate z-scores to detect outliers
            df['Temp_zscore'] = zscore(df['Data.Temperature.Avg Temp'])
            outliers = df[df['Temp_zscore'].abs() > 3]  # Z-score > 3 indicates an outlier

            if not outliers.empty:
                st.warning(f"⚠️ Found {len(outliers)} temperature outliers!")
                st.dataframe(outliers[['Date', 'Data.Temperature.Avg Temp', 'Temp_zscore']])
            else:
                st.success("✅ No significant temperature anomalies found.")

        # Seasonal Patterns: Grouping by Month and Year
        with st.expander("📅 Seasonal Patterns"):
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            seasonal_avg = df.groupby(['Year', 'Month'])['Data.Temperature.Avg Temp'].mean().unstack()

            fig, ax = plt.subplots(figsize=(10, 6))
            seasonal_avg.plot(ax=ax, marker='o', linestyle='-', color=['blue', 'green', 'red', 'purple', 'orange'])
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("Seasonal Temperature Patterns (Yearly Comparison)")
            ax.grid(True)
            st.pyplot(fig)

        # Rolling Averages: Moving averages over different windows
        with st.expander("📊 Rolling Averages (30-Day and 60-Day)"):
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Temp_MA30'] = df['Data.Temperature.Avg Temp'].rolling(window=30).mean()
            df['Temp_MA60'] = df['Data.Temperature.Avg Temp'].rolling(window=60).mean()
            df.plot(x="Date", y=["Temp_MA30", "Temp_MA60"], ax=ax)
            ax.set_ylabel("Avg Temp (°C)")
            ax.set_title("Rolling 30-Day and 60-Day Moving Averages of Temperature")
            ax.grid(True)
            st.pyplot(fig)

    else:
        st.warning("📂 Please upload the dataset to view insights.")
