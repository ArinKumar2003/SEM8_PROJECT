import pandas as pd
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="🌦️ Climate Forecast & Analysis Dashboard")

st.title("🌦️ Climate Forecast & Analysis Dashboard")
st.markdown("Upload your climate dataset with time-based entries in the 'Date' column.")

# Function to convert 'mm:ss.s' or 'HH:MM.S' format to datetime, allowing over-24 hours
def convert_time_column(df, time_col='Date'):
    def parse_time_string(t):
        try:
            # Split the time string into minutes and seconds
            parts = str(t).split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                total_seconds = minutes * 60 + seconds
                return timedelta(seconds=total_seconds)
            else:
                return None
        except Exception:
            return None

    df['Timedelta'] = df[time_col].apply(parse_time_string)

    # Set base date
    base_date = pd.to_datetime('2025-01-01')
    df['Date'] = base_date + df['Timedelta']

    # Report any failed conversions
    failed = df[df['Timedelta'].isnull()]
    if not failed.empty:
        st.warning("⚠️ Some rows couldn't be parsed and have been set to NaT:")
        st.write(failed[[time_col]])

    # Drop helper column
    df.drop(columns=['Timedelta'], inplace=True)
    return df

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your climate dataset (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty or df.columns.isnull().all():
            st.error("❌ The file is empty or doesn't contain valid column headers.")
        else:
            # Clean 'Date' column
            if 'Date' in df.columns:
                df = convert_time_column(df, time_col='Date')
                st.success("✅ Dataset successfully loaded and 'Date' column corrected!")

                # Show preview
                st.subheader("📄 Dataset Preview")
                st.dataframe(df.head())
            else:
                st.error("❌ The dataset doesn't contain a 'Date' column.")
    except pd.errors.EmptyDataError:
        st.error("❌ The file is empty. Please upload a valid dataset.")
    except pd.errors.ParserError:
        st.error("❌ There was an error parsing the CSV file. Check if it's formatted correctly.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")
