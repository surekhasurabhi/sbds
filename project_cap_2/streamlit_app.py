import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="U.S. Rent Forecast Dashboard",
    layout="wide"
)

# -------------------------------
# Load data and model
# -------------------------------

df = pd.read_csv("df_model.csv")

# -------------------------------
# Dashboard Title
# -------------------------------
st.title("U.S. Rental Price Forecast Dashboard")
st.markdown(
    "Interactive dashboard for exploring historical and forecasted rent prices "
    "using a Linear Regression model with lag features trained on Zillow Rent Index (ZORI) data."
)

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Select Location")

city = st.sidebar.selectbox(
    "City",
    sorted(df["City"].dropna().unique())
)

filtered_metros = df[df["City"] == city]["Metro"].dropna().unique()

metro = st.sidebar.selectbox(
    "Metro Area",
    sorted(filtered_metros)
)

# Filter combined
city_df = df[(df["City"] == city) & (df["Metro"] == metro)].sort_values("Date").copy()

# -------------------------------
# Historical plot
# -------------------------------
st.subheader(f"Historical Rent Trends — {city}, {metro}")

fig_hist = px.line(
    city_df,
    x="Date",
    y="RentPrice",
    labels={"RentPrice": "Median Rent ($)", "Date": "Date"}
)

st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------
# Forecasting section
# -------------------------------
st.subheader("12-Month Rent Forecast")

future_df_all = pd.read_csv("future_df_all.csv")

future_df = future_df_all[
    (future_df_all["City"] == city) &
    (future_df_all["Metro"] == metro)
]


# Combine his + forecast
hist_plot_df = city_df[["Date", "RentPrice"]].rename(columns={"RentPrice": "Rent"})
hist_plot_df["Type"] = "Historical"

forecast_plot_df = future_df[["Date", "Predicted_Rent"]].rename(columns={"Predicted_Rent": "Rent"})
forecast_plot_df["Type"] = "Forecast"

plot_df = pd.concat([hist_plot_df, forecast_plot_df])

fig_forecast = px.line(
    plot_df,
    x="Date",
    y="Rent",
    color="Type",
    labels={"Rent": "Median Rent ($)", "Date": "Date"},
    title=f"Historical vs Forecasted Rent Prices — {city}, {metro}"
)

st.plotly_chart(fig_forecast, use_container_width=True)

# -------------------------------
# Key Metrics
# -------------------------------
st.subheader("Key Metrics")

col1, col2 = st.columns(2)

# Latest actual rent
latest_actual = hist_plot_df.iloc[-1]["Rent"]

col1.metric(
    "Latest Actual Rent",
    f"${latest_actual:,.0f}"
)

# First forecasted value (next month)
next_month_forecast = forecast_plot_df.iloc[0]["Rent"]

col2.metric(
    "Next Month Forecast",
    f"${next_month_forecast:,.0f}",
    delta=f"{next_month_forecast - latest_actual:,.0f}"
)
