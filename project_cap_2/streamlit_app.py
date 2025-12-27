import streamlit as st
import pandas as pd
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
df = pd.read_csv("zillow_rent_cleaned.csv")
model = joblib.load("linreg_rent_model.pkl")

# -------------------------------
# Date handling 
# -------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# -------------------------------
# App title and description
# -------------------------------
st.title("U.S. Rental Price Forecast Dashboard")
st.markdown(
    "Interactive dashboard for exploring historical and forecasted rent prices "
    "using a Linear Regression model trained on Zillow Rent Index (ZORI) data."
)

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Select Location")

city = st.sidebar.selectbox(
    "City",
    sorted(df["City"].dropna().unique())
)

city_df = df[df["City"] == city].sort_values("Date").copy()

# -------------------------------
# Historical plot
# -------------------------------
st.subheader(f"Historical Rent Trends – {city}")

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

last_date = city_df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=12,
    freq="MS"
)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Year": future_dates.year,
    "Month": future_dates.month,
    "Population Rank": [city_df["Population Rank"].median()] * 12,
    "State": [city_df["State"].iloc[0]] * 12,
    "Metro": [city_df["Metro"].iloc[0]] * 12
})

# Model prediction
future_df["Predicted_Rent"] = model.predict(
    future_df[["Year", "Month", "Population Rank", "State", "Metro"]]
)

# -------------------------------
# Combine historical + forecast for plotting
# -------------------------------
hist_plot_df = city_df[["Date", "RentPrice"]].rename(
    columns={"RentPrice": "Rent"}
)
hist_plot_df["Type"] = "Historical"

forecast_plot_df = future_df[["Date", "Predicted_Rent"]].rename(
    columns={"Predicted_Rent": "Rent"}
)
forecast_plot_df["Type"] = "Forecast"

plot_df = pd.concat([hist_plot_df, forecast_plot_df])

fig_forecast = px.line(
    plot_df,
    x="Date",
    y="Rent",
    color="Type",
    labels={"Rent": "Median Rent ($)", "Date": "Date"},
    title=f"Historical vs Forecasted Rent Prices – {city}"
)

st.plotly_chart(fig_forecast, use_container_width=True)

# -------------------------------
# Key metrics
# -------------------------------
st.subheader("Key Metrics")

col1, col2 = st.columns(2)

col1.metric(
    "Latest Actual Rent",
    f"${hist_plot_df.iloc[-1]['Rent']:,.0f}"
)

col2.metric(
    "Next Month Forecast",
    f"${forecast_plot_df.iloc[0]['Rent']:,.0f}"
)