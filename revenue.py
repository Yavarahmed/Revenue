import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta


st.set_page_config(page_title="Revenue Insights", layout="wide")


st.title("🏨Revenue Insight")
st.caption("Bookings • Revenue • Occupancy • ADR • RevPAR • Forecasts")

# Sample Data Generator
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2024-03-31")
    hotels = ["Hotel Prime", "Hotel Elite", "Hotel Grand"]
    cities = ["Delhi", "Mumbai", "Bengaluru"]
    room_types = ["Standard", "Deluxe", "Suite"]
    data = []
    for d in dates:
        for h, c in zip(hotels, cities):
            for r in room_types:
                rooms_available = np.random.randint(30, 80)
                rooms_booked = np.random.randint(10, rooms_available)
                revenue = rooms_booked * np.random.randint(3000, 12000)
                data.append([d, h, c, r, rooms_booked, rooms_available, revenue])
    return pd.DataFrame(data, columns=[
        "Date", "Hotel_Name", "City", "Room_Type", 
        "Rooms_Booked", "Rooms_Available", "Revenue"
    ])

# Data Source
data_mode = st.radio("Choose Data Source", ["Use Sample Hotel Data", "Upload CSV"])
if data_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Hotel Booking CSV", type=["csv"])
    if not uploaded_file:
        st.stop()
    df = pd.read_csv(uploaded_file)
else:
    df = generate_sample_data()
    st.success("Sample hotel data loaded")


# Data Cleaning
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])


# Sidebar Filters
st.sidebar.header("🔍 Filters")
filtered_df = df.copy()

for col, label in [("Hotel_Name", "Hotel"), ("City", "City"), ("Room_Type", "Room Type")]:
    selected = st.sidebar.multiselect(label, sorted(df[col].unique()))
    if selected:
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

start_date = st.sidebar.date_input("Start Date", filtered_df["Date"].min().date())
end_date = st.sidebar.date_input("End Date", filtered_df["Date"].max().date())
if start_date > end_date:
    st.error("Start date must be before end date")
    st.stop()
filtered_df = filtered_df[
    (filtered_df["Date"] >= pd.to_datetime(start_date)) &
    (filtered_df["Date"] <= pd.to_datetime(end_date))
]

# Derived Metrics
filtered_df["Occupancy_Rate"] = (filtered_df["Rooms_Booked"]/filtered_df["Rooms_Available"])*100
filtered_df["ADR"] = filtered_df["Revenue"]/filtered_df["Rooms_Booked"]
filtered_df["RevPAR"] = filtered_df["Revenue"]/filtered_df["Rooms_Available"]
filtered_df["Day_Type"] = np.where(filtered_df["Date"].dt.weekday < 5, "Weekday", "Weekend")

# KPIs
st.subheader("📌 Hotel KPIs")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue", f"₹ {filtered_df['Revenue'].sum():,.0f}")
c2.metric("Occupancy Rate", f"{filtered_df['Occupancy_Rate'].mean():.1f}%")
c3.metric("ADR", f"₹ {filtered_df['ADR'].mean():,.0f}")
c4.metric("RevPAR", f"₹ {filtered_df['RevPAR'].mean():,.0f}")
c5.metric("Bookings", f"{filtered_df['Rooms_Booked'].sum()}")

# Visual Insights
st.subheader("📊 Visual Analytics")
col1, col2 = st.columns(2)

# Revenue by City
with col1:
    city_rev = filtered_df.groupby("City")["Revenue"].sum()
    fig = plt.figure()
    city_rev.plot(kind="bar")
    plt.title("Revenue by City")
    plt.ylabel("Revenue")
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# Revenue by Room Type
with col2:
    room_rev = filtered_df.groupby("Room_Type")["Revenue"].sum()
    fig = plt.figure()
    room_rev.plot(kind="bar")
    plt.title("Revenue by Room Type")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# Revenue Trend
st.markdown("**Revenue Trend Over Time**")
trend = filtered_df.groupby("Date")["Revenue"].sum()
fig = plt.figure(figsize=(10,4))
plt.plot(trend.index, trend.values)
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
st.pyplot(fig)

# Weekday vs Weekend
st.markdown("**Weekday vs Weekend Revenue**")
wd = filtered_df.groupby("Day_Type")["Revenue"].mean()
fig = plt.figure()
wd.plot(kind="bar")
plt.ylabel("Revenue")
plt.xticks(rotation=0, fontsize=10)
plt.tight_layout()
st.pyplot(fig)

# Forecast Function
def forecast_with_confidence(df, value_col, forecast_days=30, group_col=None):
    forecasts = []
    groups = df[group_col].unique() if group_col else [None]
    for g in groups:
        temp = df[df[group_col]==g] if g else df.copy()
        daily = temp.groupby("Date")[value_col].sum().sort_index()
        rolling_mean = daily.rolling(7).mean()
        std_dev = daily.rolling(7).std()
        last_date = daily.index.max()
        last_mean = rolling_mean.iloc[-1]
        last_std = std_dev.iloc[-1] if not np.isnan(std_dev.iloc[-1]) else daily.std()
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
        forecast_values = [last_mean]*forecast_days
        upper = [last_mean + last_std]*forecast_days
        lower = [last_mean - last_std]*forecast_days
        df_forecast = pd.DataFrame({
            "Date": future_dates,
            value_col + "_Forecast": forecast_values,
            "Upper": upper,
            "Lower": lower
        })
        if g:
            df_forecast[group_col] = g
        forecasts.append(df_forecast)
    return pd.concat(forecasts, ignore_index=True)


#Forecast Per Hotel

forecast_days = st.slider("Select Forecast Horizon (Days)", 7, 60, 30)
st.subheader("🔮 Forecast per Hotel")
hotels = filtered_df["Hotel_Name"].unique()
for hotel in hotels:
    st.markdown(f"**{hotel}**")
    hotel_df = filtered_df[filtered_df["Hotel_Name"]==hotel]

    rev_forecast = forecast_with_confidence(hotel_df, "Revenue", forecast_days)
    bookings_forecast = forecast_with_confidence(hotel_df, "Rooms_Booked", forecast_days)

    # Revenue chart
    fig = plt.figure(figsize=(10,4))
    daily_rev = hotel_df.groupby("Date")["Revenue"].sum()
    plt.plot(daily_rev.index, daily_rev.values, label="Actual Revenue")
    plt.plot(rev_forecast["Date"], rev_forecast["Revenue_Forecast"], linestyle=":", label="Forecast")
    plt.fill_between(rev_forecast["Date"], rev_forecast["Lower"], rev_forecast["Upper"], color='orange', alpha=0.2, label="Confidence Band")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # Bookings chart
    fig = plt.figure(figsize=(10,4))
    daily_bookings = hotel_df.groupby("Date")["Rooms_Booked"].sum()
    plt.plot(daily_bookings.index, daily_bookings.values, label="Actual Bookings")
    plt.plot(bookings_forecast["Date"], bookings_forecast["Rooms_Booked_Forecast"], linestyle=":", label="Forecast")
    plt.fill_between(bookings_forecast["Date"], bookings_forecast["Lower"], bookings_forecast["Upper"], color='green', alpha=0.2, label="Confidence Band")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Rooms Booked")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)


# Forecast Per City

st.subheader("🔮 Forecast per City")
cities = filtered_df["City"].unique()
for city in cities:
    st.markdown(f"**{city}**")
    city_df = filtered_df[filtered_df["City"]==city]

    rev_forecast = forecast_with_confidence(city_df, "Revenue", forecast_days)
    bookings_forecast = forecast_with_confidence(city_df, "Rooms_Booked", forecast_days)

    # Revenue chart
    fig = plt.figure(figsize=(10,4))
    daily_rev = city_df.groupby("Date")["Revenue"].sum()
    plt.plot(daily_rev.index, daily_rev.values, label="Actual Revenue")
    plt.plot(rev_forecast["Date"], rev_forecast["Revenue_Forecast"], linestyle=":", label="Forecast")
    plt.fill_between(rev_forecast["Date"], rev_forecast["Lower"], rev_forecast["Upper"], color='orange', alpha=0.2, label="Confidence Band")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # Bookings chart
    fig = plt.figure(figsize=(10,4))
    daily_bookings = city_df.groupby("Date")["Rooms_Booked"].sum()
    plt.plot(daily_bookings.index, daily_bookings.values, label="Actual Bookings")
    plt.plot(bookings_forecast["Date"], bookings_forecast["Rooms_Booked_Forecast"], linestyle=":", label="Forecast")
    plt.fill_between(bookings_forecast["Date"], bookings_forecast["Lower"], bookings_forecast["Upper"], color='green', alpha=0.2, label="Confidence Band")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Rooms Booked")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)


# Download Filtered Report
st.subheader("📥 Download Filtered Report")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Filtered Report (CSV)",
    csv,
    "hotel_revenue_report.csv",
    "text/csv"
)


st.subheader("💬 Ask the Assistant")
query = st.text_input("Ask: total revenue, occupancy, adr, revpar, best city, best room")

def hotel_chatbot(q, data):
    q = q.lower()
    if "total" in q:
        return f"Total revenue is ₹ {data['Revenue'].sum():,.0f}"
    if "occupancy" in q:
        return f"Average occupancy rate is {data['Occupancy_Rate'].mean():.1f}%"
    if "adr" in q:
        return f"Average ADR is ₹ {data['ADR'].mean():,.0f}"
    if "revpar" in q:
        return f"Average RevPAR is ₹ {data['RevPAR'].mean():,.0f}"
    if "best city" in q:
        city = data.groupby("City")["Revenue"].sum().idxmax()
        return f"Best performing city is {city}"
    if "best room" in q:
        room = data.groupby("Room_Type")["Revenue"].sum().idxmax()
        return f"Best performing room type is {room}"
    return "You can ask about revenue, occupancy, ADR, RevPAR, city or room type."

if query:
    st.info(hotel_chatbot(query, filtered_df))


st.markdown("---")
st.caption("Hotel Chain Revenue System ")
