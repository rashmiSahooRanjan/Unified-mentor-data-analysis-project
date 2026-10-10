import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Customer Intelligence Dashboard", page_icon="📊", layout="wide")
ROOT = Path(__file__).parent
DATA = ROOT / "data"

@st.cache_data
def load_data():
    transactions = pd.read_csv(DATA / "Nassau_Candy_Distributor.csv")
    customers = pd.read_csv(DATA / "Final_Interpreted_Customer_Segments.csv")
    profiles = pd.read_csv(DATA / "Final_Customer_Segment_Profiles.csv")
    evaluation = pd.read_csv(DATA / "Final_Model_Evaluation.csv")
    # Keep dates as supplied; shipping dates are known to be anomalous.
    customers["Cluster"] = customers["Cluster"].astype(str)
    profiles["Cluster"] = profiles["Cluster"].astype(str)
    return transactions, customers, profiles, evaluation

try:
    transactions, customers, profiles, evaluation = load_data()
except Exception as exc:
    st.error(f"Unable to load dashboard data from the data/ folder: {exc}")
    st.stop()

st.title("Customer Intelligence Dashboard")
st.caption("Nassau Candy Distributor | Customer segmentation and purchasing-behavior analytics")
st.info("Scope note: this is candy-distribution customer analysis, not real-estate buyer or investment profiling. Shipping dates are unverified and are excluded from analysis.")

# Sidebar filters
st.sidebar.header("Filters")
cluster_values = sorted(customers["Cluster"].dropna().unique(), key=lambda x: int(x))
region_col = "Main_Region"
division_col = "Preferred_Division"
ship_col = "Preferred_Ship_Mode"
regions = sorted(customers[region_col].dropna().astype(str).unique())
divisions = sorted(customers[division_col].dropna().astype(str).unique())
ship_modes = sorted(customers[ship_col].dropna().astype(str).unique())

selected_clusters = st.sidebar.multiselect("K-Means cluster", cluster_values, default=cluster_values)
selected_regions = st.sidebar.multiselect("Customer's main region", regions, default=regions)
selected_divisions = st.sidebar.multiselect("Preferred division", divisions, default=divisions)
selected_ship_modes = st.sidebar.multiselect("Preferred ship mode", ship_modes, default=ship_modes)

filtered = customers[
    customers["Cluster"].isin(selected_clusters)
    & customers[region_col].astype(str).isin(selected_regions)
    & customers[division_col].astype(str).isin(selected_divisions)
    & customers[ship_col].astype(str).isin(selected_ship_modes)
].copy()

if filtered.empty:
    st.warning("No customers match the current filters. Change one or more filters.")
    st.stop()

# KPIs based on selected customers
n_customers = filtered["Customer ID"].nunique()
sales = filtered["Total_Sales"].sum()
profit = filtered["Total_Profit"].sum()
units = filtered["Total_Units"].sum()
margin = (profit / sales * 100) if sales else 0
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{n_customers:,}")
k2.metric("Customer sales total", f"${sales:,.2f}")
k3.metric("Gross profit total", f"${profit:,.2f}")
k4.metric("Units", f"{units:,.0f}")
k5.metric("Gross profit margin", f"{margin:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Customer segments", "Geography", "Product preferences", "Model evaluation"
])

with tab1:
    st.subheader("Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Transaction rows", f"{len(transactions):,}")
    b.metric("Unique orders", f"{transactions['Order ID'].nunique():,}")
    c.metric("Distinct products", f"{transactions['Product ID'].nunique():,}")
    d.metric("Recorded total sales", f"${transactions['Sales'].sum():,.2f}")
    monthly = transactions.copy()
    monthly["Order Date Parsed"] = pd.to_datetime(monthly["Order Date"], format="%d-%m-%Y", errors="coerce")
    monthly = monthly.dropna(subset=["Order Date Parsed"])
    monthly["Month"] = monthly["Order Date Parsed"].dt.to_period("M").astype(str)
    trend = monthly.groupby("Month", as_index=False).agg(Sales=("Sales", "sum"), Gross_Profit=("Gross Profit", "sum"))
    fig = px.line(trend, x="Month", y="Sales", markers=True, title="Monthly sales by recorded order date")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Trend is descriptive of this file's recorded order dates; it is not a forecast.")

with tab2:
    st.subheader("Cluster distribution and contribution")
    cluster_summary = filtered.groupby("Cluster", as_index=False).agg(
        Customers=("Customer ID", "nunique"),
        Total_Sales=("Total_Sales", "sum"),
        Total_Profit=("Total_Profit", "sum"),
        Mean_Sales_Per_Customer=("Total_Sales", "mean"),
        Median_Sales_Per_Customer=("Total_Sales", "median"),
        Mean_Order_Count=("Order_Count", "mean"),
        Mean_Product_Count=("Product_Count", "mean")
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(cluster_summary, x="Cluster", y="Customers", color="Cluster", title="Customers per cluster"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(cluster_summary, x="Cluster", y="Total_Sales", color="Cluster", title="Sales contribution by cluster"), use_container_width=True)
    st.dataframe(cluster_summary.round(2), use_container_width=True, hide_index=True)
    feat = ["Total_Sales", "Total_Profit", "Total_Units", "Order_Count", "Product_Count", "Average_Sales_Per_Order"]
    rel = filtered.groupby("Cluster")[feat].mean()
    rel = rel / filtered[feat].mean().replace(0, np.nan)
    st.plotly_chart(px.imshow(rel, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=0, zmax=2, title="Cluster mean relative to selected-customer mean"), use_container_width=True)

with tab3:
    st.subheader("Geographic analysis")
    region_summary = filtered.groupby(region_col, as_index=False).agg(
        Customers=("Customer ID", "nunique"),
        Sales=("Total_Sales", "sum"),
        Gross_Profit=("Total_Profit", "sum")
    ).sort_values("Sales", ascending=False)
    left, right = st.columns(2)
    left.plotly_chart(px.bar(region_summary, x=region_col, y="Sales", title="Customer sales by main region"), use_container_width=True)
    right.plotly_chart(px.bar(region_summary, x=region_col, y="Customers", title="Customers by main region"), use_container_width=True)
    st.dataframe(region_summary.round(2), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Product preference analysis")
    division_summary = filtered.groupby(division_col, as_index=False).agg(
        Customers=("Customer ID", "nunique"),
        Sales=("Total_Sales", "sum"),
        Gross_Profit=("Total_Profit", "sum")
    ).sort_values("Sales", ascending=False)
    st.plotly_chart(px.bar(division_summary, x=division_col, y="Sales", color=division_col, title="Sales by most common customer division"), use_container_width=True)
    st.dataframe(division_summary.round(2), use_container_width=True, hide_index=True)
    st.caption("Preferred division is each customer's modal division; it is not an exclusive purchase category.")

with tab5:
    st.subheader("Clustering model evaluation")
    st.caption("Silhouette scores in the source evaluation file are sample estimates; other metrics were computed on the full dataset.")
    selected_k = st.selectbox("Cluster count (K)", sorted(evaluation["K"].unique()), index=list(sorted(evaluation["K"].unique())).index(4))
    comp = evaluation[evaluation["K"] == selected_k]
    left, right = st.columns(2)
    left.plotly_chart(px.bar(comp, x="Model", y="Silhouette", color="Model", title=f"Silhouette score at K={selected_k}"), use_container_width=True)
    right.plotly_chart(px.bar(comp, x="Model", y="Davies_Bouldin", color="Model", title=f"Davies–Bouldin index at K={selected_k}"), use_container_width=True)
    st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
    st.caption("K-Means with K=4 is the working interpretation candidate in this project; the selection is a practical trade-off, not a universal optimum.")

st.subheader("Customer-level records")
show_cols = [
    "Customer ID", "Cluster", "Total_Sales", "Total_Profit", "Total_Units",
    "Order_Count", "Product_Count", "Main_Region", "Preferred_Division", "Preferred_Ship_Mode"
]
st.dataframe(filtered[show_cols].sort_values("Total_Sales", ascending=False), use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered customer segments (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_customer_segments.csv",
    mime="text/csv"
)
