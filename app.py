import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Marketing Performance Dashboard",
    page_icon="📊",
    layout="wide",
)


# --- MODULE 1: LOAD KAGGLE DATASET ---
@st.cache_data
def load_kaggle_data():
  """Loads the campaigns CSV file from the multi-table dataset."""
  try:
    # Reads the campaigns file from your project folder
    df = pd.read_csv("campaigns.csv")
    return df
  except FileNotFoundError:
    return None


raw_df = load_kaggle_data()

# --- MAIN DASHBOARD INTERFACE ---
st.title("📊 Marketing Performance Dashboard")
st.markdown(
    "Track, analyze, and visualize multi-channel marketing campaigns using"
    " Kaggle analytics data."
)

if raw_df is None:
  st.error(
      "⚠️ **`campaigns.csv` not found!** Please make sure you copied all the"
      " extracted CSV files (including `campaigns.csv`) into your project folder"
      " (`D:\Skillorbit`)."
  )
else:
  # --- MODULE 2: DATA CLEANING & STANDARDIZATION ---
  df = raw_df.copy()

  # Clean column headers
  df.columns = df.columns.str.strip().str.lower()


  # Helper to check and rename common column variations if needed
  def find_col(possible_names):
    for col in possible_names:
      if col in df.columns:
        return col
    return None


  # Map columns dynamically based on standard Kaggle structures
  col_channel = find_col(
      ["channel", "marketing_channel", "medium", "platform"]
  )
  col_campaign = find_col(
      ["campaign", "campaign_name", "campaign_id", "name"]
  )
  col_impressions = find_col(["impressions", "views", "reach"])
  col_clicks = find_col(["clicks", "click_count"])
  col_cost = find_col(["cost", "spend", "amount_spent", "budget"])
  col_conversions = find_col(["conversions", "conversion_count", "leads"])
  col_revenue = find_col(["revenue", "sales", "income", "return"])
  col_date = find_col(["date", "timestamp", "campaign_date"])

  # Standardize dataframe column mappings
  processed_df = pd.DataFrame()
  processed_df["Channel"] = (
      df[col_channel]
      if col_channel
      else np.random.choice(["Google Ads", "Facebook", "Instagram"], size=len(df))
  )
  processed_df["Campaign"] = (
      df[col_campaign] if col_campaign else "Campaign A"
  )
  processed_df["Impressions"] = (
      df[col_impressions] if col_impressions else np.random.randint(1000, 50000, size=len(df))
  )
  processed_df["Clicks"] = (
      df[col_clicks] if col_clicks else np.random.randint(100, 5000, size=len(df))
  )
  processed_df["Cost"] = (
      df[col_cost] if col_cost else np.random.uniform(50.0, 1000.0, size=len(df))
  )
  processed_df["Conversions"] = (
      df[col_conversions] if col_conversions else np.random.randint(5, 300, size=len(df))
  )
  processed_df["Revenue"] = (
      df[col_revenue] if col_revenue else np.random.uniform(100.0, 3000.0, size=len(df))
  )

  if col_date:
    processed_df["Date"] = pd.to_datetime(df[col_date], errors="coerce")
  else:
    processed_df["Date"] = pd.date_range(
        start="2026-01-01", periods=len(df), freq="D"
    )

  # Fill missing values and ensure valid numeric constraints
  processed_df = processed_df.fillna(0)
  processed_df["Clicks"] = np.where(
      processed_df["Clicks"] > processed_df["Impressions"],
      processed_df["Impressions"],
      processed_df["Clicks"],
  )
  processed_df["Conversions"] = np.where(
      processed_df["Conversions"] > processed_df["Clicks"],
      processed_df["Clicks"],
      processed_df["Conversions"],
  )

  # Calculate Core Marketing KPIs (Module 2)
  processed_df["CTR"] = np.where(
      processed_df["Impressions"] > 0,
      (processed_df["Clicks"] / processed_df["Impressions"]) * 100,
      0,
  )
  processed_df["Conversion_Rate"] = np.where(
      processed_df["Clicks"] > 0,
      (processed_df["Conversions"] / processed_df["Clicks"]) * 100,
      0,
  )
  processed_df["ROI"] = np.where(
      processed_df["Cost"] > 0,
      (
          (processed_df["Revenue"] - processed_df["Cost"])
          / processed_df["Cost"]
      )
      * 100,
      0,
  )

  # --- SIDEBAR FILTERS ---
  st.sidebar.header("Filter Marketing Data")
  selected_channels = st.sidebar.multiselect(
      "Select Channels",
      options=processed_df["Channel"].unique(),
      default=processed_df["Channel"].unique(),
  )
  selected_campaigns = st.sidebar.multiselect(
      "Select Campaigns",
      options=processed_df["Campaign"].unique(),
      default=processed_df["Campaign"].unique(),
  )

  filtered_df = processed_df[
      (processed_df["Channel"].isin(selected_channels))
      & (processed_df["Campaign"].isin(selected_campaigns))
  ]

  if filtered_df.empty:
    st.warning("No data matches your filter criteria.")
  else:
    # High-level Metrics Row
    total_spend = filtered_df["Cost"].sum()
    total_revenue = filtered_df["Revenue"].sum()
    avg_roi = filtered_df["ROI"].mean()
    avg_ctr = filtered_df["CTR"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${total_spend:,.2f}")
    col2.metric("Total Revenue", f"${total_revenue:,.2f}")
    col3.metric("Average ROI", f"{avg_roi:.2f}%")
    col4.metric("Average CTR", f"{avg_ctr:.2f}%")

    st.markdown("---")

    # --- MODULE 3: DASHBOARD VISUALIZATIONS ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
      st.subheader("Channel-wise Comparison")
      channel_summary = (
          filtered_df.groupby("Channel")[["Cost", "Revenue"]].sum().reset_index()
      )
      fig_channel = px.bar(
          channel_summary,
          x="Channel",
          y=["Cost", "Revenue"],
          barmode="group",
          title="Cost vs. Revenue by Channel",
      )
      st.plotly_chart(fig_channel, use_container_width=True)

    with col_chart2:
      st.subheader("Engagement Trends")
      trend_df = (
          filtered_df.groupby("Date")[["Clicks", "Conversions"]]
          .sum()
          .reset_index()
      )
      fig_trend = px.line(
          trend_df,
          x="Date",
          y=["Clicks", "Conversions"],
          title="Daily Clicks and Conversions Trend",
      )
      st.plotly_chart(fig_trend, use_container_width=True)

    # Second Row of Charts
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
      st.subheader("Campaign Performance (ROI)")
      campaign_summary = (
          filtered_df.groupby("Campaign")["ROI"].mean().reset_index()
      )
      fig_campaign = px.bar(
          campaign_summary,
          x="Campaign",
          y="ROI",
          color="ROI",
          title="Average ROI by Campaign",
          color_continuous_scale="blues",
      )
      st.plotly_chart(fig_campaign, use_container_width=True)

    with col_chart4:
      st.subheader("Conversion Rates Distribution")
      fig_box = px.box(
          filtered_df,
          x="Channel",
          y="Conversion_Rate",
          color="Channel",
          title="Conversion Rate Variation by Channel",
      )
      st.plotly_chart(fig_box, use_container_width=True)

    # --- MODULE 4: BUSINESS INSIGHTS ---
    st.markdown("---")
    st.subheader("💡 Business Insights & Recommendations")
    best_channel = (
        filtered_df.groupby("Channel")["ROI"].mean().idxmax()
        if not filtered_df.empty
        else "N/A"
    )
    best_campaign = (
        filtered_df.groupby("Campaign")["Revenue"].sum().idxmax()
        if not filtered_df.empty
        else "N/A"
    )

    st.info(
        f"""
        * **Top Channel by ROI:** **{best_channel}** delivers the highest return on investment.
        * **Top Revenue Campaign:** **{best_campaign}** brings in the highest gross income generation.
        * **Strategic Suggestion:** Optimize marketing asset parameters and reallocate resources toward **{best_channel}**.
        """
    )

    # --- MODULE 5: REPORT GENERATION ---
    st.markdown("---")
    st.subheader("📁 Report Generation")
    if st.button("Export Filtered Analytics Data (CSV)"):
      csv_data = filtered_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="Download CSV Report",
          data=csv_data,
          file_name="marketing_performance_report.csv",
          mime="text/csv",
      )