import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="BBBN Fraud System", layout="wide")

st.title("🧠 BBBN – Healthcare Fraud Intelligence")
st.caption("Behavior–Belief–Boosted Fraud Network")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH / "provider_with_decision.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
decision_filter = st.sidebar.multiselect(
    "Decision",
    options=df["Decision"].unique(),
    default=list(df["Decision"].unique())
)

filtered = df[df["Decision"].isin(decision_filter)]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Providers", len(filtered))
col2.metric("Flagged", (filtered["Decision"] == "FLAG").sum())
col3.metric("Blocked", (filtered["Decision"] == "BLOCK").sum())

st.divider()

# Table
st.subheader("Provider Risk Table")
st.dataframe(
    filtered.sort_values("fraud_probability", ascending=False),
    use_container_width=True
)

# Provider drill-down
st.subheader("🔍 Provider Detail")
pid = st.selectbox("Select Provider", filtered["Provider"].unique())

row = filtered[filtered["Provider"] == pid].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Fraud Probability", f"{row['fraud_probability']:.3f}")
c2.metric("Behavior Risk", f"{row['behavior_risk']:.3f}")
c3.metric("Anomaly Score", f"{row['anomaly_score']:.3f}")

st.success(f"Final Decision: {row['Decision']}")
