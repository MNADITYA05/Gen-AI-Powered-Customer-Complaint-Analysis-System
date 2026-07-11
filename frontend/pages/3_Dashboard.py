"""
Dashboard page — analytics and trends across all complaints.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.utils.auth_guard import require_auth
from frontend.utils.api_client import list_complaints

require_auth()

st.title("📊 Dashboard")
st.caption("Analytics and trends across all complaints.")

# ── Load data ─────────────────────────────────────────────────────────────────

data = list_complaints(limit=500)
items = data.get("items", [])
total = data.get("total", 0)

if not items:
    st.info("No complaints in the database yet. Submit some from the **Submit Complaint** page.")
    st.stop()

df = pd.DataFrame(items)
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["date"] = df["created_at"].dt.date

# ── KPI row ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Complaints", total)
c2.metric("Open Cases",    (df["status"] == "open").sum()        if "status" in df.columns else "—")
c3.metric("Resolved",      (df["status"] == "resolved").sum()    if "status" in df.columns else "—")
c4.metric("Critical",      (df["severity"] == "critical").sum()  if "severity" in df.columns else "—")

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("By Category")
    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig = px.pie(cat_counts, names="Category", values="Count", hole=0.4)
    fig.update_layout(margin=dict(t=20, b=20, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("By Severity")
    sev_order = ["low", "medium", "high", "critical"]
    sev_counts = df["severity"].value_counts().reindex(sev_order, fill_value=0).reset_index()
    sev_counts.columns = ["Severity", "Count"]
    colors = {"low": "#2ecc71", "medium": "#f1c40f", "high": "#e67e22", "critical": "#e74c3c"}
    fig2 = px.bar(sev_counts, x="Severity", y="Count",
                  color="Severity", color_discrete_map=colors)
    fig2.update_layout(margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("By Status")
    if "status" in df.columns:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig3 = px.bar(status_counts, x="Status", y="Count", color="Status")
        fig3.update_layout(margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

with col_right2:
    st.subheader("Complaints Over Time")
    if "date" in df.columns:
        time_df = df.groupby("date").size().reset_index(name="Count")
        fig4 = px.line(time_df, x="date", y="Count", markers=True)
        fig4.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig4, use_container_width=True)

# ── Emotion breakdown ──────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Emotion Distribution")
if "emotion" in df.columns:
    emo_counts = df["emotion"].value_counts().reset_index()
    emo_counts.columns = ["Emotion", "Count"]
    fig5 = px.bar(emo_counts, x="Count", y="Emotion", orientation="h")
    fig5.update_layout(margin=dict(t=20, b=20), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig5, use_container_width=True)

# ── Raw data ───────────────────────────────────────────────────────────────────

with st.expander("Raw data table"):
    show_cols = [c for c in ["complaint_text", "category", "severity", "emotion", "status", "source", "created_at"] if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True)
