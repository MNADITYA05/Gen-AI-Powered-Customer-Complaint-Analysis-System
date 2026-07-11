"""
Cases page — view, filter and manage complaint cases.
"""
import streamlit as st
import pandas as pd
from frontend.utils.auth_guard import require_auth
from frontend.utils.api_client import (
    list_complaints, update_complaint_status,
    get_similar_complaints, rebuild_rag_index,
)

require_auth()

st.title("📋 Cases")
st.caption("Browse and manage all complaint cases.")

# ── Filters ───────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
with col1:
    filter_category = st.selectbox(
        "Category", ["All", "ATM_FAILURE", "FRAUD_DETECTION", "UX_ISSUES"]
    )
with col2:
    filter_severity = st.selectbox("Severity", ["All", "low", "medium", "high", "critical"])
with col3:
    filter_status = st.selectbox("Status", ["All", "open", "in_progress", "resolved", "closed"])

# ── Fetch ─────────────────────────────────────────────────────────────────────

data = list_complaints(
    limit=200,
    category=None if filter_category == "All" else filter_category,
    severity=None if filter_severity == "All" else filter_severity,
    status=None if filter_status == "All" else filter_status,
)

items = data.get("items", [])
total = data.get("total", 0)

st.markdown(f"**{total}** complaints found &nbsp; (showing {len(items)})")

if not items:
    st.info("No complaints match your filters.")
    st.stop()

# ── Summary badges ────────────────────────────────────────────────────────────

df = pd.DataFrame(items)

status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Open",        status_counts.get("open", 0))
c2.metric("In Progress", status_counts.get("in_progress", 0))
c3.metric("Resolved",    status_counts.get("resolved", 0))
c4.metric("Closed",      status_counts.get("closed", 0))

st.markdown("---")

# ── Table ─────────────────────────────────────────────────────────────────────

SEVERITY_ICON = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
STATUS_ICON   = {"open": "🔵", "in_progress": "🟡", "resolved": "✅", "closed": "⬛"}

display_df = df[["id", "complaint_text", "category", "severity", "emotion", "status", "source", "created_at"]].copy()
display_df["severity"] = display_df["severity"].apply(lambda s: f"{SEVERITY_ICON.get(s,'')} {s}")
display_df["status"]   = display_df["status"].apply(lambda s: f"{STATUS_ICON.get(s,'')} {s}")
display_df["complaint_text"] = display_df["complaint_text"].str[:120] + "…"
display_df.columns = ["ID", "Complaint", "Category", "Severity", "Emotion", "Status", "Source", "Created"]

st.dataframe(display_df.drop(columns=["ID"]), use_container_width=True, height=380)

# ── Case detail & update ──────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Update a case")

complaint_ids = [item["id"] for item in items]
selected_id = st.selectbox(
    "Select complaint to update",
    complaint_ids,
    format_func=lambda cid: next(
        (f"[{i['status'].upper()}] {i['complaint_text'][:80]}…" for i in items if i["id"] == cid), cid
    ),
)

selected = next((i for i in items if i["id"] == selected_id), None)
if selected:
    st.markdown(f"**Full text:** {selected['complaint_text']}")
    st.markdown(
        f"Category: `{selected['category']}` &nbsp; Severity: `{selected['severity']}` &nbsp; "
        f"Emotion: `{selected['emotion']}` &nbsp; Source: `{selected.get('source','—')}`"
    )

    with st.form("update_form"):
        new_status = st.selectbox(
            "New status",
            ["open", "in_progress", "resolved", "closed"],
            index=["open", "in_progress", "resolved", "closed"].index(
                selected.get("status", "open") if selected.get("status") in
                ["open", "in_progress", "resolved", "closed"] else "open"
            ),
        )
        note = st.text_area("Add a note (optional)", placeholder="e.g. Escalated to fraud team.")
        update_btn = st.form_submit_button("Update", type="primary")

    if update_btn:
        result = update_complaint_status(selected_id, new_status, note)
        if result:
            st.success(f"Case updated to **{new_status}**.")
            st.rerun()

# ── Similar Cases (RAG) ───────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔍 Similar Cases")
st.caption("Semantically similar complaints retrieved via local RAG (sentence-transformers + FAISS).")

if selected:
    col_rag1, col_rag2 = st.columns([3, 1])
    with col_rag1:
        k_val = st.slider("Number of similar cases", min_value=1, max_value=10, value=5)
    with col_rag2:
        find_btn = st.button("Find Similar", type="primary", use_container_width=True)

    if find_btn:
        with st.spinner("Searching for similar cases…"):
            rag_result = get_similar_complaints(selected_id, k=k_val)

        similar = rag_result.get("similar", [])
        info    = rag_result.get("engine_info", {})

        st.caption(
            f"Index size: **{info.get('index_size', '?')}** complaints  |  "
            f"Model: `{info.get('embed_model', '?')}`"
        )

        if not similar:
            st.info("No similar cases found. The RAG index may need to be rebuilt (Admin → Rebuild RAG Index).")
        else:
            for i, case in enumerate(similar, 1):
                sim_pct = round(case["similarity_score"] * 100, 1)
                with st.expander(
                    f"#{i}  [{case['category']}]  {SEVERITY_ICON.get(case['severity'],'')} {case['severity'].upper()}"
                    f"  —  {sim_pct}% similar  [{STATUS_ICON.get(case['status'],'')} {case['status']}]"
                ):
                    st.markdown(f"**Case ID:** `{case['id']}`")
                    st.markdown(f"**Emotion:** {case['emotion']}")
                    st.markdown(f"**Snippet:** {case['snippet']}")

# ── Admin: Rebuild RAG index ──────────────────────────────────────────────────

if st.session_state.get("role") == "admin":
    st.markdown("---")
    with st.expander("⚙️ Admin — Rebuild RAG Index"):
        st.caption(
            "Rebuilds the FAISS similarity index from all complaints in the database. "
            "Run this after bulk imports or the first time you use Similar Cases."
        )
        if st.button("Rebuild RAG Index", type="secondary"):
            with st.spinner("Building index…"):
                res = rebuild_rag_index()
            st.success(f"Index rebuilt — {res.get('indexed', 0):,} complaints indexed.")
