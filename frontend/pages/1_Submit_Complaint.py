"""
Submit Complaint page — real users submit complaints, get instant AI classification.
"""
import streamlit as st
from frontend.utils.auth_guard import require_auth
from frontend.utils.api_client import analyze_complaint, upload_csv

require_auth()

st.title("📝 Submit a Complaint")
st.caption("Submit a complaint below. Our AI instantly categorises it and flags the severity.")

tab_single, tab_bulk = st.tabs(["Single Complaint", "Bulk Upload (CSV)"])

# ── Single submission ─────────────────────────────────────────────────────────

with tab_single:
    with st.form("complaint_form"):
        text = st.text_area(
            "Describe your complaint",
            placeholder="e.g. My ATM card was declined three times despite having sufficient balance...",
            height=180,
        )
        submitted = st.form_submit_button("Analyse & Submit", use_container_width=True, type="primary")

    if submitted:
        if len(text.strip()) < 20:
            st.error("Please describe your complaint in at least 20 characters.")
        else:
            with st.spinner("Analysing..."):
                try:
                    result = analyze_complaint(text.strip())
                    st.success("✅ Complaint received and classified.")

                    col1, col2, col3 = st.columns(3)
                    severity_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}

                    with col1:
                        st.metric("Category", result.get("category", "—"))
                    with col2:
                        sev = result.get("severity", "—")
                        st.metric("Severity", f"{severity_color.get(sev, '')} {sev}")
                    with col3:
                        st.metric("Emotion detected", result.get("emotion", "—"))

                    conf = result.get("confidence", {})
                    if conf:
                        st.markdown("**Confidence scores**")
                        c1, c2, c3 = st.columns(3)
                        c1.progress(conf.get("category", 0), text=f"Category {conf.get('category', 0):.0%}")
                        c2.progress(conf.get("emotion", 0),  text=f"Emotion {conf.get('emotion', 0):.0%}")
                        c3.progress(conf.get("severity", 0), text=f"Severity {conf.get('severity', 0):.0%}")

                    st.info(
                        "Your complaint has been logged. Our team will review it and get back to you. "
                        "You can track its status in the **Cases** page."
                    )
                except Exception as exc:
                    st.error(f"Could not classify complaint: {exc}")

# ── Bulk CSV upload ───────────────────────────────────────────────────────────

with tab_bulk:
    st.markdown("""
    Upload a CSV file containing multiple complaints.

    **Required column:** `complaint_text`

    **Optional columns:** `category`, `severity`, `emotion`, `customer_name`, `customer_id`, `channel`, `location`

    Missing `category`, `severity`, and `emotion` values are predicted automatically if the model is trained.
    """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        if st.button("Upload & Process", type="primary"):
            with st.spinner("Processing..."):
                result = upload_csv(uploaded.read(), uploaded.name)
                if result:
                    st.success(
                        f"✅ Uploaded **{result['stored']}** complaints "
                        f"(skipped {result['skipped']} invalid rows out of {result['total_rows']} total)."
                    )
                    st.page_link("pages/2_Cases.py", label="View Cases →")

    with st.expander("Download a sample CSV template"):
        import io
        import pandas as pd
        sample = pd.DataFrame({
            "complaint_text": [
                "My ATM card was declined despite sufficient balance.",
                "I noticed an unauthorised transaction on my account.",
            ],
            "customer_name": ["John Doe", "Jane Smith"],
            "channel": ["ATM", "Mobile App"],
        })
        buf = io.BytesIO()
        sample.to_csv(buf, index=False)
        st.download_button("Download template.csv", buf.getvalue(), "template.csv", "text/csv")
