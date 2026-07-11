"""
Admin: Train Model — retrain the ML classifier on stored complaints.
Restricted to admin role only.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
from frontend.utils.auth_guard import require_auth
from frontend.utils.api_client import train_model, model_info, model_runs

require_auth(admin_only=True)

st.title("⚙️ Admin: Train Classifier")
st.caption("Retrain the ML model on all stored complaints. Admin only.")

st.warning(
    "Training runs on **all complaints in the database** (real + synthetic). "
    "The new model replaces the previous one immediately after training.",
    icon="⚠️",
)

st.markdown("---")

# ── Current model status ──────────────────────────────────────────────────────

st.subheader("Current model")
info = model_info()

col1, col2, col3 = st.columns(3)
if info.get("is_trained"):
    col1.metric("Category classes", len(info.get("category_classes", [])))
    col2.metric("Emotion classes",  len(info.get("emotion_classes",  [])))
    col3.metric("Severity classes", len(info.get("severity_classes", [])))
    st.success(f"✅ Model loaded — vocabulary size: {info.get('vectorizer_vocab_size', '—')}")
else:
    st.warning("No trained model found. Generate data first, then train.")

st.markdown("---")

# ── Train button ──────────────────────────────────────────────────────────────

if st.button("🚀 Train Now", type="primary", use_container_width=True):
    with st.spinner("Training… this may take 30–60 seconds"):
        result = train_model()

    if result:
        st.success(f"✅ {result.get('message', 'Training complete.')}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Category accuracy", f"{result.get('category_accuracy', 0):.1%}")
        m2.metric("Emotion accuracy",  f"{result.get('emotion_accuracy',  0):.1%}")
        m3.metric("Severity accuracy", f"{result.get('severity_accuracy', 0):.1%}")
        st.caption(f"Trained on {result.get('training_samples', 0):,} samples · MLflow run: `{result.get('mlflow_run_id','—')}`")

st.markdown("---")

# ── Training history ──────────────────────────────────────────────────────────

st.subheader("Training history")
runs_data = model_runs()
runs = runs_data.get("runs", [])

if runs:
    runs_df = pd.DataFrame(runs)
    runs_df["created_at"] = pd.to_datetime(runs_df.get("created_at"), errors="coerce")
    display_cols = [c for c in ["created_at", "category_accuracy", "emotion_accuracy",
                                "severity_accuracy", "training_samples", "is_active"] if c in runs_df.columns]
    st.dataframe(runs_df[display_cols], use_container_width=True)

    if len(runs) > 1:
        metric_cols = [c for c in ["category_accuracy", "emotion_accuracy", "severity_accuracy"] if c in runs_df.columns]
        if metric_cols:
            fig = px.line(
                runs_df.sort_values("created_at"),
                x="created_at",
                y=metric_cols,
                markers=True,
                title="Accuracy over training runs",
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No training runs yet.")
