"""
Streamlit entry point — Login / Home page.
Run with:  streamlit run frontend/app.py
Or:        make frontend
"""
import streamlit as st
from frontend.utils.api_client import health_check, login, register

st.set_page_config(
    page_title="Complaint Analysis System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────

if not st.session_state.get("authenticated"):
    st.title("🏦 Customer Complaint Analysis System")

    api_ok = health_check().get("status") == "ready"
    if not api_ok:
        st.error("❌ Cannot reach the API. Make sure `make api` is running.")
        st.stop()

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter username and password.")
            else:
                result = login(username, password)
                if result is None:
                    st.error("Incorrect username or password.")
                else:
                    # Fetch profile to get role
                    import httpx, os
                    base = os.environ.get("API_BASE_URL", "http://localhost:8000")
                    me_resp = httpx.get(
                        f"{base}/auth/me",
                        headers={"Authorization": f"Bearer {result['access_token']}"},
                        timeout=10,
                    )
                    me = me_resp.json() if me_resp.is_success else {}
                    st.session_state["authenticated"] = True
                    st.session_state["token"]         = result["access_token"]
                    st.session_state["username"]      = me.get("username", username)
                    st.session_state["role"]          = me.get("role", "agent")
                    st.rerun()

    with tab_register:
        st.caption("New users are registered as **agents**. Contact an admin to get admin access.")
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_user")
            new_email    = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            reg_submit   = st.form_submit_button("Create account", use_container_width=True)

        if reg_submit:
            if not new_username or not new_email or not new_password:
                st.error("All fields are required.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                user = register(new_username, new_email, new_password)
                if user:
                    st.success(f"Account created! You can now log in as **{user['username']}**.")

    st.stop()

# ── Sidebar (authenticated) ───────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Complaint Analysis")
    st.markdown("---")
    role = st.session_state.get("role", "agent")
    username = st.session_state.get("username", "")
    st.markdown(f"👤 **{username}** &nbsp; `{role}`")
    st.markdown("---")

    api_status = health_check()
    if api_status.get("status") == "ready":
        st.success("✅ API connected")
    else:
        st.error("❌ API unreachable")

    if role == "admin":
        try:
            from frontend.utils.api_client import model_info
            info = model_info()
            if info.get("is_trained"):
                st.success("✅ Model trained")
            else:
                st.warning("⚠️ No trained model")
        except Exception:
            pass

    st.markdown("---")
    if st.button("Log out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── Home page ─────────────────────────────────────────────────────────────────

st.title("🏦 Customer Complaint Analysis System")
role = st.session_state.get("role", "agent")

if role == "admin":
    st.markdown("""
    Welcome back, **Admin**. Use the sidebar to navigate.

    | Page | Who | What it does |
    |------|-----|-------------|
    | **Submit Complaint** | All users | Submit a real complaint for instant AI classification |
    | **Cases** | All users | View, filter, and manage complaint cases |
    | **Dashboard** | All users | Analytics and trend charts |
    | **Admin: Generate Data** | Admin only | Generate synthetic training data |
    | **Admin: Train Model** | Admin only | Retrain the ML classifier |
    """)
else:
    st.markdown("""
    Welcome! Use the sidebar to navigate.

    | Page | What it does |
    |------|-------------|
    | **Submit Complaint** | Submit a complaint for instant AI classification |
    | **Cases** | View and manage complaint cases |
    | **Dashboard** | Analytics and trend charts |
    """)
