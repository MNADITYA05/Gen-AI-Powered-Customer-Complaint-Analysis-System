"""
Auth helpers shared across all Streamlit pages.
Call require_auth() at the top of every page that needs login.
"""
import streamlit as st


def require_auth(admin_only: bool = False) -> None:
    """
    Stop page rendering if the user is not authenticated.
    If admin_only=True, also stop if the user's role is not 'admin'.
    """
    if not st.session_state.get("authenticated"):
        st.error("Please log in to access this page.")
        st.page_link("app.py", label="← Go to Login")
        st.stop()

    if admin_only and st.session_state.get("role") != "admin":
        st.error("🔒 This page is restricted to administrators.")
        st.stop()


def current_user() -> dict:
    return {
        "username": st.session_state.get("username", ""),
        "role":     st.session_state.get("role", "agent"),
        "token":    st.session_state.get("token", ""),
    }


def is_admin() -> bool:
    return st.session_state.get("role") == "admin"
