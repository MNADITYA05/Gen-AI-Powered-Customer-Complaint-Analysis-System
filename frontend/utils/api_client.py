"""
Thin HTTP client used by all Streamlit pages to call the FastAPI backend.
All requests include the JWT Bearer token stored in st.session_state.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx
import streamlit as st


def _base_url() -> str:
    if url := os.environ.get("API_BASE_URL"):
        return url
    try:
        return st.secrets.get("API_BASE_URL", "http://localhost:8000")
    except Exception:
        return "http://localhost:8000"


def _auth_headers() -> dict:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _get(path: str, params: dict = None) -> dict:
    url = f"{_base_url()}{path}"
    try:
        resp = httpx.get(url, params=params, headers=_auth_headers(), timeout=30)
        if resp.status_code == 401:
            st.session_state.clear()
            st.error("Session expired. Please log in again.")
            st.stop()
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at **{_base_url()}**. Is the backend running? (`make api`)")
        st.stop()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
        st.stop()


def _post(path: str, body: dict, timeout: float = 120.0) -> dict:
    url = f"{_base_url()}{path}"
    try:
        resp = httpx.post(url, json=body, headers=_auth_headers(), timeout=timeout)
        if resp.status_code == 401:
            st.session_state.clear()
            st.error("Session expired. Please log in again.")
            st.stop()
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at **{_base_url()}**. Is the backend running? (`make api`)")
        st.stop()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
        st.stop()


def _patch(path: str, body: dict, timeout: float = 30.0) -> dict:
    url = f"{_base_url()}{path}"
    try:
        resp = httpx.patch(url, json=body, headers=_auth_headers(), timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at **{_base_url()}**.")
        st.stop()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
        st.stop()


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(username: str, password: str) -> Optional[dict]:
    """Returns token dict on success, None on failure (wrong credentials)."""
    url = f"{_base_url()}/auth/login"
    try:
        resp = httpx.post(url, json={"username": username, "password": password}, timeout=10)
        if resp.status_code == 401:
            return None
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at **{_base_url()}**. Is the backend running? (`make api`)")
        st.stop()


def register(username: str, email: str, password: str) -> Optional[dict]:
    """Returns user dict on success, raises on error."""
    url = f"{_base_url()}/auth/register"
    try:
        resp = httpx.post(url, json={"username": username, "email": email, "password": password}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"Cannot reach the API at **{_base_url()}**.")
        st.stop()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"Registration failed: {detail}")
        return None


def get_me() -> dict:
    return _get("/auth/me")


# ── Health ────────────────────────────────────────────────────────────────────

def health_check() -> dict:
    url = f"{_base_url()}/health/ready"
    try:
        resp = httpx.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "unreachable"}


# ── Complaints ────────────────────────────────────────────────────────────────

def submit_complaint(text: str) -> dict:
    return _post("/api/v1/complaints/analyze", {"text": text})


def upload_csv(file_bytes: bytes, filename: str) -> dict:
    url = f"{_base_url()}/api/v1/complaints/upload"
    try:
        resp = httpx.post(
            url,
            files={"file": (filename, file_bytes, "text/csv")},
            headers=_auth_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot reach the API.")
        st.stop()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"Upload failed: {detail}")
        st.stop()


def list_complaints(
    page: int = 1,
    limit: int = 50,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    params = {"page": page, "limit": limit}
    if category:
        params["category"] = category
    if severity:
        params["severity"] = severity
    if status:
        params["status"] = status
    return _get("/api/v1/complaints", params=params)


def update_complaint_status(complaint_id: str, status: str, note: str = "") -> dict:
    body = {"status": status}
    if note:
        body["note"] = note
    return _patch(f"/api/v1/complaints/{complaint_id}/status", body)


# ── Models ────────────────────────────────────────────────────────────────────

def train_model() -> dict:
    return _post("/api/v1/models/train", {}, timeout=300.0)


def model_info() -> dict:
    return _get("/api/v1/models/info")


def model_runs() -> dict:
    return _get("/api/v1/models/runs")


def get_similar_complaints(complaint_id: str, k: int = 5) -> dict:
    return _get(f"/api/v1/complaints/{complaint_id}/similar", params={"k": k})


def rebuild_rag_index() -> dict:
    return _post("/api/v1/complaints/rag/rebuild", {}, timeout=120.0)


def analyze_complaint(text: str) -> dict:
    return _post("/api/v1/complaints/analyze", {"text": text})


def batch_analyze(texts: list) -> dict:
    return _post("/api/v1/complaints/batch-analyze", {"texts": texts}, timeout=120.0)
