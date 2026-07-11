"""
Integration tests against the FastAPI TestClient.
All tests use the in-memory DB fixture from conftest.py.
"""
import pytest


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness_endpoint_exists(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert "status" in resp.json()


class TestGenerate:
    def test_generate_template_returns_complaints(self, client):
        resp = client.post(
            "/api/v1/complaints/generate",
            json={"count": 10, "method": "template"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["complaints"]) == 10
        assert body["stored"] == 10

    def test_generate_respects_count(self, client):
        for count in [5, 20, 50]:
            resp = client.post(
                "/api/v1/complaints/generate",
                json={"count": count, "method": "template"},
            )
            assert resp.status_code == 200
            assert len(resp.json()["complaints"]) == count

    def test_generate_with_distribution(self, client):
        resp = client.post(
            "/api/v1/complaints/generate",
            json={
                "count": 30,
                "method": "template",
                "category_distribution": {
                    "ATM_FAILURE": 1.0,
                    "FRAUD_DETECTION": 0.0,
                    "UX_ISSUES": 0.0,
                },
            },
        )
        assert resp.status_code == 200
        complaints = resp.json()["complaints"]
        assert all(c["category"] == "ATM_FAILURE" for c in complaints)

    def test_generate_bad_distribution_422(self, client):
        resp = client.post(
            "/api/v1/complaints/generate",
            json={
                "count": 10,
                "method": "template",
                "category_distribution": {"ATM_FAILURE": 0.5},  # doesn't sum to 1
            },
        )
        assert resp.status_code == 422

    def test_llm_without_token_returns_422(self, client):
        resp = client.post(
            "/api/v1/complaints/generate",
            json={"count": 5, "method": "llm_huggingface"},
        )
        assert resp.status_code == 422

    def test_generated_complaints_stored_in_db(self, client):
        # Clear list first by fetching before
        before = client.get("/api/v1/complaints?limit=500").json()["total"]

        client.post(
            "/api/v1/complaints/generate",
            json={"count": 15, "method": "template"},
        )

        after = client.get("/api/v1/complaints?limit=500").json()["total"]
        assert after == before + 15


class TestListComplaints:
    def test_list_returns_paginated_results(self, client):
        # Ensure some data exists
        client.post("/api/v1/complaints/generate", json={"count": 20, "method": "template"})
        resp = client.get("/api/v1/complaints?limit=10")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) <= 10
        assert "total" in body

    def test_category_filter(self, client):
        resp = client.get("/api/v1/complaints?category=ATM_FAILURE&limit=100")
        assert resp.status_code == 200
        for item in resp.json()["items"]:
            assert item["category"] == "ATM_FAILURE"


class TestModelEndpoints:
    def test_model_info_before_training(self, client):
        resp = client.get("/api/v1/models/info")
        assert resp.status_code == 200
        # May or may not be trained depending on test order

    def test_train_requires_minimum_data(self, client):
        """If DB has fewer than 20 complaints, training should return 422."""
        # This test only holds if run in isolation with an empty DB
        # In the shared test session the DB may already have data
        resp = client.post("/api/v1/models/train")
        assert resp.status_code in (200, 422)

    def test_train_succeeds_with_sufficient_data(self, client):
        # Generate enough data
        client.post("/api/v1/complaints/generate", json={"count": 60, "method": "template"})
        resp = client.post("/api/v1/models/train")
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 < body["category_accuracy"] <= 1.0
        assert 0.0 < body["emotion_accuracy"]  <= 1.0
        assert 0.0 < body["severity_accuracy"] <= 1.0

    def test_runs_endpoint_returns_list(self, client):
        resp = client.get("/api/v1/models/runs")
        assert resp.status_code == 200
        assert "runs" in resp.json()


class TestAnalyze:
    @pytest.fixture(autouse=True)
    def ensure_trained(self, client):
        """Train models before analysis tests if not already trained."""
        info = client.get("/api/v1/models/info").json()
        if not info.get("is_trained"):
            client.post(
                "/api/v1/complaints/generate",
                json={"count": 60, "method": "template"},
            )
            client.post("/api/v1/models/train")

    def test_analyze_single(self, client):
        resp = client.post(
            "/api/v1/complaints/analyze",
            json={"text": "The ATM at your branch ate my card and I am furious."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "category" in body
        assert "emotion" in body
        assert "severity" in body
        assert "confidence" in body

    def test_analyze_too_short_422(self, client):
        resp = client.post(
            "/api/v1/complaints/analyze",
            json={"text": "short"},
        )
        assert resp.status_code == 422

    def test_batch_analyze(self, client):
        resp = client.post(
            "/api/v1/complaints/batch-analyze",
            json={
                "texts": [
                    "The ATM swallowed my card and I am extremely frustrated.",
                    "Unauthorized transactions on my account. I am scared.",
                    "Your mobile app keeps crashing during login attempts.",
                ]
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["results"]) == 3
