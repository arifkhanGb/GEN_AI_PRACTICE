from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_create_and_get_job() -> None:
    response = client.post(
        "/api/v1/jobs",
        json={"message": "Index repository"},
    )

    assert response.status_code == 202

    data = response.json()
    assert data["status"] == "queued"
    assert data["message"] == "Index repository"

    job_response = client.get(f"/api/v1/jobs/{data['job_id']}")

    assert job_response.status_code == 200
    assert job_response.json()["job_id"] == data["job_id"]
    assert job_response.json()["status"] == "queued"


def test_get_unknown_job_returns_404() -> None:
    response = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
