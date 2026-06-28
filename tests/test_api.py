import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app
from api.auth import create_access_token


@pytest.fixture
def valid_token():
    return create_access_token("test-user")


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_screen_endpoint(valid_token):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/screen",
            json={"material_ids": ["mp-123"], "top_k": 5},
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
