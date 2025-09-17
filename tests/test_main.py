# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from ..main import app
from io import BytesIO

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_parse_csv_endpoint():
    csv_content = b"name,age,city\nAlice,25,NYC\nBob,30,LA"
    response = client.post(
        "/api/parse-file",
        files={"file": ("data.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "schema" in data
    assert "preview" in data
    assert "understanding" in data
    assert "inspection" in data
    assert "cleaning" in data
    assert "descriptive" in data
    assert "visualizations" in data
    assert "relationships" in data
    assert "advanced" in data
    assert "insights" in data
    assert "metadata" in data

    assert isinstance(data["insights"], list)
    assert len(data["insights"]) > 0  