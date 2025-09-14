import pytest
from fastapi.testclient import TestClient
from main import app
from io import BytesIO

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_parse_csv_small():
    csv_content = b"name,age,city\nAlice,25,NYC\nBob,30,LA"
    response = client.post(
        "/api/parse-file",
        files={"file": ("data.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "schema" in data
    assert len(data["preview"]) == 2
    assert data["metadata"]["original_file_size_mb"] < 0.1
    assert data["metadata"]["sampled"] is False

def test_parse_csv_large_sampled():
    # Tạo file CSV giả lập 20MB (khoảng 100k dòng)
    rows = ["name,age\n"] + [f"User{i},{i % 100}\n" for i in range(100000)]
    csv_content = "".join(rows).encode("utf-8")
    
    response = client.post(
        "/api/parse-file",
        files={"file": ("large_data.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["original_file_size_mb"] > 10
    assert data["metadata"]["sampled"] is True  # Phải là sample
    assert len(data["preview"]) <= 10  # Chỉ lấy 10 mẫu