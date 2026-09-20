from backend.services.report_service import report_service

def test_report_service_generation():
    res = report_service.generate_report(state="Punjab", district="Ludhiana", year=2017)
    assert "report_id" in res
    assert "AGRICULTURAL INTELLIGENCE REPORT" in res["markdown_content"]
    assert "Executive Summary" in res["markdown_content"]
    assert "Scientific Limitations" in res["markdown_content"]
    assert res["summary_metrics"]["predicted_yield"] > 0

def test_report_api_endpoint(client):
    payload = {
        "state": "Tamil Nadu",
        "district": "Thanjavur",
        "year": 2017,
        "report_type": "comprehensive"
    }
    response = client.post("/api/reports/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Tamil Nadu"
    assert len(data["markdown_content"]) > 100
