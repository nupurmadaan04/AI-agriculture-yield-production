def test_decision_support_endpoint(client):
    response = client.get("/api/decision-support")
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert "regional_situation" in data
    assert len(data["regional_situation"]) > 0
    assert "model_signals" in data
    assert "ai_insight" in data
    assert "scientific_disclaimer" in data
