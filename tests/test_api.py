def test_health_check(client):
    response = client.get("/health")
    # In CI without models, it might be 200 or return unhealthy status payload
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_no_model(client):
    # This will test the /predict endpoint. 
    # Depending on whether artifacts exist during the test, it might return 503 or 400.
    payload = {
        "features": {
            "AMT_INCOME_TOTAL": 50000,
            "DAYS_BIRTH": -15000
        }
    }
    response = client.post("/predict", json=payload)
    
    # Accept 200 (if model loaded & predicts), 400 (if missing feature schema validation), or 503 (if model not loaded)
    assert response.status_code in [200, 400, 503]
    
    if response.status_code == 200:
        assert "probability" in response.json()
