def test_health_is_always_ok(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness_reports_models_loaded_state(api_client):
    r = api_client.get("/readiness")
    assert r.status_code == 200
    body = r.json()
    assert "database" in body
    assert "models_loaded" in body
    assert body["status"] in ("ok", "degraded")


def test_readiness_degraded_when_db_unreachable(api_client, monkeypatch):
    import src.api.routers.health as health_module

    class _BrokenEngine:
        def connect(self):
            raise ConnectionError("no db")

    monkeypatch.setattr(health_module, "get_engine", lambda: _BrokenEngine())

    r = api_client.get("/readiness")
    body = r.json()
    assert body["database"] == "unreachable"
    assert body["status"] == "degraded"
