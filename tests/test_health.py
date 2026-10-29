"""GET /health"""


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # extra info field: model readiness (fake service reports ready)
    assert body["model"] == "ready"


def test_health_is_fast_even_while_model_loading():
    from app.asr.base import ASRService

    class LoadingService(ASRService):
        @property
        def status(self):
            return "loading"

        def transcribe(self, audio, language):  # pragma: no cover
            raise AssertionError("should not be called")

    from tests.conftest import make_client

    with make_client(LoadingService()) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "loading"}
