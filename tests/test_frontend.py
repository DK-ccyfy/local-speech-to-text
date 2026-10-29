"""Frontend assets are served correctly (browser flow smoke tests)."""


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "语音识别" in body
    assert "开始录音" in body
    assert "停止录音" in body
    assert "上传" in body


def test_static_assets_served(client):
    for path, kind in [
        ("/static/app.js", "javascript"),
        ("/static/style.css", "css"),
        ("/static/recorder-worklet.js", "javascript"),
    ]:
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert kind in resp.headers["content-type"], path


def test_app_js_uses_required_api_and_limits(client):
    js = client.get("/static/app.js").text
    assert "/api/transcribe" in js
    assert "MAX_SECONDS = 60" in js
    assert "getUserMedia" in js
    assert "NotAllowedError" in js  # permission-denied hint
    # synchronous re-entry guard: a double-clicked 开始录音 must not start
    # two audio pipelines (second one could never be stopped)
    assert "state.starting" in js


def test_error_box_hidden_attribute_actually_hides(client):
    """Regression: author `display:flex` used to defeat the HTML hidden attribute,
    leaving a permanent empty red box on the page."""
    css = client.get("/static/style.css").text
    assert "[hidden]" in css and "display: none" in css
