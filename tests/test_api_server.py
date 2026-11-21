import socket

import httpx

from hearth.app import App
from hearth.config import Config
from conftest import wait_for


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_api_serves_on_good_addresses_while_an_unavailable_one_is_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_API_TOKENS", "ha:tok")
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path)
    cfg.audio.source = "none"
    cfg.mqtt.enabled = cfg.summary.enabled = False
    cfg.api.hosts = ["127.0.0.1", "203.0.113.7"]        # the second is not an address of this machine
    cfg.api.port = free_port()
    app = App(cfg)
    app.start()
    try:
        url = f"http://127.0.0.1:{cfg.api.port}"
        assert wait_for(lambda: _ok(url + "/healthz"), timeout=10)
        r = httpx.get(url + "/api/v1/status", headers={"Authorization": "Bearer tok"})
        assert r.status_code == 200 and r.json()["recording"] is False
        assert any(t.name == "api-203.0.113.7" and t.is_alive() for t in app.threads)   # still trying, not dead
    finally:
        app.stop()


def _ok(url):
    try:
        return httpx.get(url, timeout=1).status_code == 200
    except httpx.HTTPError:
        return False
