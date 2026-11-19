"""The stage-one test suite must never use live API/network access."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pytest


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    import requests, httpx

    def blocked(*args, **kwargs):
        raise AssertionError("Live network is forbidden in stage-one tests")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)
