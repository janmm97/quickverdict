"""The official TypeSafe Python SDK talks to quickverdict unchanged, only `base_url` differs."""

import socket
import threading
import time

import pytest

ts = pytest.importorskip("typesafe_sdk")
uvicorn = pytest.importorskip("uvicorn")


@pytest.fixture(scope="module")
def base_url(engine):
    from quickverdict.server import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 60
    while not server.started:
        assert t.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(5)


def test_official_sdk_round_trip(base_url):
    with ts.TypeSafeClient(api_key="local", model="quickverdict", base_url=base_url) as client:
        r = client.system_one(
            {"ticket": "My card was charged twice."},
            {
                "topic": ts.Choice(criteria={"billing": None, "shipping": None, "other": None}),
                "urgency": ts.Score(criteria=["low", "medium", "high"]),
                "refund": ts.Noul(instructions="Does the customer want money back?"),
            },
        )
        assert r.answers["topic"].choice in {"billing", "shipping", "other"}
        assert abs(sum(r.answers["topic"].probabilities.values()) - 1) < 1e-4
        assert 0 <= r.answers["urgency"].score <= 2
        assert 0 <= r.answers["refund"].noul <= 1
        assert client.models.list().models
