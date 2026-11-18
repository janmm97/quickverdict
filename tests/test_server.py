import gzip
import json

import pytest
import zstandard
from fastapi.testclient import TestClient

from quickverdict.confidence import choice_confidence, score_confidence
from quickverdict.engine import RequestError
from quickverdict.server import MAX_BODY, create_app

BODY = {"model": "m", "state": "hi", "questions": {"q": {"type": "noul", "instructions": "Greeting?"}}}


class FakeEngine:
    model_id = "fake-model"

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if state == "too long":
            raise RequestError("max_tokens_exceeded: 9 > 8")
        return {"answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 5, "output_tokens": 0}}


@pytest.fixture
def client():
    return TestClient(create_app(FakeEngine(), api_key="secret"))


AUTH = {"Authorization": "Bearer secret"}


def post(client, body, headers=None):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return client.post("/v1/systemone", content=data,
                       headers={"Content-Type": "application/json", **AUTH, **(headers or {})})


def test_ok_and_request_id(client):
    r = post(client, BODY)
    assert r.status_code == 200
    assert r.json() == {"model": "fake-model", "answers": {"q": {"type": "noul", "noul": 0.9}},
                        "usage": {"input_tokens": 5, "output_tokens": 0}}
    assert r.headers["x-typesafe-request-id"]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret"}])
def test_auth(headers):
    c = TestClient(create_app(FakeEngine(), api_key="secret"))
    assert c.post("/v1/systemone", json=BODY, headers=headers).status_code == 401
    assert c.get("/v1/models", headers=headers).status_code == 401


def test_no_key_means_open():
    c = TestClient(create_app(FakeEngine()))
    assert c.post("/v1/systemone", json=BODY).status_code == 200


def test_models(client):
    m = client.get("/v1/models", headers=AUTH).json()["models"]
    assert m[0]["name"] == "fake-model"


@pytest.mark.parametrize("encoding,pack", [("gzip", gzip.compress), ("zstd", zstandard.compress)])
def test_compressed_bodies(client, encoding, pack):
    assert post(client, pack(json.dumps(BODY).encode()), {"Content-Encoding": encoding}).status_code == 200
    assert post(client, b"not compressed", {"Content-Encoding": encoding}).status_code == 400


def test_bombs_and_limits(client):
    bomb = gzip.compress(b" " * (MAX_BODY + 10))
    assert post(client, bomb, {"Content-Encoding": "gzip"}).status_code == 413
    assert post(client, b"{}", {"Content-Encoding": "br"}).status_code == 415
    deep = {"model": "m", "state": json.loads("[" * 70 + "]" * 70), "questions": BODY["questions"]}
    assert post(client, deep).status_code == 400


@pytest.mark.parametrize("raw", [b"\xef\xbb\xbf{}", b"{", b'{"model":"m","state":NaN,"questions":{}}', b"\xff"])
def test_bad_json_is_400(client, raw):
    assert post(client, raw).status_code == 400


def test_validation_is_422(client):
    bad = {**BODY, "questions": {"q": {"type": "choice", "criteria": {}}}}
    r = post(client, bad)
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][0] == "body"
    assert post(client, {**BODY, "extra": 1}).status_code == 422


def test_engine_errors_and_assets_are_400(client):
    assert post(client, {**BODY, "state": "too long"}).status_code == 400
    with_asset = {**BODY, "assets": {"img": {"type": "image", "url": "https://example.com/a.png"}}}
    assert post(client, with_asset).status_code in (400, 422)


def test_confidence_matches_typesafe_docs():
    assert choice_confidence([0.25] * 4) == 0
    assert choice_confidence([1, 0, 0, 0]) == 1
    assert score_confidence([0, 0.57, 0.43]) == pytest.approx(0.355, abs=1e-3)  # docs example
    assert score_confidence([0, 1, 0]) == 1
    assert score_confidence([1 / 3] * 3) == 0  # uniform = no confidence
