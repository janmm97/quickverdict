import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from quickverdict.engine import RequestError
from quickverdict.remote import RemoteEngine, key_logprobs
from quickverdict.schema import SystemOneRequest


def lp(**probs):
    return [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]


def test_key_logprobs_sums_variants_and_floors_missing():
    top = [{"token": "A", "logprob": math.log(0.5)}, {"token": " A", "logprob": math.log(0.2)},
           {"token": "B", "logprob": math.log(0.25)}]
    a, b, c = (math.exp(x) for x in key_logprobs(top, ["A", "B", "C"]))
    assert a == pytest.approx(0.7) and b == pytest.approx(0.25)
    assert c == pytest.approx(0.05)  # the leftover mass, below every key that was seen


def test_key_logprobs_missing_never_outranks_seen():
    a, b = (math.exp(x) for x in key_logprobs(lp(A=0.01), ["A", "B"]))
    assert b <= a


class Fake(RemoteEngine):
    def __init__(self, **kw):
        super().__init__("http://localhost:1/v1", "m", **kw)
        self.bodies = []

    def _post(self, body):
        self.bodies.append(body)
        prompt = body["messages"][1]["content"]
        if "Yes or No" in prompt:
            top = lp(Yes=0.9, No=0.1)
        elif "level" in prompt:
            top = lp(**{"2": 0.8, "1": 0.2})
        else:  # put the mass on whichever key shows "cat" in this ordering
            key = next(line[0] for line in prompt.splitlines() if line.endswith(") cat"))
            top = lp(**{key: 0.9})
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


def test_system_one_remote_matches_local_shape():
    req = SystemOneRequest.model_validate({"model": "m", "state": {"pet": "meow"}, "questions": {
        "animal": {"type": "choice", "criteria": {"dog": None, "cat": None, "cow": None}},
        "loud": {"type": "score", "criteria": ["silent", "quiet", "loud"]},
        "pet": {"type": "noul", "instructions": "Is it a pet?"}}})
    eng = Fake()
    out = eng.system_one(req.state, req.questions)
    a = out["answers"]
    assert a["animal"]["choice"] == "cat" and a["animal"]["probabilities"]["cat"] > 0.8
    assert a["loud"]["score"] == pytest.approx(1.8, abs=0.01)
    assert a["pet"]["noul"] == pytest.approx(0.9)
    assert len(eng.bodies) == 5  # 3 orderings + 1 + 1
    assert out["usage"] == {"input_tokens": 50, "output_tokens": 5}
    b = eng.bodies[0]
    assert b["logprobs"] is True and b["max_tokens"] == 1 and b["temperature"] == 0
    assert '"pet": "meow"' in b["messages"][1]["content"]


def test_rejects_non_http_base():
    with pytest.raises(ValueError):
        RemoteEngine("file:///etc/passwd", "m")


class Handler(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):  # noqa: N802
        Handler.seen = {"path": self.path, "auth": self.headers.get("Authorization"),
                        "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))}
        status, data = (200, {"choices": [{}]}) if self.headers.get("Authorization") else (401, {"error": "no key"})
        if "redir" in self.path:
            status = 307
        self.send_response(status)
        if status == 307:
            self.send_header("Location", "http://evil.example/steal")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def backend():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/"
    srv.shutdown()


def test_http_errors_become_request_errors(backend):
    q = SystemOneRequest.model_validate({"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}}).questions
    with pytest.raises(RequestError, match="401"):
        RemoteEngine(backend, "m").system_one("s", q)
    with pytest.raises(RequestError, match="no logprobs"):
        RemoteEngine(backend, "m", api_key="k").system_one("s", q)
    assert Handler.seen["path"] == "/v1/chat/completions" and Handler.seen["auth"] == "Bearer k"
    with pytest.raises(RequestError, match="307"):  # never follow: it would carry the key to another host
        RemoteEngine(backend + "redir", "m", api_key="k").system_one("s", q)
    with pytest.raises(RequestError, match="unreachable"):
        RemoteEngine("http://127.0.0.1:9/v1", "m", timeout=2).system_one("s", q)
