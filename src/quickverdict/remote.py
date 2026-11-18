"""Remote backend: the same method, scored by any OpenAI-compatible server that returns logprobs
(vLLM, llama.cpp, Ollama, SGLang, LM Studio, or a hosted open-model provider)."""

import http.client
import json
import logging
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from .engine import SYSTEM, BackendError, RequestError, aggregate, plan, text_of

log = logging.getLogger("quickverdict")
TOP = 20  # the most top_logprobs OpenAI-style servers return


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # urllib would resend the Authorization header to any host
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def key_logprobs(top: list[dict], keys: list[str]) -> list[float]:
    """Log-probability of each answer key from the first token's top_logprobs.
    Tokens that strip to the same key ("A", " A") are summed; keys outside the top list split the
    probability mass the list leaves over, so they rank below every key that was seen."""
    mass = dict.fromkeys(keys, 0.0)
    for t in top:
        k = t["token"].strip()
        if k in mass:
            mass[k] += math.exp(t["logprob"])
    if not any(mass.values()):
        seen = ", ".join(repr(t["token"]) for t in top[:3])
        raise BackendError(f"backend's first token was not an answer key (top: {seen}): "
                           "the model must reply with the bare key (no preamble or reasoning mode)")
    missing = [k for k in keys if mass[k] == 0.0]
    if missing:
        rest = max(1.0 - sum(math.exp(t["logprob"]) for t in top), 1e-12) / len(missing)
        floor = min([math.exp(t["logprob"]) for t in top], default=1.0)
        for k in missing:
            mass[k] = min(rest, floor)
    return [math.log(max(mass[k], 1e-300)) for k in keys]


class RemoteEngine:
    def __init__(self, api_base: str, model: str, api_key: str | None = None, permutations: int = 3,
                 temperature: float = 1.0, concurrency: int = 8, timeout: float = 120.0):
        u = urlparse(api_base)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError("--api-base must be an http(s) URL, e.g. http://localhost:8080/v1")
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.model_id, self.api_key = model, api_key
        self.permutations, self.temperature = permutations, temperature
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(max(1, concurrency))  # shared: caps calls to the backend across requests

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        try:
            with _OPENER.open(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            log.warning("backend %s: %s", e.code, e.read()[:300].decode("utf-8", "replace"))
            raise BackendError(f"backend returned {e.code}") from e
        except (OSError, http.client.HTTPException) as e:  # URLError, reset, timeout, truncated body
            raise BackendError(f"backend unreachable: {e}") from e
        except ValueError as e:  # JSONDecodeError: an HTML error page, a proxy, the wrong URL
            raise BackendError("backend did not return JSON: check --api-base ends with /v1") from e

    def _score(self, state_text: str, prompt: str, keys: list[str]) -> tuple[list[float], dict]:
        out = self._post({
            "model": self.model_id, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": TOP,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"State:\n{state_text}\n\n{prompt}"}],
        })
        try:
            top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError("backend returned no logprobs: it must support logprobs + top_logprobs") from e
        return key_logprobs(top, keys), out.get("usage") or {}

    def system_one(self, state, questions: dict) -> dict:
        items = plan(questions, self.permutations)
        if any(len(k) > 1 and k not in ("Yes", "No") for _, _, keys, _ in items for k in keys):
            # shortcut: first-token scoring can't see multi-token keys; lift if backends expose prompt logprobs
            raise RequestError("the remote backend scores one token per answer: "
                               "Choice needs <= 26 options, Score <= 10 levels")
        s = text_of(state)
        results = list(self.pool.map(lambda it: self._score(s, it[1], it[2]), items))
        answers = aggregate(questions, [(qid, opts) for qid, _, _, opts in items],
                            [sc for sc, _ in results], self.temperature)
        usage = {"input_tokens": sum(u.get("prompt_tokens", 0) for _, u in results),
                 "output_tokens": sum(u.get("completion_tokens", 0) for _, u in results)}
        return {"answers": answers, "usage": usage}
