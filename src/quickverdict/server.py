"""HTTP server speaking TypeSafe's System One API, so the official SDKs work with `base_url`."""

import gzip
import hmac
import io
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .engine import Engine, RequestError
from .schema import SystemOneRequest

MAX_BODY = 32 * 1024 * 1024
MAX_DEPTH = 64
RELEASE_DATE = "2026-11-18"


def _err(status: int, detail) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers={"x-typesafe-request-id": uuid.uuid4().hex})


def _decompress(body: bytes, encoding: str) -> bytes:
    """One layer of gzip or zstd, capped at MAX_BODY after decompression (zip-bomb safe)."""
    if encoding == "gzip":
        stream = gzip.GzipFile(fileobj=io.BytesIO(body))
    else:
        import zstandard  # lazy: only needed for zstd bodies

        stream = zstandard.ZstdDecompressor(max_window_size=8 * 1024 * 1024).stream_reader(io.BytesIO(body))
    data = stream.read(MAX_BODY + 1)
    if len(data) > MAX_BODY:
        raise OverflowError
    return data


def _depth(obj, level: int = 1) -> int:
    if isinstance(obj, dict):
        return max([_depth(v, level + 1) for v in obj.values()], default=level)
    if isinstance(obj, list):
        return max([_depth(v, level + 1) for v in obj], default=level)
    return level


def _reject_constant(name):
    raise ValueError(f"non-finite number {name}")


def create_app(engine: Engine, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="quickverdict", version="0.1.0")

    def unauthorized(request: Request) -> bool:
        if not api_key:
            return False
        sent = request.headers.get("authorization", "")
        return not hmac.compare_digest(sent.encode(), f"Bearer {api_key}".encode())

    @app.get("/v1/models")
    async def models(request: Request):  # async: never queues behind GPU work in the threadpool
        if unauthorized(request):
            return _err(401, "invalid API key")
        return {"models": [{"name": engine.model_id, "release_date": RELEASE_DATE,
                            "description": "Local System One model served by quickverdict."}]}

    @app.post("/v1/systemone")
    async def system_one(request: Request):
        if unauthorized(request):
            return _err(401, "invalid API key")
        body = bytearray()
        async for chunk in request.stream():  # stop reading at the cap instead of buffering everything
            body += chunk
            if len(body) > MAX_BODY:
                return _err(413, "request body exceeds 32 MiB")
        body = bytes(body)
        encoding = request.headers.get("content-encoding", "").strip().lower()
        if encoding not in ("", "identity", "gzip", "zstd"):
            return _err(415, f"unsupported Content-Encoding: {encoding}")
        if encoding in ("gzip", "zstd"):
            try:
                body = _decompress(body, encoding)
            except OverflowError:
                return _err(413, "decompressed body exceeds 32 MiB")
            except Exception:
                return _err(400, f"malformed {encoding} body")
        if body.startswith(b"\xef\xbb\xbf"):
            return _err(400, "body must be UTF-8 JSON without a BOM")
        try:
            raw = json.loads(body.decode("utf-8"), parse_constant=_reject_constant)
            too_deep = _depth(raw) > MAX_DEPTH
        except RecursionError:  # ~1,000 levels: deeper than the parser or _depth can go
            too_deep = True
        except (UnicodeDecodeError, ValueError) as e:
            return _err(400, f"invalid JSON body: {e}"[:500])
        if too_deep:
            return _err(400, f"JSON nesting exceeds {MAX_DEPTH} levels")
        try:
            req = SystemOneRequest.model_validate(raw)
        except ValidationError as e:
            errs = [{"loc": ["body", *err["loc"]], "msg": err["msg"], "type": err["type"]}
                    for err in e.errors(include_url=False)]
            return _err(422, errs)
        if req.assets:
            return _err(400, "assets (images) are not supported by quickverdict yet")
        try:
            # sync GPU work runs in the threadpool so the event loop stays free
            result = await run_in_threadpool(engine.system_one, req.state, req.questions)
        except RequestError as e:
            return _err(400, str(e))
        return JSONResponse({"model": engine.model_id, **result},
                            headers={"x-typesafe-request-id": uuid.uuid4().hex})

    return app
