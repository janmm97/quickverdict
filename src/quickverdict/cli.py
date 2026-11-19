"""`quickverdict serve`: start the local System One server."""

import argparse
import os
import sys

DEFAULT_MODEL = "auto"  # biggest model the GPU fits, see hardware.py


def _positive(v: str) -> float:
    f = float(v)
    if not f > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return f


def main(argv=None):
    p = argparse.ArgumentParser(prog="quickverdict")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve POST /v1/systemone and GET /v1/models")
    s.add_argument("--model", default=os.environ.get("QUICKVERDICT_MODEL", DEFAULT_MODEL),
                   help="Hugging Face model id or local path, or auto (default): the biggest model your GPU fits "
                        "(env QUICKVERDICT_MODEL); with --api-base, the backend's own model name")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default=None, help="cuda, xpu, mps, cpu, or auto to split the model across all GPUs "
                                                         "(default: the first of cuda/xpu/mps found, else cpu)")
    s.add_argument("--dtype", default=None,
                   help="float16, bfloat16, float32 (default: bfloat16 on GPUs that have it, else float16; "
                        "float32 on CPU)")
    s.add_argument("--permutations", type=int, default=3,
                   help="option orderings averaged per Choice to cancel order bias (1 = off)")
    s.add_argument("--temperature", type=_positive, default=1.0, help="calibration temperature")
    s.add_argument("--chunk-tokens", type=int, default=2048, help="max new tokens per packed forward pass")
    s.add_argument("--load-in-4bit", action="store_true", help="nf4 weights, ~4x less VRAM (needs the [4bit] extra)")
    s.add_argument("--compile", action="store_true", help="torch.compile the model: faster, slow first request")
    s.add_argument("--api-base", default=os.environ.get("QUICKVERDICT_API_BASE"),
                   help="score with an OpenAI-compatible server that returns logprobs instead of a local model, "
                        "e.g. http://localhost:8080/v1 (env QUICKVERDICT_API_BASE). Its key: env "
                        "QUICKVERDICT_BACKEND_API_KEY")
    s.add_argument("--concurrency", type=int, default=8, help="parallel requests to --api-base")
    a = p.parse_args(argv)
    api_key = os.environ.get("QUICKVERDICT_API_KEY")
    if a.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print(f"warning: serving on {a.host} without QUICKVERDICT_API_KEY: anyone who can reach this port "
              "can use the model", file=sys.stderr)

    import uvicorn

    from .server import create_app

    if a.api_base:
        from urllib.parse import urlparse

        from .remote import RemoteEngine

        if a.model == "auto":  # a backend can't be sized from here; keep the old default name
            from .hardware import SMALLEST

            a.model = SMALLEST
        try:
            engine = RemoteEngine(a.api_base, a.model, api_key=os.environ.get("QUICKVERDICT_BACKEND_API_KEY"),
                                  permutations=a.permutations, temperature=a.temperature,
                                  concurrency=a.concurrency)
        except ValueError as e:
            p.error(str(e))
        host = urlparse(a.api_base).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            print(f"note: every state you send is forwarded to {host}", file=sys.stderr)
        local = [f for f in ("device", "dtype", "load_in_4bit", "compile") if getattr(a, f)]
        if local or a.chunk_tokens != 2048:
            print("note: --device, --dtype, --load-in-4bit, --chunk-tokens and --compile are ignored "
                  "with --api-base", file=sys.stderr)
    else:
        from .engine import Engine

        if a.model == "auto":
            from .hardware import auto_model

            a.model, a.device, nf4, why = auto_model(a.device)
            a.load_in_4bit = a.load_in_4bit or nf4
            print(why, file=sys.stderr)
        engine = Engine(a.model, device=a.device, dtype=a.dtype, permutations=a.permutations,
                        temperature=a.temperature, chunk_tokens=a.chunk_tokens, load_in_4bit=a.load_in_4bit,
                        compile=a.compile)
    uvicorn.run(create_app(engine, api_key=api_key), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
