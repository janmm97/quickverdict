"""`quickverdict serve`: start the local System One server."""

import argparse
import os
import sys

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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
                   help="Hugging Face model id or local path (env QUICKVERDICT_MODEL)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default=None, help="cuda, cpu, ... (default: cuda if available)")
    s.add_argument("--dtype", default=None, help="float16, bfloat16, float32 (default: float16 on GPU)")
    s.add_argument("--permutations", type=int, default=3,
                   help="option orderings averaged per Choice to cancel order bias (1 = off)")
    s.add_argument("--temperature", type=_positive, default=1.0, help="calibration temperature")
    s.add_argument("--chunk-tokens", type=int, default=2048, help="max new tokens per packed forward pass")
    s.add_argument("--load-in-4bit", action="store_true", help="nf4 weights, ~4x less VRAM (needs the [4bit] extra)")
    s.add_argument("--compile", action="store_true", help="torch.compile the model: faster, slow first request")
    a = p.parse_args(argv)
    api_key = os.environ.get("QUICKVERDICT_API_KEY")
    if a.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print(f"warning: serving on {a.host} without QUICKVERDICT_API_KEY: anyone who can reach this port "
              "can use the model", file=sys.stderr)

    import uvicorn

    from .engine import Engine
    from .server import create_app

    engine = Engine(a.model, device=a.device, dtype=a.dtype, permutations=a.permutations,
                    temperature=a.temperature, chunk_tokens=a.chunk_tokens, load_in_4bit=a.load_in_4bit,
                    compile=a.compile)
    uvicorn.run(create_app(engine, api_key=api_key), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
