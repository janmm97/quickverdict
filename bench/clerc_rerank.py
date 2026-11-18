"""Re-rank CLERC legal passages exactly as TypeSafe's rerank cookbook does, with a local quickverdict engine.

Recipe (docs.typesafe.ai/cookbooks/rerank_typesafe): 170 CLERC rows pooled into one corpus (3,565 passages),
40 queries, BM25 shortlist of 30, one noul question per (query, candidate), sort by noul.
Published Jev numbers on this slice: top-1/5/10 = 30/57/65 %; BM25 alone: 5/15/38 %.

    pip install -e ".[bench]"
    python bench/clerc_rerank.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

CLERC_FILE = "https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/main/teva_train_dir/train_data.jsonl.gz"
N_ROWS, N_QUERIES, TOP_K = 170, 40, 30
QUESTION = {
    "type": "noul",
    "instructions": (
        "The query excerpt comes from a US federal court opinion and was written "
        "immediately around a citation to a precedent; the citation itself has been "
        "removed. Could the candidate passage be from that cited precedent — does it "
        "establish the specific legal proposition the query excerpt invokes at its "
        "citation point?"
    ),
    "criteria": {
        "true": (
            "The candidate passage states or establishes the specific rule, standard, "
            "holding, or fact pattern that the query excerpt attributes to its removed "
            "citation."
        ),
        "false": (
            "The candidate passage is merely on a similar topic or doctrine; it does not "
            "supply the specific proposition the query excerpt relies on."
        ),
    },
}


def cid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_slice(seed: int = 0) -> dict:
    """Same rows, same seed, same order as the cookbook's build_slice."""
    from datasets import load_dataset

    rows = []
    for row in load_dataset("json", data_files=CLERC_FILE, streaming=True, split="train"):
        if row.get("positive_passages") and len(row.get("negative_passages") or []) == 20:
            rows.append(row)
        if len(rows) >= 1000:
            break
    rng = random.Random(seed)
    corpus, pool = {}, []
    for row in rng.sample(rows, N_ROWS):
        gold = row["positive_passages"][0]["text"]
        corpus[cid(gold)] = gold
        for neg in row["negative_passages"]:
            corpus[cid(neg["text"])] = neg["text"]
        pool.append({"qid": str(row["query_id"]), "query": row["query"], "gold": cid(gold)})
    return {"queries": rng.sample(pool[20:], N_QUERIES), "corpus": dict(sorted(corpus.items()))}


def bm25_rankings(corpus: dict, queries: dict, k: int = 100) -> dict:
    import bm25s

    cids = list(corpus)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([corpus[c] for c in cids], stopwords="en"))
    qids = list(queries)
    idxs, _ = retriever.retrieve(bm25s.tokenize([queries[q] for q in qids], stopwords="en"), k=min(k, len(cids)))
    return {q: [cids[i] for i in idxs[r]] for r, q in enumerate(qids)}


def hits(ranked: dict, golds: dict, n: int) -> float:
    return sum(golds[q] in ranked[q][:n] for q in golds) / len(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--cache", default="bench/.cache/clerc_slice.json", help="where the data slice is kept")
    ap.add_argument("--out", default=None, help="write per-pair scores + summary JSON here")
    args = ap.parse_args()

    path = Path(args.cache)
    if path.exists():
        ds = json.loads(path.read_text("utf-8"))
    else:
        ds = build_slice()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ds), "utf-8")
    corpus = ds["corpus"]
    queries = {q["qid"]: q["query"] for q in ds["queries"]}
    golds = {q["qid"]: q["gold"] for q in ds["queries"]}
    cands = {q: r[:TOP_K] for q, r in bm25_rankings(corpus, queries).items()}
    recall = hits(cands, golds, TOP_K)
    print(f"corpus {len(corpus)} passages, {len(queries)} queries, gold in top {TOP_K}: {recall:.0%}")

    import torch

    from quickverdict.engine import Engine
    from quickverdict.schema import SystemOneRequest

    engine = Engine(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit)
    qs = SystemOneRequest.model_validate({"model": "m", "state": "", "questions": {"q": QUESTION}}).questions
    scores, times = {}, []
    for q in queries:
        scores[q] = {}
        for c in cands[q]:
            t = time.perf_counter()
            out = engine.system_one({"query_excerpt": queries[q], "candidate_passage": corpus[c]}, qs)
            if engine.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            scores[q][c] = out["answers"]["q"]["noul"]
    reranked = {q: sorted(cands[q], key=lambda c: -scores[q][c]) for q in queries}

    times.sort()
    gpu = torch.cuda.get_device_name() if engine.device == "cuda" else "cpu"
    summary = {
        "model": args.model + (" (nf4)" if args.load_in_4bit else ""), "hardware": gpu, "calls": len(times),
        "median_ms": round(1000 * times[len(times) // 2], 1), "p90_ms": round(1000 * times[int(len(times) * 0.9)], 1),
        "bm25": {n: hits(cands, golds, n) for n in (1, 5, 10)},
        "quickverdict": {n: hits(reranked, golds, n) for n in (1, 5, 10)},
        "jev_published": {1: 0.30, 5: 0.57, 10: 0.65},
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "scores": scores}, indent=1), "utf-8")


if __name__ == "__main__":
    main()
