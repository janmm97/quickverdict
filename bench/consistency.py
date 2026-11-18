"""Self-consistency and latency, exactly as TypeSafe's consistency_choice cookbook measures them.

Recipe (docs.typesafe.ai/cookbooks/consistency_choice_cookbook): 40 borderline moderation posts (a seeded
draw from 128 combinations), 8 Choice questions (37 options), each post sent 15 times. Per option: the mean
|p_i - p_j| over all 105 pairs of repeats = the chance that asking again flips a decision at a random cutoff.
Each post scores its least stable option. Published Jev 1.14: 1.0% over all options, 4.0% average post,
6.7% worst post, 0/40 posts above 1 in 10; ~103 ms per call.

    python bench/consistency.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import copy
import itertools
import json
import random
import time
from statistics import mean

NUM_SAMPLES, NUM_POSTS = 15, 40
POST = {
    "post_id": "P-88213",
    "author": {"user_id": "user_4471", "account_age_days": 38, "prior_strikes": 1, "followers": 210},
    "context": {"surface": "public reply", "in_reply_to": "another user defending a game patch",
                "community": "example-game-debates"},
    "content": {"text": "", "has_link": True, "link_domain": "chat.example", "language": "en"},
    "reports": {"user_reports": 4, "report_reasons": ["harassment", "spam", "threat"]},
}
PERSONAL = (
    "Are you seriously this dense? Anyone who defends that patch is a complete clown and "
    "should be embarrassed to even post here. People like you are what's ruining this "
    "community and honestly you need to be dealt with."
)
IMPERSONAL = (
    "Is this patch serious? Whoever signed off on it is a complete clown and should be "
    "embarrassed. This is what's ruining this community."
)
INVITE = " Come say it to my face, invite's right here."
CLOSINGS = {
    "end-channel": " Keep it up and I'll end your whole channel.",
    "find-you": " Keep it up and I'll find out where you live.",
    "report": " Keep it up and I'll report your whole channel.",
    "no-closing": "",
}
SWITCHES = ("personal", "invite", "strikes", "reports", "new_account")
QUESTIONS = {
    "category": ("What is the single most applicable content-policy category for this post?", {
        "None": "No policy violation of any kind.",
        "Harass": "Insults or demeans a person, with no threat of harm and no protected-class attack.",
        "Hate": "Attacks a person or group over a protected characteristic (race, religion, gender, ...).",
        "Violence": "Makes a credible threat of harm or incites violence against someone.",
        "Spam": "Unsolicited promotion or link spam, with no personal attack.",
        "Sexual": "Sexual or adult content."}),
    "primary_risk": ("What is the primary moderation risk that should drive triage for this post?", {
        "Harassment": "Personal attack or targeted abuse is the main risk.",
        "Violence": "A threat of harm or intimidation is the main risk.",
        "LinkAbuse": "External-link or off-platform coordination risk is the main risk.",
        "AccountHistory": "Prior account history or repeat behavior is the main risk.",
        "LowRisk": "No meaningful moderation risk is present."}),
    "target": ("Who or what is the content primarily directed at?", {
        "None": "Not directed at anyone in particular.",
        "Person": "Directed at one specific individual.",
        "Group": "Directed at a protected group or class.",
        "Platform": "Directed at the community or platform itself, not a person."}),
    "action": ("What enforcement action should be taken on this post?", {
        "Allow": "Leave the post up with no action.",
        "Warn": "Leave the post up but attach a warning label.",
        "Remove": "Remove the post, but do not penalize the account.",
        "Strike": "Remove the post and add a strike to the account.",
        "Escalate": "Take no automated action; hold for a human decision."}),
    "queue": ("Which single moderation queue should own this post?", {
        "Auto": "Auto-resolve; no human queue needed.",
        "General": "General moderation queue.",
        "Threat": "Threat / violence response queue.",
        "Spam": "Spam and platform-abuse queue.",
        "TSLead": "Trust-and-safety lead / senior queue."}),
    "link_handling": ("How should any external link or off-platform invite in the post be handled?", {
        "Allow": "Leave the link in place.",
        "RmLink": "Strip or disable the link but keep the post.",
        "Brigade": "Treat the link as coordinated brigading and action it as abuse.",
        "Escalate": "Send the link to a specialist to assess before acting."}),
    "review_path": ("Who should make the final call on this post?", {
        "Auto": "Automated action; no human review.",
        "Human": "A frontline human moderator makes the call.",
        "Senior": "A senior or specialist reviewer is required.",
        "Legal": "Route to legal or law-enforcement escalation."}),
    "severity": ("What is the overall severity of this post?", {
        "None": "No violation.",
        "Low": "Rude or dismissive, but essentially harmless.",
        "Medium": "Personal harassment with no clearly credible threat.",
        "High": "Harassment together with a threat that could be read as credible."}),
}


def build_post(closing, personal, invite, strikes, reports, new_account) -> dict:
    post = copy.deepcopy(POST)
    post["content"]["text"] = (PERSONAL if personal else IMPERSONAL) + (INVITE if invite else "") + CLOSINGS[closing]
    post["content"]["has_link"] = invite
    post["content"]["link_domain"] = "chat.example" if invite else None
    post["author"]["prior_strikes"] = 3 if strikes else 0
    post["author"]["account_age_days"] = 2 if new_account else 38
    post["reports"] = ({"user_reports": 4, "report_reasons": ["harassment", "spam", "threat"]} if reports
                       else {"user_reports": 0, "report_reasons": []})
    return post


def posts() -> dict:
    combos = list(itertools.product(CLOSINGS, *[(False, True)] * len(SWITCHES)))
    return {"-".join([c[0], *(n for n, on in zip(SWITCHES, c[1:]) if on)]): build_post(*c)
            for c in random.Random(0).sample(combos, NUM_POSTS)}


def change_chances(samples: list[dict]) -> dict:
    """Per option: mean |p_i - p_j| over every pair of repeats."""
    out = {}
    for key, (_, opts) in QUESTIONS.items():
        for opt in opts:
            vals = [s[key][opt] for s in samples]
            pairs = list(itertools.combinations(vals, 2))
            out[f"{key}={opt}"] = sum(abs(a - b) for a, b in pairs) / len(pairs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default=None, help="write per-post results + summary JSON here")
    args = ap.parse_args()

    import torch

    from quickverdict.engine import Engine
    from quickverdict.schema import SystemOneRequest

    engine = Engine(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit, compile=args.compile)
    qs = SystemOneRequest.model_validate({"model": "m", "state": "", "questions": {
        k: {"type": "choice", "instructions": ins, "criteria": opts} for k, (ins, opts) in QUESTIONS.items()
    }}).questions
    sync = torch.cuda.synchronize if engine.device == "cuda" else (lambda: None)
    all_posts = posts()
    engine.system_one(next(iter(all_posts.values())), qs)  # warm-up (and compile), not timed

    times, least, avg, per_post = [], [], [], {}
    for pid, post in all_posts.items():
        samples = []
        for _ in range(NUM_SAMPLES):
            t = time.perf_counter()
            ans = engine.system_one(post, qs)["answers"]
            sync()
            times.append(time.perf_counter() - t)
            samples.append({k: ans[k]["probabilities"] for k in QUESTIONS})
        ch = change_chances(samples)
        least.append(max(ch.values()))
        avg.append(mean(ch.values()))
        per_post[pid] = {"least_stable": least[-1], "choices": {k: ans[k]["choice"] for k in QUESTIONS}}

    times.sort()
    summary = {
        "model": args.model + (" (nf4)" if args.load_in_4bit else "") + (" +compile" if args.compile else ""),
        "hardware": torch.cuda.get_device_name() if engine.device == "cuda" else "cpu",
        "calls": len(times), "posts": NUM_POSTS, "repeats": NUM_SAMPLES,
        "median_ms": round(1000 * times[len(times) // 2], 1), "p90_ms": round(1000 * times[int(len(times) * 0.9)], 1),
        "avg_all_options": mean(avg), "least_stable_avg_post": mean(least), "least_stable_worst_post": max(least),
        "posts_above_1_in_10": sum(v > 0.10 for v in least),
        "jev_published": {"avg_all_options": 0.010, "least_stable_avg_post": 0.040,
                          "least_stable_worst_post": 0.067, "posts_above_1_in_10": 0, "latency_ms": 103},
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "posts": per_post}, f, indent=1)


if __name__ == "__main__":
    main()
