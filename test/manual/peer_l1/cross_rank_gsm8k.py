"""gsm8k accuracy where every answer of one pass is conditioned on cross-rank KV.

Each question gets its own few-shot prefix (shots drawn per question from a pool
disjoint from the questions), so no two prompts share pages and a cross-rank
request cannot be served by the receiving rank's own tree.

  cold    question i on rank i % 8 (computes and caches the whole prompt)
  cross   same prompt on rank (i + 1) % 8: peer_l1 restores it from rank i % 8,
          an L2 server from UMBP, a device-only server recomputes it
  repeat  same prompt on rank (i + 1) % 8 again: a local device hit

Prompt format and answer parsing follow benchmark/gsm8k (as accuracy_hires.py).

    python cross_rank_gsm8k.py --url http://127.0.0.1:30001 --data test.jsonl --out arm.json
"""

import argparse
import ast
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

INVALID = -9999999
STOP = ["Question", "Assistant:", "<|separator|>"]


def answer_value(text):
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def ask(url, rank, prompt):
    response = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "routed_dp_rank": rank,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 512,
                "stop": STOP,
            },
        },
        timeout=1800,
    )
    response.raise_for_status()
    out = response.json()
    cached = out["meta_info"].get("cached_tokens_details") or {}
    return out["text"], cached.get("device", 0), cached.get("host", 0)


def run_pass(url, name, prompts, labels, rank_of, parallel):
    start = time.monotonic()
    with ThreadPoolExecutor(parallel) as pool:
        results = list(
            pool.map(lambda i: ask(url, rank_of(i), prompts[i]), range(len(prompts)))
        )
    texts = [r[0] for r in results]
    correct = sum(answer_value(t) == label for t, label in zip(texts, labels))
    summary = dict(
        name=name,
        accuracy=correct / len(prompts),
        correct=correct,
        invalid=sum(answer_value(t) == INVALID for t in texts),
        device_tokens=sum(r[1] for r in results),
        host_tokens=sum(r[2] for r in results),
        seconds=round(time.monotonic() - start, 1),
    )
    print(json.dumps(summary), flush=True)
    return summary, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30001")
    parser.add_argument("--data", required=True)
    parser.add_argument("--questions", type=int, default=400)
    parser.add_argument("--shots", type=int, default=15)
    parser.add_argument("--shot-pool", type=int, default=300)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lines = [json.loads(line) for line in open(args.data) if line.strip()]
    pool = lines[: args.shot_pool]
    questions = lines[args.shot_pool : args.shot_pool + args.questions]
    rng = random.Random(args.seed)
    prompts, labels = [], []
    for item in questions:
        shots = rng.sample(pool, args.shots)
        prefix = "".join(
            f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots
        )
        prompts.append(prefix + f"Question: {item['question']}\nAnswer:")
        labels.append(answer_value(item["answer"]))

    report = dict(vars(args), passes=[])
    passes = [
        ("cold", lambda i: i % 8),
        ("cross", lambda i: (i + 1) % 8),
        ("repeat", lambda i: (i + 1) % 8),
    ]
    texts = {}
    for name, rank_of in passes:
        summary, texts[name] = run_pass(
            args.url, name, prompts, labels, rank_of, args.parallel
        )
        report["passes"].append(summary)
        time.sleep(1)  # Publication of the pass's pages (one scheduler poll).
    for a, b in (("cold", "cross"), ("cold", "repeat"), ("cross", "repeat")):
        report[f"identical_{a}_{b}"] = sum(x == y for x, y in zip(texts[a], texts[b]))
    report["texts"] = texts
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("identical")}))


if __name__ == "__main__":
    main()
