"""Controlled cross-rank prefix probe (design Phase 2 step 1).

For each prefix length: cold on rank a (recompute), then the same prompt on rank
a+1 (restored from rank a's HBM on a peer_l1 server, from UMBP on an L2 server,
recomputed on a device-only server), then once more on a+1 (device hit).
--concurrency 8 runs eight such chains at once on disjoint rank pairs.

    python prefix_probe.py --url http://127.0.0.1:30001 --lengths 8192,32768 --reps 3

Every chain uses a fresh salt; a chain whose cold request hit any cache aborts.
The reported latency is a one-token request end to end, i.e. TTFT plus a fixed
HTTP/detokenize overhead that is the same for all three phases.
"""

import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

# One record is about 45 tokens with the DSv4 tokenizer.
TOKENS_PER_RECORD = 45


def prompt(salt, tokens):
    records = "\n".join(
        f"Record {i}: the archive received {i % 17 + 3} books on Monday and "
        f"{i % 23 + 2} books on Tuesday. The librarian checked their labels and "
        "placed them on the numbered shelf."
        for i in range(max(1, tokens // TOKENS_PER_RECORD))
    )
    return (
        "<｜begin▁of▁sentence｜><｜User｜>"
        f"Session {salt}. Read these inventory records:\n{records}\n"
        "How many records are there?<｜Assistant｜><think>"
    )


def one(url, rank, text, timeout):
    start = time.monotonic()
    response = requests.post(
        url + "/generate",
        json={
            "text": text,
            "routed_dp_rank": rank,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    meta = response.json()["meta_info"]
    return dict(
        seconds=time.monotonic() - start,
        prompt_tokens=meta["prompt_tokens"],
        cached=meta.get("cached_tokens_details") or {},
    )


def chain(url, rank, tokens, timeout):
    text = prompt(uuid.uuid4().hex, tokens)
    cold = one(url, rank, text, timeout)
    if sum(cold["cached"].values()):
        raise RuntimeError(f"cold request hit a cache: {cold}")
    time.sleep(0.5)  # Let the owner publish (one scheduler poll) or L2 write through.
    peer = (rank + 1) % 8
    restore = one(url, peer, text, timeout)
    local = one(url, peer, text, timeout)
    return dict(
        rank=rank, tokens=cold["prompt_tokens"], cold=cold, restore=restore, local=local
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30001")
    parser.add_argument("--lengths", default="8192,32768,65536,131072,262144")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    # Discarded warm-up chain: first requests on a fresh server carry one-time costs.
    chain(args.url, 0, 8192, args.timeout)
    for tokens in map(int, args.lengths.split(",")):
        rows = []
        for rep in range(args.reps):
            ranks = [(i + rep) % 8 for i in range(args.concurrency)]
            with ThreadPoolExecutor(args.concurrency) as pool:
                rows += list(
                    pool.map(lambda r: chain(args.url, r, tokens, args.timeout), ranks)
                )
        for row in rows:
            print(json.dumps(dict(label=args.label, **row)), flush=True)

        def med(phase):
            return statistics.median(r[phase]["seconds"] for r in rows)

        def restored(phase):
            return statistics.median(
                r[phase]["cached"].get("host", 0) + r[phase]["cached"].get("device", 0)
                for r in rows
            )

        print(
            json.dumps(
                dict(
                    label=args.label,
                    summary=True,
                    concurrency=args.concurrency,
                    prompt_tokens=statistics.median(r["tokens"] for r in rows),
                    cold_s=med("cold"),
                    restore_s=med("restore"),
                    local_s=med("local"),
                    restore_cached_tokens=restored("restore"),
                    speedup_vs_recompute=med("cold") / med("restore"),
                    n=len(rows),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
