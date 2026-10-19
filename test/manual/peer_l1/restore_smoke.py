"""Real-weight DEP8 probe; run the server with verify_bytes/check_hooks enabled.

Exercises a long cold prefix, cross-DP restore, local reuse, concurrent restores,
and a fresh cold read after flushing every directory. Text is reported, while
the server's byte verifier supplies the exact-copy correctness check.
"""

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:39087")
    parser.add_argument("--records", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    session = uuid.uuid4().hex
    output_lock = Lock()

    def prompt(salt):
        records = "\n".join(
            f"Record {i}: the archive received {i % 17 + 3} books on Monday and "
            f"{i % 23 + 2} books on Tuesday. The librarian checked their labels "
            "and placed them on the numbered shelf."
            for i in range(args.records)
        )
        return (
            "<｜begin▁of▁sentence｜><｜User｜>"
            f"Verification session {salt}. The archive's codeword is ORCHID.\n"
            f"Read these independent inventory records:\n{records}\n"
            "What is the archive's codeword? Reply with the codeword."
            "<｜Assistant｜><think>"
        )

    text = prompt(session)

    def generate(label, rank, text):
        start = time.monotonic()
        response = requests.post(
            args.url + "/generate",
            json={
                "text": text,
                "routed_dp_rank": rank,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        output = response.json()
        meta = output["meta_info"]
        result = {
            "label": label,
            "rank": rank,
            "seconds": time.monotonic() - start,
            "prompt_tokens": meta["prompt_tokens"],
            "cached": meta.get("cached_tokens_details") or {},
            "text": output["text"],
        }
        with output_lock:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    cold = generate("cold", 0, text)
    assert sum(cold["cached"].values()) == 0
    time.sleep(0.2)  # Let the owner's next scheduler poll publish completed pages.
    peer = generate("peer", 1, text)
    assert (
        peer["cached"].get("host", 0) >= 2048
    ), "cross-DP restoration was not exercised"
    local = generate("local", 1, text)
    assert local["cached"].get("device", 0) >= 2048
    assert local["cached"].get("host", 0) == 0
    control = generate("new_prefix", 2, prompt(session + "-control"))
    assert sum(control["cached"].values()) == 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda rank: generate("concurrent_peer", rank, text), range(2, 6)
            )
        )
    assert all(result["cached"].get("host", 0) >= 2048 for result in results)

    def flush():
        deadline = time.monotonic() + 30
        while True:
            response = requests.post(args.url + "/flush_cache", timeout=10)
            if response.ok and "not flushed" not in response.text.lower():
                break
            assert time.monotonic() < deadline, response.text
            time.sleep(0.1)

    flush()
    after = generate("after_flush", 1, text)
    assert sum(after["cached"].values()) == 0, "flush left a peer directory readable"
    response = requests.post(
        args.url + "/pause_generation", json={"mode": "retract"}, timeout=30
    )
    response.raise_for_status()
    try:
        flush()
    finally:
        response = requests.post(args.url + "/continue_generation", json={}, timeout=30)
        response.raise_for_status()
    after = generate("after_paused_flush", 1, text)
    assert sum(after["cached"].values()) == 0, "paused flush left stale peer pages"
    print(
        "PASS: peer/local/cold attribution, concurrent restores, ordinary and paused flush",
        flush=True,
    )


if __name__ == "__main__":
    main()
