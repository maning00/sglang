"""CPU provider adapter costs; fake transport timings are NOT engine RPC RTTs.

Run with this worktree on PYTHONPATH and external Rust/cache directories. No
GPU or native storage client is constructed. Native lookup/grant/retire RTTs
remain an engine-provider integration measurement.
"""

import argparse
import json
import time
from array import array
from dataclasses import asdict
from pathlib import Path

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.kv_memory_provider import CapacityShape
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import get_context
from sglang.test.kv_memory_provider import build_provider_fixture


def benchmark(backend, iterations):
    provider, allocator, tree, _ = build_provider_fixture(backend=backend)
    source = allocator.full_attn_allocator
    samples = {
        name: []
        for name in (
            "reserve_take",
            "tree_insert",
            "offer_confirm",
            "fake_retire_progress",
        )
    }

    def measure(name, callback):
        start = time.perf_counter_ns()
        result = callback()
        samples[name].append((time.perf_counter_ns() - start) / 1000)
        return result

    for iteration in range(iterations):

        def take():
            reservation = provider.reserve(
                CapacityShape("benchmark", iteration, 0, 1, 0, 4)
            )
            with source.reservation_scope(reservation, "extend"):
                return source.alloc(4)

        indices = measure("reserve_take", take)
        measure(
            "tree_insert",
            lambda: tree.insert(
                InsertParams(key=RadixKey(array("q", range(4))), value=indices)
            ),
        )
        provider.progress()
        measure("offer_confirm", lambda: tree.confirm_detach(tree.offer_candidates(1)))
        measure("fake_retire_progress", provider.progress)
    accounting = provider.accounting()
    assert accounting.cached == accounting.capacity and not accounting.violations()
    output = {}
    for name, values in samples.items():
        values.sort()
        output[name + "_us"] = {
            f"p{q}": values[min(len(values) - 1, int((len(values) - 1) * q / 100))]
            for q in (50, 95, 99)
        }
    return {
        "backend": backend,
        "iterations": iterations,
        "transport": "fake",
        "timings": output,
        "idle_accounting": asdict(accounting),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    with get_context().override_server_args():
        results = [
            benchmark(backend, args.iterations) for backend in ("python", "rust")
        ]
    encoded = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
