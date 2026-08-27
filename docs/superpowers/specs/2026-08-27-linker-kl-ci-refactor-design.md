# Linker KL CI Refactor Design

## Goal

Refactor the GLM-5.2 and DeepSeek-V4 Flash Direct Linker CI tests so that
Mooncake process management is shared and all KL workloads come from
`UnifiedRadixTreeTestMixin`. Every KL case must prove that a prefix was loaded
back from Direct remote storage.

## Shared Mooncake services

Add `MooncakeTestServices` under `python/sglang/test/mooncake_utils.py`. The
object owns the HTTP metadata server, `mooncake_master`, and an external
`mooncake_client` store. It provides:

- dynamically allocated service and metrics ports;
- RDMA device and local-host discovery with environment overrides;
- fail-fast startup and readiness checks;
- the environment variables required by an SGLang server;
- master metrics access and an offload wait primitive; and
- idempotent process-group cleanup.

The helper is composed into tests rather than inherited. This keeps server
lifecycle ordering explicit and avoids coupling Mooncake cleanup to the test
class MRO. The external store remains bounded to 4 GiB because the CI only
needs a small deterministic eviction workload.

## Linker KL tests

Each model test inherits `UnifiedRadixTreeTestMixin` and `CustomTestCase`.
Model classes keep only their model-specific launch flags, CI registration,
KL threshold, page size, and server lifecycle. Both Linker launch
configurations remove `--skip-tokenizer-init` and initialize the common Mixin
input set with
`sglang.test.kl_multiturn_utils.get_input_ids(model, num_samples=18)`. This is
the same real-data path used by the HiCache KL tests: it formats
`THUDM/LongBench-v2` context and question text, then tokenizes it with the
model tokenizer.

All three Mixin KL cases remain enabled:

1. ordinary multi-turn logprob replay;
2. prefill-cache-hit branching; and
3. decode-cache-hit branching.

GSM8K is skipped because these files are Linker KL regression tests. Each KL
method wraps the inherited implementation, resets a per-test remote-token
counter, and asserts that the counter is positive after the KL comparison.
Prefill and decode cache callbacks retain model page-alignment checks while
accumulating `cached_tokens_details.host`. This makes Direct load-back a hard
condition of every KL test rather than an incidental log message.

`--max-total-tokens=8192`, bounded generation lengths, and branch interleaving
create L1 pressure. These values may be tuned during remote validation, but
the three KL cases, positive Direct remote assertion, and model KL thresholds
must remain.

## Validation

Run formatting, lint, compilation, CI-registration discovery, and focused
unit tests for the Mooncake lifecycle helper locally. Then sync only the
incremental files to `test-2-sglang-home:/sgl-workspace/sglang` and run both
registered tests against the existing local GLM-5.2 and DeepSeek-V4 Flash
checkpoints. A passing remote run requires all three KL cases to pass, every
case to report positive Direct remote tokens, and no Mooncake transfer failure.
