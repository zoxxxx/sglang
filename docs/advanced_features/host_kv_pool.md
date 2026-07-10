# Host-Centric KV Pool Prototype

This document describes an experimental KV ownership model for SGLang PD
disaggregation. It is a research prototype, not a storage hierarchy feature.

## Ownership Model

Conventional PD disaggregation assigns a request and its KV cache to a decode
worker before prefill completes. The decode worker allocates HBM and receives KV
directly from the selected prefill worker.

The Host KV Pool prototype separates logical ownership from execution:

```text
Request -> Host KV Object -> temporary Decode attachment
                    |
                    +-> Mooncake-backed Host KV Pool (authoritative copy)
```

- Prefill publishes a sealed KV object to a shared Mooncake-backed pool.
- The object is identified by `host_kv_id`, independently of a decode worker.
- The router dispatches to its selected decode worker after the object is ready.
- Decode attaches the object to temporary HBM and runs normally.
- GPU HBM is execution residency, not the logical owner of the object.

This first prototype implements late binding. It does not yet evict or migrate
KV while a decode request is running.

## Request Flow

1. The router generates a unique `host_kv_id` and dispatches prefill.
2. Prefill stages GPU KV pages in host memory and writes them to Mooncake.
3. Prefill writes metadata last, sealing the object only after all pages succeed.
4. The router dispatches decode after prefill returns.
5. Decode reads metadata and attaches the object's pages to allocated HBM slots.
6. Request completion, abort, or transfer failure schedules asynchronous object
   deletion from Mooncake.

Objects use rank-scoped keys such as
`hostkv:<id>:tp<rank>:pp<rank>:page<n>` and
`hostkv:<id>:tp<rank>:pp<rank>:meta`. Cleanup is idempotent per request and only
removes keys belonging to that object and source rank.

## Configuration

Install Mooncake Transfer Engine as described in
[PD Disaggregation](pd_disaggregation.md#mooncake). Start both worker groups with
the Host KV Pool flag and a Mooncake storage backend:

```bash
# Prefill worker arguments
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --disaggregation-mode prefill \
  --disaggregation-host-kv-pool \
  --hicache-storage-backend mooncake \
  --hicache-storage-backend-extra-config "$MOONCAKE_CONFIG" \
  --port 30000

# Decode worker arguments
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --disaggregation-mode decode \
  --disaggregation-host-kv-pool \
  --hicache-storage-backend mooncake \
  --hicache-storage-backend-extra-config "$MOONCAKE_CONFIG" \
  --port 30001
```

The flag does not require the HiCache hierarchy to be enabled. When HiCache is
disabled, the prototype creates a private host staging controller for Mooncake.

Run the model gateway in PD Host KV mode:

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --pd-host-kv-pool \
  --prefill http://127.0.0.1:30000 8998 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

The Rust PD router and Python MiniLB both generate `host_kv_id` values. The
MiniLB path additionally supports explicit prefill and decode DP-rank selection
for controlled experiments.

## Fixed-Rank Benchmark

`benchmark/host_kv_fixed_rank_bench.py` sends prefill and decode requests
sequentially for each sample while fixing their DP ranks. Prompts contain a
unique prefix to prevent benchmark requests from sharing their body.

```bash
python benchmark/host_kv_fixed_rank_bench.py \
  --prefill-url http://127.0.0.1:30000 \
  --decode-url http://127.0.0.1:30001 \
  --prefill-dp-rank 0 \
  --decode-dp-rank 1 \
  --prompt-chars 32000 \
  --max-tokens 256 \
  --requests 128 \
  --concurrency 16 \
  --output /tmp/host-kv-fixed-rank.json
```

The output records per-stage latency, failures, selected rank, token usage, and
aggregate latency percentiles.

## Preliminary Evaluation

An initial 4-prefill/4-decode DeepSeek-V3.2 experiment compared this prototype
with worker-owned PD transfer on the same eight nodes. The trace contained 1,024
requests with 32.8K average input tokens, 54.5K P90, and 99.5K maximum.

| Metric | Worker-owned PD | Host KV Pool | Change |
| --- | ---: | ---: | ---: |
| Request throughput | 1.3657 req/s | 1.3608 req/s | -0.36% |
| Mean TTFT | 219.41 s | 197.41 s | -10.03% |
| Mean end-to-end latency | 232.66 s | 214.15 s | -7.96% |
| Mean inter-token latency | 35.79 ms | 44.44 ms | +24.18% |
| Peak decode HBM token usage | 99% | 82% | -17 points |
| Maximum running requests per DP rank | 4 | 6 | +50% |

The result indicates that late binding removes decode-side KV preallocation and
creates HBM headroom. It moves backpressure to the prefill/Host KV queue and
allows more concurrent decodes, improving TTFT while increasing per-request
inter-token latency. Request throughput was effectively unchanged in this run.

These numbers are directional rather than a final performance claim: each mode
was measured once, and the lifecycle cleanup added after the run still requires
a long-running cluster validation.

## Current Limitations

- Mooncake is the only supported storage backend.
- MHA and MLA KV pools are supported; other pool types are rejected.
- Decode radix caching, decode KV offload, speculative decoding, and HiSparse
  are incompatible with this prototype.
- Active decode requests keep their KV resident in HBM until completion.
- Mid-decode detach, eviction, migration, and reattachment are not implemented.
- Host capacity still requires admission control and bounded-lifetime testing.
- The router waits for prefill completion before dispatching decode; it does not
  yet optimize placement using Host KV occupancy or transfer cost.

The next research step is to treat HBM as a managed cache: schedule attachment
globally, evict cold request state under pressure, and reattach the same KV
object to another decode worker without repeating prefill.
