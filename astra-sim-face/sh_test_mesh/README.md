# ASTRA-sim FACE Request Mapping on a 54-NPU Wafer Mesh

This directory contains a trace-driven implementation of the FACE request
mapping policy for ASTRA-sim's analytical backend. A FACE physical compute die
is represented by one ASTRA-sim NPU. A FACE instance is a configured rectangular
group of NPUs that can execute both Prefill and Decode.

## Repository-local hardware configuration

Physical mesh, D2D, local HBM, remote-memory bandwidth, and peak-compute
parameters are authored once in this repository:

`@astra-sim-face/sh_test_mesh/hardware/face_case5_config_c.json`

The FACE trace CSV selects the `validation-160gib` capacity profile and the
repository-local no-memory-expansion source. NPU count is derived from the mesh, while the
ASTRA-Sim-native files are generated automatically in:

`@astra-sim-face/sh_test_mesh/generated/runtime_config`

Generated files are not edit points.

## Checked-in instance layout

FACE does not publish a final optimal instance configuration list for this
modeled wafer layout.
The default scenario therefore uses a deterministic, non-optimality claim:

- Instance 0: rows 3-5, columns 2-3, ranks 20, 21, 26, 27, 32, 33.
- Instance 1: rows 0-2, columns 2-3, ranks 2, 3, 8, 9, 14, 15.
- Instance 2: rows 3-5, columns 0-1, ranks 18, 19, 24, 25, 30, 31.
- Instance 3: rows 3-5, columns 4-5, ranks 22, 23, 28, 29, 34, 35.
- Instance 4: rows 6-8, columns 2-3, ranks 38, 39, 44, 45, 50, 51.
- Instance 5: rows 0-2, columns 0-1, ranks 0, 1, 6, 7, 12, 13.
- Instance 6: rows 0-2, columns 4-5, ranks 4, 5, 10, 11, 16, 17.
- Instance 7: rows 6-8, columns 0-1, ranks 36, 37, 42, 43, 48, 49.
- Instance 8: rows 6-8, columns 4-5, ranks 40, 41, 46, 47, 52, 53.
- Every instance has shape 3x2 and TP degree 6.
- Every instance owns both a Prefill queue and a Decode queue.

The configuration order starts from the center instance, then its four direct
neighbors, then the four corners. This preserves the same 3x3 tiling while
avoiding a corner-first deterministic tie-break that can strand free HBM behind
increased weighted-distance edges during long multi-session workloads.

Equal instance sizes let the online graph emitter pair KV shards by relative
rank when a request moves between instances.

## Default workload: all requests in the normalized first three minutes

The simulated model is Meta LLaMA 2 7B: 32 decoder layers, hidden size 4096,
32 attention heads (128 dimensions per head), SwiGLU intermediate size 11008,
and vocabulary size 32000 in BF16/FP16 (2 bytes per element). FACE TP=6 does
not divide 4096, 11008, 32000, or 32, so ET generation assigns whole heads
unevenly (two ranks own six heads; four ranks own five) and exactly partitions
the remaining model dimensions. No model padding is introduced.

The default queue is the validated request-level window `0 <= derived arrival
<= 180s`: 678 sessions and 9,179 requests. `request_queue_session_limit=0`
means all of these sessions are used. Each session's first visible request is
turn zero with zero history, even if the source trace recorded an earlier
`prefix_len`; only prompt and decode tokens completed inside this window form
later context. Prefill ranges from 1 to 161,734 tokens, Decode from 1 to
32,000 tokens, and every window-local final context is below 1,000,000 tokens.

`@agent-traces/TraceLab_ASTRA_WSC_empirical_arrival_compute_80_100_120_v1/derived/compute_100_trunc1M_first_3_minutes/astra_compute_100_trunc1M_first_3_minutes_request_queue.csv`

## FACE mapping implemented by the planner

The pure Python planner (`face_scheduler.py`, kept as the shared policy library
and test oracle for the online schedulers) performs a deterministic
discrete-event pass:

1. `p_chunk` is the explicit configuration value 512 for both ordinary
   Prefill and history recompute chunks.
2. A new Prefill request chooses the instance with the lexicographically minimum
   key `(remaining chunk count, last enqueued arrival time, config order)`.
3. Each instance processes one FCFS Prefill chunk per estimated iteration while
   every active Decode request advances by one token.
4. After Prefill, the planner constructs `Instance_map` from current weighted
   shortest instance distances satisfying `distance <= D2D_BW / DRAM_BW`.
5. For each candidate, LUT lookup matches instance size, Prefill chunk size, and
   Decode batch size exactly, then selects the nearest Decode token length.
6. Decode chooses the minimum per-die incremental cost
   `(T_prime - T) / instance_size`, with config order as the final deterministic
   tie-break.
7. Completed KV stays as one complete session on its final Decode instance.
   Per-rank HBM reserves exact whole-head shards for a 1,000,000-token cache;
   on pressure the oldest inactive resident sessions are deleted by
   `(last_completion_ns, session_id)`. A resident history moves through NoC
   before Prefill; a deleted history is recomputed from its window-local
   logical context. These cache decisions never change FACE mapping.

The paper does not publish its numerical LUT or attention tile choices. The
checked-in planner builds only the required scheduling-time LUT columns with a
documented analytical Roofline estimate and exports the concrete rows used by
each run. It does not fabricate operator tile sizes.

Pure planner and tests:

`@astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`

`@astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`

FACE trace-configuration loader:

`@astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py`

Default trace configuration:

`@astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv`

## Online execution adaptation

FACE is a live host scheduler. The online strategy routes make FACE decisions
in-process at each decision boundary (arrival / prefill drain / decode
completion) and emit the resulting per-rank graph batches to the ASTRA-sim
execution-driven engine; decisions are not fixed ahead of time. The planner's
LUT models Prefill/Decode overlap at the iteration level, so Transformer nodes
in one ASTRA-sim rank's batch may serialize.

The default `trace_granularity=request_aggregated` keeps the real request queue
practical. For each request phase and rank it folds the real Prefill chunks or
Decode steps, plus identical Transformer layers, into 17 operator-category
nodes. The aggregate FLOPs, tensor/HBM bytes, optional remote reads, and
All-Reduce payload bytes equal the sums of the token-expanded trace. It
intentionally compresses repeated collective invocation and startup latency, so
it is a workload-volume-preserving approximation rather than a cycle-exact
replacement for `token_expanded`. Small workloads can still select
`token_expanded` in the trace config.

The selected TraceLab first-arrival offsets are around `1.7e16`-`2.0e16` ns,
which exceed the exact-integer range of an IEEE-754 `double`. The analytical
network adapter therefore keeps its 64-bit event time as `long double` when
returning ASTRA-sim time. This prevents a rounded callback timestamp from
missing the corresponding internal event-map key by one nanosecond.

Time-precision fix:

`@astra-sim-face/astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc`

Session history remains cumulative. KV stays on wafer instead of using the old
off-chip gateway pool. If the next turn maps Prefill to a different instance,
the previous KV shards cross the wafer mesh first. If Prefill and Decode map to
different instances, cumulative KV shards are transferred before Decode.

## Key configuration files

Single hardware source:

`@astra-sim-face/sh_test_mesh/hardware/face_case5_config_c.json`

Hardware-free system template:

`@astra-sim-face/sh_test_mesh/system/llama2_7b_roofline_template.json`

Repository-local no-memory-expansion source:

`@astra-sim-face/sh_test_mesh/remote_memory/no_memory_expansion.json`

Scenario and inference-group selection:

`@astra-sim-face/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv`

Request queue:

`@astra-sim-face/sh_test_mesh/workload/workload_request_queue_tracelab.csv`

Generated runtime files (do not edit):

`@astra-sim-face/sh_test_mesh/generated/runtime_config`

## Run and validate

The online strategy routes are the supported pipeline (route 3 = strategy,
route 4 = strategy + sensing). Inputs are materialized per
`traces/PROVENANCE.md` (the only allowed source is
`agent-traces/tracelab/astra_compute_20.csv`), then the plan directory is
produced by the materializer:

```bash
cd sh_test_mesh/workload/llama2_7b_inference
# 1. materialize the request-queue input (rules: traces/PROVENANCE.md)
# 2. materialize the plan directory + runtime_config four-piece set
python3 plan_materializer.py
cd ../../..
# 3. run the online strategy (route 3) and the sensing variant (route 4)
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <abs request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <abs request_csv>
# legacy second variant (strategy mode + trace_config_legacy.csv)
bash sh_test_mesh/run_scripts/run_online_strategy_legacy.sh <run_dir> <request_csv> <legacy_gen>
# metrics post-processing of a run's cpp.log
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
```

Unit tests (workload layer + sh_test_mesh contracts):

```bash
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py test_checkpointing.py -q
cd ../.. && python3 -m pytest tests/ -q
```

Each generated plan directory contains:

- `manifest.json`: queue-derived per-request facts (9 fields; existence
  credential for the online runner's `--plan-dir`).
- `metrics_manifest.json`: synthetic-prerun metrics manifest (rank-attribution
  fields are placeholders; request-level metrics are trustworthy).

Generated files are under:

`@astra-sim-face/sh_test_mesh/generated`
