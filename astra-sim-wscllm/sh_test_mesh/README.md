# ASTRA-sim WSC-LLM PD Request Mapping on a 54-NPU Wafer Mesh

This directory implements WSC-LLM-style Prefill/Decode disaggregation in the
trace-generation planner. The placement unit is one existing instance. One
instance still contains exactly the same 3x2 rectangle of six ASTRA-sim NPUs
and uses its existing TP=6 communicator; rank membership, model, requests,
hardware source, generated runtime semantics, and simulator C++ are unchanged.

## Repository-local physical configuration

The physical mesh, D2D, local HBM, remote-memory bandwidth, and peak-compute
parameters are authored once in this repository:

`@astra-sim-wscllm/sh_test_mesh/hardware/face_case5_config_c.json`

This scenario selects the `validation-160gib` capacity profile. It defines only the
WSC-LLM-specific instance roles and rank membership in its scenario CSV:

`@astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv`

NPU count, network dimensions, network performance, system hardware fields,
and communicator dimensions are derived automatically. Runtime files are
generated in:

`@astra-sim-wscllm/sh_test_mesh/generated/runtime_config`

These generated files must not be edited.

Repository-local system template:

`@astra-sim-wscllm/sh_test_mesh/system/llama2_7b_roofline_template.json`

The no-memory-expansion remote-memory setting is embedded in the
`remote-memory` section of the hardware source above (the config resolver
reads it directly -- there is no standalone remote-memory source file).

## Dedicated 6P/3D instance layout

The config now carries an explicit `phase_role` for every inference group.
Decode-only instances occupy the middle row of the 3x3 instance tiling; all
top/bottom-row instances are Prefill-only:

| Config index | Role | Physical location | Ranks |
|---:|---|---|---|
| 0 | Decode | center | 20, 21, 26, 27, 32, 33 |
| 1 | Prefill | north | 2, 3, 8, 9, 14, 15 |
| 2 | Decode | west middle | 18, 19, 24, 25, 30, 31 |
| 3 | Decode | east middle | 22, 23, 28, 29, 34, 35 |
| 4 | Prefill | south | 38, 39, 44, 45, 50, 51 |
| 5 | Prefill | northwest | 0, 1, 6, 7, 12, 13 |
| 6 | Prefill | northeast | 4, 5, 10, 11, 16, 17 |
| 7 | Prefill | southwest | 36, 37, 42, 43, 48, 49 |
| 8 | Prefill | southeast | 40, 41, 46, 47, 52, 53 |

The planner validates that both roles exist and that the maximum Manhattan
distance from any Decode instance center to the wafer center is no greater
than the minimum corresponding Prefill distance.

WSC-LLM does not publish the final LLaMA2-7B/workload instance allocation.
The checked-in 6P:3D = 2:1 split is a deterministic simulation assumption, not
an optimality claim. It is grounded in both paper examples: Fig. 7 uses
`NP=8, ND=4`, and Fig. 8 depicts three eight-die Prefill queues plus three
four-die Decode queues, again 24:12 dies. The user-required six-NPU instance is
kept fixed instead of reproducing the paper's offline instance-size/TP search.

Authoritative role and rank configuration:

`@astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv`

## Static Prefill-to-Decode mapping policy

Routing is built once at whole-instance granularity. For each Prefill instance,
the planner finds the nearest Decode instance, enumerates equal-length shortest
instance paths, then selects a globally deterministic combination ordered by:

1. minimum total hops;
2. minimum shared-edge occurrences;
3. config-order/path signature.

The checked-in layout produces six one-hop, edge-disjoint routes:

```text
P1 -> D0
P4 -> D0
P5 -> D2
P6 -> D3
P7 -> D2
P8 -> D3
```

An `alpha >= 1` adjusted-cost metadata field is retained for describing shared
links. WSC-LLM does not publish a general alpha formula, and alpha is irrelevant
for this layout because every route is edge-disjoint. Decode is never selected
again from runtime load, timing-LUT deltas, or a dynamically expanded candidate
set.

## Runtime queue and phase behavior

## KV lifecycle override

The request-queue input is the materialized first-30-seconds recompute window
of `agent-traces/tracelab/astra_compute_20.csv` (placeholder state; nothing is
checked in). The first visible request of every session starts with zero
history; no source prefix-context sidecar is loaded. Prefill and history
recompute chunks are fixed at 512 tokens.

Completed KV remains on the static Decode target; terminal sessions are not
automatically released. Every physical TP rank reserves the exact whole-head
shard for one decimal-million-token KV cache. If a rank falls below this
watermark, inactive resident sessions on that instance are deleted in
`(last_completion_ns, session_id)` order. A later turn uses NoC migration when
its KV is still resident and recomputes all window-local history when it was
deleted. Neither action changes the least-request-count Prefill choice or the
static P→D route. Cache actions are recorded per request in the online
decision log; deletes have zero compute cost and no remote-memory traffic.

When a request arrives, only Prefill instances are eligible. The unpublished
WSC-LLM phrase “least occupied queue” is interpreted as the number of requests
currently in `QP`, including the active FCFS head. Config order is the final
unpublished tie-break. Remaining chunks and last-arrival time are not used.

Each Prefill-only instance executes one FCFS chunk per planner iteration. Each
Decode-only instance advances every request in its active continuous batch by
one token per iteration, preserving FCFS insertion order. No iteration can
contain both phases. Prefill completion sends the request directly to the
Decode queue fixed by the static route and records its queue depth and route.

The analytical table is now a phase-timing LUT only. Every entry is either a
Prefill workload or a Decode workload; it does not choose an instance or alter
the static mapping.

Planner implementation and tests:

`@astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/wsc_llm_scheduler.py`

`@astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/test_wsc_llm_scheduler.py`

## Session-KV residency and legacy allocator boundary

The default `session_lru_recompute` policy does not use a `Relevant(P,D)`
multi-location allocation domain. A session's complete KV is resident in one
TP instance (the completed request's static Decode target), or it is fully
deleted. Before Prefill, the planner reserves the fixed Decode target's final
KV footprint and then admits the original Prefill choice. A target that is
temporarily constrained by active KV keeps the request at that fixed mapping
until a completion permits a retry; neither P nor D is remapped.

`WscRelevantKvAllocator` remains in the scheduler only as a regression-tested
legacy-policy implementation. It is not selected by the checked-in workload,
does not model the default session lifecycle, and must not be used to spill a
default session across P/D instances.

No off-chip expansion is used: the `remote-memory` section of the hardware
source selects `NO_MEMORY_EXPANSION` (embedded there; no standalone
remote-memory source file exists).

## Model, requests, and ET boundary

The model remains Meta LLaMA 2 7B: 32 layers, hidden size 4096, 32 attention
heads, SwiGLU size 11008, vocabulary 32000, and two bytes per element. TP=6 is
unchanged. Whole heads are distributed exactly (two ranks receive six heads and
four receive five), while other dimensions use exact uneven shards without
model padding.

The request-queue input is not checked in (request-neutral bare repo): the
caller materializes the official first-30-seconds window of
`agent-traces/tracelab/astra_compute_20.csv` (recompute variant) via the
traces/ script and passes it in explicitly. Window-visible turn zero always
starts at zero history, and the explicit Prefill/history-recompute chunk size
is 512.

All Prefill operators are emitted on the selected Prefill instance's six ranks;
all Decode operators are emitted on its fixed Decode instance's six ranks. KV
movement pairs equal relative TP ranks.

The scheduler models Decode continuous batching for timing. The default
`request_aggregated` emission folds each request's Decode phase into its own
aggregated node set rather than one fused cross-request batch node.
Aggregation preserves total FLOPs, tensor/HBM bytes, and All-Reduce payload,
while compressing repeated invocations and startup latency.

Trace-configuration loader:

`@astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/generate_wsc_llm_trace.py`

## Run and validate

The online strategy routes are the supported pipeline (route 3 = strategy,
route 4 = strategy + sensing). This is a request-neutral bare repo: the
request-queue input is materialized by the caller per the repository plan doc
§3 steps 0-1 (recompute variant, same rule family as the face repo's
`traces/derive_20_first_30_seconds.py`; only allowed source:
`agent-traces/tracelab/astra_compute_20.csv`) and passed in explicitly; the
plan directory is then produced by the in-repo materializer:

```bash
cd sh_test_mesh/workload/llama2_7b_inference
# 1. caller materializes the request-queue input (plan doc §3 steps 0-1)
# 2. materialize the plan directory + runtime_config four-piece set
python3 plan_materializer.py
cd ../../..
# 3. online strategy (route 3) and sensing variant (route 4)
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <abs request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <abs request_csv>
# legacy second variant (strategy mode + trace_config_legacy.csv)
bash sh_test_mesh/run_scripts/run_online_strategy_legacy.sh <run_dir> <request_csv> <legacy_gen>
# metrics post-processing of a run's cpp.log
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
```

Unit tests (workload layer + sh_test_mesh contracts):

```bash
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_wsc_llm_scheduler.py test_wsc_llm_legacy_online_scheduler.py -q
cd ../.. && python3 -m pytest tests/ -q
```

Each generated plan directory contains `manifest.json` (queue-derived 9
fields) and `metrics_manifest.json` (synthetic-prerun; rank attribution is
placeholder).

`@astra-sim-wscllm/sh_test_mesh/generated`
