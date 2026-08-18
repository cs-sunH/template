# ASTRA-sim FACE Request Mapping with HBM-Aware Session KV Migration

This directory contains a trace-driven implementation of the FACE request
mapping policy for ASTRA-sim's analytical backend, extended with per-NPU HBM
accounting and historical session KV migration. One FACE physical compute die
is represented by one ASTRA-sim NPU. A FACE instance is a configured rectangular
group of NPUs that executes both Prefill and Decode.

The extension does not change FACE's Prefill queue selection, Decode candidate
construction, LUT lookup, or Prefill-to-Decode instance selection. FACE chooses
the execution instance first. Historical KV location is then resolved, and any
required transfer delays execution without remapping the request.

## Repository-local hardware configuration

Physical mesh, D2D, local HBM, remote-memory bandwidth, and peak-compute
parameters are authored only once in this repository:

`@astra-sim-sh/sh_test_mesh/hardware/face_case5_config_c.json`

This scenario selects the `validation-160gib` local-HBM capacity profile in
its trace CSV. The profile selection is scenario-specific; all common hardware
values remain in the repository-local source. The mesh NPU count is derived from rows and
columns rather than configured separately.

The remote-memory source below defines only the edge-pool policy and its own
latency. Boundary ranks and bandwidth are derived from the repository-local hardware:

`@astra-sim-sh/sh_test_mesh/remote_memory/edge_remote_memory_pool.json`

At load time, the repository-local resolver generates the ASTRA-Sim-native system,
network, remote-memory, and communicator files in the runtime directory:

`@astra-sim-sh/sh_test_mesh/generated/runtime_config`

These files are derived output and must not be edited manually.

The trace configuration sets `kv_reserve_context_tokens=1000000`. The reserve
is evaluated independently for the exact whole-head shard stored by each NPU.

## Checked-in instance layout

FACE does not publish a final optimal instance configuration list for this
modeled wafer layout.
The default scenario therefore uses a deterministic layout without claiming
optimality:

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
neighbors, then the four corners. It remains the final deterministic tie-break
for Prefill; Decode uses it only when both per-die cost and current aggregate
remaining HBM capacity are equal.

## Default workload: requests arriving in the first three minutes

The simulated model is Meta LLaMA 2 7B: 32 decoder layers, hidden size 4096,
32 attention heads with 128 dimensions per head, SwiGLU intermediate size
11008, vocabulary size 32000, and BF16/FP16 elements of 2 bytes.

TP=6 does not divide the head count. Each complete session KV cache is therefore
stored on exactly one TP instance using exact whole-head shards: two relative
TP ranks own six heads each and four own five heads each. The six shard sizes
sum exactly to the complete session KV size; no model or KV padding is added,
and one session is not capacity-split across multiple instances.

The default config reads the selected native `astra_compute_100_trunc1M.csv`
through its checked-in normalized queue. It retains only requests in the
inclusive simulation-time window `0 <= t <= 180,000,000,000 ns`: request zero
uses `arrival_time`, and each following request adds the preceding row's
`human_time` or `tool_time`. A session can therefore end at its last in-window
request rather than retaining a later tail:

- source session IDs cover `0` through `677` (678 sessions, 9,179 requests);
- Prefill length range 1-161,734 tokens;
- Decode length range 1-32,000 tokens;
- session first-arrival offsets span 26,249,000-179,437,413,000 ns, without
  rebasing; the latest retained request is at 179,937,766,000 ns.

The adapter moves each native row's outgoing `human_time` or `tool_time`
onto the following request's closed-loop interval. The 20 retained nonterminal
rows with neither value are explicitly represented as a zero-nanosecond wait.

Native source CSV:

`@agent-traces/TraceLab_ASTRA_WSC_empirical_arrival_compute_80_100_120_v1/astra_compute_100_trunc1M.csv`

Normalized request queue (the prefix-context sidecar is intentionally not
loaded):

`@agent-traces/TraceLab_ASTRA_WSC_empirical_arrival_compute_80_100_120_v1/derived/compute_100_trunc1M_first_3_minutes/astra_compute_100_trunc1M_first_3_minutes_request_queue.csv`

For every session, the first request observable in this three-minute window is
treated as turn zero with no prior KV or prompt context. The trace therefore
does not reconstruct any context from before the window; later retained
requests accumulate only context produced by earlier retained requests from
the same session.

## Request mapping implemented by the planner

The pure Python planner performs a deterministic discrete-event pass before ET
generation:

1. `p_chunk` is fixed at 512 tokens.
2. The selected 9,179-row queue has arithmetic mean Decode length
   `463.43926353633293` tokens. Active Decode load uses this mean and becomes zero
   once a request has generated at least that many tokens.
3. Each instance records remaining Roofline service-time load in nanoseconds:
   the unfinished fraction of its running Prefill chunk, all queued Prefill
   chunks, and all active Decode requests. Prefill chunk work uses prior-session
   KV plus the processed prompt; Decode work uses the complete current context.
4. A new Prefill request chooses the lexicographically minimum key
   `(total remaining task load in ns, last arrival time, config order)`;
   a never-used instance has arrival time `-1`. A partially resident session
   remains pinned to the instance holding its prefix layers.
5. Each instance processes one FCFS Prefill chunk per estimated iteration while
   every active Decode request advances by one token.
6. After Prefill, the planner constructs `Instance_map` from current weighted
   shortest instance distances satisfying
   `distance <= D2D_BW / DRAM_BW`.
7. Each candidate LUT lookup exactly matches instance size, Prefill chunk size,
   and Decode batch size, then selects the nearest Decode token length.
8. Decode chooses the minimum per-die incremental cost
   `(T_prime - T) / instance_size`. Candidates tied at the minimum cost prefer
   the greatest current sum of remaining HBM bytes across all NPU ranks in the
   instance; config order is used only if that capacity is also equal.
9. HBM capacity does not override a lower Decode cost, and current KV location
   or transfer cost does not affect selection. After the cost/capacity decision,
   the HBM/KV subsystem prepares the selected instance before execution.

Pure planner and tests:

`@sh_test_mesh/workload/llama2_7b_inference/face_scheduler.py`

`@sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py`

FACE ET integration:

`@sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py`

## Per-NPU HBM accounting

Every physical NPU independently tracks:

- model-weight shard bytes following the exact whole-head attention, uneven
  FFN, and uneven vocabulary extents for its relative TP rank;
- locally resident session KV shard bytes;
- total used HBM bytes;
- remaining HBM bytes.

The invariant is:

```text
used_hbm_bytes = model_weight_bytes + local_kv_bytes
remaining_hbm_bytes = capacity_bytes - used_hbm_bytes
```

A session records a model-derived resident-layer prefix. Its state is one of:

- `local_hbm`: all `L` layers are local;
- `partial_hbm_remote`: layers `[0, ceil(L/2))` remain on the owning TP
  instance and `[ceil(L/2), L)` are remote;
- `remote_memory`: all layers are remote.

`L` always comes from the configured model; no layer count is hard-coded.

## Historical KV preparation before a request

FACE first maps the request to its Prefill instance. A new session has no
historical KV and starts directly. A continuing session prepares its history on
that mapped Prefill instance through one of four deterministic paths:

1. **Local hit.** If all historical shards already reside on the selected
   instance, the request reuses them without communication.
2. **Other local instance.** Each whole-head shard moves between corresponding
   relative TP ranks along a minimum-hop physical NoC path. The destination must
   receive the complete session before inference starts, and source HBM is
   released only after the corresponding destination transfer completes.
3. **Unified remote pool.** Each destination rank selects the configured remote
   edge rank with minimum Manhattan hop count; equal-distance edges are ordered
   by rank ID. The shard is loaded from the unified pool at that edge and then
   transferred through the NoC to the destination rank. Restore completion and
   the instance-wide dependency gate must finish before inference starts.
4. **Resident prefix plus remote suffix.** Prefill is pinned to the instance
   retaining the prefix. The suffix restore starts in a parallel ET branch.
   The first iteration executes resident layers immediately; its first suffix
   operator has both the prefix-compute predecessor and target-HBM DMA completion
   as dependencies. Thus it waits only if the suffix is not ready in time.

Full-local, NoC-migrated, and fully remote histories retain the complete-KV
readiness barrier. A partial history instead uses a resident-prefix readiness
barrier and a suffix-boundary gate, removing the previous whole-session restore
barrier.

Every remote load is followed by an explicit target-HBM DMA node. On each NPU,
if inference and this DMA simultaneously have outstanding HBM bytes, the runtime
gives each exactly 50% of that NPU's HBM bandwidth. When either side finishes
its HBM work, the survivor immediately returns to 100%; this is event-driven,
not a fixed 2x-duration approximation.

The pool has a unified logical address space, so a restore may use a different
edge from the one used for the earlier store.

After Prefill, FACE independently selects the Decode instance with its LUT cost
policy and the HBM-capacity tie-break described above. If it differs from the
Prefill instance, the current cumulative KV moves between corresponding
relative TP ranks through the same minimum-hop local-instance transfer
mechanism. The Decode mapping is not reconsidered after this decision, and an
equivalent Decode-group readiness barrier completes before any Decode rank
starts computation. The session's local authoritative location after completion
is the Decode instance.

## Two-stage FIFO migration after request completion

After a request completes, every rank in the owning TP instance checks whether
its remaining HBM can hold its exact 1M-token KV shard reserve. If any rank is
below its reserve, locally resident historical sessions are ordered by:

```text
(last_completion_ns, session_id)
```

Reclamation has two strictly ordered stages:

1. Visit inactive, fully local sessions in FIFO order and offload only their
   latter `floor(L/2)` layers. Recheck the watermark after every session.
2. Only if every eligible full-local session has been halved and the watermark
   is still unmet, revisit resident sessions in FIFO order and offload each
   remaining prefix, making that session fully remote.

An executing session is excluded in both stages. If no eligible history can
satisfy the reserve, the planner records `reserve_unmet` and terminates.

For a shard already on an edge rank, the edge issues the remote store directly.
For an internal rank, the shard first follows a minimum-hop physical NoC path
to its nearest configured edge, with edge rank as the deterministic tie-break.
The edge issues `MEM_STORE`, and a store-completion ACK gates reclamation at
the source. Local HBM is not released when the send begins.

The configured remote-memory edges are all 26 physical boundary ranks of the
9x6 mesh:

```text
0,1,2,3,4,5,6,11,12,17,18,23,24,29,30,35,36,41,42,47,48,49,50,51,52,53
```

The source owns the boundary-selection policy, latency, and logical-pool name:

`@astra-sim-sh/sh_test_mesh/remote_memory/edge_remote_memory_pool.json`

The concrete edge list is derived from the repository-local mesh and the
bandwidth is derived from the repository-local hardware source.

## Online execution adaptation (Chakra node semantics)

FACE is a live host scheduler. The online strategy routes compute mappings, HBM
state transitions, session locations, FIFO evictions, and routes at each
decision boundary, then emit the resulting per-rank graph batches to the
execution-driven engine (54 ranks).

The emitted graph expresses remote store as NoC transfer, edge `MEM_STORE`, and
completion ACK. Remote restore is edge `MEM_LOAD`, NoC delivery, and target-HBM DMA. For a
partial session, dependency-chain checkpoint/restore creates parallel suffix
load and prefix-compute branches that rejoin at the suffix boundary. The C++
local-HBM fluid model dynamically enforces the per-NPU 50/50 sharing rule.

The default `trace_granularity=request_aggregated` folds repeated Transformer
layers, Prefill chunks, and Decode steps into operator-category nodes. Aggregate
FLOPs, tensor/HBM bytes, KV migration bytes, and collective payload bytes are
preserved, while repeated collective startup costs are compressed. Small
workloads can still select `token_expanded`.

The selected ASTRA-compute sessions have first-arrival offsets from 26,249,000
to 179,437,413,000 ns. The analytical network adapter retains event time as
`long double` when returning ASTRA-sim time so large 64-bit nanosecond
event-map keys remain stable.

Time-precision implementation:

`@astra-sim-sh/astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc`

Each generated `manifest.json` records the FACE mapping decisions together
with:

- per-NPU model, local KV, used, and remaining bytes, plus the configured
  reserve context length;
- session KV locations and exact shard sizes;
- local migrations, remote stores, remote restores, edge choices, paths, and
  byte counts;
- per-request completion HBM snapshots and transfer-completion dependencies;
- FIFO order, eviction reasons, and any `reserve_unmet` result;
- LUT queries, candidate costs, planning iterations, and final state.

## Key configuration files

Single hardware source:

`@astra-sim-sh/sh_test_mesh/hardware/face_case5_config_c.json`

Hardware-free system template:

`@astra-sim-sh/sh_test_mesh/system/llama2_7b_roofline_template.json`

Shared resolver and generated-file contract:

`@astra-sim-sh/sh_test_mesh/config_resolver.py`

Edge-attached remote-memory policy:

`@astra-sim-sh/sh_test_mesh/remote_memory/edge_remote_memory_pool.json`

Default trace configuration:

`@astra-sim-sh/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv`

Generated runtime files (do not edit):

`@astra-sim-sh/sh_test_mesh/generated/runtime_config`

## Run and validate

Run from the repository root. The online strategy routes are the supported
pipeline (route 3 = strategy, route 4 = strategy + sensing). Inputs are
materialized per `traces/PROVENANCE.md` (only allowed source:
`agent-traces/tracelab/astra_compute_20.csv`), then the plan directory is
produced by the materializer:

```bash
# 1. materialize the 30s request-queue input (rules: traces/PROVENANCE.md)
cd sh_test_mesh/workload/llama2_7b_inference
python3 traces/materialize_first_30s.py \
  /home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv traces/
# 2. materialize the plan directory + runtime_config four-piece set
python3 plan_materializer.py
cd ../../..
# 3. online strategy (route 3) and sensing variant (route 4)
bash sh_test_mesh/run_scripts/run_online_strategy.sh <run_dir> <abs request_csv>
bash sh_test_mesh/run_scripts/run_online_strategy_sensing.sh <run_dir> <abs request_csv>
# metrics post-processing of a run's cpp.log
bash sh_test_mesh/run_scripts/run_metrics_postprocess.sh <run_dir>/cpp.log
```

Unit tests (workload layer + sh_test_mesh contracts):

```bash
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py test_checkpointing.py -q
cd ../.. && python3 -m pytest tests/ -q
```

Each generated plan directory contains `manifest.json` (queue-derived 9
fields; sidecar_restore variant) and `metrics_manifest.json` (synthetic-prerun; rank
attribution is placeholder).

Generated artifacts:

`@sh_test_mesh/generated`
