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
neighbors, then the four corners. It is also the final deterministic tie-break
for otherwise equal FACE mapping decisions.

## Default workload: requests arriving in the first three minutes

The simulated model is Meta LLaMA 2 7B: 32 decoder layers, hidden size 4096,
32 attention heads with 128 dimensions per head, SwiGLU intermediate size
11008, vocabulary size 32000, and BF16/FP16 elements of 2 bytes.

TP=6 does not divide the head count. Each complete session KV cache is therefore
stored on exactly one TP instance using exact whole-head shards: two relative
TP ranks own six heads each and four own five heads each. The six shard sizes
sum exactly to the complete session KV size; no model or KV padding is added,
and one session is not capacity-split across multiple instances.

The default config reads the normalized queue exported from the selected native
`astra_compute_100_trunc1M.csv`. It retains only requests in the inclusive
simulation-time window `0 <= t <= 180,000,000,000 ns`: request zero uses
`arrival_time`, then each following request adds the preceding row's
`human_time` or `tool_time`. Source session IDs `0` through `677` contribute
678 sessions and 9,179 requests; their later request tails are excluded.
Arrival timestamps are not rebased:

- Prefill length range 1-161,734 tokens;
- Decode length range 1-32,000 tokens;
- first-arrival range 26,249,000-179,437,413,000 ns; latest retained request
  arrival is 179,937,766,000 ns.

The adapter moves a native row's outgoing `human_time` or `tool_time` to
the following request's interval. It explicitly uses a zero-nanosecond interval
for the 20 retained nonterminal rows that omit both values.

The canonical sidecar is intentionally not consumed here: the existing FACE
instance mapping and KV-management logic continue to use the repository's
original request semantics.

Authoritative source and normalized workload input:

`@../../agent-traces/TraceLab_ASTRA_WSC_empirical_arrival_compute_80_100_120_v1/astra_compute_100_trunc1M.csv`

`@../../agent-traces/TraceLab_ASTRA_WSC_empirical_arrival_compute_80_100_120_v1/derived/compute_100_trunc1M_first_3_minutes/astra_compute_100_trunc1M_first_3_minutes_request_queue.csv`

## FACE mapping preserved by the planner

The pure Python planner performs a deterministic discrete-event pass before ET
generation:

1. `p_chunk` is the fixed `prefill_chunk_size` in `trace_config.csv`; the
   checked-in workload uses `p_chunk=512`.
2. A new Prefill request chooses the instance with the lexicographically minimum
   key `(remaining chunk count, last enqueued arrival time, config order)`.
3. Each instance processes one FCFS Prefill chunk per estimated iteration while
   every active Decode request advances by one token.
4. After Prefill, the planner constructs `Instance_map` from current weighted
   shortest instance distances satisfying
   `distance <= D2D_BW / DRAM_BW`.
5. Each candidate LUT lookup exactly matches instance size, Prefill chunk size,
   and Decode batch size, then selects the nearest Decode token length.
6. Decode chooses the minimum per-die incremental cost
   `(T_prime - T) / instance_size`, with config order as the final tie-break.
7. The selected instance is not changed in response to HBM pressure, current KV
   location, or transfer cost. The HBM/KV subsystem prepares that selected
   instance before execution.

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

A session has one authoritative location: one local TP instance or the unified
remote KV pool. A local location represents all six exact whole-head shards of
that session on the six ranks of the instance.

## Historical KV preparation before a request

FACE first maps the request to its Prefill instance. A new session has no
historical KV and starts directly. A continuing session prepares its history on
that mapped Prefill instance through one of three deterministic paths:

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

After history preparation and any Prefill-capacity evictions, a one-byte
TP-group All-Reduce acts as an explicit readiness barrier. No Prefill rank may
start Transformer computation until all six target KV shards are ready.

The pool has a unified logical address space, so a restore may use a different
edge from the one used for the earlier store.

After Prefill, FACE independently selects the Decode instance with its original
LUT policy. If it differs from the Prefill instance, the current cumulative KV
moves between corresponding relative TP ranks through the same minimum-hop
local-instance transfer mechanism. The Decode mapping is not reconsidered, and
an equivalent Decode-group readiness barrier completes before any Decode rank
starts computation. The session's local authoritative location after
completion is the Decode instance.

## FIFO migration after request completion

After a request completes, every rank in the owning TP instance checks whether
its remaining HBM can hold its exact 1M-token KV shard reserve. If any rank is
below its reserve, locally resident historical sessions are ordered by:

```text
(last_completion_ns, session_id)
```

The earliest session is migrated as one complete unit: all six whole-head
shards move to the remote pool. The check repeats until every rank satisfies its
reserve or no eligible local history remains. If the configured capacity or
current state makes the reserve unattainable, the planner stops after finite
work and records `reserve_unmet`; it never loops indefinitely.

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

The emitted graph expresses remote store as NoC transfer, edge `MEM_STORE`,
and completion ACK. Remote restore is expressed as edge `MEM_LOAD`, NoC
delivery, and a dependency gate before compute. Prefill and Decode each add an
explicit TP-group readiness collective after their KV preparation stage. These
dependencies preserve the required ordering while the online scheduler makes
each decision at runtime.

The default `trace_granularity=request_aggregated` folds repeated Transformer
layers, Prefill chunks, and Decode steps into operator-category nodes. Aggregate
FLOPs, tensor/HBM bytes, KV migration bytes, and collective payload bytes are
preserved, while repeated collective startup costs are compressed. Small
workloads can still select `token_expanded`.

The selected ASTRA-compute first-arrival offsets range from `26,249,000` to
`179,437,413,000` ns.
The analytical network adapter retains event time as `long double` when
returning ASTRA-sim time so 64-bit event-map keys are not missed by one
nanosecond.

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
python3 traces/derive_20_first_30_seconds.py \
  /home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv traces/
# 2. materialize the plan directory + runtime_config + face_lut.csv
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
fields), `metrics_manifest.json` (synthetic-prerun; rank attribution is
placeholder), and `face_lut.csv` (contract-9 frozen LUT).

Generated artifacts:

`@sh_test_mesh/generated`
