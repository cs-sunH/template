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

`@astra-sim-sh_2.0/sh_test_mesh/hardware/face_case5_config_c.json`

This scenario selects the `validation-160gib` local-HBM capacity profile in
its trace CSV. The profile selection is scenario-specific; all common hardware
values remain in the repository-local source. The mesh NPU count is derived from rows and
columns rather than configured separately.

The edge-pool remote-memory policy (memory type, latency, boundary selection,
logical-pool name) is embedded in the `remote-memory` section of the hardware
source above; the resolver reads it directly and derives the concrete boundary
ranks and bandwidth from the repository-local hardware. There is no standalone
remote-memory source file.

At load time, the repository-local resolver generates the ASTRA-Sim-native system,
network, remote-memory, and communicator files in the runtime directory:

`@astra-sim-sh_2.0/sh_test_mesh/generated/runtime_config`

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

## Default workload: materialized first-30-seconds window

The simulated model is Meta LLaMA 2 7B: 32 decoder layers, hidden size 4096,
32 attention heads with 128 dimensions per head, SwiGLU intermediate size
11008, vocabulary size 32000, and BF16/FP16 elements of 2 bytes.

TP=6 does not divide the head count. Each complete session KV cache is therefore
stored on exactly one TP instance using exact whole-head shards: two relative
TP ranks own six heads each and four own five heads each. The six shard sizes
sum exactly to the complete session KV size; no model or KV padding is added,
and one session is not capacity-split across multiple instances.

This bare template repository ships no checked-in request queue (placeholder
state); the runner fails closed on a missing `--request-queue-csv` input. The
only allowed source trace is `agent-traces/tracelab/astra_compute_20.csv`, and
the official input is its first-30-seconds window (sidecar_restore variant:
the turn-0 prefix is kept as historical KV, see the context sidecar),
materialized by the traces/ script (the run commands below show the exact
invocation):

`@astra-sim-sh_2.0/sh_test_mesh/workload/llama2_7b_inference/traces/materialize_first_30s.py`

## Request mapping implemented by the planner

The pure Python planner performs a deterministic discrete-event pass before ET
generation:

1. `p_chunk` is fixed at 512 tokens.
2. Active Decode load uses the arithmetic-mean Decode length of the
   materialized request queue and becomes zero once a request has generated
   at least that many tokens.
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

## Typed two-class, two-stage eviction after request completion

Each queued request carries a `next_trigger_type` column (`human`/`tool`,
materialized from the source trace's per-row human_time/tool_time return
path). When a request completes, `mark_complete` records that type on the
session; an idle session's class is `tool` only when its latest completed
request was followed by a tool-call return, everything else -- including
sessions with no recorded successor -- is the `human` class.

After a request completes, every rank in the owning TP instance checks
whether its remaining HBM can hold its exact 1M-token KV shard reserve. If
any rank is below its reserve, reclamation visits trigger classes in the
fixed order `human` then `tool`. Within each class, eligible sessions are
ordered by:

```text
(last_completion_ns, session_id)
```

and reclamation has two strictly ordered stages, giving the four-segment
order `human-half -> human-full -> tool-half -> tool-full`:

1. Visit inactive, fully local sessions of the class in oldest-completed
   order and offload only their latter `floor(L/2)` layers. Recheck the
   watermark after every session.
2. Only if every eligible full-local session of the class has been halved
   and the watermark is still unmet, revisit resident sessions of the same
   class in the same order and offload each remaining prefix, making that
   session fully remote.
3. Move to the next trigger class only after both stages of the previous
   class are exhausted.

The watermark is rechecked after every single eviction, so the pass stops as
soon as the reserve is met. An executing session is excluded in both stages.
If no eligible history can satisfy the reserve, the planner records
`reserve_unmet` and terminates.

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

The boundary-selection policy, latency, and logical-pool name are owned by the
`remote-memory` section of the repository-local hardware source (the resolver
reads it directly; there is no standalone remote-memory source file). The
concrete edge list is derived from the repository-local mesh and the
bandwidth is derived from the repository-local hardware source.

## Online execution adaptation (Chakra node semantics)

FACE is a live host scheduler. The online strategy routes compute mappings, HBM
state transitions, session locations, typed-class evictions, and routes at each
decision boundary, then emit the resulting per-rank graph batches to the
execution-driven engine (54 ranks).

The emitted graph expresses remote store as NoC transfer, edge `MEM_STORE`, and
completion ACK. Remote restore is edge `MEM_LOAD`, NoC delivery, and target-HBM DMA. For a
partial session, dependency-chain checkpoint/restore creates parallel suffix
load and prefix-compute branches that rejoin at the suffix boundary. The C++
local-HBM fluid model (`hbm-bandwidth-contention`, default on) enforces a strict
N-way equal split of the per-NPU HBM bandwidth across every active HBM user
(inference COMP, KV-restore DMA, NoC p2p comm data endpoints, pool-traffic
endpoints), with event-driven reallocation on every completion; NoC/edge
pass-through traffic is never charged (only data endpoints are, exactly once
per byte flow; see README section D).

The default `trace_granularity=request_aggregated` folds repeated Transformer
layers, Prefill chunks, and Decode steps into operator-category nodes. Aggregate
FLOPs, tensor/HBM bytes, KV migration bytes, and collective payload bytes are
preserved, while repeated collective startup costs are compressed. Small
workloads can still select `token_expanded`.

The selected ASTRA-compute source trace carries first-arrival offsets from
`94,835,000` to `44,609,097,174,000` ns. The analytical network adapter retains event time as
`long double` when returning ASTRA-sim time so large 64-bit nanosecond
event-map keys remain stable.

Time-precision implementation:

`@astra-sim-sh_2.0/astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc`

Each generated `manifest.json` records the FACE mapping decisions together
with:

- per-NPU model, local KV, used, and remaining bytes, plus the configured
  reserve context length;
- session KV locations and exact shard sizes;
- local migrations, remote stores, remote restores, edge choices, paths, and
  byte counts;
- per-request completion HBM snapshots and transfer-completion dependencies;
- FIFO order within each trigger class, eviction reasons, and any
  `reserve_unmet` result;
- LUT queries, candidate costs, planning iterations, and final state.

## Key configuration files

Single hardware source:

`@astra-sim-sh_2.0/sh_test_mesh/hardware/face_case5_config_c.json`

Hardware-free system template:

`@astra-sim-sh_2.0/sh_test_mesh/system/llama2_7b_roofline_template.json`

Shared resolver and generated-file contract:

`@astra-sim-sh_2.0/sh_test_mesh/config_resolver.py`

Edge-attached remote-memory policy: embedded in the `remote-memory` section of
the single hardware source above (resolver reads it directly; there is no
standalone remote-memory source file).

Default trace configuration:

`@astra-sim-sh_2.0/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv`

Generated runtime files (do not edit):

`@astra-sim-sh_2.0/sh_test_mesh/generated/runtime_config`

## Run and validate

Run from the repository root. The online strategy routes are the supported
pipeline (route 3 = strategy, route 4 = strategy + sensing). Inputs are
materialized per `traces/materialize_first_30s.py` (only allowed
source: `agent-traces/tracelab/astra_compute_20.csv`; its stdout is the
authoritative provenance record), then the plan directory is
produced by the materializer:

```bash
# 1. materialize the 30s request-queue input (traces/materialize_first_30s.py)
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
cd sh_test_mesh/workload/llama2_7b_inference && python3 -m pytest test_face_scheduler.py -q
cd ../.. && python3 -m pytest tests/ -q
```

Each generated plan directory contains `manifest.json` (queue-derived 9
fields; sidecar_restore variant) and `metrics_manifest.json` (synthetic-prerun; rank
attribution is placeholder).

Generated artifacts:

`@sh_test_mesh/generated`
