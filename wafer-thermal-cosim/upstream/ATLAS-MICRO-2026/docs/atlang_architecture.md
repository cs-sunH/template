# ATLang Implementation Architecture

## 1. Purpose

ATLang is the ATLAS-owned, simulator-only implementation of the programming
interface shared with the TileLang frontend. It avoids TileLang, TVM, and TIR
runtime dependencies while preserving the operator programming model needed by
ATLAS. The implementation is divided into four ownership layers: public
language, neutral IR, AST capture, and simulator backend.

## 2. End-to-End Flow

```text
frontend/ops/atlang operator source
    -> frontend/atlang/language public objects and calls
    -> frontend/atlang/capture Python AST replay
    -> frontend/atlang/ir SimulatorMetadataSnapshot / AtlangKernel
    -> frontend/atlang/simulator cloud or edge extraction
    -> data_placement.yaml + operator_description.yaml + simulator logs
    -> atlasim cycle simulation
```

`@A.main` starts this flow. It parses the decorated function, builds an
`AtlangKernel` shell, and runs extraction immediately when a captured
`system_config` is present. Final latency, energy, generated paths, placement,
task, and communication fields are attached to that shell.

## 3. Public Programming Interface

`frontend/atlang/language/` owns syntax visible to operator authors:

- `entry.py`: `A.main` and the parser entry.
- `objects.py`: `Tensor`, `Buffer`, `CoreArray`, `SPMD`, `MPMD`, and `Kernel`
  context objects.
- `control.py`: `Serial` and host-range handling.
- `memory.py`: `alloc`, `copy`, `fill`, and `clear`.
- `gemm.py`: matrix-compute calls.
- `vector.py` and `reduction.py`: element-wise, scalar, and reduction helpers.
- `communication.py`: `send` and `recv`.
- `math.py`: symbolic helpers such as `ceildiv` and `infinity`.
- `__init__.py`: the supported `import frontend.atlang.language as A` surface.

Add or change user-visible call signatures here. Reject unsupported arguments
at this boundary instead of silently storing backend-only metadata.

## 4. Neutral IR

`frontend/atlang/ir/` defines structures shared by capture and extraction:

- `dtype.py`: dtype registry and byte-size semantics.
- `expr.py`: symbols, calls, arithmetic, comparisons, and substitution/walk
  helpers for captured expressions.
- `nodes.py`: tensor declarations/accesses, actions, loops, kernel regions,
  operator regions, core-array contexts, metadata snapshots, and extraction
  result fields.
- `kernel.py`: the `AtlangKernel` shell and extraction-result attachment.

New simulator-independent semantics belong in this layer. Keep backend paths,
YAML formatting, and cloud/edge policy out of neutral IR dataclasses.

## 5. AST Capture and Parser

`frontend/atlang/capture/` converts inspectable Python source into the neutral
IR:

- `source.py` loads and validates decorated function source.
- `parser.py` is the public parse orchestration entry.
- `replay.py` evaluates supported expressions and statements in source order.
- `frame.py` maintains active core-array/operator/kernel/loop state, validates
  declarations, and constructs the kernel snapshot.
- `access.py` normalizes tensor and local-buffer windows into `TensorAccess`.
- `actions.py` lowers memory, compute, reduction, and communication calls into
  `OpAction` records.

When adding syntax, decide whether it creates a value/expression, a structural
region, or a modeled action. Extend replay and the corresponding normalization
module, then emit neutral IR; do not invoke simulator code from the parser.

## 6. Simulator Backend

`frontend/atlang/simulator/` translates captured IR into current ATLAS
simulator inputs:

- `extract.py` selects cloud, edge, or general extraction.
- `io.py` provides plain-YAML conversion, process-output redirection, and safe
  multiprocessing helpers.
- `common/` owns shared symbolic evaluation, tensor layout, DRAM/NoC task
  helpers, system checks, general task materialization, and general autotune.
- `cloud/` owns cloud placement, GEMM, decode attention, communication, task
  descriptions, and final kernel execution/aggregation.
- `edge/` owns edge placement, GEMM cache/DSE, task descriptions, softmax and
  channel overheads, and final kernel execution/aggregation.

Cloud- or edge-specific behavior should remain in its respective directory.
Only semantics genuinely shared by both paths belong in `common/`.

## 7. Operator Sources and Validation

`frontend/ops/atlang/` is the source of truth for user programs:

- `cloud.py`: transformer-layer execution on the 4x4 cloud mesh.
- `edge.py`: edge GEMM/attention execution and channel behavior.
- `general_spmd.py`: fixed and autotuned general SPMD examples.
- `general_mpmd.py`: fixed and autotuned static core-group MPMD examples.

`tests/inference_test_atlang.py` is the end-to-end regression entry.

## 8. Extension Checklist

1. Add the user-facing object or function under `language/` and export it from
   `language/__init__.py`.
2. Add neutral IR fields or node types under `ir/` only when existing records
   cannot represent the new semantics.
3. Teach `capture/replay.py`, `capture/access.py`, or `capture/actions.py` to
   lower the source construct without backend dependencies.
4. Materialize the new IR in `simulator/common/`, `simulator/cloud/`, or
   `simulator/edge/` according to ownership.
5. Add a focused operator example under `frontend/ops/atlang/` and validate it
   through the appropriate inference test.
6. Keep source semantics explicit. Do not repair an incorrectly described
   operator with extractor-side name remapping or hidden aliases.

