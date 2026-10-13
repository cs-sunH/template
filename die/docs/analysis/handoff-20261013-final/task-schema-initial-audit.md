# Typed task schema audit (read-only)

Audit date: 2026-10-13. This is a scratch review of the current tree; no
tracked file was changed. The review covers `die/AGENTS.md`, `die/docs/PROJECT.md`,
`die/docs/HANDOFF.md`, the root coarse-grain design document, `Compiler.cc`,
`Mapping.cc`, existing compiler fixtures, and the task producers/GraphBatch
parsers in all five AstraSim trees.

## Findings

### 1. Missing v2 dimensions can compile as a different, smaller operation

`normalize_new_task()` requires an object-valued `tensor_shapes`, but only
normalizes and checks `m/n/k` for `gemm`. Other dimensions are left absent
([Compiler.cc](/workspace/template/die/src/Compiler.cc:327), [Compiler.cc](/workspace/template/die/src/Compiler.cc:362)).
`Mapping.cc` then supplies executable defaults: `M=1`, `H=1`, `F=H`, `V=H`
([Mapping.cc](/workspace/template/die/src/Mapping.cc:1517)); attention and QKV
also default `q_heads`, `kv_heads`, and `head_dim` to 1 (or `H` for the QKV
head dimension) ([Mapping.cc](/workspace/template/die/src/Mapping.cc:969),
[Mapping.cc](/workspace/template/die/src/Mapping.cc:1552)). Score transforms
default head count to 1 ([Mapping.cc](/workspace/template/die/src/Mapping.cc:1543)).

Examples of silent under-modeling include `{"op_kind":"norm", "tensor_shapes":{}}`
becoming a one-element norm; `mlp_activation` without `ffn` becoming a width-1
activation; and attention with spans but no head fields becoming a one-head,
one-dimension operation. Some missing combinations happen to return no mapping
or fail later, but that does not make the contract safe: each supported op
should fail at normalization with a path naming its missing dimension.

The one zero exception is also inconsistent: `m` accepts zero in `get_alias`,
while `M` and every other primary shape require positive values; GEMM later
rejects zero, but non-GEMM ops can reach Mapping with `m=0` and create zero
work. The generic Mapping `shape_u` accepts zero as a valid integer and does
not itself enforce operation-specific positivity ([Compiler.cc](/workspace/template/die/src/Compiler.cc:335),
[Mapping.cc](/workspace/template/die/src/Mapping.cc:122)).

### 2. New-schema TP metadata and the dimensions consumed by Mapping diverge

The design document says `tensor_shapes` are actual local M/N/K, head, and FFN
dimensions and `tp_slice` records TP/rank/local slices ([design doc](/workspace/template/die内仿真器设计文档_最粗粒度参数响应版.md:252)).
In the v2 normalizer, a local value is copied *from* a shape into `tp_slice`
when absent, but no canonical local shape is populated from `tp_slice`; explicit
`tp_slice` local values are not reconciled with shape values
([Compiler.cc](/workspace/template/die/src/Compiler.cc:346)). `tp_slice` also
does not use `reject_unknown()`.

Mapping reads only `tensor_shapes` for these work dimensions. For example,
attention reads `q_heads`/`kv_heads` from shapes and never reads
`tp_slice.q_heads_local` / `kv_heads_local`; its fallback `q_heads_local` is
looked up in `tensor_shapes`, not in `tp_slice` ([Mapping.cc](/workspace/template/die/src/Mapping.cc:969)).
FFN and logits similarly consume `tensor_shapes.ffn` and `.vocab`, falling back
to `hidden`, without consulting `tp_slice.ffn_local` or `.vocab_local`
([Mapping.cc](/workspace/template/die/src/Mapping.cc:1517)). Therefore:

- Only `tp_slice.q_heads_local` present: attention uses fallback 1.
- `tensor_shapes.q_heads` plus a different `tp_slice.q_heads_local`: Mapping
  uses the shape value and silently ignores the local shard.
- Only `tp_slice.ffn_local` / `vocab_local` present: Mapping uses `hidden` for
  those widths.
- A global `q_heads`/`ffn`/`vocab` in shapes plus local TP values can overstate
  per-rank work without warning.

The legacy adapter intentionally does different canonicalization: it writes
global `hidden`, but puts TP-local `ffn`, `vocab`, `q_heads`, and `kv_heads`
into `tensor_shapes` ([Compiler.cc](/workspace/template/die/src/Compiler.cc:232)).
This mixed global/local shape is part of the current v1-to-v2 Mapping contract;
do not change it as collateral work while fixing direct v2 input.

Before implementation, v2 needs one unambiguous rule. Recommended direction:
all dimensions in `tensor_shapes` are the per-rank dimensions Mapping consumes;
`tp_slice` is identity/provenance and consistency metadata, not a second source
of work sizes. If global dimensions are needed for validation, represent them
separately with explicitly global names and define how a local shard is
obtained. Do not silently divide global sizes by TP degree: the five producers
use uneven `shard_extent`, and a rank-local task already carries actual local
work.

`q_heads` versus `num_q_heads` needs a deliberate decision too. Current v2
normalization treats them as aliases, while v1 `model_dims.num_q_heads` is
global and `tp_slice.q_heads_local` is local. Those names cannot safely be
treated as the same unit across dialects. Similarly, `ffn` and `ffn_local`,
`vocab` and `vocab_local` need explicit local/global semantics rather than
primary-key precedence.

### 3. Aliases and parallel declarations can disagree without an error

`get_alias(primary, fallback, ...)` picks the primary whenever it exists and
never compares the fallback. This affects `m/M`, `n/N`, `k/K`,
`q_heads/num_q_heads`, `kv_heads/num_kv_heads`, `ffn/ffn_local`, and
`vocab/vocab_local` ([Compiler.cc](/workspace/template/die/src/Compiler.cc:335)).
For example, `m: 2, M: 200` compiles with `m=2`; the ignored value remains in
the task fingerprint. `execution_profile` also silently wins over
`execution_profile_id` when both are present ([Compiler.cc](/workspace/template/die/src/Compiler.cc:342)).
Span aliases have the same issue: `q_len` wins over `q_tokens`, `kv_len` over
`kv_tokens`, explicit position keys over their alias, and `batch` over
`count` ([Compiler.cc](/workspace/template/die/src/Compiler.cc:297)).

Require either one spelling, or permit multiple spellings only if their
normalized values agree. For pairs with different global/local meanings,
reject ambiguous simultaneous use until the schema names those units. Reject
unknown `tensor_shapes` and `tp_slice` keys once the supported per-op field
sets are defined. The existing `swiglu`/`mlp_variant` conflict check is a
useful precedent ([Compiler.cc](/workspace/template/die/src/Compiler.cc:330)).

### 4. GQA width relationships must stay operation-scoped

Legacy v1 currently checks `hidden == num_q_heads * head_dim`
([Compiler.cc](/workspace/template/die/src/Compiler.cc:210)). The five current
Python adapters derive MHA from their single `num_heads` setting
(`num_q_heads == num_kv_heads == num_heads`, `head_dim=hidden/num_heads`); they
also compute per-rank local counts with `shard_extent`. They do not currently
emit GQA as a distinct configuration. All five emit `schema_version: 1` under
`die_op_spec`; GraphBatch parsing preserves the JSON and the C++ coarse
compiler dispatches it through the old normalizer. See the adapter functions
in `astra-sim-{wscllm,wscllm-LRU,face,face-LRU,joint}/sh_test_mesh/workload/llama2_7b_inference/online/die_model_adapter.py`
and the corresponding `astra-sim-*/astra-sim/workload/execution_driven/ParsedGraphBatch.cc`.

Do not promote the old equality into a new global constraint. In an operation
such as QKV projection, the input K can be the model hidden width while the
local projected Q/K/V width is `q_heads_local*head_dim +
2*kv_heads_local*head_dim`; attention inner width and residual/model width are
separate quantities. The new schema should require the dimensions its selected
operator consumes and validate only relationships the operator contract
defines. Standard GQA's current Mapping rule is local `q_heads % kv_heads == 0`
([Mapping.cc](/workspace/template/die/src/Mapping.cc:972)); local counts alone
do not encode a nonstandard or rank-crossing global head assignment. Such a
mapping must remain unsupported unless the task explicitly supplies the
mapping or a supported shard rule.

### 5. Span aliases and progression bounds need normalization-time decisions

New spans correctly reject zero `q`, `kv`, and compact `count`; regular spans
allow `batch` or `count` to default to 1, and compact spans require `count`
([Compiler.cc](/workspace/template/die/src/Compiler.cc:281)). `kv_stride` is
allowed to be zero. These can be valid choices, but the accepted meaning must
be explicit and tested. Current gaps:

- In both regular and compact forms, omitted query start becomes `kv-q` when
  `kv >= q`, otherwise 0; `q > kv` is not rejected at parse time.
- In compact form, overflow checks use run `count` before multiplying by
  top-level `batch`, while normalized `count` is `count*batch`; the full final
  query/KV progression is not validated at that point.
- `batch` and `count` can coexist in a regular span, with `batch` silently
  taking precedence.
- Some progression constraints are deferred to `Mapping::spans_of` and produce
  a generic `unsupported_mapping` instead of a schema-path validation error.

Define whether `batch` is multiplicity of identical shapes or an extension of
the compact sequence. Validate the final expanded count and both final
absolute-position ranges with checked arithmetic. Keep zero stride only if
repeated identical lengths/positions are supported; otherwise reject it for a
multi-item run. Do not add an arbitrary small count/stride cap without a model
limit: require positive values, safe `uint64` progression bounds, and clear
unsupported-domain errors.

### 6. Unsupported operator names do not fail at the schema boundary

`normalize_new_task()` does not validate `op_kind` against a supported set.
Unknown names eventually return no candidates and are reported as
`unsupported_workspace` by `compile()`, which conflates an unsupported
operator with a valid operator whose buffers do not fit. Validate the op enum
up front and report `unsupported_op_kind`.

## Small implementation boundary

After the parent checkpoint is complete, a narrowly scoped follow-up can add a
per-op v2 shape canonicalizer in `Compiler.cc` and a focused
`die/tests/task_schema_test.cc`, leaving `normalize_old_spec()` behavior frozen.
The canonicalizer should provide:

1. An explicit supported-op table and per-op required positive fields. At
   minimum: GEMM `m,n,k`; norm/residual `m,hidden`; QKV `m,hidden,q_heads,
   kv_heads,head_dim`; attention/qk/av `spans,q_heads,kv_heads,head_dim`;
   score transforms `spans,q_heads`; activation `m,ffn`; FFN region
   `m,hidden,ffn`; logits `m,hidden,vocab`. Before requiring these exact sets,
   confirm each Mapping branch's mathematical input/output dimensions; for
   example `o_proj` currently assumes `H×H`, which is not enough to describe
   arbitrary local attention width.
2. One alias resolver that detects contradictory duplicate spellings and
   canonicalizes equivalent aliases before building a Plan.
3. A single local-shape rule: validate local Mapping dimensions from canonical
   `tensor_shapes`, reconcile explicit `tp_slice` locals when present, and
   reject disagreement or an empty local slice when that operator needs it.
   Do not force `global_hidden == q_heads*head_dim`; do not infer local values
   by integer division.
4. Span normalization that resolves aliases only when equal, rejects
   unsupported progressions with a field path, and checks the actual
   post-batch final positions/count for overflow.

Avoid rewriting supported v1 producer semantics as part of this patch. In
particular, preserve the v1 required fields, MHA-derived `hidden`/head check,
legacy fusion enum, span representation, local shard extraction, and resulting
Plan behavior. If future producers need GQA/global-local semantics not
representable by v1, add a new schema path rather than guessing within v1.

## Compatibility and rejection fixtures

Keep the fixtures small and independent of performance ranking:

- **Legacy golden:** wrap a representative v1 adapter-produced spec in
  `die_op_spec`; assert normalization keeps global `hidden`, maps local
  q/kv/ffn/vocab into the exact current shape fields, preserves TP rank/degree,
  `m=sum(q_tokens*count)`, and retains span absolute starts. Include compact
  v1 spans and `layer_count`/`weight_passes > 1` so old behavior stays isolated.
- **Old-schema guard:** an existing accepted MHA v1 fixture still compiles;
  existing invalid/missing legacy fields keep failing. New-v2 policy changes
  must not leak into `normalize_old_spec()`.
- **Valid direct v2 GQA:** `hidden=10`, local `q_heads=4`, `kv_heads=2`,
  `head_dim=2`, spans of positive size. Assert it compiles and maps attention
  using local GQA fields without requiring `hidden == q_heads*head_dim`.
- **TP consistency:** test a non-divisible TP example with explicit global
  metadata and the producer's `shard_extent` local count; accept a consistent
  rank-local task. Reject `tensor_shapes.q_heads`/`tp_slice.q_heads_local`,
  `ffn`/`ffn_local`, and `vocab`/`vocab_local` disagreement. A local-only
  `tp_slice` must not fall back to Mapping's 1/H defaults.
- **Required-shape table:** for each supported op, remove each required field
  in turn and assert a stable `task.tensor_shapes.<field>` or `task.spans`
  diagnostic. Include zero, negative, floating point, string, and overflow
  dimensions; distinguish optional documented defaults (phase, layers/passes,
  fusion, dtype roles) from mathematical dimensions.
- **Alias conflicts:** conflicting `m/M`, `n/N`, `k/K`, head aliases, FFN/vocab
  aliases, execution-profile aliases, and span key aliases fail; equal aliases
  either canonicalize once or are explicitly rejected by policy.
- **Boundary spans:** zero and missing counts, zero/positive stride cases,
  `q>kv` default start, explicit query/KV starts near `UINT64_MAX`, and
  `count*batch` / final-position overflow are covered. Assert errors occur in
  Compiler normalization rather than later as `unsupported_workspace` or a
  wrong plan.
- **Unknowns:** unknown op, shape key, and TP key get direct unsupported/unknown
  field errors.

Existing direct-v2 coverage already includes GEMM, fused FFN, causal GQA, and
compact attention spans in `die/tests/compile_execution_test.cc`; extend that
coverage with schema assertions rather than duplicating its work/accounting
checks.

## Verification status

This pass was read-only apart from this ignored scratch artifact. No code,
tracked tests, CMake, or documentation changed, and no tests were run. Current
tree status before writing this note already had the parent's in-progress
tracked checkpoint changes; those are not attributed to this audit.
