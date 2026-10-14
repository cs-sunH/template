# Workspace capacity algebra and Plan identity replay

This checkpoint records the candidate-independent workspace proof work and a
real C ABI replay of the historical 62-task Plan identity set. It does not
change the main HANDOFF/PROJECT, rerun the public build, or claim that a
capacity bound proves a mapping is executable.

## Capacity formulas and proof boundary

For Attention span `i`, let `q_i` be query rows per request, `k_i` the first
declared KV length, `d_i` its KV stride, `n_i` its request count, and `b_i`
its batch multiplier. With local query heads `Hq`, head width `D`, layer
count `L`, and weight passes `P`, define

```text
S = Hq * sum_i [ b_i * q_i * sum_{r=0}^{n_i-1}(k_i + r*d_i) ]
R = Hq * sum_i [ b_i * q_i * n_i ]
W_materialized = L * P * (8*S + (output_dtype != fp32 ? 4*D*R : 0))
W_qk = L * P * 4*S
```

The two `8*S` terms are the FP32 score and probability buffers. The optional
cast term is the simultaneously live FP32 output temporary. Full materialized
workspace charges complete query-by-KV rectangles, regardless of causal mask
or tile dimensions. AV-only and forced-fused Attention have zero HBM
workspace. The repeat multiplier intentionally preserves existing Plan
accounting pending a separate semantic review.

For a materialized FFN with local rows `M`, local expansion `F`, activation
dtype `T`, and SwiGLU indicator `g`,

```text
W_ffn = (g ? 2 : 1) * bytes(M*F, T)
bytes(x, int4) = ceil(x/2)
```

Each INT4 intermediate is independently byte-ceiled. FFN temporary storage is
reused across serial layers/passes, so this peak does not multiply by `L*P`.
Forced fused FFN has zero HBM intermediate workspace.

These equations prove coverage when the hardware fingerprint and invariant
normalized task fields (op, phase, execution profile, local shapes, TP slice,
dtype, variant, and fusion policy) match; the exact normalized task identity
still includes each task's distinct spans. Each child query row is among
the manifest rows already charged; its KV context length is no larger than the
manifest context charged for that row; the child adds no rows/context/repeated
work; and its layer/pass counts do not exceed the accounted manifest counts.
For FFN, local `F`/dtype/variant must match and child `M` cannot exceed
manifest `M`. A token-total comparison by itself is insufficient. A P full
chunk must be represented by the manifest span union, not inferred from a
smaller capped token count. Different TP or storage/fusion identities require
a separate proof. The API preserves opaque metadata in normalized identity
and makes no executable-mapping claim.

## Captured candidate-feasibility counterexample

The baseline binary and source identities are described in
[`../workspace-capacity-proof/README.md`](../../../../work/workspace-capacity-proof/README.md)
and the short output is captured in
[`counterexample.txt`](../../evidence/workspace-capacity-proof-20261014/counterexample.txt).
The fixture uses SRAM 262,144 B/core, `rho_sram=0.25` (65,536 B limit),
BF16 Attention with `q_heads=kv_heads=1`, `head_dim=256`, and `auto` fusion.
The parent q=64/kv=64 full candidate's materialized SRAM peak is 71,752 B, so
it is filtered; its fused candidate fits at 64,204 B and the old envelope is
0 B. The contained q=8/kv=8 child gains a feasible 8,704 B materialized
candidate. The candidate-independent parent bound is 98,304 B. A
materialized-only parent has no feasible candidate but the new capacity-only
API still returns the bound.

## Implementation and focused verification

The source diff is limited to these six files (no CMake target was added):

- `Plan.hh:118` adds `WorkspaceCapacityBound`; `Capi.h:57` declares
  `die_coarse_workspace_capacity_bound_v2`.
- `Mapping.cc:230-285` factors checked Attention rectangle/query/workspace and
  FFN byte formulas shared by actual assignment and potential capacity;
  `Mapping.cc:1693` computes the candidate-independent bound.
- `Compiler.cc:524` asserts feasible envelope candidates do not exceed the
  theoretical bound; `Compiler.cc:547` adds the normalized capacity-only API.
- `Capi.cc:378` returns schema 2 / capacity-bound schema 1 with model revision,
  bound kind, exact bytes, and hardware/task identities.
- `workspace_envelope_test.cc:221` covers the feasibility counterexample,
  no-feasible-map capacity query, fusion policies, empty shards, overflow,
  unknown dtype, INT4 ceiling, repeat accounting, C ABI and cache/handle
  behavior.

Focused test command and result:

```bash
source work/activate.sh
cmake --build work/build-die --target coarse_workspace_envelope_test -j2
work/build-die/coarse_workspace_envelope_test
# PASS workspace_envelope_test
```

The repeated Attention Plan remains workspace 245,760 B with historical full
Plan fingerprint
`3283ef35e93b58457e011f770a95647589b195fbe2c5335d1fcb6cdfaf7cc3a8`.
The old workspace-envelope response/schema is unchanged. This focused suite
passed before the parent started its full public CTest; no build was run here
after that handoff.

## Historical 62-task identity through the new production C ABI

The source capture is `die/docs/analysis/os-foundation-20261014/identity_driver.cc`
(SHA-256 `aa2ee5bdc1efeacec7808c48aed156e4b16b92a3947911883bfec6aa12c64617`).
Its saved 62 full Plan summaries/hashes are
`die/docs/evidence/workspace-envelope-20261014/public/identity-current.json`
(SHA-256 `0742247a8de06341edffb0ebafd76b28266c73402b216f27ddc87f02634226be`).
The earlier comparison record establishes that saved capture as identical to
the historical baseline for all 62 entries.

[`replay_62_identity_via_cabi.py`](replay_62_identity_via_cabi.py) groups the
captured inputs by hardware fingerprint, recreates the two recorded kappa
profiles, and calls the compiled library's actual
`die_coarse_compile_batch_v2`. It compares each returned Plan hash and complete
summary object, then releases every returned handle. The compact result is
[`identity-replay.json`](../../evidence/workspace-capacity-proof-20261014/identity-replay.json):
62/62 hashes and 62/62 complete summaries match exactly. It records the
library and source SHA-256 values. Reproduce from the repository root with:

```bash
python3 die/docs/analysis/workspace-capacity-proof-20261014/replay_62_identity_via_cabi.py \
  --output die/docs/evidence/workspace-capacity-proof-20261014/identity-replay.json
```

No source or binary was copied into the evidence tree. The old Plan identity
capture and the small feasibility counterexample output are referenced in
place, avoiding duplicate full summaries and large artifacts.
