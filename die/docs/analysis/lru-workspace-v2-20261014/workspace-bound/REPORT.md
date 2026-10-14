# Manifest-bound coarse Die workspace bound

Date: 2026-10-14. Work-only derivation; no tracked source was edited.

## Result

The compile batch gives the **selected Plan** result for every candidate `DieOpSpec` kind and each legal Prefill/Decode shape in the two captured one-request manifests. This is the max of selected Plans across operation kinds, not the max across every internal mapping template. The separate formula column comes from inspecting every HBM workspace assignment in `Mapping.cc`; it bounds all candidates returned by mapping enumeration, including mappings the `compile()` latency ranking does not select:

| Fixture | Prefill ranks | Decode ranks | Max selected C ABI Plan, all adapter kinds tried | Analytical max across all Mapping candidates | Max selected Plan for kinds actually emitted |
|---|---:|---:|---:|---:|---:|
| Official tiny MHA4 | 0–3 | 4–7 | P: 16,384 B/rank; D: 260 B/rank | P: 40,960 B/rank; D: 648 B/rank | P: 0 B/rank; D: 0 B/rank |
| Larger-workspace probe | 0–3 | 4–7 | P: 65,536 B/rank; D: 1,024 B/rank | P: 65,536 B/rank; D: 1,032 B/rank | P: 65,536 B/rank; D: 1,024 B/rank |

The official tiny max selected Plan comes from `qk` in both phases. In the larger probe, selected `ffn_region` is the maximum; selected `qk` reaches 16,384 B in Prefill and 260 B in Decode. The all-mapping-template upper bound is higher for tiny because `attention_region` includes a full materialized mapping (40,960/648 B) that the C ABI's selected Plan did not choose. In the larger Decode phase, the full materialized attention candidate is 1,032 B, slightly above the selected maximum 1,024 B. The second fixture is a separate exact compiler result, not a sample from which a general formula or envelope is inferred.

The 0-byte tiny selected-Plan value applies only to the seven kinds emitted by this exact generator path and exact shape. The 16,384/260-byte values are the maximum **selected Plans** over the C++-supported adapter candidate kinds, including fallback component kinds. They do not bound every internal attention mapping. For a conservative all-mappings certificate, use the analytical all-mapping column unless the protocol guarantees the exact compiler-selected Plan identity and the runtime verifies it.

## Shape coverage

The official manifest has one request and one session, `max_same_session_span=1`, `prefill_length=64`, `decode_length=1`, `history_tokens_before=0`, and `prefill_context_tokens=64`. The trace config uses `prefill_chunk_size=512`, one layer, hidden 128, FFN 256, vocabulary 512, four Q/KV heads, BF16, Prefill ranks 0–3 and Decode ranks 4–7, each with TP=4. Thus its sole Prefill span is `(q=64, kv=64, q_start=0, kv_start=0)`; its sole Decode span is `(q=1, kv=65, q_start=64, kv_start=0)`. Each rank has one local Q and KV head, FFN shard 64, and vocabulary shard 128.

The larger probe uses the same singleton request/phase spans and rank groups, but hidden 512 and FFN 1,024 (four heads, head dimension 128, FFN shard 256). It is intentionally reported separately because a workspace bound changes with task dimensions.

I read every actual typed `die_op_spec` in the three captured responses per fixture, including the empty final response. Each fixture has 80 emitted compute specs: ten per participating rank. For each of the eight actual `(phase, hardware-rank)` pairs, normalizing away only `op_kind` and `fusion_policy` leaves exactly one shape variant across every emitted node. The probe asserts this before compiling, so it does not silently reduce a multi-shape phase to its first task.

The manifest and source establish why these are the complete legal shapes for these fixtures: the selected request/session counts are both one; Prefill makes `ceil(64/512)=1` chunk; Decode has one remaining token and `_plan_train` emits one `(1, context+1)` span. The P and D groups are disjoint, so the exact candidate universe is eight phase/rank shapes, not a Cartesian product of all ranks and phases.

Relevant implementation points:

- `online/die_model_adapter.py:112-120` defines the 15 `OP_KINDS` and the two fusion chains.
- `online/die_model_adapter.py:468-488,491-555` spells out region fusion guards and the fallback when a chain is not fusable.
- `generate_trace.py:727-769,1067-1149` creates aggregate Die specs and applies the pure-local attention/FFN fusion.
- `generate_wsc_llm_trace.py:649-687` chunks Prefill and starts per-rank Prefill emission.
- `online/wsc_llm_online_scheduler.py:833-869` derives Decode train spans; with this manifest it yields the single 65-token KV span.
- `online/graph_batch_builder.py` passes the resulting exact span/rank data into the same Die adapter used by the captured responses.

## Compiler enumeration

`compile_manifest_candidates.py` takes the exact emitted task spec for each shape and changes only `op_kind` plus the normal `fusion_policy` (`auto` for the two region kinds, `none` otherwise). It calls the official `die_coarse_compile_batch_v2` C ABI against the resolved hardware for that manifest. It reads the exact `summary.hbm_workspace_peak_bytes` returned for each Plan and releases every Plan handle. It implements no workspace formula and does not sample an empirical envelope.

There are 15 adapter spellings × 8 actual phase/rank shapes = 120 compile attempts per fixture. For each fixture, 112 return compiler summaries and the eight `local_region` attempts fail closed with `unsupported local_region: DieOpSpec v1 carries no typed mathematical graph`. The latter is in Python's candidate enum but is not a legal v1 compiled Plan. There are no unexpected compile failures. The C++ compiler stores and ranks workspace in the selected Plan (`extern/die_model/coarse/src/Compiler.cc:555-579`), and its summary publishes the exact peak as uint64 (`extern/die_model/coarse/src/Mapping.cc:981`); the C ABI batch response returns that Plan summary (`extern/die_model/coarse/src/Capi.cc:304-327,329-352,367`).

The existing batch API returns only the selected Plan per task spec. `compile()` first gets a vector from `enumerate_mappings_for_compile()`, then ranks candidates by latency, HBM payload, workspace, SRAM peak, and template/tile fields. Therefore the compile table must not be described as a maximum across internal `Mapping` candidates. I separately inspected all assignments to `Metrics.workspace` in `Mapping.cc`; there are only two. This supports the all-template bound below without claiming the C ABI returned unselected candidates.

## All-mapping analytical bound

For the current typed v1 task vocabulary, `Metrics.workspace` starts at zero. All GEMM mapping templates, vector kernels, `norm`, `residual`, `mlp_activation`, `scale_mask`, `softmax`, `qkv_rope`, `mlp_up`, `mlp_down`, `o_proj`, `logits`, and `av` leave it zero. `qkv_rope` combines GEMM and a vector RoPE Plan whose workspace is also zero. `local_region` is rejected before mapping enumeration.

The only nonzero workspace assignments in `Mapping.cc` are:

- Attention at `Mapping.cc:1452-1474`: `S = 4 × score_slots`. For full materialized attention, `score_slots = q_heads_local × Σ_s(q_s × Σ_{j=0..count_s-1}(kv_s + j×kv_stride_s))` (current legacy v1 spans have batch 1). `probability_bytes = S`, and for non-FP32 output `cast_bytes = 4 × head_dim × query_rows`, where `query_rows = q_heads_local × Σ_s(q_s×count_s)`. Thus an `attention_region` with `fusion_policy=auto` has fused causal candidates at 0 HBM workspace and a full materialized candidate at `W_att = 2S + cast_bytes` (`cast_bytes=0` for FP32 output). A `qk` candidate has `W_qk=S`; `av`, `scale_mask`, and `softmax` leave HBM workspace 0. The full path is materialized regardless of `causal`, so the upper bound does not rely on causal pruning.
- FFN at `Mapping.cc:1802-1855,1857-1994`: the materialized candidate has `W_ffn = (SwiGLU ? 2 : 1) × bytes(M×F, activations)`, with `M=Σ_s(q_s×count_s)` and F the rank-local FFN width. The two SwiGLU intermediates (or one GELU intermediate) are reused across layer/pass executions, so this workspace assignment does not multiply by `layer_instances × weight_passes`. The fused output-shard candidates set HBM workspace to zero; their temporary tiles contribute to SRAM peak, not HBM workspace. The child GEMM choices also set no HBM workspace.

Therefore, over all Python adapter kinds **and every mapping template returned by the current C++ enumerator**, a per-plan bound for an exact span set is `W = max(W_att, W_ffn)`. `W_qk` is dominated by `W_att`; all other kinds are zero. This is an analytical statement about the current `Mapping.cc` assignments, not a formula to copy into the Python manager. Changing a workspace assignment, dtype contract, op vocabulary, or enumeration path invalidates the proof and must invalidate the certificate.

These expressions are monotone in the positive span contributions: adding a span or increasing its query/count/batch/KV lengths cannot reduce S or M; increasing the local head count/head dimension or FFN width cannot reduce the respective terms. For a future representation with explicit batch, each span contribution is multiplied by batch; the currently emitted v1 spans use batch 1. Increasing layer count/weight passes does not increase HBM workspace in these assignments; those fields scale work, while the materialized buffer is reused. Increasing causal reach cannot make a fused candidate exceed the full materialized candidate.

For the fixtures, qh-local is one and output/activation dtype is BF16:

- Tiny Prefill: `S=4×1×64×64=16,384`, `cast=4×32×64=8,192`, so `W_att=40,960`; `W_ffn=2×64×64×2=16,384`.
- Tiny Decode: `S=4×1×1×65=260`, `cast=4×32×1=128`, so `W_att=648`; `W_ffn=2×1×64×2=256`.
- Larger Prefill: `S=16,384`, `cast=4×128×64=32,768`, so `W_att=65,536`; `W_ffn=2×64×256×2=65,536`.
- Larger Decode: `S=260`, `cast=4×128×1=512`, so `W_att=1,032`; `W_ffn=2×1×256×2=1,024`.

The complete finite manifest shape set can be formed before manager mutations from the materialized request rows. Write `b` for activation bytes/element, `β=2` for SwiGLU and `β=1` for GELU, and use rank-local `qh`, `hd`, and `F`.

For Prefill request r, let `P_r=prefill_length`, `H_r=history_tokens_before`, and `C` be the fixed chunk size. Its full task has `q_i=min(C,P_r-iC)` and `kv_i=H_r+iC+q_i`, for `i=0..ceil(P_r/C)-1`. Define `A_P=Σ_rΣ_i(q_i×kv_i)` and `Q_P=Σ_r P_r`. A manifest-wide conservative prefill bound is `max(8×qh×A_P + cast_P, β×b×F×Q_P)`, where `cast_P=4×qh×hd×Q_P` for non-FP32 output and zero for FP32 output. This concatenates every request's positive spans, even though P runs one request at a time; a tighter certificate compiles each request shape and takes the maximum.

For Decode request r, let `D_r=decode_length_r`, `K_r=prefill_context_tokens`, and `T_r=final_context_tokens=K_r+D_r`. Its complete possible future span is q=1 at KV contexts `K_r+1 ... T_r`, representable as one run with stride 1/count `D_r`. Define `A_D=Σ_r[D_r×K_r + D_r(D_r+1)/2]` and `Q_D=Σ_r D_r`. The all-request decode envelope is `max(8×qh×A_D + cast_D, β×b×F×Q_D)`, where `cast_D=4×qh×hd×Q_D` for non-FP32 output and zero otherwise. Replacing each exact run sum by `D_r×T_r` gives a looser bound that depends only on terminal contexts.

Prefill is one request at a time, but each request's entire prefill task is the chunk-span list above. Decode `_plan_train` includes only currently active decode members, each with its remaining token spans. A capped train or first-step/remainder split partitions those runs; it adds no padding. Thus a Decode task is a subset of the all-request full terminal-context spans. The all-request sums dominate any such subset because every per-span term in `S`, `M`, and `query_rows` is nonnegative and additive. Manager evictions, readiness changes, and queue-frontier selection can remove/partition members but cannot create longer spans than the materialized manifest. This proves completeness for a fixed manifest, including session histories already materialized there; it does not cover requests arriving outside that manifest.

These coarse all-request envelopes can substantially overreserve: for Decode they combine all future output spans into one hypothetical task even though runtime trains may be smaller. Tighter finite certificates can enumerate the actual legal train subsets/frontiers if that state space is bounded. For arbitrary online input or unbounded co-batching, no fixed phase constant follows from the two current fixtures. The exact fallback is a pure pre-admission plan probe for the frozen next candidate; the C++ probe can use the existing deterministic selected Plan and exact summary, while an all-mapping envelope API would return `max(hbm_workspace_peak_bytes)` across its internal mapping vector. Neither path needs a Python copy of the workspace equations.

Artifacts and identity:

- Full rows, exact selected summaries, shape records, input hashes, source hashes, and candidate spec hashes: `compiled-candidate-plans.json` (SHA-256 `93b7793f109e276a486a022ebee4e9c6ac87ac7162ff19efed218d025101659e`). Its `all_opkind_selected_plan_*` fields are deliberately named to distinguish selected C ABI Plans from all Mapping candidates.
- Compiler probe: `compile_manifest_candidates.py` (SHA-256 `ab57971b645abbe89c892b9d65ba0b74ffaed53e6fa3add7d073bf817ec5d9bc`); run summary and stderr are adjacent.
- C ABI: `work/lru-cpp-workspace-20261014/build/aggregate/lib/libcoarse_die_cabi.so`, SHA-256 `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b`. Its build provenance is `work/lru-cpp-workspace-20261014/source-identity/provenance.txt` (source HEAD `e2dd5af7b61895a85b122c200606017bb741a004` plus the recorded workspace-v2 patch). This identity refers to the compiler library, not the Online runtime binary.
- Official tiny manifest SHA-256 `762c85a6c301d7d37f15bfbabb09590c835dbb1c9a95cc875152812afc9788af`; larger probe manifest SHA-256 `1e5bf41485ef947b397d2c2a325472f33aee7ebfcf2cf71639930dafb9070549`.
- Both resolved hardware JSON files have SHA-256 `3321c0cd5543414ad55b4601497027130d0e0a32994a7e52cae3716e13424cda`.
- The JSON records all three response SHA-256 values per fixture, both canonical candidate-spec hashes, hashes of the five Python generator/adapter/scheduler source files, and hashes of `Mapping.cc`, `Compiler.cc`, `Capi.cc`, and `Plan.hh` used as compiler-proof provenance.

## Pre-admission decision

For a fixed manifest, the safe artifact is an immutable, manifest-bound workspace certificate generated before scheduling. Its key should include the manifest/request input digest, resolved hardware fingerprint, C ABI/compiler identity, generator/adapter source identity, candidate `op_kind` set, and canonical candidate-spec-set digest. It should record both the exact compiler-selected Plan bound and, for the strict all-template proof, the `Mapping`-candidate upper bound by `(phase, hardware rank)`. The current C ABI can supply the selected bound; a new compile-only envelope result can return the maximum across internal mapping candidates without a Python formula. Admission may use the chosen certificate bound as a fit-only phase margin; it must never be added to resident KV, reserved KV, active workspace, or the remote-pool ledger. When the real Plan is compiled, C++ must fail closed if its exact spec identity is outside the certificate or its peak exceeds the recorded bound.

This certificate is a proof only for its bound identity. The two fixture results show why a global constant is unsafe. A change in request context/span count, decode co-batch membership, model shape, TP partition, fusion guards, source identity, compiler, or hardware invalidates it. The official tiny's actually-emitted-only value of zero would also miss the `qk` fallback candidate, so it cannot stand in for the all-candidate bound.

For arbitrary online requests, either compile an exhaustive certificate for a separately proven finite set of all legal candidate train shapes, or use a two-phase compile-only `plan-probe -> admission -> commit` API. The probe should compile the frozen prospective Prefill/Decode candidate specs, return exact per-rank peaks plus a digest of those specs and the current capacity epoch, and acquire no workspace lease or KV reservation. Python then performs whole-session fit/eviction; final graph commit must echo the same proposal/spec digest and epoch, and C++ checks the actual Plan against the probe result. This reuses the existing compiler's exact summary and avoids a Python workspace formula. A post-admission blocked-demand notification cannot provide the same guarantee because the current request may already be protected/reserved in the manager.

The certificate covers required byte capacity, not reclamation timing. If LRU eviction supplies the byte margin, the task must not run until the corresponding store has actually released the source HBM capacity; current ordinary store branches can run independently of GPU COMP. Keep the existing store-tail ordering caveat as a separate required runtime proof/gate.

## Limits

The direct C ABI inventory enumerates two finite, singleton MHA4 manifests. The monotone all-span argument describes how to extend it to a fixed finite multi-request manifest, but no such combined certificate was generated or validated in this pass. These artifacts do not cover requests outside their manifest, GQA dimensions, multiple layers, arbitrary TP groups, or future dynamic fusion barriers. The larger fixture demonstrates that even with the same spans/topology, model shape changes the selected and all-mapping bounds. No C++ runtime test or full Online E2E was rerun here; the check was the 240 C ABI candidate compilations summarized above.
