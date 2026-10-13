# TX8 workbook audit and conditional reference reconstruction

Source workbook: `TX8_单die建模_精简版.xlsx` (SHA-256 `eba181c80a2d51bd5a604bad6928610ea5039824b4ec4d5d9bd023ee4bda52ee`). The XLSX is parsed directly from ZIP/XML with Python's standard library.

The workbook overview states that the original nine records were removed. Its 13 group values are a mixture of one reported anchor, a reused Norm anchor, one conditional reinterpretation, and model estimates/attributions. The 19 math-node rows do not provide 19 independent timing samples. No raw timing logs or repeat distributions are present.

## Source cache audit

`计时组13行!B16` contains formula `SUM(B2:B14)` and cached XML value `11897.602619368336` (11897.602619368335581 μs). The tool recomputes B2:B14 as 11897.602619368333762 μs, a difference of -1.82e-12 μs. `计时组13行!B15` is a separate hardcoded value `11897.60261936833` (11897.602619368330124 μs), differing from the formula cache by -5.46e-12 μs. The formula cell and its cache are preserved verbatim in `tx8_samples.json`.

Cache reconciliation: **passed** (required absolute difference < 1e-06 μs). This checks workbook accounting, not timing accuracy.

## Boundary records and identifiable reference quantities

- 246 μs at `计时组13行!B2`: `reported_anchor`, interpreted as one full RMSNorm by the workbook explanation; the raw record and repeats are absent. Under the fixed BF16 traffic count `16785408` bytes, this gives reference vector DDR service 68.233365853659 GB/s and η_vector_memory=0.341166829268293 relative to 200 GB/s.
- 3363 μs at `计时组13行!B13`: `conditional`; the original SiLU label is reinterpreted as the whole fused FFN. Under 128 TFLOP/s matrix peak, 1 T eq-op/s vector peak, and a single shared matrix/vector compute efficiency, this gives η_compute_reference=0.9079551923877489. This does not separately identify PE and Vec efficiency.
- The 140 GB/s matrix memory service rate is the workbook's selected 70% setting, not an independent measurement.
- These two inversions apply only to `tx8_reference_v1`. They do not calibrate or initialize target INT4 PE, target Vec/SFU, HBM, NoC, SRAM, launch, or pipeline parameters.

## Canonical 13-group reconstruction

All operation counts and physical byte counts are integer-valued and use the workbook shape B=1, Sq=Sk=1024, H=4096, F=14336, Hq=32, Hkv=8, d=128; BF16 inputs/weights/projection outputs; FP32 score/probability and accumulation. Each group starts compute and memory together and completes at their maximum. The AV workbook timing group maps once to the stable `av_pv` key and C++ `AV/PV` label.

| Key | Group | Fop | Eop | Bytes | Compute μs | Memory μs | Predicted μs | XLSX μs | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `norm_1` | RMSNorm_1 | 0 | 25165824 | 16785408 | 27.717033 | 246.000000 | 246.000000 | 246.000000 | reported_anchor |
| `q_proj` | Q_proj | 34359738368 | 0 | 50331648 | 295.648352 | 359.511771 | 359.511771 | 359.511771 | derived_engineering_estimate |
| `k_proj` | K_proj | 8589934592 | 0 | 18874368 | 73.912088 | 134.816914 | 134.816914 | 134.816914 | derived_engineering_estimate |
| `v_proj` | V_proj | 8589934592 | 0 | 18874368 | 73.912088 | 134.816914 | 134.816914 | 134.816914 | derived_engineering_estimate |
| `rope_qk` | RoPE_QK | 0 | 15728640 | 21495808 | 17.323146 | 315.033675 | 315.033675 | 315.033675 | derived_engineering_estimate |
| `qk_matmul` | QKT | 8589934592 | 0 | 144703488 | 73.912088 | 1033.596343 | 1033.596343 | 1033.596343 | derived_engineering_estimate |
| `score_mul_softmax` | score_mul_softmax | 0 | 402653184 | 268435456 | 443.472527 | 3934.079063 | 3934.079063 | 3934.079063 | derived_engineering_estimate |
| `av_pv` | AV/PV | 8589934592 | 33554432 | 144703488 | 110.868132 | 1033.596343 | 1033.596343 | 1033.596343 | derived_engineering_estimate |
| `o_proj` | O_proj | 34359738368 | 0 | 50331648 | 295.648352 | 359.511771 | 359.511771 | 359.511771 | derived_engineering_estimate |
| `residual_1` | res_add_1 | 0 | 4194304 | 25165824 | 4.619505 | 368.819912 | 368.819912 | 368.819912 | derived_engineering_estimate |
| `norm_2` | RMSNorm_2 | 0 | 25165824 | 16785408 | 27.717033 | 246.000000 | 246.000000 | 246.000000 | reported_anchor_reused |
| `ffn_fused` | FFN | 360777252864 | 234881024 | 369098752 | 3363.000000 | 2636.419657 | 3363.000000 | 3363.000000 | conditional |
| `residual_2` | res_add_2 | 0 | 4194304 | 25165824 | 4.619505 | 368.819912 | 368.819912 | 368.819912 | derived_engineering_estimate |

Maximum absolute group difference between the reference equations and the workbook cached values: 4.55e-13 μs. Reference formula reproduction: **passed** against the <1e-06 μs threshold.

The attention-core sum counts Q/K/V projections, RoPE, QK, scale+softmax, AV/PV, and O projection exactly once: source group sum 7304.962795064 μs; reference reconstruction 7304.962795064 μs. The workbook's B6 attention subtotal is an engineering estimate and is not included as another component.

FFN is one fused 3363 μs group. Its compute-budget attribution is gate 1034.769230769, up 1034.769230769, SiLU 242.524038462, gate-mul 16.168269231, and down 1034.769230769 μs; these sum to 3363.000000000 μs and are not five independently measured kernel times. Its memory service budget is 2636.419657143 μs.

The 13 group reference predictions sum to 11897.602619368332 μs. The attention and FFN parent values are reporting subtotals only; neither is added again to the block total. The formula cache B16 is the authoritative workbook sum cache; B15 remains separately identified as a hardcoded total.

## Evidence state

| Layer | Status | Meaning |
|---|---|---|
| XLSX cache | passed | Exact XML formula/cache extraction; recomputed sum agrees within 1e-6 μs; says nothing about prediction accuracy. |
| TX8 reference | passed | Independent 13-group equations and attention/FFN/block accounting reproduce the reconstructed workbook values. |
| New coarse-library execution | not_run | This task only reconstructs workbook/reference behavior; C ABI is not needed. |
| Independent held-out hardware validation | not_available | No auditable raw measurements or holdout set exist in this workbook. |

The reference agreement is **reproduction**, not prediction accuracy. Since the source values were themselves reconstructed, no inference about real TX8 hardware accuracy follows.
