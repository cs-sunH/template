# E5 compiled GEMM axis evidence

The evidence is from the actual `coarse_die::compile()` output and public C++
Runtime for the requested shapes. The grouped owner-Q/alpha oracle compares
the three OS selections against independently expanded K-wave work. These are
synthetic model-to-model results. The top-level `manifest.json` remains
`failed`; no precision pass is claimed.

| Probe | Actual template / tile | Oracle and comparison outcome |
|---|---|---|
| M80/N128/K128 | Weight Resident, 1×64×64 | Compiler reselected the dataflow and `tm`, so the OS wave oracle rejected this case. Plan activity totals matched the compiled summary exactly. Solo and synchronous runtime work counters strictly equal the integer plan totals; at 50 ms offset only vector work is non-equal, by −4.54747E−13 ops. |
| M80/N128/K4096 | Output Stationary, 16×64×64 | E5 comparison failed. Solo and synchronous finish error 2.432%; offset finish errors 0.569% and 1.216%. HBM busy errors are 202.275% (solo/synchronous) and 240.350% (offset); offset shared-busy error is 195,548.198%. |
| M80/N128/K4097 | Output Stationary, 16×64×64 | E5 comparison failed. Solo and synchronous finish error 2.034%; offset finish errors 0.561% and 1.216%. HBM busy errors are 202.275% (solo/synchronous) and 240.335% (offset); offset shared-busy error is 3,229.249%. |
| M65/N65/K4097 | Output Stationary, 16×64×64 | E5 comparison failed. Solo and synchronous finish error 3.827%; offset finish errors 10.782% and 0.535%. HBM busy errors are 124.951% (solo/synchronous) and 158.869% (offset); offset shared-busy error is 994.515%. |
| No-argument compatibility | Output Stationary, 16×64×64 | The original 65×65×4097 default path ran and returned raw counters; its E5 comparison remains failed. |

Completion limits remain 3% for solo and 5% for concurrent work. HBM-busy and
nonzero shared-HBM-busy limits remain 5%. The compiler selected the same OS
template and tiles for K4096/K4097 and for M80/N128 versus M65/N65 at K4097.
K128 is not a clean K-depth pair: it switches to Weight Resident, 1×64×64, and
160 output rounds instead of 10. The M/N-edge case expands into output tile
classes `{4×(16×64), 4×(16×1), 1×(1×64), 1×(1×1)}`.

## Work accounting detail

`planned_integer_conservation` checks exact integer equality between the
independent wave ledger and compiled-plan totals for the OS cases. The runtime
counter section separately compares serialized cumulative long-double values
with expected integers using Decimal arithmetic. Every mode remains within
the 1e-9 relative reconciliation threshold. Where serialized values differ,
the report preserves `strict_integer_equality: false` and the actual delta;
the tolerance result is not called exact equality.

| Shape | Solo | Synchronous two-request | 50 ms offset |
|---|---|---|---|
| M80/N128/K4096 | Strict HBM and vector equality | Vector delta −2.910383E−11 ops | Vector delta +1.4551915E−11 ops |
| M80/N128/K4097 | Strict HBM and vector equality | Vector delta −2.910383E−11 ops | Strict HBM and vector equality |
| M65/N65/K4097 | Strict HBM and vector equality | Strict HBM and vector equality | Each HBM row-read counter delta −4.54747E−13 B; writes and vector work strictly equal |
| M80/N128/K128 | Against compiled plan only; strict equality | Against compiled plan only; strict equality | Against compiled plan only; vector delta −4.54747E−13 ops |
| Default M65/N65/K4097 | Strict HBM and vector equality | Strict HBM and vector equality | Same two HBM row-read deltas as M65/N65/K4097 |

The C++ output and full Decimal reconciliation objects are retained under
`raw/` and `reports/`. The K128 fallback is explicitly scoped to compiled-plan
activity versus runtime counters because its selected dataflow is outside the
independent OS oracle's preconditions. Archived fixtures that lack the exact
serialized counters report the runtime reconciliation as `not_available`.
The nonzero residuals occur in cumulative serialized totals; any follow-up
should trace the Runtime accumulation boundary and preserve integer work
conservation separately from its long-double display value.

## Files and rerun

`manifest.json` records all probe statuses and selected mappings;
`source_identity.json` records source and binary hashes; `logs/` holds the
build recheck, utility test, runner output, and per-probe driver stderr. The
portable runner and build instructions are in
[`../../analysis/gemm-axis-20261014/README.md`](../../analysis/gemm-axis-20261014/README.md).
It accepts an explicit `--driver` path, so it does not require this session's
build directory. Its run command exits nonzero because the E5 accuracy probes
remain failed, while retaining every report and raw output.
