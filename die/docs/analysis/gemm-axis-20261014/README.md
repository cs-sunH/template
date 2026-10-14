# Compiled GEMM E5 shape probes

This portable package runs the compiled `coarse_die::compile()` driver and the
public C++ Runtime, then applies the independent grouped owner-Q/alpha
discrete K-wave oracle. These are synthetic model-to-model probes, not silicon
measurements or an accuracy certificate. Accuracy failures stay in the
reports.

Build the utility in any chosen directory after sourcing the project setup:

```sh
source /workspace/cloud-setup/activate.sh
cmake -S die -B /path/to/build-gemm-axis -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/build-gemm-axis --target coarse_compiled_gemm_oracle_driver -j2
```

Run the packaged script with that binary path and a chosen evidence directory:

```sh
python3 die/docs/analysis/gemm-axis-20261014/run_e5_axis.py \
  --driver /path/to/build-gemm-axis/coarse_compiled_gemm_oracle_driver \
  --output-dir die/docs/evidence/gemm-axis-20261014
```

The runner locates the repository from its own location, accepts any explicit
driver path, preserves every raw driver response, and exits nonzero when a
probe or its accuracy comparison fails. The no-argument default case checks
that the original 65×65×4097 driver behavior remains available.

The reports keep two work checks separate. `planned_integer_conservation`
uses exact integer equality for the independent wave ledger and compiled plan.
`runtime_cumulative_serialization_reconciliation` compares serialized
long-double counters with that integer ledger using Decimal arithmetic; it
records strict equality separately from the existing 1e-9 relative
reconciliation threshold and includes each signed and absolute delta. A
within-tolerance reconciliation is not an exact-equality result. Archived raw
fixtures without the required serialized counters report
`not_available`.

The smallest useful analytical follow-up for the matched OS cases is a
fixed-class period split:

`T = T_prologue + N_full × T_full_K + T_final_K`.

Apply it to HBM-busy and shared-busy increments, count M/N edge classes by
run length, and treat each full-K body as a discrete load/compute readiness
cycle. The final-K compute reserves an output slot and enables the final
store. For concurrent requests, reuse a period only while active owners,
stage membership, slot/credit state, and the grouped-Q/alpha LP region remain
fixed; split at the 50 ms arrival and each state boundary, and stop if the
state does not return. This keeps the next oracle small and avoids the heavy
rational scratch work.
