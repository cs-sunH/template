# Grouped Runtime allocator Q-priority fixtures

This scratch driver submits hand-built `Plan` objects to the actual
`coarse_die::Runtime` implementation and reads rates and fairness weights from
`Runtime::query_progress`. It complements the separate eight-case pure-Compute
allocator comparison under `work/audit-busy/grouped-allocator-runtime/`; it
does not duplicate those scenarios.

The cases target canonical-Q selection that the pure-Compute set did not cover:

| Scenario | Coverage |
| --- | --- |
| `local_precedes_sram_for_compute_q` | Compute activities have positive LOCAL, SRAM, and GATE work; `canonical_basis` selects LOCAL. |
| `sram_sum_is_fallback_q` | Compute activities have SRAM read/write work and no LOCAL or HBM payload; Q is read plus write bytes. |
| `hbm_payload_precedes_local_and_sram` | HBM activities also carry larger LOCAL/SRAM work; Q still comes from the HBM payload. Companion Compute activities derive Q from LOCAL. |
| `zero_rate_hbm_activity_excluded_from_owner_denominator` | Submit at positive HBM0 capacity, set current HBM0 capacity to zero at time 0, then check the blocked HBM activity reports rate/alpha zero and each owner's remaining active alpha sums to one. |

Every case has two owners, one epoch per owner, no credit deltas, and no
credit-boundary pressure. Only `COMPUTE_GATE`, `HBM0`, and `HBM1` have finite
peak capacities. The scratch hardware fixture assigns `1e30` to every other
Runtime resource and marks those Runtime-only domains enabled; no activity uses
them. This resource setup is synthetic and makes no claim about physical
topology or compiled GEMM behavior. Compute activities include one unit of
GATE work plus the specified LOCAL/SRAM work, so their Q is obtained through
the production `canonical_basis(Activity)` helper rather than manually set.
HBM stage bytes populate both the matching row work and HBM payload matrix.

The exact Runtime fields are `query_progress.activities[].rate` (unit
`progress_per_second`) and `query_progress.activities[].alpha`. Each result
also contains the input activity work, canonical Q and its selected basis,
peak/current capacity maps, returned activity status, actual rate and alpha.
All rates in `runtime-rates.json` came from the Runtime query; the driver does
not insert hand-calculated expected rates.

## Baseline and command

Before a parallel C API-only rebuild, the existing static archive was copied
to `baseline/libcoarse_die.a`. Its SHA-256 is
`e0864d0eba1402a39654e572afc23a624a667fe7e04952a64486d99d6a560a9a`. The
archive was built against public source digest
`e877a4ddc2dd5e0eff332fc8c01fcc33cc3ea92cfb8aac99c856b45a24f48b7a`; the
parallel C API change does not replace the Runtime code linked by this driver.
The report records the archive, driver source and binary hashes, compiler,
exact compile command, and exact run command.

From `/workspace/template`, run:

```sh
bash work/grouped-allocator-driver/build_and_run.sh
```

The script rebuilds only this scratch C++ executable, verifies the copied
archive SHA, runs four small zero-time rate queries against
`die/configs/example_hardware.json`, and writes `runtime-rates.json`. Compiler
output is in `compile.log`; un-enriched Runtime output is retained as
`rates.raw.json`.

This is a synthetic allocator/fairness fixture, not a compiler certificate, a
compiled-GEMM workload, or a performance benchmark.
