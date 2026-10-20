# Acceptance status

Two conclusions are recorded **separately**: the *engineering* closure achieved
by this change, and the *physical accuracy* of the target model (still open).

Reproduce the engineering matrix offline:

```
cd template/astra-sim-joint
python sh_test_mesh/run_scripts/thermal_acceptance.py --output-dir local-results/thermal
```

It writes `acceptance_summary.json` with one status per entry and never folds a
`not_run` native/business test into "all passed".

## Verified in this change (offline)

| Check | Evidence | Status |
|---|---|---|
| V01 standalone package | `extern/thermal_cosim` installs/imports and resolves its packaged configs with no sibling directory; `extern/thermal_cosim/tests` | pass |
| V02 hardware/mapping export | `config_resolver.export_thermal_hardware_facts` (54 dies, 144 GB, 4710.4 GB/s, missing facts reported); `joint/test_thermal_hardware_export.py` | pass (needs `libcoarse_die_cabi.so`) |
| V03 J/W ledger | `sum(P)*dt + external == total`; per-action/block/external split; `tests/test_thermal_session.py` | pass |
| V04 thermal state | snapshot keeps coolant + ledger; fork discard leaves committed temperature/time/ledger unchanged | pass |
| V05 thermal time | ns->s at the boundary only; overlap/regression/gap rejected; explicit idle windows advance cooling | pass |
| V06 activity progress | cross-window contiguity, duplicate commit idempotence | pass |
| V13 three modes | `off` no session; `shadow` produces a cap but never installs it; `closed_loop` installs a 0.5 COMPUTE_GATE update and bumps the epoch | pass |
| V14 replay/convergence | `thermal_replay.py --verify` reproduces recorded temperature (0.0 K delta) and energy residual ~1e-13 J | pass |
| E2 online-service lifecycle | `online_service` creates one run-long thermal session from `JOINT_THERMAL_MODE`, records it in the run manifest, writes its audit and closes it on every exit path; closed_loop refuses to start while the C++ install path is absent | pass |
| activity sampler | coarse-runtime cumulative served counters → delta `ActivityWindow` (refuses incomplete/non-monotonic accounting) | pass |
| cap installer | envelope fraction → absolute COMPUTE_GATE rate → `die_coarse_update_capacity_v2` (offline recorded against a runtime ABI stub) | pass |
| real runtime ABI | `online/test_coarse_runtime_abi.py` drives a **real** `libcoarse_die_cabi.so`: model→runtime→advance→`update_capacity(COMPUTE_GATE,0.5)` and observes `capacity[0]` 1.0→0.5; the declared `served_by_category` keys match the real payload | pass |
| F projection capacity events | `resource_projection.CoarseCapacityEvent` + wire serializer; **real** `libjoint_projection.so` defers the predicted finish when a 40 ns `0.5` COMPUTE_GATE cap is frozen, with `capacity_events_processed == 1` (`joint/test_native_coarse_projection.py`) | pass |
| F production live update | `CoarseDieRuntime::update_capacity` (old-rate integrate → install → same-tick settle → stale completion invalidation/re-schedule); `die_execution_lifecycle_test.cc` case T3 asserts half-speed HBM0 yields a **later** completion tick and that −1/NaN/inf rates throw | pass |
| online build tree | `build/astra_analytical/build_congestion_aware/` produces the online binary + all four native libs (`-j4`, no swap growth) | pass |
| minimal online run | 1-request `off`-baseline smoke via `joint_runner.py --combo none` completes (exit 0, peak RSS ≈92 MB) | pass |
| joint regressions | the 44-test Python subset in the plan's appendix B still passes; `joint_config` thermal switch validated | pass |

Counts from the last run: **7 pass, 0 fail, 1 not_run**.

## Online closed loop — three-arm evidence (the completion criterion)

The C++ periodic thermal epoch (`SH_THERMAL_EPOCH_NS`, default off) plus the
`thermal_activity`/`thermal_cap` additive siblings drive the loop. Measured on
the minimum-cost 1-request run (`--combo none`, peak RSS ≈92–127 MB, swap 0):

| run | epochs_delivered | cap_events_applied | cap_events_skipped | completion_ns | e2e_ns | prefill_ns |
|---|---|---|---|---|---|---|
| off | — | — | — | 1852108869 | 1757273869 | 1059149316 |
| shadow | 19 | **0** | **200** | 1852108869 | 1757273869 | 1059149316 |
| closed_loop | 27 | **12528** | 0 | 2639218534 | 2544383534 | 1814546853 |

* **shadow is physically equal to off** (every column identical): the proposed
  cap carries `applied=false`, so the C++ consumer records it and never installs
  it. `cap_events_skipped=200`, `cap_events_applied=0`.
* **closed_loop changes real execution**: completion **+42.5 %**, e2e **+44.8 %**,
  prefill **+71.3 %**. Source: the fixture's hot die reaches 322.3 K ≥ the 304 K
  policy threshold → compute gate drops to 0.5 → `CoarseDieRuntime::update_capacity`
  hits the live runtimes (`cap_events_applied=12528`).
* Audit files land under `<run>/bridge/thermal/`: `activity_windows`,
  `power_windows`, `thermal_feedback`, `resource_envelopes`, `thermal_epochs`,
  `thermal_decisions`, plus `resolved_hardware.json`, `physical_mapping.json`,
  `thermal_manifest.json`.

## Remaining honest gaps

| Item | State |
|---|---|
| rank → thermal die | C++ fills the **rank number** as `source_id die<N>` (placeholder: it has no cooling-side map); the Python parser resolves rank → thermal die via `MappingSnapshot.rank_to_die`. |
| multi-die domain granularity | a multi-domain envelope emits one event per domain with the same `resource_index`; C++ installs on **every** rank holding a coarse runtime (last event wins per epoch). Feedback precision is therefore thermal-block/aggregate precision, not per-Core. |
| unmodelled actions | only `pe`/`vec`/`hbm_read`/`hbm_write` are modelled; SRAM/NoC/D2D/DRAM activity is reported, never assumed zero. |
| cap granularity | COMPUTE_GATE only; no independent HBM/NoC/D2D control domains yet. |
| DVFS depth | fixed-voltage, frequency-only; a compute-gate throughput fraction approximated as `f/f_base`. Not a voltage/OPP model. |
| window_start_ns | the provider is called per delivery, so it is the previous thermal delivery tick rather than strictly the previous thermal epoch. |
| joint selection (V12) | the cap reaches execution and the projection; the closed-loop *selection* comparison (a hot fixture that changes the instance/action choice) is not yet asserted. |
| standalone distribution | the package installs standalone and `run_all.sh` covers the thermal suites, but the full "copy the joint tree elsewhere and run the loop" distribution test is not yet automated. |

## Physical accuracy — separate and still open

The target-hardware energy facts do **not** match the reference energy table:
strict validation fails closed, which is asserted by
`test_target_hardware_does_not_match_the_reference_energy_table` (see
`joint/test_thermal_hardware_export.py`).  The joint task's **model-trust gate
for the RC candidate is still `failed`**.  No engineering test in this change
substitutes for target calibration, and `target_calibrated` stays `false` on the
synthetic fixture.  Regenerating action coefficients, SRAM macro organisation,
OPP and thermal physics for the real device is plan phase **H** and remains
incomplete.
