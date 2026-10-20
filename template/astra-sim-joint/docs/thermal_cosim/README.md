# Embedded thermal co-simulation (`extern/thermal_cosim`)

This directory holds the thermal/power core that the joint repository embeds as
a first-class internal dependency.  The design and phased plan live in
`/home/sunhao/thermal/docs/thermal_joint_integration_plan.md`; this document
records what is implemented, how to run it, and — separately — what is *not* yet
verified.

## Responsibility split

> joint provides the workload activity, the software mapping and the parsed
> hardware; the embedded thermal component turns those into **power** and then
> **thermal feedback**; joint turns the feedback into a throttle and a joint
> decision, and applies the same throttle to both cost prediction and real
> execution.

| Object | Single owner |
|---|---|
| requests, P/D phases, KV, migration, candidate commit | joint scheduler / ledger |
| simulation clock, task progress, resource occupancy | joint execution engine |
| peak hardware, Core/Die/NoC/memory organisation | joint parsed hardware snapshot |
| thermal geometry, materials, cooling | thermal profile (`wafercosim/configs/thermal`) |
| action energy, background power, spatial projection | embedded thermal core |
| solid/coolant temperature + energy ledger | thermal session |
| thresholds, frequency steps, hysteresis, recovery | joint thermal policy |
| final resource rate and the instant it applies | joint runtime |

The three grids (Astra rank/network, per-die Core grid, thermal block/floorplan)
stay distinct and are never inferred from one another.  A rank number is never
used as a thermal block index.

## Layout

```
extern/thermal_cosim/
  pyproject.toml                 # installable: pip install -e ./extern/thermal_cosim
  wafercosim/
    config.py                    # schema v2: geometry/materials/cooling only
    thermal.py                   # LiquidRC + snapshot/fork/restore/reset/close
    action_energy.py             # EnergyModel (dynamic J, background W, domains)
    power.py                     # ActivityWindow -> PowerWindow (new)
    contracts.py                 # stable data contracts + mapping builder (new)
    session.py                   # ThermalSession: power, committed timeline, fork (new)
    paths.py                     # packaged-config resolution (new)
    fixtures.py                  # named synthetic fixtures (new)
    configs/thermal|energy|fixtures
  tests/                         # thermal-core + contract tests
  LICENSE, THIRD_PARTY.md, repos.lock.json, SOURCE_MANIFEST.json
  docs/                          # upstream design notes (input contract etc.)
  reference/                     # optional native bridges + offline examples
```

Configuration ships as package data, so `wafercosim.config_path(...)` resolves
the thermal/energy profiles without a sibling directory or an absolute
`PYTHONPATH`.

## Install

```
python -m pip install -e ./extern/thermal_cosim
python -m unittest discover -s extern/thermal_cosim/tests -p 'test_*.py'
```

## Run

The joint side owns the run configuration (`thermal_mode` = `off` | `shadow` |
`closed_loop`, plus references to the thermal profile, energy table and policy):

```
JOINT_THERMAL_MODE=shadow \
JOINT_THERMAL_CONFIG=sh_test_mesh/workload/llama2_7b_inference/thermal/synthetic_run_config.json \
python ...
```

`off` (the default) preserves existing behaviour.  `shadow` consumes real
activity and produces temperature and a *proposed* cap but never installs it —
that is the zero-drift comparison arm.  `closed_loop` additionally applies the
cap (the caller/runtime installs it) and bumps the resource epoch.

Offline acceptance and replay:

```
python sh_test_mesh/run_scripts/thermal_acceptance.py --output-dir local-results/thermal
python sh_test_mesh/run_scripts/thermal_replay.py \
  --run-dir local-results/thermal/closed_loop --verify
```

## Build and run the online binary (verified minimal chain)

```
cd template/astra-sim-joint
mkdir -p build/astra_analytical && cat > build/astra_analytical/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.22)
project(AstraSim_Analytical_Build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
EOF
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware \
  -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
# cap parallelism: the ASTRA-SIM + protobuf translation units are memory heavy
cmake --build build/astra_analytical/build_congestion_aware \
  --target AstraSim_Analytical_Congestion_Aware_Online -j4
```

Minimum-cost online smoke (1 request, ~7 s, peak RSS ~92 MB; `--combo none`
is required unless you intend the full T+J+E combo):

```
python3 sh_test_mesh/workload/llama2_7b_inference/traces/materialize_20_30s.py \
  <compute-trace.csv> \
  sh_test_mesh/workload/llama2_7b_inference/request_queue_placeholder.csv 94835001 1.0
python3 sh_test_mesh/workload/llama2_7b_inference/plan_materializer.py
python3 sh_test_mesh/run_scripts/joint_runner.py sh_test_mesh/runs/min_off_baseline \
  sh_test_mesh/workload/llama2_7b_inference/request_queue_placeholder.csv --combo none
```

## Units

Simulation time ns; thermal API time s; frequency Hz; throughput op/s or byte/s;
action counts op/byte; energy J; power W; temperature K; geometry m.  The only
`ns -> s` conversion happens in `ThermalSession.advance_committed`; the host's
`perf_counter_ns` is used for overhead accounting only.

## Provenance

`SOURCE_MANIFEST.json` records the source commit, per-file sha256 and the
vendoring note.  `THIRD_PARTY.md` and `repos.lock.json` carry the upstream
licences and locked versions; the offline tool checkouts are intentionally not
embedded.
