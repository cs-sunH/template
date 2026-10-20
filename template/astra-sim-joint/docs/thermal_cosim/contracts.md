# Data contracts (implemented surface)

All types below are in `extern/thermal_cosim/wafercosim/contracts.py` and are
validated fail-closed.  They are the *implemented* subset of the plan's §5; the
C++ types that mirror them on the wire are not part of this change.

## HardwareSnapshot

Produced by the joint resolver (`sh_test_mesh/config_resolver.py`,
`export_thermal_hardware_facts`) from the same `ResolvedHardware` / coarse C ABI
that materialises the runtime configs.  It carries: schema version, a canonical
`content_hash`, source path (with sha256), physical source list, units, mesh
rows/cols, per-die Core grid + clock + precision, per-resource peaks, SRAM/HBM/
DRAM/NoC organisation and D2D bandwidth, and the normalized
`energy_hardware_facts` for the energy table.

`from_json` re-verifies `content_hash`, so a run cannot be replayed against a
different hardware snapshot.  Fields the host cannot state are listed in
`missing_host_facts` (see the target export below) rather than filled from a
table's reference domain.

## MappingSnapshot

Built by `build_mapping(blocks, ...)` from the thermal floorplan plus an explicit
`rank_to_die` map.  It holds `mapping_epoch`, `hardware_hash`,
`source_block_weights` (per-source distribution, non-negative and summing to 1),
`rank_to_die`, `source_to_destination`, `external_sinks` and the stable
`block_ids`.  Energy with no on-wafer block (external DRAM) is an explicit sink;
it is never averaged across dies.

## ActivityWindow

Incremental half-open `[start_ns, end_ns)` facts: identity/sequence, hardware and
mapping versions, resource epoch, `(source_id, action_kind, count)` actions and
`(source_id, active_ns, idle_ns)` dwell, operating point, coverage and
provenance.  Unknown actions are rejected (never treated as zero).

## PowerWindow

Per-block `power_w` aligned with the stable `block_ids`, plus the dynamic /
background / external energy ledger and diagnostics (operating-point model,
temperature basis, per-action energy).  Invariant:
`sum(P)*dt + external == total_energy_j`.

## ThermalFeedback

Thermal epoch, effective simulation ns, committed s, block K, coolant K, per-die
block-max K, energy residual / input / removed, model + calibration identity,
state version, energy-domain status, and the hardware/mapping/resource epochs.

## ResourceEnvelope

Joint-owned policy output: per control domain the step, compute-throughput
fraction, resource ceiling, temperature margin, reason and `applied` flag, plus
`effective_from_ns` / `as_of_ns` and `resource_epoch`.  Produced by
`joint/thermal_policy.py`, never by the thermal core.

## Operating point

First version: a declared fixed voltage with frequency scaling only.  Dynamic
energy per action stays in the calibrated domain; a lower frequency only reduces
how many actions complete.  `P ∝ f³` is deliberately not applied.  The model is
recorded as `fixed_voltage_frequency_only` in every power window.
