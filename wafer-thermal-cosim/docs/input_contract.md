# Thermal-core input contract

## Thermal configuration

Thermal configurations use `schema_version: 2`. The schema is intentionally
incompatible with version 1: a thermal input describes the thermal geometry and
physical parameters, not a task, software mapping, memory policy, or performance
model. The solver receives power and a time step; it returns temperatures.

`geometry` contains the die grid and per-die source-block rectangles. The HBM
and router rectangles use coordinates relative to the lower-left corner of one
die. The reference files preserve the former built-in rectangles as explicit
configuration data:

The following is an excerpt showing the schema boundary and source rectangles;
the thermal block omits required material and cooling values, so this is not a
complete runnable config. Use the packaged files as complete examples.

```json
{
  "schema_version": 2,
  "geometry": {
    "die_columns": 4,
    "die_rows": 2,
    "module_width_m": 0.025,
    "module_height_m": 0.032,
    "compute_side_m": 0.01732,
    "compute_groups_per_side": 2,
    "hbm": {
      "x_m": 0.0,
      "y_m": 0.01747,
      "width_m": 0.01626,
      "height_m": 0.00492
    },
    "router": {
      "x_m": 0.01747,
      "y_m": 0.0,
      "width_m": 0.001,
      "height_m": 0.01732
    }
  },
  "thermal": {
    "backend": "rc_liquid",
    "window_s": 0.01,
    "initial_k": 300.0,
    "inlet_k": 300.0
  }
}
```

For these reference values, `hbm.y_m` and `router.x_m` equal
`compute_side_m + 0.00015`. The rectangles preserve prior assumptions; they are
not a measured or validated physical floorplan. Hosts should keep die columns,
rows, and source locations consistent with the thermal geometry. The actual
package placement, interposer, material stack, and coolant-channel layout still
need an external physical description and review.
The compute area and HBM/router rectangles must fit within the declared module
width and height; the validator rejects overflowing source blocks.

The rest of `thermal` holds solver and material values: time step/window,
initial and inlet temperatures, silicon properties, RC capacities and
conductances, coolant properties/flow, and optional native-backend parameters
such as `three_dice`. Calibration metadata remains attached to exploratory
reference configurations so that assumptions and failed scientific gates stay
visible. Unconsumed throttle/resume/cooldown controls are not part of this
input.

## Energy inputs and host facts

Thermal configs do not contain an energy fallback. The one retained fallback
is in `configs/fixtures/tiny-analytical.json` for the energy-core unit test; it
is not a packaged thermal-run default. When a host uses the energy API, it
selects an energy table and passes the target system facts explicitly:

```python
energy = EnergyModel(calibration_config, hardware_facts=host_hardware_facts)
```

`host_hardware_facts` is supplied by the caller and covers the concrete resource
facts required to check the selected table. Its current fields are flat
resource facts (`sram_capacity_bytes_per_die`, `pe_ops_s_per_die`,
`vec_ops_s_per_die`, `hbm_capacity_bytes_per_die`, and `dram_capacity_bytes`)
plus `compute` (`weight_bytes`, `activation_bytes`), `noc` (`nx`, `ny`,
`flit_width_bits`, `num_vcs`, `packet_size_flits`, `router`, `routing`), `hbm`
and `dram` (`protocol`, `config`), and `sram` (`macro_capacity_bytes`,
`macro_count_per_die`, `word_width_bits`, `access_bytes`). It is not inferred
from `geometry`, an energy-table domain, or a `calibration.hardware` object.
The NoC dimensions describe the energy model's network, separately from the
thermal die array and the Core grid inside a die. A future host adapter must
map their identities explicitly rather than treating those dimensions as equal.
For memory resources, the actual host macro capacity and macro count per die
are host facts. Macro counts stored in an energy table remain source/reference
organization only; any resulting organization scaling uses the host facts.
Other artifact paths and hashes in energy-table metadata document historical
generation provenance; they are not files loaded by the EnergyModel at runtime.
These fields define a normalized host-fact contract, not a raw joint JSON object
that can be passed through unchanged. The caller converts PE/Vec rates into the
selected energy table's operation and precision basis; joint INT4 throughput or
equivalent-op units need explicit conversion when the table uses a different
basis. SRAM macro organization also requires physical/architectural data from
the host owner. Do not infer actual host resources or macro counts from an
energy table's reference domain. The host assembles these facts at the call
site; they should not be duplicated in a second independent runtime hardware
JSON file.

Calibration settings such as the energy-table path, strictness, acceptance of
assumed coefficients, and
`memory_organization_sensitivity_accept: {"hbm": true, "dram": true}` remain
configuration choices. The latter records an explicit acceptance of the
table's organization sensitivity; it does not claim that the table's hardware
organization is the actual host organization or replace host facts. The old
performance `service_table` is not part of the energy API or thermal input.

## Migration from schema version 1

The reference configs under `configs/thermal/` have been migrated. Existing
version-1 configs require explicit migration before use:

- Rename `hardware.nx` and `hardware.ny` to `geometry.die_columns` and
  `geometry.die_rows`. Retain only module dimensions, compute-side length, and
  compute groups from the former hardware block; add explicit `geometry.hbm`
  and `geometry.router` rectangles.
- Remove the old `model`, `workload`, `energy`, and `simulation` blocks and
  performance-only hardware resources. Remove unused
  `thermal.throttle_k`/`resume_k`/`throttle_ratio`/`idle_cooldown_s` fields.
- If the energy API is used, move actual target-system descriptors out of
  `calibration.hardware` and pass them as `hardware_facts` to `EnergyModel`
  using the fields above.
  Keep energy-table selection and strict/assumption controls as calibration
  settings. Remove the obsolete `service_table` entry.

This boundary prepares host-owned inputs for a future integration. It does not
claim that a joint runtime currently exports power windows, calls the thermal
solver, or consumes a thermal feedback loop.
