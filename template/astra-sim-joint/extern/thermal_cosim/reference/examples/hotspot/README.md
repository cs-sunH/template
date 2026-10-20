# HotSpot persistent-step example

This example exercises the block-level RC model with one persistent HotSpot
model and temperature state. Each non-comment input row advances the model by
`dt` seconds using one power value in watts for each floorplan block. The
bridge writes cumulative time and the resulting block temperatures in kelvin.

From the repository root, build and run it with:

```sh
bash scripts/build_hotspot_bridge.sh
cd examples/hotspot
../../build/hotspot_bridge --flp wafer_minimal.flp --config wafer_minimal.config \
  < wafer_minimal.steps
```

The first stdout line is a CSV header beginning with `# time_s`; diagnostics
and HotSpot initialization messages go to stderr. The `die0`, `die1`, and
`side_hbm` blocks are adjacent regions in a simplified 15 mm by 5 mm plane.
They demonstrate spatial coupling and time-varying power only. The sample uses
HotSpot's standard block RC model, so it does not model wafer stack layers,
liquid channels, or HBM-specific materials. Use a configured grid stack and
`--detailed-3d --microchannels` for HotSpot's compact microchannel path.

The build script expects the HotSpot checkout at the repository's
`upstream/HotSpot` directory. Set `HOTSPOT_ROOT=/path/to/HotSpot` if it is elsewhere;
`CC` selects the bridge compiler and `BUILD_JOBS` controls parallelism for the
upstream make.
