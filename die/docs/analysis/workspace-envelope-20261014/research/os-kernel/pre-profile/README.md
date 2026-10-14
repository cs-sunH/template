# Scratch C++ Solo13 kernel

This is an isolated, standard C++17 implementation study for one owner and one
core. It reads the existing Runtime driver's class-facts JSON reports and does
not calculate resource work, bytes, typed operations, or private-time inputs.
`solo13_kernel.hh` rebuilds `GemmOSGeometry` from the report and verifies every
one of its at-most-32 records against that independent geometry before caching
the three actual Load/Compute/Store `Activity` facts per class.

Run the compiler and checks from the repository root with:

```bash
bash work/os-solo-cpp-13-20261014/build_and_check.sh
```

The kernel keeps three active-stage slots, two input queue slots, two output
queue slots, a fixed 32-class facts cache, fixed 256 snapshot slots, checked
u128 geometry/cursor/return arithmetic, and one global 256 expanded-event
budget for the life of the kernel. It performs exact one-owner 13-row
progressive filling with the class facts' PRIVATE limit, and accumulates exact
service/busy/typed-op/HBM ledgers. It uses the frozen checked 256-bit rational
state and 768-bit scratch prototype. Any arithmetic/geometry overflow fails
closed.

Identity skips compare canonical rational progress, live stage/queue topology,
all translated wave/class facts, and the retired-output offset. K-interior
identity and output-classword/contiguous-shape guards cap a skip before the
next K or M/N class boundary. External stop times are exact rational seconds;
the exact repeat count is clamped to the stop. Resume retains the same event
budget. A capacity update invalidates saved identity snapshots and keeps the
event count. The optional `unpowered-cap` test disables skips only to verify
fail-closed behavior at event 256; it is not a scheduler fallback.

`check_kernel.py` compares full and intermediate ledgers/state to the Python
exact `Fraction` implementation, then compares small full and paused prefixes
to the independent readiness/allocator oracle in
`work/os-solo-stop-20261014/check_stops.py`. It covers strict FP32, native and
converted INT4 (including fractional kappa), dense tail geometry, 17 ns stop,
pause/resume, capacity invalidation, backwards-stop rejection, and huge K and
output grids. The huge cases are compared to exact Fraction powering rather
than expanded tile-by-tile. `kernel_validation.json` records facts/source
hashes, compiler-run timings, statuses, event counts, and exact skip counts.
`build_and_check.log` captures the real command output.

The runtime is continuous exact internal time. It does not call the public
integer-nanosecond Runtime clock; its pause boundaries therefore do not make
the public wake-rounding claim. The large-shape Fraction comparison is the
existing exact study, not an independent huge-grid expansion. These checks
cover only the captured facts and single-owner/single-core synthetic
capacities. They make no production Runtime, multi-owner, C ABI, 50 us / 2 ms,
or descriptor-certificate claim. Nothing here is registered with CMake or
integrated into tracked production source.

Recorded development failures: the first compile with `-pedantic` rejected the
existing geometry header's GNU `unsigned __int128` typedefs, so the reproducible
build uses C++17 warnings-as-errors without `-pedantic`; the first smoke run
also exposed a progressive-fill loop that did not stop after freezing every
stage. Both issues were corrected before the captured passing suite. The
deliberate unpowered 650-wave dense run is retained as a `failed_closed` test at
exactly the global 256-event cap.
