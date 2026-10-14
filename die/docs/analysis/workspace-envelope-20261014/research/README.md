# Frozen research for handoff

The user requested handoff and immediate stop on 2026-10-14. No new development
task was started after that instruction. These are isolated studies and partial
work, not activated production code.

- `store-release/`: unapplied three-file prototype patch, real builder/generator
  fixture, baseline/prototype OFF JSON and report. Both versions passed the
  existing 10+13+18 tests; the new fixture passed 5/5. OFF bytes match. The
  patch tracks remote-store source release separately from same-session pool
  write tails, and gates destination writes/Compute. C++ NodeStore GC was
  audited only, not replayed; full Online and migration's own source-release
  coverage still need verification. Review the activation flag before applying.
- `manifest-envelope/`: a frozen **unexecuted draft** and its own HANDOFF.
  No finite multi-request certificate or test result was produced. Inputs and
  dependency locations are recorded there; do not treat this as admission E.
- `os-kernel/`: standard C++ solo 13-domain kernel, fixed-width rational
  arithmetic, facts, checks, baseline and profiling/sanitizer state. Its own
  freeze note identifies actual completed commands and partial measurements.
  It is not registered with public CMake or production Runtime.

Recreate each original root `work/` location from the corresponding source
directory before running its scripts. The OS checker also needs the earlier
`os-single-owner-13-20261014/current`, `os-solo-stop-20261014` and
`os-fixedwidth-13-20261014` work trees. Their sources and evidence were delivered
by PR #9 under `analysis/os-solo-13-20261014` and its evidence tree; the new
kernel package includes its facts and fixed rational header as well.

For the store prototype, recreate `baseline-src` and `prototype-src` under its
original work directory by copying the current LRU `sh_test_mesh` source tree
and the official generated Chakra module. Check `BASELINE_SOURCE.json`, then
apply `prototype.patch` to the prototype copy (paths are repository-relative).
The old `run_prototype.sh` assumes those copied trees exist. Do not overwrite
the production tree during this reconstruction. The patch remains an experiment
until source release, matching communication edges, GC and real Online have
been validated together.
