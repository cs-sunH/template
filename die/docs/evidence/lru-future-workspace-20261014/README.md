# LRU future-workspace checkpoint evidence

The default and capped actual Online results, journals and binary/source
identities are in `run/`, `run_capped/` and `source-identity/`. The base SHA
`912b748` is the checked-out context; per-file hashes identify the dirty source
actually compiled and executed. These runs are generated fixtures, not hardware
measurements. Rebuild instructions are in
[the analysis package](../../analysis/lru-future-workspace-20261014/README.md).

Public final build/CTest passed 30/30. The actual C ABI client passed 13 tests
with a deliberately incompatible legacy library supplied explicitly. Final
Python Online discovery passed 182 tests and 15 subtests, with two recorded
skips (optional LRU oracle and unavailable archived pre-envelope library); the
dedicated client invocation covered the incompatible-library check. Explicit
LRU two-ABI config tests passed 15/15. The focused CTest subset passed 5/5;
GraphBatchCommitterTest, including new source ACK/GC fixtures, passed separately.
These scopes are recorded independently, not added as unique coverage counts.

The old envelope ABI's 224 successful responses and 16 expected errors are
unchanged. The separate [62-task identity replay](../workspace-capacity-proof-20261014/README.md)
keeps all Plan hashes and complete summaries unchanged. Canonical23 source
digest is `9d025e25699e44b708997da5e4bc9c45e60c75b02166be931282f20921c0c4f4`.

`harness-failures.json` records the stale schema/provenance/run-path failures
and their fixes. Neither a no-test harness failure nor an old-auditor rejection
was reported as a passing integration run. No production timing/granularity,
hardware-calibration, full-compatibility or overhead certificate is issued here.
