# Workspace capacity proof evidence

This evidence folder is intentionally small. It contains the original
counterexample's six-line output and the compact result of replaying all 62
historical Plan hashes and complete summaries through the built C ABI. It does
not copy the old binary, the 62-entry source capture, a build cache, or a full
duplicate summary log.

- [`counterexample.txt`](counterexample.txt): preserved old-library SRAM
  candidate-set result and repeated Attention Plan identity.
- [`identity-replay.json`](identity-replay.json): 62/62 exact Plan hash and
  complete summary matches via `die_coarse_compile_batch_v2`.
- [`source-identity.txt`](source-identity.txt): source, input capture, shared
  library and test identities.
- [`focused-test.txt`](focused-test.txt): the earlier capacity test command and
  result; it is separate from the parent's running full public CTest.
- [`../analysis/workspace-capacity-proof-20261014/README.md`](../../analysis/workspace-capacity-proof-20261014/README.md):
  formulas, sufficient proof conditions, counterexample interpretation,
  implementation diff map and replay command.
