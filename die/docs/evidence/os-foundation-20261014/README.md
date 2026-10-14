# OS facts and LRU manager development checkpoint

This checkpoint resumes from merged PR #7, main `fba3b8e544b65728be384b43ecfeb355fddd028f`.
It is continued development, not a stop/handoff request or engineering acceptance.

- Public default Release build and 29/29 CTest pass. OS production accounting now
  consumes bounded per-K geometry classes; its five-stage timing remains unchanged.
  The 62 captured selected task hashes and complete summaries are byte-identical
  to the baseline. The class, wave, q/core, tail, and single-Core resource-fact
  tests cover the new foundation; they are not a K-lifecycle timing certificate.
- `public/runtime-readiness-oracle.json` compares an offline C++ wrapper around
  actual Runtime with independent Fraction readiness/13-row allocation in six
  small fixtures. Terminal payload/typed work is exact. Public integer-ns wakes
  produce at most 2 ns start/finish differences. Lifetime long-double counter
  rounding is reported explicitly, including the native INT4 residual. The
  wrapper permits one active L/C/S each and two input/output slots. Its submission
  budget is distinct from production LP/discovery limits. No cycle skipping or
  production descriptor integration is claimed.
- The actual compiled production E5 comparison remains failed at the unchanged
  3%/5% thresholds; E8/E16 remain unavailable. `public/compiled-oracle.json` keeps
  this failure, which the new offline driver does not replace.
- LRU Python manager regressions: 45 pass using its official protoc-generated
  Chakra binding. Absolute workspace state remains outside KV rollback; fit and
  invariants include it. Whole-session LRU and remote/pool ownership are retained.
  Workspace-only impossible margins block without useless eviction or deep-gap;
  release permits a later normal eviction/restore. Protocol/C++ wiring is separate
  ongoing work and is not covered by these manager tests.
- The canonical 23-file publication and all five dual-ABI builds/config suites
  pass (83/83, no skips). Focused five-repository results pass 16/16 (3+4+3+3+3),
  with the pre-v2 LRU source/binary identity kept separately. The source digest is recorded in source-sync
  and ABI/config JSON, not inferred from a branch name.
- `base-online/` is a fresh tiny two-rank run of committed `fba3b8e`, including
  the final marker teardown correction. C++/Python exit 0/0, 20 typed COMP,
  3 successful schema-2 acks, positive FFN workspace, 31 successful terminals,
  REQUEST_COMPLETE at 36,185 ns. It predates the OS facts checkpoint and does
  not certify the subsequent LRU port or broader business/performance coverage.

TX8 reference coefficients and target priors are unchanged; no independent
target measurement or heldout data appeared. No new host performance claim is
made here. Large raw strace/runtime outputs remain reproducible under work/;
their identities and commands are retained in the Online report.

Reconstruction scripts are in `../../analysis/os-foundation-20261014/`.
Build with `source /workspace/cloud-setup/activate.sh` in this environment;
installed Protobuf uses FindProtobuf with `PROTOBUF_FROM_SOURCE` unset. Source
and binary evidence describe actual bytes, independently of the later commit ID.
