# LRU workspace v2 reconstruction and next pressure work

The implementation preserves the LRU variant's whole-session eviction,
remote_store/pool/history, store→restore ordering, routing and callback behavior.
Workspace is separately owned capacity, using the exact compiled Plan peak and
rank/node/generation lease identity. The bridge installs an absolute full-rank
revision view before policy and returns the post-policy capacity snapshot.
Preflight validates every rank before graph mutation; changed ranks join touched
ranks for a real issue pass. Typed hardware rates remain distinct from generic
TFLOPS, and coarse metrics use actual Runtime service counters/raw denominators.

The final official 8-rank fixture and its reconstruction scripts are in
`online-reproduction/`; measured identities and logs are in the matching
[evidence tree](../../evidence/lru-workspace-v2-20261014/README.md).

The `aggregate/CMakeLists.txt` is the actual aggregation scaffold. For the saved
runner's original work layout, copy it to `work/five-os-20261014/aggregate/`, use
the explicit `ASTRA_SIM_ROOT` for LRU, and configure as described in the measured
CPP README. Build the nine test executables as well as Online/two ABIs before a
fresh CTest run; a command that builds only lifecycle cannot recreate nine tests
on a clean build tree. The test names and exact successful commands are in the
CPP provenance and CTest log. This Cloud environment uses activation followed
by `unset PROTOBUF_FROM_SOURCE`, with system FindProtobuf.
`bash die/docs/analysis/lru-workspace-v2-20261014/rebuild_cpp.sh` supplies that
complete configure/build/test command sequence. The individual commands were
used for the measured run; the packaged wrapper was syntax checked, not rerun
as a second clean build certificate.

Generate the checkout's official Chakra Python binding using
`python-checks/schema-generation.txt`, including its ignored source-relative copy
needed for generator root discovery. The broad Python runner preloads the actual
generated module and checks its path; it does not mock the generator or routing.
Copy `python-checks/run_python_checks.py` to
`work/lru-workspace-20261014/final/` and run from the repository root with activated
Python and explicit `DIE_MODEL_CABI_PATH` / `DIE_COARSE_CABI_PATH` pointing to the
actual LRU aggregate ABI build. It writes its report into that work directory.
One optional oracle module remains absent and is an explicit skip.

The following are read-only audits, not implemented or pressure-tested fixes:

- `workspace-demand-strategy.md`: why post-commit blocked-demand alone cannot
  safely recover protected/reserved KV; phase fit-only envelopes or an actual
  two-stage proposal must precede admission, and cold-store release requires
  physical ordering.
- `liveness-pressure-proposal.md`: an actual blocked-ready Plan can leave GPU,
  workspace and event queue empty. Closed Online fails loudly; it cannot make
  capacity change by self-retry.
- `gqa-input-audit.md`: base and LRU public producers still only accept num_heads.
  GQA needs a common group layout across config/trace/typed task/weights/KV,
  transfers, remote storage, SLO tooling and readiness. Existing typed Q/KV
  consumers alone are not an end-to-end producer.

Normal-path E2E does not certify pressure, multi-request restore/history, all
metric record types, generalized GQA, the other three v2 variants or latency/
host budgets. The next implementation uses these saved facts rather than
repeating the completed foundation tests.

`workspace-bound/REPORT.md` derives the two nonzero Mapping workspace formulas
and their monotonicity. Its 240 actual CABI probes cover two singleton manifests
(224 compiled, 16 local_region v1 rejections as required). Selected Plan maxima
are distinct from unselected-template bounds: tiny P/D 16,384/260 B selected
versus 40,960/648 B all-template; larger P/D 65,536/1,024 B selected versus
65,536/1,032 B all-template. The fixed-manifest all-span argument is a proposed
generalization; no multi-request envelope or fit integration was implemented.
These equations stay in the compiler: production Python must not copy them.
