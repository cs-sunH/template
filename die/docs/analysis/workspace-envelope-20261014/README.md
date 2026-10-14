# Workspace envelope checkpoint reconstruction

The public API is described in [API.md](API.md). It uses the actual compiler
mapping vector, exact uint64 bytes, and the existing owned JSON/free/error ABI.
It does not create a Plan handle or alter existing selected-Plan identities.
The new unit test is registered in the standalone `die/` build:

```bash
cmake -S die -B work/envelope-public -DCMAKE_BUILD_TYPE=Release
cmake --build work/envelope-public --parallel 2
ctest --test-dir work/envelope-public --output-on-failure
python3 die/tools/sync_repositories.py --check
python3 die/docs/analysis/workspace-envelope-20261014/replay_captured_envelopes.py \
  --library work/envelope-public/lib/libcoarse_die_cabi.so \
  --evidence die/docs/evidence/workspace-envelope-20261014/legacy-v1 \
  --output work/envelope-public/legacy-replay.json
```

`probe_legacy_specs.py` preserves the original measured driver. It was run from
`work/workspace-envelope-20261014/` and requires that historical work tree.
The portable replay above uses the delivered task JSON and exact hardware
bytes. The selected-Plan baseline remains in the PR #9 workspace-bound evidence;
no Python workspace or timing formula is used by either driver.

[lru-pressure.md](lru-pressure.md) records real one- and two-Plan blocked-ready
fixtures, the initial rank-clock failure, exact commands and old/new binary
identities. The new four-variant rebuild evidence links the current public
static archive; it is separate from the earlier pressure-only build.

The [Python client package](python-client/) preserves its original 12-file
manifest, commands, raw failures and passing standard-library unittest logs.
Set `DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY` to the new build. The missing-symbol
negative test accepts an explicitly supplied old coarse ABI; rebuild PR #9 in
an isolated checkout if the historical archived binary is unavailable. No
binary is delivered in this checkpoint.

The [manager floor package](manager-floor/) documents the optional immutable
full-rank `E`, fit with `max(A,E)`, permanent versus temporary rejection, and
the six suites (57 tests). Its runner must be copied to
`astra-sim-wscllm-LRU/work/lru-prospective-admission-20261014/` before execution,
then run from the LRU checkout. Generate the real official Chakra binding by
the instructions in `../lru-workspace-v2-20261014/python-checks/`; no dependency
stub or policy monkeypatch is used. The complete online/config regression
runner in the new evidence directory runs from the repository root with both
new ABI paths supplied.

[joint-newline-audit.md](joint-newline-audit.md) describes the new active JOINT
implementation imported by main `fecf81c`. The pure coarse library is additive;
the new JOINT business adapter/protocol has not been ported. The archived
`astra-sim-joint_old` tree is preserved and supplies no current test result.

The manager floor is not activated by the scheduler. Complete finite-manifest
coverage, actual-Plan bound checks and source-release ordering for eviction and
cross-instance migration remain required before that wiring.
