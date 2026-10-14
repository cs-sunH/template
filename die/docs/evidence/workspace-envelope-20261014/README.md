# Compiler workspace envelope and admission foundation

This is a continued-development checkpoint, not overall simulator acceptance.
The effective source hashes and release manifests are in `source-identity.json`.
Historical build HEAD values identify dirty-tree context, not committed inputs.

| Check | Actual result |
|---|---|
| Public standalone CTest | 30/30 |
| Existing task hash and full summary comparison | 62/62 identical |
| Actual legacy-v1 envelope queries | 224 successes, 16 expected local-region rejects |
| Replay through new production ABI | All 240 responses/errors identical |
| Python envelope client | 11/11, real C ABI |
| Manager floor and related suites | 57/57, no skips |
| Complete LRU online regression | 154 passed, 1 skipped (optional oracle absent) |
| Typed hardware/config/scheduler regression | 44/44 |
| Five copied trees, two ABI targets each | 10 builds succeeded |
| Five config resolver suites | 83/83, no skips |
| Four current business adapter focused suites | 18/18; four Online targets rebuilt |

`public/`, `legacy-v1/`, `manager-floor/`, `python-client/`, `python/`,
`five-abi/`, and `four-focused/` preserve the measured commands/logs/results.
The complete client/manager reconstruction packages are under the companion
analysis directory. The first JOINT config attempt failed because its missing-
library fixture restored the environment override before discovery; the test
fixture was corrected, production resolver unchanged, and the 13-test suite
passed. The failed attempt remains delivered.

All five coarse libraries from the repository ABI builds have SHA-256
`7e9b9670b8c06c8600688acf6446acc5f112a97e83315a26ea8d554040cc5600`.
The standalone public library tested earlier has SHA-256
`b4778f15831110814eb68de6cf3cf02995d65aa480fff929946f0e68504ea80d`;
build configurations are separately recorded. The exact canonical 23-file
source digest is `e0b6fd1a083ebe89cc73956f84f07524e6c3944fbcd4cc28a91a947d66cbbf78`.

LRU diagnostics store bounded observations from the last issue pass. They are
not a live blocked-node census, create no retry timer, and do not cure the
pressure dead end. The optional manager floor defaults to zero and is unused
by production scheduling. `max(A,E)` relies on normal serial Compute issue;
direct/concurrent test calls bypassing that gate do not carry this certificate.
`E` is not added to actual workspace, KV, reservations or remote accounting.

New main `fecf81c` replaced the active JOINT tree and archived the old tree.
This checkpoint adds only the pure canonical library and corrects its config
test isolation. Its legacy config tests do not establish coarse business
integration. No production Online E2E was rerun for this checkpoint, and no
new OS production timing, TX8 fit, target heldout data or host-budget result is
claimed. Store/migration source-release ordering and manifest-bound activation
are continuing tasks. `ARTIFACTS.json` identifies the files actually delivered.
