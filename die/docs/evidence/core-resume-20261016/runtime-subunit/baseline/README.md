# Current Core baseline red and unverified synchronous reduction

`runtime_credit_slack_test.cc` is byte-identical to the verified archive source.
It was actually compiled once with the current headers/configs and linked to
`work/build-core-release/libcoarse_die.a`, the fresh public51 library. It returned
1 with the expected positive-credit/headroom half-second assertion. Full compiler
and execution argument arrays, all source/header/config/library hashes before
and after, executable SHA and return codes are in `identity.json`; build/test
stdout/stderr logs are retained. No old executable or numerical patch was used.
No performance inference follows from this regression run.

`synchronous-contract.json` extracts only the original epoch4 numerical contract,
capacities and observed stalled state from the verified strict-projection log.
It is explicitly historical evidence. Its old hardware/Plan identity is not
imported, changed or represented as current hardware.

`synchronous_three_activity_candidate.cc` builds a new manual synthetic Plan
against a resolved current Core descriptor. Activity work, HBM bytes, Q, PRIVATE,
credit capacities/initial inventory and signed cumulative endpoints come directly
from that extracted contract. It does not depend on NormLongRow or other
unadopted mapping. Its new hardware/Plan identity labels the synthetic scope.

The reduction is source preparation only: it was not compiled or run. It may
pass the existing Runtime, because omitting earlier epochs changes the exact
fractional entry time and earlier rounded service/allocation history that led to
the archived final p=0.9999999999997531472231. The earlier multi-epoch failure
cannot be claimed reproduced from this source without actual new evidence.
Do not seed an old stalled snapshot with a replacement hardware fingerprint to
manufacture equivalence. If the reduction passes, preserve that result and next
derive the predecessor boundary history or a separate analytical manual case;
do not change credit/progress tolerances until it fails.

The original independent tiny-credit red already supplies a current-source
minimal numerical failure without requiring this larger reduction to be red.
