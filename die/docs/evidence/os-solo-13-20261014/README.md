# Solo 13-domain research evidence, 2026-10-14

The actual facts producer was compiled from `die/tests/os_lifecycle_runtime_driver.cc`
and the public `coarse_die` library. `current/producer_identity.json` records the
exact source/library/binary hashes and fixture identities. Its Git HEAD is context;
the file hashes identify the inputs. No production timing change was made.

The exact study matches expanded and separately implemented continuous-clock
oracles on strict FP32, native INT4, fractional converted INT4 and converted storage.
Dense 65×65×4097 matches 1,310 expanded events using 72 powered events. Large K
(64,000,001) and 2,000,010 output tiles each complete in 72 events with exact
service, payload and typed ledgers. Global cap=1 fails closed. Detailed return maps,
classword guards and every resource ledger are in `current/study_validation.json`
and `current/powered_*.json`.

The public integer-nanosecond checker keeps its existing 2 ns bound. Three probes
exceed it: converted-storage finish 3 ns; 129×129×2 with 64×64×1 tiles start/finish
6/6 ns; and K=1 two-output-slot pressure start/finish 3/3 ns. They are recorded as
deviations. The 257×257×2, 128×128×1 probe passes at 0/1 ns. Only the K=1 pressure
fixture actually reaches output occupancy two. Injected queue tests establish the
nonfinal/full transition rule, without claiming that a multi-K schedule reaches it.

`stops/stop_validation.json` records 226 independent exact prefix comparisons
across six fixtures, followed by resume and changed-HBM-capacity checks. The dense
case uses at most 99 events with pause/resume, or 104 with capacity change. Large-K
and large-output two-stop/resume runs match no-stop exact ledgers in 102 and 100
events respectively. These huge comparisons use the same mathematical model, not
an independently expanded large-grid oracle.

These are bounded offline mathematical results, not production cycle skipping,
fixed-width numerical certification, a multi-owner/cancellation proof, host
performance measurements, or the original production latency/busy acceptance.
The production compiled E5 failure and unavailable E8/E16 are unchanged. TX8
reference coefficients and target priors remain frozen.

`ARTIFACTS.json` hashes all delivered evidence. Rebuild sources and commands are
in [analysis](../../analysis/os-solo-13-20261014/README.md). Generated executables,
library archives and caches are intentionally reconstructed from the recorded inputs.

`fixedwidth/bitwidth_audit.json` instruments arithmetic results and conservative
raw cross-product widths across 11 geometry/expanded runs and eight external-stop
runs. Normalized widths peak at 213 numerator/153 denominator bits; inferred
scratch peaks at 271/218 bits. Checked 768-bit scratch therefore covers these
observations, while 256-bit scratch does not. `rational256_test_results.json`
records 2,247 Fraction differential operations for the standalone C++17 primitive
and its sanitizer self-test. It is neither a complete C++ lifecycle nor a
production numerical/performance certificate. Research sources are in the
matching analysis `fixedwidth/` directory.
