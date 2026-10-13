# Cache response and grouped-reference checkpoint

Date: 2026-10-14 Asia/Shanghai. Baseline is merged PR #4, main
`47079d88fcf3a8408743a282804cd2098dee8557`. This is a continuing development
checkpoint; model accuracy and complete five-repository business acceptance
remain unfinished.

`verification.json` records exact source/binary identities and validation
scope. Public default targets built and 22/22 CTest passed. All five vendors
share the 20-file source digest `05c3f91c…e755`; this phase requested both ABI
targets and ran existing selected CTest executables (17/17) plus explicit-ABI
configuration tests (83/83). CMake rebuilt some core dependencies while building
the ABI. Online/AstraSim executables were not rebuilt in this phase; their
previous complete requested build is in the PR #4 evidence.

The cache retains escaped response fragments under the same 4096/64MiB limits.
The new boundary test actually reaches the byte limit; it does not separately
trigger the count limit or inject malloc failures. Seven default task hashes
and summaries remain unchanged. Their accounted cache bytes increase from
141094 to 164810 because the newly retained response strings are counted.

`performance/` preserves all three shapes × 30 samples for C++ cold/hot
recompilation and true same-model C ABI miss/hit, inputs, helper source,
commands, environment, CPU/RSS and source/binary hashes. Caller timing matches
the original baseline. Hit P95 is 10.426/14.292/18.788µs, meeting the 20µs target
on this fixture set. Attention P99 exceeds 20µs. The 128³ cold compile/miss
still exceeds 2ms. C++ core changes were absent, so its timing deltas are
controls. Setup attempts that produced no measurements are retained; missing
`/usr/bin/time` was replaced by a Python outer CPU/RSS wrapper without changing
the benchmark's internal clocks. Binaries are deliberately not delivered;
rebuild and place the reusable probe in the `work/` paths recorded by the runner.

`compiled-grouped-oracle-report.json` reuses the previously recorded real
compiled-driver output because its Runtime binary is unchanged. The independent
reference now uses owner Q/alpha with compute Q=LOCAL bytes, including when
LOCAL capacity is nonbinding. Legacy equal-progress comparisons are preserved
alongside it. The production comparison still **fails** the unchanged 3%/5%
limits; E8/E16 remain unavailable. Exact scratch lifecycle checks and the
qualified affine proof are separately saved under
`../../analysis/bounded-lifecycle-20261014/` and do not certify production.
