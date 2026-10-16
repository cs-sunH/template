# Read-only packing row storage, fourth candidate

This patch is based on the three-patch source identified in `manifest.json`.
It is not applied to production. Runtime already builds `RatioRow` coefficients,
then copies every coefficient vector into an identical `PackingRow` before
calling the nonnegative allocator. Both row types expose the same double
coefficient sequence and bound. The allocator only reads rows during the call.

The new private function template reads either row type. Its validation,
coefficient traversal, owner arithmetic, all floating operations and ordering,
rounding, final resource certification, output swapping and failure handling
are unchanged. The old `allocate_packing` signature remains as a wrapper.
Runtime directly supplies its original coefficient rows, removing their extra
vector and coefficient copies. No row, validation or numerical decision is
removed. Input rows stay alive for the whole synchronous call; no view escapes.

The test header was freshly compiled. The seven affected relink tests passed,
including the independent owner-LP Fraction oracle, split/symmetric ownership,
semantic/credit guards and 1092 complete candidate materializations. The full
two-configuration public capture, including caches and eight live traces,
equals the baseline exactly. The old numerical fork rejection remains.

This is a current-session affected-object experiment, not a final clean CMake
build or complete public51 rerun. Only Execution was replaced in the previously
identified candidate archive; every other member is byte-identical. Static and
shared identities, actual link commands and loader resolution are retained in
`../../../evidence/core-resume-20261016/perf-row-view/`.

Five instrumented diagnostic samples confirm example OS C++ new calls fall
51→49→34 at J1 and 58→52→37 at J8/J32 for baseline, three-patch and four-patch
versions respectively. The engineering-point ratio path stays 71/74/74 in the
four-patch version. Instrumented CPU times vary and do not certify a speed gain.
Both serial normal30 budgets subsequently failed, with zero added cgroup throttles and no concurrent builds/tests. Example cold128 P95 was 6.545091ms and OS J32 53.187us; engineering-point cold128 was 4.297845ms and OS J1/J8/J32 662.642/121.652/1814.584us. All samples remain in evidence. No candidate is adopted, no old failure is replaced, and no unique cause for these wall-clock tails is asserted.
