# OS scratch-kernel recovery evidence

`identities.json` records source, binary, facts, test, profile, and benchmark
hashes plus tool availability. Subdirectories preserve the reconstructed
baseline, post-profiler/reset full-suite and sanitizer results, pre/post
fixed-width operation profiles, the primitive differential test, and the n30
benchmark data. The n30 record overlapped concurrent C++ builds and remains a
contaminated diagnostic, not an isolated performance measurement.

The corresponding analysis is
[`REPORT.md`](../../analysis/os-kernel-recovery-20261014/REPORT.md). The fixed
width variant source and standalone test are also copied under
`die/docs/analysis/os-kernel-recovery-20261014/fixedwidth_variant/` for review.

The public-clock API boundary and retained exploratory deviations are in [`current/public_clock_contract.md`](current/public_clock_contract.md); per-source/fact hashes and the seven-scenario comparison summary are in [`current/public_clock_manifest.json`](current/public_clock_manifest.json).
