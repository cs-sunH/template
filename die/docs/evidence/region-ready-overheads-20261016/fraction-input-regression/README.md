# Fractional ready-delay input regression

This directory records a correctness probe, not a performance benchmark.

The frozen public library SHA256 is `aa3e68b2a8d871cf63f12500e6391e49f8c7b2ad60d2d2163e611ac6a5bc602a`.
The rerun stage is **after region-overhead audit deferral and owner-key deferral, before the fractional-input fix**, on uncommitted source above Git HEAD `f4d576bc4c066b7afdae6699bc377e9a0ff628a7`.
This is a pre-fix baseline for this specific regression. It is not the clean `f4d576b` main library and is not attributed to the original earlier probe's library, whose SHA was not captured at that earlier invocation.

`result.json` records the library identity, exact dyadic expected values, observed source hashes, Git status and parsed responses. Individual `*.raw.json` files preserve actual ABI output; `stderr.txt` is the probe's captured stderr. `observed-source/` and `observed-source.patch` preserve the source state inspected around this rerun. The source hashes and frozen library remained unchanged during the probe; the probe did not rebuild or independently certify the library's build provenance.

All three hardware configurations resolve successfully. The controls `0.5 + 0 ns` and `0.25 + 0.5 ns` compile. The valid input `0.5 + 2^-65 ns`, with the same exact integer ready delay `ceil(...) = 1 ns`, fails compilation with `DIE_OVERFLOW` / `equivalent work fractional sum is not exactly representable`. This failure comes from exact fractional audit arithmetic, rather than uint64 ready-time overflow.

To reproduce against a selected library without overwriting this captured result:

```bash
python work/region-ready/fraction-input-regression/reproduce.py \
  --repo /workspace/template \
  --library /workspace/template/work/region-ready/fraction-input-regression/library-under-test.so \
  --output /workspace/template/work/region-ready/fraction-input-regression-rerun \
  --stage pre_fraction_fix_frozen_library_rerun
```

The local ELF copy is a scratch preservation aid and should not be committed as a GitHub deliverable. The script, configs, raw JSON and source identities are portable evidence.

Committed evidence excludes the local ELF and duplicated observed-source tree. Their original SHA/paths stay in raw manifest; rebuild frozen f4d576b plus observed-source.patch in an isolated tree. The replay script recreates local copies when explicitly run.
