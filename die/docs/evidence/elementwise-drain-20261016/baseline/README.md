# Frozen a0d baseline — nine additional elementwise cases

This directory contains a fresh independent Release build from **`a0d327aeeb4a966faab56b426ed88d61e5607478`**, followed by only the current nine-case `run_extra_cases.py` suite. No old 17-case suite or performance benchmark was rerun. Nothing in production source was modified.

The balanced 511/512 completion errors **already exist at a0d** and have exactly the same actual/reference times in the separately recorded phase3 results:

| Case, solo | a0d actual / reference | a0d error | Phase3 error |
|---|---:|---:|---:|
| balanced 511 | 49.921876 / 39.9609375 ms | **24.926689% failed** | **24.926689% failed** |
| balanced 512 | 50 / 40 ms | **25% failed** | **25% failed** |

Phase3 removes these cases' solo compute-busy overstatement: a0d errors **50.097847% / 50%**, phase3 approximately zero. This is a busy improvement, not a completion-error fix or full model acceptance.

The full comparison also exposes two phase3 regressions in offset **shared HBM busy**, which must remain failed:

| Case, offset | a0d relative error | Phase3 relative error |
|---|---:|---:|
| compute 511 | 14.427507%, within20 but unaccepted | **25.428048%, failed** |
| compute 512 | 14.406780%, within20 but unaccepted | **25.423729%, failed** |

For compute 511, shared-HBM reference is 39.193939 ms; actual changes from 44.848647 to 49.160193 ms. For compute 512, reference is 39.333333 ms; actual changes from 45 to 49.333333 ms. These new failures concern shared-busy timing, not completion. Solo compute-profile completion error also increases from near zero to **1.840838% / 4.536560% / 4.545455%** for 321/511/512, still within the 10% completion target. All nine baseline ledgers pass, and every compared reference record is exactly equal between baseline and phase3. No threshold or busy definition was changed.

`comparison.json` preserves all nine cases × solo/sync/offset × both reference semantics, including completion, first-output and original busy metrics. Phase3 data was read from the existing sibling `extra/`, not rerun here; its source identity and each report hash are bound in the comparison. Process exits are all zero, but the JSON model-comparison failures above are real and must not be inferred as passing from exit codes.

## Source and build identity

`source-identity.json` contains every archived source/test/config/validation/helper file SHA, archive SHA, driver source SHA, compiler/CMake versions, ELF build ID, static-library/driver/compile-command identities, and scope. The current Python oracle is byte-for-byte equal to the a0d archived oracle. The nine-case runner and oracle snapshots are preserved alongside this README.

- Driver ELF SHA256: `5e9acd5b4ea7e7654255baa34b3972e1d2c3f714435ebf81c3294d9af6b4576f`
- `libcoarse_die.a` SHA256: `fc09d79c0428b5323656faf6e986e47b362a662c9772359c180ab6e0c626f538`
- Fresh configure: exit0; driver Release build: exit0, 33.96 s; nine-case runner: exit0, 0.65 s. These are execution logs, not speed-budget measurements.

All original driver JSON, stderr, reports and report logs are retained under `extra/`; exact top-level/per-case commands and exits are in `commands.json` and `extra/suite.json`. `files.sha256.json` binds this evidence package.

## Reproduction

The exact original archive/build/run procedure is saved as `run_baseline.py`; it assumes `/workspace/template` and refuses an already existing source directory. Activate `/workspace/cloud-setup/activate.sh`, use a fresh assigned work directory, and retain its new logs separately.

To replay the saved case runner independently of future edits to the repository's validation tools, put its saved snapshot at `SOURCE_ROOT/die/docs/analysis/elementwise-drain-20261016/run_extra_cases.py` inside a fresh archived a0d source tree. Its relative-root calculation then selects that tree's `die/validation/elementwise_service_oracle.py`, whose hash is verified equal here. Point `--driver` at the isolated Release `coarse_elementwise_oracle_driver` and `--output` at a fresh directory; do not overwrite these original raw results.

This is small synthetic model-to-model evidence. It is not TX8/reference calibration, independent target-hardware measurement, five-repository integration, or a proof that phase3 has completed accuracy acceptance.
