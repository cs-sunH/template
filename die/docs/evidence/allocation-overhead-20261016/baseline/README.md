# Allocation-overhead frozen public C ABI baseline

Fresh Release source was archived from `eba9e972e0514dddb4913dace1c00c57c3f63076`, independently built as `coarse_die_cabi`, and passed to the corrected `capture_public_results.py`. Configure, build and capture all returned zero; build took 37.89 seconds. This is a new actual capture, not recovered output from the initial failed script.

`public-before.json` contains all eight task inputs and complete hardware/compile/estimate public results. `source-identity.json` binds the complete archived source, script, Python wrapper, configuration inputs, library/static-library/compile-command SHAs, ELF build ID and actual dependencies. Script/wrapper/config files remained identical to their recorded baseline. `commands.json` and original logs preserve every phase. The observed library SHA in the output exactly matches the independently built artifact. No performance benchmark or old oracle ran.

The parent reported that the first capture attempt destroyed its model while compile handles were still live, causing status9; it did not write before data. That failure remains explicitly recorded in the identity and summary. This successful run used the corrected finally block to release each compile plan handle before model destruction. No initial before artifact was reused or invented.

For a strict after comparison, compare the complete `results` object; the outer `library_sha256` intentionally identifies distinct compiled libraries. No other output field is designated for exclusion. The main agent owns that after capture/comparison; this directory records only the frozen before side.

The exact fresh archive/build/capture procedure is saved in `run_baseline.py`, with `/workspace/template` paths and a refusal to reuse an existing source directory. Use a fresh task work directory for replay and preserve new results separately. This evidence does not certify hardware calibration, full-workload behavior, or speed budgets.
