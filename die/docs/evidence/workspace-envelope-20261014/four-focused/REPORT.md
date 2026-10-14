# Four-repository focused rebuild against canonical23

This run rebuilt the base, FACE, FACE-LRU, and WSL-LRU Online target and requested C++ focused tests, in that order. All evidence is saved in this directory. It did not rebuild either ABI target or run a production Online workload.

The canonical 23-file sync check passed before and after every variant with source digest `e0b6fd1a083ebe89cc73956f84f07524e6c3944fbcd4cc28a91a947d66cbbf78`. Per-variant coarse source and C++/CMake build-input manifests match byte for byte across each build. Each variant's `libcoarse_die_cabi.so` hash is `7e9b9670b8c06c8600688acf6446acc5f112a97e83315a26ea8d554040cc5600`; ABI binaries were hashed from their existing root-built outputs, not rebuilt here.

| Variant | Focused CTest | Online SHA-256 | Coarse C ABI SHA-256 |
|---|---:|---|---|
| `astra-sim-wscllm` | 3/3 | `112487c8011621c75cb610a557098560614643261c59546d3fed19083787c004` | `7e9b9670b8c06c8600688acf6446acc5f112a97e83315a26ea8d554040cc5600` |
| `astra-sim-face` | 3/3 | `2e228e8ca9592bb71697d28efda51f53092395d9ee52edd987f7260a5a89b42d` | `7e9b9670b8c06c8600688acf6446acc5f112a97e83315a26ea8d554040cc5600` |
| `astra-sim-face-LRU` | 3/3 | `fab7758e8b1013c1537df91ed2b71e0350189e92b209f72116cbb1b212c63274` | `7e9b9670b8c06c8600688acf6446acc5f112a97e83315a26ea8d554040cc5600` |
| `astra-sim-wscllm-LRU` | 9/9 | `9292443d927a03d942177739d957c6496f6f1ddb8ee9963763fb009eed09dba0` | `7e9b9670b8c06c8600688acf6446acc5f112a97e83315a26ea8d554040cc5600` |

The three non-LRU variants selected `DieModelAdapterTest`, `DieExecutionLifecycleTest`, and `LocalHbmDieMemoryTest`. WSL-LRU selected the nine tests recorded in `commands.txt` and `results.json`, including `BridgeLoopbackTest` and the two-Plan lifecycle test. Total focused result: 18/18. All four Online targets built successfully.

During the WSL-LRU build, the parallel Python workspace audit changed `session_kv_manager.py` and `test_workspace_admission_manager.py`. This changed the repository-wide tracked diff; those files are outside the C++/CMake inputs used by these targets. The recorded C++/CMake source manifests and canonical coarse source manifest remained unchanged. The evidence does not claim the full worktree was frozen.

The new JOINT model has no business-focused C++ integration test in this set and is reported as `not_yet_ported`. No JOINT Online target or JOINT business focused tests were run.

Exact commands are in `commands.txt`; build and CTest logs are in `logs/`; target and source SHA-256 details are in `results.json` and `source-identity/`.

The complete 25-file coarse subtree manifests are identical across the four copied trees. Tool versions are in environment.log.
The Online-linked coarse static library hashes are: e44a0d43a99a739cde8a49fba2df12590e01662c525c7193b62132bfde7537e0.
All four captured Online link commands reference the coarse static library; exact commands and hashes are in source-identity/.
