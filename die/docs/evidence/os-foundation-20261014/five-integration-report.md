# Five-repository canonical23 build and configuration verification

All outputs for this verification are under `work/five-os-20261014/`. The final repository sync check passed for 23 files with source digest `9fb747016e69200da9ef461bc4e0a9d00343b746a4d70081e3c489401e613a47` (`logs/source-sync-check-final.log`).

Configured the four non-base variants in independent `work/five-os-20261014/build/<repo>` directories. The base `astra-sim-wscllm` reused the existing aggregate build under `work/online-baseline-20261014/integration-build/build/astra-sim-wscllm`. All builds used CMake Release and `--parallel 2`. `PROTOBUF_FROM_SOURCE` was unset, so CMake selected system protobuf 3.21.12 from `/workspace/cloud-setup/sysroot/usr`; Python protobuf remained 6.33.5 (Python 3.12.14).

Both `coarse_die_cabi` and `die_model_cabi` built in all five variants. The explicit `test_config_resolver*.py` suites passed 83/83 with no skips. Adapter/Lifecycle/HBM focused targets passed 16/16 CTest cases: base 3, WSL-LRU 4, FACE 3, FACE-LRU 3, JOINT 3. The selected focused set is the 16-test set documented by the existing final verification recipe. Per this task scope, no full five-variant Online business E2E was rerun.

| Variant | Config tests | Focused CTest | `libcoarse_die_cabi.so` SHA-256 | `libdie_model_cabi.so` SHA-256 |
|---|---:|---:|---|---|
| astra-sim-wscllm | 20/20 | 3/3 | `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b` | `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68` |
| astra-sim-wscllm-LRU | 15/15 | 4/4 | `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b` | `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68` |
| astra-sim-face | 18/18 | 3/3 | `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b` | `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68` |
| astra-sim-face-LRU | 17/17 | 3/3 | `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b` | `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68` |
| astra-sim-joint | 13/13 | 3/3 | `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b` | `4f294213d775637abedb0cd9ea2bc579a6450c48d35a5712e05fab4a5c26e477` |

The detailed focused test executable SHA-256 values, exact build/test commands, and per-run logs are in `focused-results.json`, `abi-config-results.json`, and `logs/`. CABI output hashes are also recorded in the ABI results JSON. The older WSL-LRU Adapter/CMake/source inputs were hashed before the parallel LRU C++ v2 edits at `source-identity/astra-sim-wscllm-LRU-focused-inputs.sha256`; the 4/4 focused test and binary identity above belong to that captured pre-v2 version. Those six pre-v2 binaries are also archived under `source-identity/lru-pre-v2-binaries/` with hashes in `source-identity/lru-pre-v2-binary-sha256.txt`.

Environment versions and the full command recipe are in `environment.txt` and `commands.txt`. The verification scripts and evidence did not modify tracked files.
