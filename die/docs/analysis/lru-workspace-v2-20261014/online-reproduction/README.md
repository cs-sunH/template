# Reproducing the WSCLLM-LRU workspace-v2 Online run

This folder packages exact fixture bytes, scripts, schema source identity, dirty-source evidence, and build inputs. It does not contain executable binaries, raw strace data, or generated protobuf Python snapshots. See [PROJECT.md](../../../PROJECT.md), [HANDOFF.md](../../../HANDOFF.md), [workspace demand strategy](../workspace-demand-strategy.md), [liveness pressure proposal](../liveness-pressure-proposal.md), and [GQA input audit](../gqa-input-audit.md). This run uses existing MHA/GQA1 wiring; it does not validate GQA2.

## Source and build identity

The measured E2E binary came from the LRU source tree based on HEAD `e2dd5af7b61895a85b122c200606017bb741a004`, branch `codex/die-lru-workspace-v2-20261014`, with dirty source changes recorded in `source-identity/identity-summary.json`, `all-files-sha256.txt`, `tracked-diff-from-head.patch`, and `untracked-source/`. To reconstruct that historical dirty tree, check out exactly that e2 baseline, apply `tracked-diff-from-head.patch` from the workspace root, then copy each entry under `untracked-source/` to `astra-sim-wscllm-LRU/<same relative path>`. This patch is contextual to the e2 baseline only. **Do not apply it to the current source/checkpoint commit**, which already contains these source changes.

For a clean checkout at the current source commit, run `bash die/docs/analysis/lru-workspace-v2-20261014/rebuild_cpp.sh` (the root helper [`rebuild_cpp.sh`](../rebuild_cpp.sh)). It configures the aggregate build, builds Online, both C ABI libraries and all nine focused C++ test targets, then runs the nine tests. It recreates the complete focused suite; the original run's C++ artifacts and logs remain under the sibling evidence package. The recorded Online SHA-256 is `dab727d8b664d2785f86718d36874ad6295b5110618d3e86e4db9cb1c2dbc1ae`; coarse C ABI SHA-256 is `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b`.

The recorded environment was Python 3.12.14, protobuf 6.33.5, `protoc` 3.21.12, CMake 3.31.10 and C++ 14.2.0. Start Online/fixture shells with `source /workspace/cloud-setup/activate.sh` and `unset PROTOBUF_FROM_SOURCE`. The source schema is [et_def.proto](schema/chakra/et_def.proto); generated-module hashes, byte counts and protoc commands are in [SCHEMA_IDENTITIES.json](SCHEMA_IDENTITIES.json). Generated `.py` snapshots are intentionally omitted. If the binding is needed, generate it inside the LRU checkout at its ignored, source-relative `work/chakra-python-root` path:

```sh
PROTO_DIR="$PWD/astra-sim-wscllm-LRU/extern/graph_frontend/chakra/schema/protobuf"
PY_ROOT="$PWD/astra-sim-wscllm-LRU/work/chakra-python-root"
PY_OUT="$PY_ROOT/extern/graph_frontend/chakra/schema/protobuf"
mkdir -p "$PY_OUT"
protoc --proto_path="$PROTO_DIR" --python_out="$PY_OUT" "$PROTO_DIR/et_def.proto"
```

This path is under the LRU checkout (and matched by `.gitignore` `work/`); placing it in the template-root `work/` does not preserve the LRU-relative module/root lookup.

## Restore the fixture and run

Copy `work-layout/work/lru-online-workspace-20261014/e2e-large-workspace-probe/` to the same relative location under a clean repository root. It contains the official fixture CSVs byte-for-byte with their provenance, hardware/system templates, precompile specs/results, manifest/runtime config, and runner/audit scripts. The fixture has one request on a 4x2 RingTopology: P ranks 0–3 and D ranks 4–7, MHA/GQA1 (one layer, hidden 512, FFN 1,024, vocabulary 512, Q=4/KV=4/head dimension 128), prefill 64/decode 1, and 144,000,000,000 bytes/rank derived from the coarse C ABI. No capacity or workspace demand override is used.

After `rebuild_cpp.sh` passes, from the repository root run:

```sh
python3 work/lru-online-workspace-20261014/e2e-large-workspace-probe/prepare_workspace_e2e.py
python3 work/lru-online-workspace-20261014/e2e-large-workspace-probe/precompile_official_ffn.py
bash work/lru-online-workspace-20261014/e2e-large-workspace-probe/run_online_e2e.sh
```

The runner refuses to overwrite an existing `run/`; the preserve wrapper archives an existing run only when its archive path is new. The final evidence recorded three schema-v2 ACKs, 124/124 graph terminals, eight positive-workspace plans (65,536 bytes on P ranks; 1,024 bytes on D ranks), and ten balanced acquire/release operations inferred per rank. Workspace was zero at bridge-boundary snapshots, so the run does not claim a directly sampled positive live lease. `iteration` and `memory_anchor` were not emitted; base-only watermark window keys are unmapped for LRU's flat-zero/no-planner-replay watermark behavior.
