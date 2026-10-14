# Three-variant capacity API publication check

The three scoped repositories each verified all 23 release-manifest file hashes
and recomputed source digest `9d025e25699e44b708997da5e4bc9c45e60c75b02166be931282f20921c0c4f4`.
Each was configured in an isolated `work/five-capacity-source/` build tree from
`work/integration-aggregate` and built only `coarse_die_cabi` and
`die_model_cabi`. The existing config-resolver suites passed 55/55 with no
skips. Both libraries were byte-identical across the three variants.

`die_coarse_workspace_capacity_bound_v2` was dynamically loaded and called from
each newly built library. The same Attention fixture returned the versioned
136-byte bound and matching hardware/task fingerprints. It returned no
executable-mapping claim.

The byte-identical coarse C ABI library was also used for the historical
62-task identity replay: all 62 Plan hashes and full summaries matched. The
224 captured legacy envelope responses and 16 expected rejections matched
exactly. Detailed counts, commands, and source/library identities are in
[`three-variants.json`](three-variants.json).

Reproduce the configured ABI build from the repository root with:

```bash
source work/activate.sh
cmake -S work/integration-aggregate -B work/five-capacity-source/wscllm \
  -DASTRA_SIM_ROOT="$PWD/astra-sim-wscllm" \
  -DBUILDTARGET=congestion_aware \
  -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build work/five-capacity-source/wscllm --parallel 2 \
  --target coarse_die_cabi die_model_cabi
```

For FACE and FACE-LRU, use the corresponding repository root and isolated build
directory recorded in the JSON. The explicit config suite command is
`python3 -m unittest discover -s <repo>/sh_test_mesh/tests -p 'test_config_resolver*.py' -v`
with `DIE_COARSE_CABI_PATH` and `DIE_MODEL_CABI_PATH` set to that build's two
libraries. The historical replay scripts and their inputs are referenced in
the workspace-capacity and workspace-envelope analysis records; no binaries or
large captured payloads were copied here.
