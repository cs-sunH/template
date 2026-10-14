# LRU v2 / typed metrics checkpoint evidence

The final dirty-tree inputs were based on PR #8 main
`e2dd5af7b61895a85b122c200606017bb741a004`; measured source hashes identify the
actual build/test inputs. The final Online binary SHA is
`dab727d8b664d2785f86718d36874ad6295b5110618d3e86e4db9cb1c2dbc1ae`.
The coarse and legacy CABI hashes are in `cpp/source-identity/` and Online
provenance. Old build/attempt identities remain separate, not reused as final.

`cpp/metrics-alias-fix-nine-ctest.log` records 9/9 focused tests after the final
metric compatibility fix; its build and source manifests are alongside it.
Typed rank/local_hbm tests cover zero/nonzero service, raw denominators, windows,
category isolation, cancellation and old non-coarse output. The coarse branch
retains `hbm_shared_definition` with the correct owner definition and the original
restore-issued counter, distinct from Runtime restore-served bytes.

`python/python-checks.json` records the final source's 143 online passes/1 skip
and 44 typed-hardware/config/scheduler passes. The skip names the absent optional
die_model_oracle. The actual protoc-generated Chakra binding hash and 76 Python
source file hashes are saved. Two earlier runner environment setup failures are
preserved; the final activated environment and explicit ABI paths passed.
Expected negative ledger-reconciliation test output is not a suite failure.

The official Online fixture uses 8 ranks in a 4×2 mesh with disjoint 2×2 P/D
groups, original `session_lru_tiered`, MHA Q=KV=4, one layer, hidden512/FFN1024/
vocab512, prefill64/decode1 and descriptor-derived 144GB/rank. Final C++/Python/
audit exit0; three schema2 request/response/ack exchanges, 124 successful graph
terminals, REQUEST_COMPLETE and 24 fitting full-rank capacity rows. Eight actual
FFN Plans have positive workspace: 65,536B on P0–3 and 1,024B on D4–7. Revision
changes infer ten balanced typed-Plan acquire/release pairs per rank, including
zero-workspace Plans. The bridge only samples phase-end active0; no direct
positive live-lease sample is claimed.

Legacy LRU emitted key/type checks pass after restoring the two metric fields.
Base-only watermark window bounds remain unmapped; LRU still emits a flat-zero
watermark without planner replay. This fixture emits no iteration or memory_anchor
records. These explicit coverage limits do not become fabricated records.

`online/` contains curated decoded wire/Plan/metric/journal evidence, actual
source/binary provenance, earlier failures/zero-workspace smoke and strace hashes.
Large trace files and binaries are rebuilt, not committed. The
[analysis directory](../../analysis/lru-workspace-v2-20261014/README.md) contains
the official fixture/reproduction tools and read-only pressure/GQA audits.
This is a normal-path integration checkpoint, not whole-business, pressure,
generalized GQA, production latency or CPU/RSS acceptance.

`workspace-bound/` additionally records 240 compile-only CABI attempts against
the two finite singleton manifests: 224 successful summaries and 16 expected
local_region-v1 rejections. This is selected-Plan enumeration over adapter op
kinds, not enumeration of every internal Mapping template. The separately
derived all-template workspace bound is documented in analysis and remains
unimplemented; no pressure/pre-admission certificate is implied.
