# Current A+B isolated five-repository scope

All results here use the freshly built A+B source, not the previous frozen Core
migration libraries. Formal vendors still contain canonical24/digest23160c…7e9b;
the publisher has not run. Native/E2E/CPU and overall engineering acceptance are
separate from these completed ABI and Python checks.

The isolated snapshot copied actual tracked five-repository sources. Its33-file
public SDK digest is25c51345c5ddcedb45b21c8b4dc89805c1f9ea2166d49792ddaf639ccbf91013.
Five real protoc generations preceded the4493-file freeze. The prepared manifest
and exact dependency URLs/hashes are in the public-ab evidence. No source symlink
back to an old SDK or Python time formula substitutes for the new libraries.

[ABI/Python evidence](../../evidence/runtime-continue-20261016/sdk-abi-python/)
preserves104 JSON/log files, including every command, source/library identity,
raw success/skip output and the final strict runner summary. The text archive
was roundtrip/hash verified and includes no ELF/object/library/cache.

| Repository | Real libraries built | Configuration tests | Python consumers |
|---|---:|---:|---|
| FACE | legacy+coarse |21/21 |86 passed,0 skipped |
| FACE-LRU | legacy+coarse |20/20 |80 passed,0 skipped |
| WSCLLM | legacy+coarse |23/23 |64 passed,0 skipped |
| WSCLLM-LRU | legacy+coarse |18/18 |62 passed,1 original optional skip |
| JOINT | legacy+coarse |16/16 |92 passed,0 skipped |

All10 ABI builds and98 configurations passed. All five coarse shared libraries
are byte-identical to the actual public A+B library:
`6eb9cf20e8fe2ff1630367b5bdf9e28e4edd42d9034a865f3db534a223c48663`.
Five actual typed-hardware consumer probes passed, including JOINT's real oracle
entry. FACE/FACE-LRU actual cost consumers and the FACE/FACE-LRU/JOINT availability
probe passed, retaining original full/cost/fallback behavior and R>1 guard refusal.
Those guard tests are not successful R4 business executions.

The33 runner phases are32 passed/1 failed. The failed phase is WSCLLM-LRU's
original missing optional `online.die_model_oracle` client. Its unittest process
returns0 with `OK (skipped=1)`, but the strict runner correctly marks that phase
failed. The384 successful Python cases do not turn the skip into a pass. No test,
producer input or availability check was modified to remove this gap.

New current-source native builds are running in isolated directories. Until
their actual test commands finish, do not reuse old22-pass/4-R4-failure native
evidence as if it tested A+B. No full business or CPU≤10% comparison has passed,
the public normal30 budgets remain failed, target raw data remain unavailable,
and engineering status is **NOT ACCEPTED**.
