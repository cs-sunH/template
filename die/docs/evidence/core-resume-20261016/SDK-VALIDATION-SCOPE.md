# Isolated five-repository migration scope

The frozen candidate built all 10 coarse/legacy shared libraries and passed all 98 configuration tests. This is partial integration validation, not engineering acceptance or an SDK release. Formal vendor SDKs remain unchanged canonical24. Exact source digests, raw result paths and distinct failure/correction identities are in sdk-validation-scope.json.

| Repository | Config | Actual Python consumers | Native C++ |
| --- | --- | --- | --- |
| FACE | 21/21 | Frozen 85/86; independent test-fixture correction 86/86 | 3/4; unchanged R4 lifecycle guard failure |
| FACE-LRU | 20/20 | 80/80 | 3/4; unchanged R4 lifecycle guard failure |
| WSCLLM | 23/23 | 64/64 | 3/4; unchanged R4 lifecycle guard failure |
| WSCLLM-LRU | 18/18 | 62 pass + 1 original optional skip | 3/4; unchanged R4 lifecycle guard failure |
| JOINT | 16/16 | 92/92 | 10/10 plus native Python 6/6 |

JOINT's selected current isolated suites contain 124 test cases: config16 + actual Python92 + native C++10 + native Python6. Its native C++ names and six Python names are preserved in native-resumed/ logs. ABI builds and separate hardware probes are not test-case counts. This does not establish full business or CPU-budget acceptance.

Four native lifecycle suites genuinely fail on the original R4 emitted FFN path. The body-only not_available guard remains effective; original fixtures were not shortened to R1. WS/WSLRU complete earlier lifecycle subcases, then fail in original Case O; FACE/FACE-LRU fail in original Case J. Later assertions are not passed. Native C++ totals are 26 cases:22 pass,4 fail,0 skip. Native resume retained FACE/FACE-LRU terminal outcomes instead of rerunning them; the interruption and original partial WS build log remain in native-interrupted/.

FACE's original frozen Python failure is an obsolete hardware numeric pin (6197 expected vs6192 actual). An independently identified copy changes only that test fixture and passes86/86; the seven-component real ABI decomposition is separately saved as face-legacy-pin-decomposition.json. This correction does not replace or erase the original failed result, frozen source or native R4 failures.

WSLRU's optional Python oracle module remains absent and its62+1skip suite still fails the runner's zero-skip criterion. The separate wslru-producer/ bridge uses the real existing WSLRU producer and existing FACE ctypes client against the fresh WSLRU libraries: all original five R4 operations pass the legacy protocol roundtrip; separate valid coarse R1 controls pass; all unchanged coarse R4 payloads return not_available and are rejected by full/cost/estimate_one. The bridge neither adds a production client nor claims the skipped original test passed.

The historical generic JOINT typed probe failed because it entered the legacy authoring resolver. joint-actual-hardware-consumer.json independently records a successful query through the real existing online.die_model_oracle.DieModelOracle. The historical failed probe remains in sdk-abi/.

These SDK records freeze public release source digest2e3e7e2628b7b02785328c0fb212d9332121c265d8a2f1c6fc4f06b2f4ff30f5, snapshot manifest3cad402edce1ebfc9050feb7fcf835b3bbcd9ee900492ee53067f4d5070ea9b4, and coarse library8030d49d518f84f7657882a3e888ab70504148a263268cd543f0ed53bb5a3de6. Later public performance/numerical changes have a distinct identity and are not accepted by these builds. Public normal30 performance failures, full business integration, five end-to-end CPU budgets and independent target-hardware measured accuracy remain outside this acceptance scope.
