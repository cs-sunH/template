# Old typed task compatibility through real parsers

Measured source base: `8447b76f68f5899fdf8d06bde103efb30cc0fc51` plus the four source files identified in the per-variant hashes. Date: 2026-10-15 Asia/Shanghai (raw build timestamps retain UTC).

WSCLLM and WSCLLM-LRU preserve both the original schema-1 `die_op_spec` JSON and its parsed legacy diagnostic object. Their coarse compiler accepts that original task; the prepared-Plan preflight incorrectly rejected it solely because the diagnostic was present. The fix removes that predicate while retaining raw-task/Plan/hardware/bytes/ledger-owner checks.

Each actual parser regression retains the FACE emitter FFN task verbatim, adapts only GraphBatch rank/id for a one-rank fixture, and executes prepare → acquire → Workload issue → canonical Runtime finish. It asserts both parsed pointers, the exact prepared/executing Plan fingerprint, 256 workspace bytes, one successful terminal, and released GPU/lease/Runtime ownership. Existing negative owner/task/bytes and blocked/retry/cancel cases remain in the same lifecycle target. Each target built and passed CTest 1/1 using its separate existing aggregate build.

JOINT is deliberately unchanged: its parser requires the new `die_task` wire separately from legacy `die_op_spec`, and Workload rejects legacy normalization for that profile. The WS compatibility requirement does not expand that independent boundary. FACE uses the old typed producer too; its pending bridge work is separate and is not certified by these tests.

No canonical model, calibration, capacity policy or target-accuracy claim changes. Commands, logs and measured source/binary hashes are adjacent. Rebuild the existing three-subdirectory AstraSim/backend/frontend aggregate with `ASTRA_SIM_ROOT` for the matching variant, `BUILDTARGET=congestion_aware`, and `NETWORK_BACKEND_BUILD_AS_LIBRARY=ON`; the command files record the actual local build directories/tool activation.
