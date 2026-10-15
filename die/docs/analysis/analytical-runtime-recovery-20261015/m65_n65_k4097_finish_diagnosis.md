# 65×65×4097 solo-finish diagnostic

This is a read-only, single-Core synthetic model comparison, not a production precision certificate. The actual plan finishes at 0.219017413 s versus the grouped discrete oracle at 0.235290865 s: the analytical curve is 16.273452 ms (6.9163%) too short.

Work is not missing: planned COMPUTE_GATE work is 0.21881257268544269 s, matching the independent ledger and the runtime compute-busy integral. SRAM and LOCAL are explicitly nonbinding at 1e30 B/s, so `seal_activity`'s private max cannot explain the delta. The oracle's compute intervals contain 16.273452 ms of idle gaps after the same 0.2048 ms initial load. In the actual runtime, finish minus compute-busy is 0.204840 ms, matching that initial load plus the 40 ns final 1×1 store. The offline trace records most reference gaps while processing M/N edge tiles.

The shape has 4 tiles of 16×64, 4 of 16×1, one 1×64, and one 1×1. For a full K=64 slice, the fixture's two HBM rows each take 204.8 µs, 43.52 µs, 166.4 µs, and 5.12 µs respectively. Corresponding matrix-compute times are 828.437 µs, 12.944 µs, 51.777 µs, and 0.809 µs. The full tile is compute-bound; every M/N-tail class is load-bound. The final K=1 slice scales bytes and matrix work together and is not the source of this bottleneck switch.

In `Mapping.cc`, the first output is extracted with its whole-tile compute time (lines 917–938); the remaining 585 input units and 8 outputs share one cumulative service epoch and a single linear credit segment (lines 944–976). This preserves total work but lets the full-tile aggregate replenish credit across narrow and M/N-tail classes. The grouped oracle waits for each loaded K unit before its compute stage and exposes the resulting gaps.

A possible analytic lower-bound direction is a bounded chain over contiguous service segments: maximize load-prefix service plus compute-suffix service at segment boundaries. This does not establish a production encoding. Equal classes can interleave in row-major order, and multi-Core competition, output slots, and stores still need treatment. Fixed 5/8/16 description groupings are candidates for investigation only; no exactness or error guarantee is claimed.

Supporting machine-readable figures: [m65_n65_k4097_finish_diagnosis.json](m65_n65_k4097_finish_diagnosis.json). Inputs: `work/analytical-curves/axis/m65_n65_k4097_tm16_tn64_tk64.{raw,report}.json`.
