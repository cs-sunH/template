# 80×128×4096 conditional busy bound

**Latest policy (2026-10-15):** the user replaced the former 3%/5% limits with a 10% target and a documented 20% fallback. The J=5/8/16 lower bounds below still exceed both. The bound stops forcing failure at J=232 for 10%, and J=224 for 20%; these counts remain necessary only for this conditional bound to become non-decisive, not sufficient for acceptance. The original 5% calculation is retained below as historical analysis.

This is a solo, single-Core synthetic v3 diagnostic. It does not change the busy metric or acceptance threshold and does not prove that all finite curve representations fail.

The actual Plan has ten identical 16×64 output tiles and 64 full-K consumptions per tile: `N=640` input units, input-credit capacity `B=2`, and a 7-description Plan under budget 8. Both Compute activities have one linear `C_in` segment. Their `C_in` slopes and true `COMPUTE_GATE` work give the same maximum physical consumption rate:

```text
first-output: 64 / 0.053019972695555445 = 1207.09228515625 units/s
cumulative:  576 / 0.477179754259999   = 1207.09228515625 units/s
```

Each input unit is 20,480 read bytes (10,240 bytes on each HBM row) and 131,072 Vec-equivalent operations. At the fixture rates, that is 204.8 µs of HBM service per row versus 828.437 µs of Compute-gate work per unit; all ten tiles have the same ratio. The reference busy reconstructs as `(6,553,600 read bytes + 20,480 write bytes) / 50,000,000 B/s = 0.1314816 s` per row.

The load partition is exact: first-K epoch supplies 1 unit; first-output epoch starts with that carried 1, loads 63, and consumes 64; cumulative epoch starts and ends at zero, loads 576, and consumes 576; final drain has no input consumption. Thus 640 credits are loaded and consumed exactly once. The only nonzero compute-epoch initial credit matches the prior endpoint, and the next reset is 0→0: no extra `B` is needed for this Plan. For arbitrary epoch resets that inject unbacked input credit, the general bound must subtract that injection; a safe coarse correction is `R*B` for `R` such reset boundaries.

Allowing compute stalls or a variable consumption rate bounded by `c_max`, the no-load gaps can consume at most `B` units apiece. With at most `J` positive HBM-load on-runs and no unbacked reset, at most `(J+1)*B` units can be consumed without load service. Since the remaining input units are consumed at no more than `c_max`,

```text
HBM-positive load busy >= max(0, (N - (J+1)B) / c_max).
```

Against the independent solo reference busy of 0.131481600 s, the lower bounds are 0.520258482 s for J=5, 0.515287860 s for J=8, and 0.502032866 s for J=16 (3.96×, 3.92×, and 3.82× reference). Even charging one extra full `B` at an internal boundary leaves the J=16 bound at 0.500375992 s. The measured actual busy is 0.528788613 s, consistent with these lower bounds.

This bound stops forcing a >5% mismatch only once J reaches 236; it falls to the reference itself at J=240. Those are thresholds for this bound to become non-decisive, not sufficient run counts for acceptance. The seven serialized descriptions do not prove `J<=7`: J must count every actual connected positive-load-service run, including hidden allocator/tangent switches. Under a binary transition model, J=236 requires at least 470 on/off transitions in the most favorable endpoint pattern. Concurrent arrivals, offsets, and other owner events are outside this solo calculation and must be included if they split load service.

The measured Plan itself already fails the standalone busy check: 0.528788613 s versus 0.131481600 s (about 302.18% relative error). The conditional result says a candidate with a proved `J<=16` and these reset assumptions cannot meet the unchanged 5% target. The actual 7-description count does not prove `J<=7`, so without a hidden-switch bound the design conclusion remains `tradeoff_unresolved`; this is not an impossibility result for all finite curves and does not motivate changing acceptance. Full data and calculations are in [m80_n128_k4096_busy_bound.json](m80_n128_k4096_busy_bound.json).
