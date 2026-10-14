# Nested K/output affine composition (scratch result)

This verifier covers only the saved strict 1-Core OS FP32 GEMM trace at offset `1/20`, with grouped LOCAL-Q allocation and full M/N tiles. It does not certify production or general offsets.

The 520 intervals reduce to two identical 65-K output segments. Each K macro contains four event intervals. Per output, eight category templates cover K-first, 58 repeated middle-K steps, and six K-final/output boundary steps; the middle map is exact state identity. One output segment therefore needs 32 event-template steps, and the complete saved cycle needs 64 category-template steps plus an affine composition. No wave array is created.

The per-output boundary map is `x' = 849475/2156112 x + 548564055014789/8288669491200000` with fixed point `1645692165044367/15069183193600000`. The template runner checks every selected deadline tie, strictly positive event duration, all nonselected deadline margins, progress bounds, grouped LP rates/Q, and the bounded two-input/two-output lifecycle. Squaring this map exactly reproduces the archived 520-event affine map and its elapsed-time ledger. LOCAL bytes follow the saved C-stage work/Q ledger and are not treated as a fourth physical LP capacity; busy time is integrated as a positive-rate union by domain.

The exact event-order domain is `[7514576096093/71249638400000, 244491137/1235600000]`, containing the contraction orbit `[1645692165044367/15069183193600000, 317921483913953891933450891003571855541/1629703480392930105404480315562393600000]`. The smallest selected interval is `1/312500` s and the smallest nonselected margin is `14060678425108148397248290804887552497659/5453085325217179744957762253468769177600000000` s.

The earliest full M/N category boundary is wave `64935`. Starting cursors `[197,136]` allow `995` complete one-output segments (equivalent to 497 full two-output cycles plus one segment) before the tail. One more segment crosses the owner-0 tail boundary.

The repeat evaluator uses outward-rounded Q128 intervals and binary composition of a fixed 3×3 state/ledger map. It certifies the external stop after `123` complete segments; the next segment's lower elapsed-time bound exceeds the stop. The 995-segment tail-bound state interval has width `7/170141183460469231731687303715884105728`. Stored endpoints are checked at 256 bits; multiplication scratch is checked at 512 bits.

Remaining scope: the event templates were selected from the archived trace and this does not prove a bounded event-order discovery algorithm. The result is limited to offset `1/20`, one Core, full M/N, and the listed three-resource LP. Other offsets, tails, multi-Core bases, WR fences, cancellation, and capacity changes need separate maps. A stop inside a segment leaves a residual for the ordinary event integrator.

Reproduce with:

```sh
python3 /workspace/template/work/affine-compose/nested_k_output.py
```

The script writes `nested-k-output.json`, `nested-k-output.affine`, and this report beside itself. It reads archived evidence without modifying it.
