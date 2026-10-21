# P0: K-wave readiness, busy union, and slot retirement audit

Read-only audit at base `f3b53eb906df788eecf613642106aa974d5d24a7`. Read `AGENTS.md`, `docs/HANDOFF.md`, root design §§8.1–8.6 and §§13.1–13.3, the requested source, and the already saved compiled-oracle report. I did not rerun historical tests or change production/documentation files.

## Finding

The current E5 macro is structurally too coarse for an unbounded repeated K-wave stream. This is stronger than saying E5's current fixture fails: the specific saved E5 result fails the busy gate, while E8/E16 are not implemented and remain not_run; no result supports saying either hypothetical E8 or E16 passes or fails that one fixture. But no fixed count of ordinary affine epochs can certify arbitrary repeat counts while retaining the finite-buffer wave ordering and a repeat-independent event cap.

`append_epoch` receives output rounds, not K consumption waves (`Mapping.cc:860–864`). For R>2 it divides aggregate L/C/S flow by R and emits the five cases at lines 712–717. In the saved fixture R=10, Tk=65: the plan reports 650 consumption waves but the epoch credit repeats only 10 output-round units. Each macro compute flow therefore covers the entire K sequence for its output-round share. Its interior credit uses R−2 aggregate input and output units (`push_epoch`, lines 680–691); it does not encode the 65 per-output K readiness/retirement transitions.

At a full input-credit boundary the allocator adds the signed tangent constraint `dL−dC <= 0` for equal input-unit deltas (`Execution.cc:1064–1089`). Under the same-owner max-min allocation, active L and C receive the same normalized progress scale (`Execution.cc:1130–1144, 1180–1195`); when C is compute-bound, a positive L is consequently spread across the C interval. `integrate` records busy as a boolean union: if any positive-rate activity has positive HBM work, it adds the whole interval to HBM busy (`Execution.cc:1369–1388`). It does not weight busy by utilization. This explains how a small byte stream can keep HBM marked busy across most of a long compute macro.

The saved compiled-oracle evidence gives concrete scale: single request actual HBM busy 0.1681156809 s versus oracle 0.0747344 s (relative error 1.2495087, i.e. 124.95%); compute busy is 0.2188125727 s. At 50 ms offset the HBM/shared busy errors are 158.87%/478.94%. The oracle completed 650 loads and 650 computes plus 10 stores, with 1311 offline events for one request. This is model-to-model evidence, not device data. The report correctly marks E8/E16 unavailable/not_run.

## Small counterexample and bound

Take N repeated K consumptions with two input slots. Let each load occupy HBM for l=B/H seconds at peak row rate H, and each compute take c seconds, with c much greater than l. In the steady discrete schedule, while C_i runs for c, the next L can complete in l; after the second slot is occupied, loading stops until C_i releases its input at completion. Thus each cycle has an HBM burst of length l and a gap about c−l. Ignoring O(1) startup/drain and small final stores, oracle busy is N*l while execution duration/compute is N*c.

For a single affine repeated segment, once input credit is full, L and C stay tangent. If both remain active, runtime HBM busy is approximately N*c, so relative busy error is approximately (c−l)/l. Choosing c=20*l makes that about 1900%, despite exact total bytes and operations. Splitting the same affine aggregate into any fixed E=5/8/16 still has only O(E) opportunities to stop/restart the load stream. A burst can get at most the two-slot lead before compute must drain an input; batching more waves violates slot/readiness semantics. Either L remains positive through compute and busy is inflated, or L is withheld and compute stalls after the bounded prefetch, damaging finish time. The required O(N) burst/gap transitions cannot be encoded by a fixed number of ordinary affine epochs.

This is a class-level impossibility claim for fixed E independent of N, not a claim about the unimplemented E8/E16 outcome for the saved 650-wave fixture. A few affine segments could be tuned to match one busy scalar while abandoning the actual readiness/slot schedule; that is not a valid certificate under §§8.1–8.6.

## Readiness and retirement rules

The independent oracle reserves an input slot for an active load, keeps it occupied while its exact loaded wave is ready or computing, and frees it only when that compute completes (`compiled_gemm_oracle.py:425–452`). Compute can start only from the loaded queue. A non-final K compute has no output/store result. For final K, an output slot is reserved before that compute starts; completion makes the output store-ready; store completion retires it and is the only point that frees the slot (`435–457`, `522–540`, `562–567`). The reservation covers the in-progress final compute, ready output, and active store. This must remain distinct from output production: production occurs only when final-K compute completes. Current aggregate output credit rises continuously with the whole macro C progress, while aggregate S spreads writes across the round; neither is the oracle staircase. For WR, preserve the design's column-group fence: drain old computation/writes, retire slots, release resident weight, then begin the next group.

## Bounded representation that fits the design

Use finite run-length categories, not one descriptor per tile: for example output-tile M/N full/tail classes, K full non-final repeat, K tail or final-full atom, final-output marker, and WR group-boundary marker. Each class carries exact per-consumption L/C/S work and bytes; class counts come from the existing closed-form counters. Keep exact row partition and work totals; do not multiply audit-only repeat_count twice.

Attach a repeated-period curve to each homogeneous class. Within one period store a bounded number of service phases and per-resource active/owner masks, plus the finite-state slot transitions. Input readiness changes only at exact load completion and compute consumption; final output reservation starts at final-K compute admission, production at its completion, and retirement at store completion. A category repeat count denotes repeated periods; it does not emit one Runtime event per repetition.

During stable capacity/owner membership, solve a period's service and slot-state transfer once, then advance whole periods in bulk: add exact per-period resource work, multiply the per-period busy-union measure, update slot counts/phase, and handle only the partial prefix/suffix around an external arrival or capacity change. For multiple owners, retain per-row owner-active masks and compute the union/shared intersection over repeated phase curves, including phase offsets. Keep a bounded category/phase/event cap; if a combined pattern cannot be integrated within it, fail closed as granularity unavailable/failed rather than expanding per wave. This can keep internal events O(categories × phases × external changes), independent of R*Tk, and matches §8.2's fixed piecewise-curve requirement and §13's event budget.

## Design and code references

- Root design §6.2 distinguishes output_round_count from consumption_wave_count=R*Tk and says OS compute follows per-core K order; its final output writes only at final K. WR also requires a column-group fence.
- §§8.1–8.2 require independent L/C/S activity progress, exact finite-buffer readiness/retirement curves, periodic categories, and no per-repeat event expansion. §8.6 says E5/8/16 are candidates to validate, not correctness premises.
- §§13.1–13.2 require bounded descriptors/events independent of tile count. E8/E16 must stay not_run until implemented and measured.
- `validation/compiled_gemm_oracle.py:425–457, 522–567` is the discrete readiness/retirement reference; lines 597–628 preserve the 5% busy/shared-busy gates.
