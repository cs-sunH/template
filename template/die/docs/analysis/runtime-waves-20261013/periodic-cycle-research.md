# Bounded exact cycle integration for two K-full owners

This note is scratch research for the P0 runtime work. It only changes files under `/workspace/template/work/audit-busy/`.

## Scope and exactness conditions

The scheduler is modeled as a deterministic hybrid state machine while capacity, fairness policy, owner membership, and the current category run are fixed. Each owner has at most three live primitive activities (load, compute, store). A wave-stage transition is atomic at completion; resource work and progress remain continuous while an activity is live. Input/output readiness and slot reservation are discrete state, not fractional credit:

- A load reserves an input slot at admission, becomes ready at load completion, and the slot remains occupied while its exact wave computes. Compute can only take a ready input; compute completion releases the input slot.
- A final-K compute reserves an output slot at admission. Its completion changes the slot from reserved to store-ready. Store may start only then; the slot is retired and freed only when store completes. Non-final K has no output/store transition.
- Full-K repetition is one run-length class. K-final/tail, M/N-tail, and WR column-group fences are fixed transition classes. Do not create a state/event per repeated wave.

For the repeat proof, use exact state values. Integer bytes/ops are exact; each accepted finite binary64 capacity/work coefficient can be converted to its exact binary rational. Solve the active-set/max-min LP with a deterministic tie rule and an exact rational basis, or independently certify that the production solver selected the same unique basis and exact rates. A degeneracy/tolerance boundary, unsupported coefficient, or unbounded rational growth is not a near-match: it is uncertified and must fail closed. Event times, progress, credit/slot transitions, and integrated counters for the proof path stay rational until conversion to the public numeric format.

The proof applies only before the next exogenous event and before the next category boundary. A fixed-capacity, fixed-owner window is autonomous: its next activity rates and stage transitions depend only on the current state. Replaying an exactly repeated normalized state therefore replays the same trajectory and the same physical positive-rate masks.

## State key

At each settled event boundary, after all zero-time stage completions/readiness transitions and deterministic allocation, make this exact key:

```text
K = (
  plan/pattern fingerprint,
  capacity-generation and exact capacity vector,
  fair-owner policy and deterministic LP/tie-break version,
  sorted active owner identities/generations,
  for each owner in that order:
    category-run id in [0,7],
    phase within the bounded descriptor word (at most 8 word positions),
    up to 3 live (activity id, stage, exact progress, positive-rate status),
    input FIFO category tags and reserved/ready counts,
    output FIFO category tags and reserved/ready/store-active counts,
    exact signed-credit/slot state,
  solved LP basis / exact rate-signature,
  active physical-resource and per-row owner masks
)
```

The key omits absolute time, absolute repeated-wave ordinals, accumulated served work/busy durations, and remaining repeat counts. In particular, do not key on K index modulo a potentially huge Tk: K-full non-final is a homogeneous run, and K-final/tail is a separate bounded category transition. The run's remaining count limits a skip but is not part of the repeating state. For equal-cost homogeneous waves, slot contents need category tags and FIFO order, not tile IDs. Include all state that can change future eligibility, service, or positive-rate masks. The period counter is kept separately as an additive category-count vector. If any omitted field can affect future transition behavior, it belongs in the key or in the skip bound/transition rule.

Use exact rational equality for the key. A rounded hash may be used as an index, but a hit must be followed by exact fieldwise equality. Do not merge two states because their progress differs by epsilon.

## Exact period integral and skip bound

For each ordinary event interval, after solving rates, form the real active mask from activities whose rate is strictly positive and whose physical work coefficient for that resource is positive:

```text
M_r = { owner | exists live activity a:
        rate[a] > 0 and work[a,r] > 0 }

busy_union[r]  += dt iff |M_r| > 0
busy_shared[r] += dt iff |M_r| >= 2
```

This is the same positive-rate union definition as `Execution::integrate`; it does not substitute bytes/capacity or a duty-factor approximation. For the public two-row HBM counters, preserve the implementation's unions exactly: `row_busy[row]` and `row_shared[row]` use that row's owner mask; global `hbm_busy` adds `dt` if either row mask is nonempty; global `hbm_shared` adds `dt` if either row has at least two owners. Do not sum the two row durations into the global union. Also integrate exact resource work, HBM bytes by row/direction, Vec/INT4 counters, and per-category/per-owner completed-wave deltas.

When the exact key at `(t1, stats1, ordinals1)` is seen again at `(t2, stats2, ordinals2)`, require `T=t2−t1>0`. The observed loop deltas are:

```text
T       = t2 − t1
Δstats  = stats2 − stats1
Δcount  = ordinals2 − ordinals1
```

Because the normalized state and static environment are identical, each full replay adds exactly `T`, `Δstats`, and `Δcount`, and returns to the same key. It also reproduces the same resource-owner masks, so multiplying the observed union/shared-busy integrals by an integer is exact.

Let `h` be the next external-event/advance target, `rem[c,o]` the remaining repetition count in the current category for owner `o`, and `Δcount[c,o]` the loop count increment. Do not cross a category boundary in a skip. Compute:

```text
k_time = floor((h − t2) / T)
k_cat  = min over positive Δcount[c,o] of
         floor(rem[c,o] / Δcount[c,o])
k      = min(k_time, k_cat, public-time-range limit)
```

If a loop has no positive work/count progress and zero busy masks, jump directly to `h` as idle time. If it has positive duration but cannot be bounded by the external target, category counts, or task completion, reject it as a non-advancing recurrence. If `k>0`, add `k*Δstats`, increment counts by `k*Δcount`, advance time by `k*T`, and leave normalized activity/slot state unchanged. If a category count reaches zero exactly at the endpoint, settle that category transition before another allocation. Then simulate the exact residual prefix to `h`.

This is a joint key over both owners, so it works for arbitrary phase offset: the relative phase is the pair of exact owner phase/progress states. Do not assume the owners start in phase or use a guessed common period. A period may take more than one owner cycle; any exact positive repeated joint state is sufficient.

## Probe bound and fail-closed behavior

For each static owner-set/capacity-generation window, store at most 257 keys and inspect at most 256 event-to-event transitions. There are at most eight K/full-tail/category transition classes; category boundaries stop a repeat-count skip and are processed explicitly. The event/probe budget is global for the static window, not reset per activity or per wave.

If no exact repeated key is found in 256 probes and the state still has repeated work, return `granularity_validation=failed/unavailable` (with the current exact prefix preserved for diagnostics). Do not enumerate another wave, switch to per-wave event mode, round a near-key into a hit, or alter the busy threshold. A completed finite task within 256 events can still return its exact result; the failure applies when the requested advance requires an unproved repeated continuation.

## External event and prefix recovery

Treat every external event as a hard horizon. Always integrate the old state and old rates up to the event time, including any partial period; never skip across it.

```text
advance_to(horizon):
  while now < horizon:
    h = min(horizon, next arrival, cancellation, capacity/profile change,
            next fixed category boundary)
    run exact event steps only up to h, recording keys/stats/counts
    if an exact key repeats:
      jump whole loops only if endpoint <= h and category count bounds hold
    if 256 probes expire without a repeat and work remains:
      fail closed
    settle all completions/slot changes tied at h
    if h is external:
      apply same-tick order: completion/retirement, cancellation,
      arrival, then capacity/profile change
      preserve already served counters and current owner phase/slot state
      increment owner/capacity generation as appropriate
      discard the old repeat table and start a new 256-probe window
```

For an arbitrary arrival offset `τ`, run/skip the existing solo owner only to `τ`: use its solo detected period for whole cycles, then simulate the exact fractional remainder. At `τ`, settle a completion tied with arrival first, insert the new owner in its exact initial/readiness state, keep the older owner's phase/progress/queues unchanged, and invalidate the solo key table. The first joint key includes both exact phases; no rounding `τ mod T` is needed.

Cancellation at `tc` similarly integrates joint service exactly to `tc`, settles completed work, records only served prefix counters, cancels the owner and releases its reserved slots, then redetects the remaining solo/joint cycle. Capacity change first integrates under old capacities to its exact timestamp, installs the new capacity generation, recomputes rates, and discards prior cycle keys. Neither operation rewrites prefix busy/work counters. If an external timestamp is exactly a cycle endpoint, allow the whole-cycle jump to the endpoint, settle completions/retirements, then apply the external event in the stated order.

## Two-owner closed-form check

The scratch fixture in `periodic_two_owner.py` has one shared HBM row (`H=1`), one shared compute gate (`P=1`), a full-K load of 2 units, and a full-K compute of 10 units. Each owner joins a warmed interior with one current input ready and starts one next-wave load alongside its current compute. Solo compute cycle is 10 s. With both computes active, each receives 1/2 compute capacity, so the joint full-K period is `T=20 s`; both computes remain active for the full period. A load takes 2 s alone and at most 4 s under two-way sharing, so it completes before the next 20 s compute boundary and does not stall C.

For any relative load-start phase, let `d=min(phase mod T, T−(phase mod T))` be the circular distance between load starts. If `d<2`, two HBM jobs overlap; processor sharing gives their union interval 4 s and shared interval `4−2d` s. If `d>=2`, they do not overlap; union is still 4 s and shared time is zero. Thus one exact joint period has HBM union 4 s, shared HBM `max(0,4−2d)` s for `d<2` (otherwise zero), compute union 20 s, HBM work 4, and compute work 20. This closed form is only a cross-check for the chosen fixture; the general algorithm uses observed exact positive-rate masks, not this formula.

## Differential scratch check

Run from the workspace root:

```sh
python3 work/audit-busy/periodic_two_owner.py
```

The script compares exact-Fraction no-skip event enumeration against the bounded repeat detector and independently checks every detected joint period against the closed-form mask integral above. It used ten rational arrival offsets (including `1/7`, `19/10`, `101/10`, and `1000/7`) and a horizon of 73 joint periods plus a `7/13`-second suffix. All aggregate counters and final normalized states matched exactly. Detected solo/joint periods were 10 s/20 s where observed; joint period integrals matched `HBM busy=4`, `compute busy=20`, and the phase-specific shared HBM formula. The largest joint repeat was certified in 7 event-to-event probes; total accelerated event intervals were at most 12, below the 256-probe cap. The naive run used 147–323 event intervals per sample while the accelerated run skipped 71–85 full periods. This small fixture validates skip arithmetic and phase-offset union handling, not the production LP or all category transitions.
