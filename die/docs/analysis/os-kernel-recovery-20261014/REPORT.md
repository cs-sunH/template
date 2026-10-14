# OS scratch-kernel recovery and integration review

Date: 2026-10-14. This is research-only recovery of the Solo13 lifecycle kernel.
No public production source, CMake file, main `HANDOFF.md`, `PROJECT.md`, or
variant source was changed for this work. Scratch implementation and tests live
under `work/os-kernel-recovery/`; captured evidence is indexed in
[`../evidence/os-kernel-recovery-20261014/identities.json`](../../evidence/os-kernel-recovery-20261014/identities.json).

## Baseline and available tools

The rebuilt pre-profiler baseline passed its complete differential suite. Its
identity was kernel header `bc21d044…`, kernel source `affee811…`, validated
binary `d4cf4554…`, fixed-width rational header `f7f02d21…`, and repository HEAD
`fecf81c5…`. The original historical binaries were not available; this is a
reconstructed and rebuilt baseline, with full hashes and build output in the
evidence directory.

The recovered post-profiler/reset scratch kernel now passes the complete suite
with the fixed-width optimization below. Its identity is kernel header
`2b65003a…`, source `affee811…`, fixed-width header `cd72650b…`, and binary
`4e80ec99…`. The saved full-suite record covers 17 scenarios, including the
256-event fail-closed boundary, pause/resume, capacity invalidation, large-K,
large-output, reset, and the exact rational overflow boundaries.

`g++` was available on the research agent's PATH. That standalone run did not
use the parent's local CMake/protoc prefixes; `perf` was unavailable. It used
C++17 compilation without a production build or hardware-counter profile.
The n30 timing run overlapped concurrent C++ builds;
its samples are retained as contaminated diagnostics, not called isolated
measurements. No second n30 run was started while those builds were active.

## Fixed-width normalization bottleneck

An instrumented large-output stop/resume run counted 13,241 GCD calls and
31,426 rational normalizations. Of those GCDs, 9,131 returned one. Before the
optimization, the same run made 26,482 768-bit long-division calls, 18,262 of
them with divisor one, and executed 1,402,106 division bit steps. These counts
locate the work in repeated normalization, binary GCD, and bitwise long
division; the inclusive timers overlap and cannot be added together.

The scratch rational copy now applies two exact shortcuts:

- Binary GCD exits when the smaller odd operand reaches one and restores the
  common power of two. The returned GCD is unchanged.
- Normalization copies the numerator and denominator when the computed GCD is
  one instead of invoking two divisions that return those same operands.

The normalized state remains 256 bits, scratch remains 768 bits, and repeat
counts remain checked u128. The post-optimization profile executed 168,854 GCD
iterations (37.5% fewer), 8,220 long divisions (68.9% fewer), and 735,681
division bit steps (47.5% fewer); divisor-one divisions fell to zero. The
9,131 unit-GCD normalizations are unchanged. The instrumented profile timers
varied sharply under concurrent compilation, so these work-count reductions
are not a wall-time speedup claim.

The standalone primitive passes 2,247 Python `Fraction` differential cases,
including 320 seeded random rational pairs and the existing 256/768/u128
boundary cases. The full lifecycle Fraction/prefix suite and ASan/UBSan smoke
also pass. The sanitizer smoke explicitly passes the maximum 256-bit numerator
and u128 repeat count, then fails closed at 2^256 numerator/denominator and
u128-overflow boundaries. The earlier sanitizer assertion failure is retained:
its `2^200` stop overflowed the normalized 256-bit slot before the intended
u128 assertion. The smoke input was corrected to `2^128`, which reaches the
intended fail-closed branch without widening the arithmetic domain.

## Benchmark evidence

The preserved n30 sample reports P95 cold-full / run-only-full times of:

| Fixture | Cold full | Run-only full |
| --- | ---: | ---: |
| dense | 150.956 ms | 194.390 ms |
| large K | 188.346 ms | 194.190 ms |
| large outputs | 159.631 ms | 170.886 ms |

Cold samples include file read, JSON parsing, facts/geometry construction, and
the full scratch lifecycle; run-only samples reset a reusable kernel before
timing. The host exposed three hardware threads, and multiple large C++ builds
were active during collection. The values are therefore contaminated and the
timing scope is not a direct Runtime update or class-plan compile measurement.
They are far from a useful performance demonstration; an isolated run is still
needed before making any performance claim. `perf` was unavailable.

The project design separately specifies P95 query 20 us, cold class-plan
compile 2 ms, Runtime update 50 us, and Online overhead at most 10%. These
full-lifecycle scratch timings do not measure those individual scopes, so they
cannot certify or precisely fail a particular budget. They do show that this
research path is not yet credible evidence of a 2 ms-class implementation.

## Public integer-nanosecond and multi-K semantics

See the concise standalone [public clock note](PUBLIC_CLOCK.md).


`Runtime::next_event()` reads the private long-double event delta and returns
`now_ns + max(1, ceil(delta - 1e-15 ns))`. `Runtime::advance_to(target)` still
settles internal activity and epoch events at their sub-nanosecond times while
advancing to that integer target. Terminal `finish_ns` is `ceil(exact_ns -
epsilon)`; the call returns with public `now_ns == target`. The caller then
collects terminal executions and pumps any newly eligible work. One integer
wake may therefore contain several internal settlements, while a new external
stage starts only at the caller's integer boundary.

The saved `os_lifecycle_runtime_driver` is an offline one-request L/C/S pump:
it submits independent one-activity Plans, calls `next_event()` and
`advance_to()`, then collects and pumps. It does not prove when a compiled
multi-epoch Plan changes its own internal epoch. The continuous Fraction
oracle starts each next stage at an exact event; the public-wake oracle models
the integer caller boundary. Thus continuous-vs-public deltas can accumulate
across caller pump points. The residual actual-vs-public-oracle deltas of
`(2, 3)` ns for converted storage, `(6, 6)` ns for requested output pressure,
and `(3, 3)` ns for K=1 remain deviations, not proof of an algorithmic
rounding allowance. A one-tick difference at a ceil boundary can change the
next caller wake and propagate through later starts; the present exact-rational
oracle does not emulate the C++ double allocation and long-double event
arithmetic closely enough to assign those residuals. No tolerance was raised.

A scratch public-tick comparison reuses the C++-captured stage facts but uses
exact Fraction allocation. Its event-key sets match all seven saved scenarios.
Maximum start/finish errors against the actual Runtime are: strict small `(2,
2)` ns, native INT4 `(2, 2)`, converted fractional `(2, 2)`, converted storage
`(2, 3)`, requested output pressure `(6, 6)`, K=2 pressure `(0, 1)`, and K=1
pressure `(3, 3)`. Converted storage, requested pressure, and K=1 exceed the
existing 2 ns comparison bound. This is retained as a failed exploratory
comparison, not a production clock certificate. The exact comparison output
and identities are in the evidence bundle's `current/public_tick_comparison.json`.

The queue contract needs only a small state machine. Each of at most two output
slots follows `Free -> Reserved -> Ready -> Storing -> Free`; final-K Compute
reserves at its start, its terminal makes the slot Ready, Store start makes it
Storing, and Store terminal frees it. Final-K Compute requires a Free slot;
nonfinal K does not reserve output. The actual single-request logs reach output
occupancy two for K=1 and one for K=2 and requested pressure. None contains a
reachable attempt where a ready final Compute is blocked by both full slots.
The saved driver has no external request competition, so it cannot supply the
requested full-output wait witness. A future Runtime acceptance trace should
show this finite-state transition and blocked start under the target K/shapes
and real competing work, rather than infer reachability from an injected queue
state.

## Scope and next work

The class-cache limit remains 32; the current fixture facts use fewer classes.
The scratch kernel keeps its three stage slots, two input/two output slots,
256 fixed history snapshots, fixed 256-event lifetime expanded-work cap, and
checked 256/768-bit rational domain. These are not Runtime's 96-active limit or
256 solver-pivot count. There is no GMP production path, per-tile fallback, or
cap increase. Production accuracy gates remain 3% exclusive finish and 5%
concurrent completion/busy; the profile/recovery suite is not a certificate at
those thresholds.

Before Runtime wiring, keep the public integer-wake contract at this API boundary and require a real competing-work trace for any full-output wait claim. No compiled multi-epoch translation proof, reachable output-blocked witness, or isolated performance result is established here. An isolated benchmark remains pending until the parent confirms the host is clear.
