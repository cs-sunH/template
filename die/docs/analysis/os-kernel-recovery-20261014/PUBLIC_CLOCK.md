# Public integer clock and output-slot findings

## Public integer-nanosecond and multi-K semantics

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
