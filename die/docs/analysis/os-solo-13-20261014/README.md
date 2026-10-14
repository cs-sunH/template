# Actual-facts solo lifecycle and external boundaries

This is an offline exact `Fraction` study. It consumes the actual C++ compiler's
single-Core L/C/S facts across all 13 resource rows, PRIVATE bounds, canonical Q,
physical HBM payload and typed operations. It does not modify production timing.

The constant two-input/two-output queues allow one active Load, Compute and Store.
Final-K Compute reserves an output slot; Store completion retires it. Nonfinal-K
Compute does not require an output slot. Lazy geometry retains at most 32 classes
(the probes use at most 12), never a tile/core array. Identity returns require
exact live progress equality and invariant translated class facts. Whole-row
classwords preserve the N tail and stop before the M tail. The global event budget
is 256, including continuation across external stops; exhausted budgets fail closed.

To reconstruct in a clean checkout from the repository root:

```bash
mkdir -p work/os-single-owner-13-20261014/current work/os-solo-stop-20261014
cp die/docs/analysis/os-solo-13-20261014/solo_facts_13.py work/os-single-owner-13-20261014/
cp die/docs/analysis/os-solo-13-20261014/current/* work/os-single-owner-13-20261014/current/
cp die/docs/analysis/os-solo-13-20261014/stops/* work/os-solo-stop-20261014/
cp die/docs/evidence/os-solo-13-20261014/current/facts_*.json work/os-single-owner-13-20261014/current/
python3 work/os-single-owner-13-20261014/current/study.py
python3 work/os-solo-stop-20261014/check_stops.py
```

`current/run_probes.sh` can rebuild the actual facts producer after the public
`work/build-die` build is configured. Its environment activation path is specific
to the current Cloud image; adapt tool paths on another environment. Rebuilding
replaces the copied facts and records a new producer identity. The small oracle's
explicit expansion is diagnostic only. It is not a production fallback.

`stops/solo_stop_13.py` additionally clamps return powering before an exact
absolute external boundary and integrates a partial stage when the boundary falls
inside an allocation interval. Resuming retains queues, progress, service and the
execution-wide event count. A capacity change discards old return discoveries.
The independent prefix checker compares every queue/progress coordinate and all
13 service/busy rows at 226 boundaries across six small fixtures. Capacity changes
and reverse-time rejection are checked separately. Large-K/output cases stop at
one-third and two-thirds and match exact no-stop powering, without independent
large-grid expansion.

See [evidence](../../evidence/os-solo-13-20261014/README.md) for source/binary
identities, measured results and failed public-nanosecond comparisons. Fixed-width
lowering, external competing owners, cancellation, multicore and production
performance remain separate work.

`fixedwidth/` contains the subsequent operation-width audit and standalone
standard C++17 rational primitive. Canonical slots are 256 bits, with checked
768-bit scratch and u128 repeat counts. The measured normalized widths fit
within 213/153 bits, while naïve 256-bit cross-product scratch fails. Fraction
differential checks cover 2,247 operations; the sanitizer check covers the C++
self-test. Copy that directory to `work/os-fixedwidth-13-20261014/` to run its
documented commands after the facts and stop studies above are reconstructed.
The width measurements cover those traces; they do not prove universal input
coverage or the production CPU budget. The primitive is not used by Runtime.
