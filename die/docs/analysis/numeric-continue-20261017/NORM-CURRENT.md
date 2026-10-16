# Current Runtime still reproduces the seven-epoch Norm red

The actual `54b8f6f…38aab` final Runtime library was freshly used to compile
the unchanged current-Core manual seven-epoch fixture after the numerical
checkpoint. This is a real new run, not old A+B evidence relabeled as current.
The fixture uses explicit Core32 geometry and visibly synthetic service rates;
it does not restore the archived long-row compiler or calibrate hardware.

Both arrivals still stall at5130029297ns with no next event. All11 public
snapshots are byte-identical to the previously saved A+B fixture. There are6
observed channels whose outward-rounded physical-credit interval is entirely
outside capacity. Terminal physical totals are not reached or accepted. The
positive-time/ready guards therefore do not repair the capacity1 snapping defect.
The58 passing registered tests do not certify this separate research fixture.

The new source, contract, exact runner, commands, actual library/object/ELF hashes,
all snapshots and failed ledger observations are losslessly preserved in
`../../evidence/numeric-continue-20261017/norm-current-runtime/`.
Its summary binds every archive member hash and the snapshot comparison. No ELF
is committed. Restore its members to their original `work/norm-runtime-current/`
paths in a fresh checkout, then use a fresh result label and freshly built library:

```sh
python3 work/norm-runtime-current/run_probe.py \
  --library work/NEW_BUILD/libcoarse_die.a \
  --label NEW_LABEL --compiled-source-root die/src --cxx g++
```

The library-to-source mapping must also match the build recorder. The runner's
source-root mapping is an explicit caller claim, not inferred from a filename.

Independent arithmetic research in [norm-affine/README.md](norm-affine/README.md)
improves cancellation on supplied progress but rejects the initial structural
headroom pivot after actual sign errors. It is not a Runtime fix. The next small
experiment adds fixed-size FMA product residuals on the same frozen inputs;
only a successful arithmetic checkpoint can justify a new physical-view Runtime
experiment. Earlier snapshot minus4ULP and service-role failures remain gates.
