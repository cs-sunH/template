# Terminal-count audit for the tiny FACE Online runs

The saved FACE legacy and FACE coarse runs both finished with `terminal_total=186`, with all 186 successful and no skipped or other terminals. Their C++ phase-5 counters also agree: 3 graph batches, 186 total nodes, 1 assignment, 3 KV actions, 2 watches, and 0 future alarms. The run artifacts are:

- `runs/astra-sim-face/legacy-20261014T215647Z-48224/`
- `runs/astra-sim-face/coarse-20261014T215649Z-48266/`

Both use source HEAD `ca604f5e49dc6df387888a33c75b0134642d42f3` and Online binary SHA-256 `522a16dcd26649ea1201d7854b395e297338ea8335933730bdd63a6cda61c78f`.

The persisted per-delivery split is the same for both FACE profiles: `ARRIVAL` emits 96 nodes, `PREFILL_DRAIN` emits 90, and `DECODE_COMPLETION`/`REQUEST_COMPLETE` emits 0. The first decision selects one prefill instance; the second selects decode instance 0. The train ledger shows one prefill chunk in the initial pass and two spans in the next pass for both. Coarse and legacy differ in predicted decode timing and candidate count, while retaining the same terminal/node totals. The coarse decision carries the expected `coarse_die_v2` disclosure.

The 192-terminal artifact previously cited is `work/joint-reproduction/runs/legacy-20261014T161352Z-10833/cpp.log`, whose simulator summary explicitly says `repo_variant=astra-sim-joint`. It reports a different delivery split (96, 96, 0), 192 terminals/nodes, 2 assignments, and 0 KV actions. It is not a FACE legacy run. The six-node gap is present in that other repository's second delivery (96 versus FACE's 90); it is not a coarse-versus-legacy difference in the saved FACE runs.

The FACE archives preserve aggregate delivery counts, train ledger, manifests, and request journal, but not the raw GraphBatch response IR: successful bridge responses are consumed and removed by the production bridge, and the materialized plan directory contains only its manifests. C++ metrics contain one `rank_compute` row per physical rank but no graph node-kind histogram. Therefore these saved artifacts establish the FACE totals and stage split, but do not support assigning the six-node cross-repository difference to a specific node kind or marker. No node-kind claim is inferred from terminal totals alone.
