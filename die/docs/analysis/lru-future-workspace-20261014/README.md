# LRU manifest workspace and source-release reproduction

This checkpoint adds a capacity-only C++ lifetime formula before SRAM candidate
filtering, installs a fixed per-rank manifest margin, checks the actual Plan
against that margin, and gates capacity reuse on source transport completion.
The margin is `max(active_workspace, future_workspace)` for fit. It is neither
a KV debit nor a fabricated active lease. The capacity formula does not prove
that a task has an executable mapping.

`online/test_manifest_workspace_envelope.py` reconstructs its four-request
fixture from tracked seeds in a temporary directory. It covers real P emission,
all TP ranks, 55 valid D cohort/offset combinations, capped/uncapped trains and
first-step/remainder splits using the real generator and C ABI. It does not
depend on a previous account's ignored work directory. The exact candidate
filter counterexample and formula scope are in
[workspace-capacity-proof](../workspace-capacity-proof-20261014/README.md).

The five scripts here recreate the separate actual eight-rank Online fixture:
two sessions, two turns each, MHA4, hidden512/FFN1024/vocab512, BF16, one layer,
P64/700 and P96/900, D2/5 and D3/3. Capacity is 144,000,000,000 B/rank from the
official coarse hardware descriptor; no capacity, demand or policy override is
used. Restore the scripts under `work/lru-future-workspace-20261014/e2e/`, build
the LRU aggregate under `work/build-lru`, and generate the repository-relative
Chakra Python binding as described in
[the prior rebuild instructions](../lru-workspace-v2-20261014/online-reproduction/README.md).

The aggregate CMake scaffold is the prior three-line subdirectory scaffold.
Configure Release with `ASTRA_SIM_ROOT` set to the LRU checkout,
`BUILDTARGET=congestion_aware`, and `NETWORK_BACKEND_BUILD_AS_LIBRARY=ON`.
Build `AstraSim_Analytical_Congestion_Aware_Online`, `coarse_die_cabi` and
`die_model_cabi`. CMake3.31.10/protoc3.21.12/Python protobuf4.25.8 were used here;
`work/activate.sh` is local tooling, not an assumed cross-account dependency.

```sh
python3 work/lru-future-workspace-20261014/e2e/prepare_workspace_e2e.py
bash work/lru-future-workspace-20261014/e2e/run_online_e2e.sh
python3 work/lru-future-workspace-20261014/e2e/audit_capacity_online.py
SH_RUN_DIR="$PWD/work/lru-future-workspace-20261014/e2e/run_capped" \
SH_TRAIN_MAX_ITER=2 \
bash work/lru-future-workspace-20261014/e2e/run_online_e2e.sh
SH_RUN_DIR="$PWD/work/lru-future-workspace-20261014/e2e/run_capped" \
python3 work/lru-future-workspace-20261014/e2e/audit_capacity_online.py
```

The runner needs a current `source-identity/` package: source file hashes,
dirty diff, base/branch context, actual Online/two-ABI binary hashes and
provenance. The committed evidence package supplies the identity of the
measured runs. A rebuild must record its own identity rather than relabel that
historical package. Raw strace and large recompiled Plan JSON are regenerated;
they are not committed. The runner refuses to overwrite an existing run.

Both actual runs passed the corrected strict wire/Plan/capacity/DAG audit. The
default run had 16 batches, 480 typed COMP Plans, 48 positive-workspace Plans
and 764 successful terminals; the capped run had 19/560/56/880. Every actual
Plan was below its rank bound, 128/152 capacity rows fit, eight migration-source
ACKs had capacity-consumer dependencies, all leases returned to zero and all
nodes were collected. The same immutable certificate was used in both runs.
Positive workspace was not sampled at bridge boundaries; balanced lease pairs
are inferred from revisions, not a sampled occupancy timeline. Remote-store/
restore pressure was not observed in these Online inputs; its ordering is
covered separately by real-builder and C++ cross-batch/GC fixtures.

This is software integration evidence. Production E5 finish/busy accuracy,
target held-out data, all business paths and the host-overhead budget remain
unaccepted. TX8 and target priors were unchanged. Auxiliary modeling follows
the user's formula-first principle in the design document §1.4.
