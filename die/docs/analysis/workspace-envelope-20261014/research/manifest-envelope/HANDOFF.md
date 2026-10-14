# Manifest envelope scratch handoff

Status: **draft only; not executed or validated**. The scratch directory contains `generate_validate.py` (40,898 bytes). Its `inputs/`, `results/`, `logs/`, and `generated/` directories are empty. No command is still running, and no finite envelope, C ABI comparison, or validation result was produced. Treat the script as an unverified draft, not a reusable generator or certificate.

The draft was intended to derive a fixed manifest and compare actual scheduler/graph-builder train subsets against a finite envelope by calling the real C ABI. It was not run, so fixture generation, config loading, scheduler state setup, protobuf imports, operation/rank/phase queries, and actual-subset bounds are all unverified. Also unfinished are recorded source/binary identities for this study, result/audit files, and README instructions. Any command to resume is only a proposal, not a verified reproduction command:

```sh
cd /workspace/template
source /workspace/cloud-setup/activate.sh
python3 work/lru-manifest-envelope-20261014/generate_validate.py
```

Candidate inputs and implementation sources identified before the stop:

- Prior official LRU positive-workspace fixture: `work/lru-online-workspace-20261014/e2e-large-workspace-probe/` (8-rank 4x2; P ranks 0–3, D ranks 4–7; MHA4, hidden 512, FFN 1024, vocab 512, bf16, one layer). This is an existing generated E2E fixture, not measured source-model data. The draft intended to use it as a seed and generate a separate scratch manifest; it did not do so.
- Real envelope library: `/workspace/template/die/work/lru-workspace-envelope-20261014/build/lib/libcoarse_die_cabi.so`, recorded SHA-256 `b4778f15831110814eb68de6cf3cf02995d65aa480fff929946f0e68504ea80d` in the earlier client work. Recheck identity before any future use.
- Official generated Chakra Python module: `astra-sim-wscllm-LRU/work/chakra-python-root/extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py`.
- Intended production helpers to call (not reimplemented): `DieModelContext.from_service` / `build_die_spec`, GraphBatchBuilder prefill/train emission helpers, and `WscLlmOnlineScheduler._plan_train`.
- The `local_region` adapter operation is expected to be rejected: the legacy `DieOpSpec` v1 input does not encode a typed mathematical graph. This expectation was not exercised by this draft.

No tracked adapter, scheduler, service, or resolver source was changed for this scratch task. The earlier client files `online/die_workspace_envelope.py` and `online/test_die_workspace_envelope.py` are outside this scratch task and were left untouched. Freeze this draft and its empty output directories at handoff; do not infer successful manifest coverage or monotonicity from the script contents.
