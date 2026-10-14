> 此代理报告记录当时的中间验证。其后 root 已对最终源显式 ABI 运行完整在线 discovery 106/106、typed hardware/train 35/35，无跳过；详见 verification.json 与最终日志。下文保留原报告，不把中间的“未重跑”描述当作最终状态。

# Python workspace bridge validation

Date: 2026-10-14

The vendored protobuf source at `work/deps/protobuf-21.12/python` is incomplete
for generated Chakra modules: it lacks `google.protobuf.descriptor_pb2`.
Installed protobuf **4.25.8** into `/workspace/template/work/deps/python-protobuf`
for test execution. Runtime: Python **3.12.14**.

From
`/workspace/template/astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference`:

```sh
python3 -m py_compile session_kv_manager.py online/online_scheduler_base.py online/wsc_llm_online_scheduler.py online/decision_bridge.py online/online_service.py online/test_workspace_admission.py
PYTHONPATH=/workspace/template/work/deps/python-protobuf python3 -m unittest discover -s online -p 'test_*.py'
git diff --check
```

Results: py_compile succeeded; all **102** discovered online Python tests passed
in 0.241 s; `git diff --check` succeeded.

After that full discovery run, the workspace suite grew from 11 to 15 focused
cases, including cross-epoch capacity, release retry, rank parser validation,
workspace-only cold-session reclamation, strict v2 ack fields, and atomic
over-capacity workspace rejection. The current suite was run separately after
the final manager and ack validation changes:

```sh
PYTHONPATH=/workspace/template/work/deps/python-protobuf python3 -m unittest online.test_workspace_admission
```

Result: all **15** workspace tests passed in 0.008 s. The full discovery suite
was not rerun after the final fit-policy and ack validation changes.

The final Python paths were syntax-checked:

```sh
python3 -m py_compile session_kv_manager.py online/online_scheduler_base.py online/wsc_llm_online_scheduler.py online/decision_bridge.py online/online_service.py online/test_workspace_admission.py
```

Result: succeeded. Scoped whitespace validation also succeeded:

```sh
git diff --check -- sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py sh_test_mesh/workload/llama2_7b_inference/online/decision_bridge.py sh_test_mesh/workload/llama2_7b_inference/online/online_scheduler_base.py sh_test_mesh/workload/llama2_7b_inference/online/online_service.py sh_test_mesh/workload/llama2_7b_inference/online/wsc_llm_online_scheduler.py sh_test_mesh/workload/llama2_7b_inference/online/test_workspace_admission.py
```

Result: succeeded.

Read-only wire cross-check against the current C++ serializer/parser confirmed
the v2 names and rank fields: request schema_version=2 plus
workspace_state(protocol_version=1, ranks[rank, workspace_revision,
active_workspace_bytes]); response schema_version=2,
source_delivery_sequence, and workspace_capacity_snapshot with per-rank
workspace_revision, capacity_bytes, model_weight_bytes, resident_kv_bytes,
reserved_request_bytes; ack schema_version, delivery_sequence, batch_id, and
boolean success. C++ continues to emit the preexisting delivery_epoch field;
no additional sequence field was introduced.

To verify the package version using the same import path:

```sh
PYTHONPATH=/workspace/template/work/deps/python-protobuf python3 -c 'from importlib.metadata import version; print(version("protobuf"))'
```

Output: `4.25.8`.
