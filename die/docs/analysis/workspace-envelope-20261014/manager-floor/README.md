# Prospective admission audit and manager floor

The design and production wiring audit is in [`design.md`](design.md). The
only tracked-source changes in this follow-up are:

- `sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py`
- `sh_test_mesh/workload/llama2_7b_inference/test_workspace_admission_manager.py`

The optional constructor map is full-rank, exact-u64, copied, and read-only.
The default map is zero, preserving the legacy no-view path. KV admission and
reservation extension use `max(actual_workspace, prospective_floor)`; the
external workspace snapshot and revision API continue to describe actual
active workspace only. E is outside journal rollback. The online scheduler is
not wired to compute or install E; the remote-store and cross-instance
migration source-release gap described in `design.md` also remains an
activation blocker.

## Test environment and identity

- LRU branch: `codex/die-workspace-envelope-20261014`
- LRU HEAD at test time: `fecf81c5574755209c5d58ecdb982843ff349042`
- Python: `/workspace/cloud-setup/venv/bin/python3` (3.12.14)
- Python protobuf runtime: 6.33.5
- Schema used by the focused runner: actual protoc-generated
  `work/chakra-python-root/extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py`
  (ignored work output; SHA-256
  `4608144d0aab16cde4d259e8ff56c0c7b947e181c722f7b8a50d07441487ab1b`).
- The runner preloads that real generated module before `generate_trace`
  inserts the checkout root into `sys.path`. It does not stub protobuf or
  monkeypatch routing/generator behavior.

The exact manager/test source hashes after the final edit are:

```text
8e5de6ff9926bc88a2b9f2b788568e71f8010c856005bb6818f58a3c1d615ed3  sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py
584660dc82c2372335ec85a671146c623a85cfdf73ead4e74769634f1cbb9684  sh_test_mesh/workload/llama2_7b_inference/test_workspace_admission_manager.py
```

Run a focused suite with:

```bash
source /workspace/cloud-setup/activate.sh
python3 work/lru-prospective-admission-20261014/run_focused_suite.py test_workspace_admission_manager
```

## Results

All six suites below passed with the official generated schema available to
the full real manager paths. Logs are in this directory.

| Suite | Result |
|---|---:|
| `test_workspace_admission_manager` | 16 passed |
| `test_wscllm_kv_incremental_invariants` | 8 passed |
| `test_tiered_eviction_sequence` | 5 passed |
| `test_kv_delta_journal` | 14 passed |
| `test_admission_net_credit` | 8 passed |
| `test_workspace_admission_v2` | 6 passed |

`git diff --check` and `py_compile` for the changed Python files and runner
passed. The first direct invocation under the host Python failed on missing
`google.protobuf`; rerunning through the configured cloud Python environment
and real protoc-generated schema passed. No source workaround was added.
