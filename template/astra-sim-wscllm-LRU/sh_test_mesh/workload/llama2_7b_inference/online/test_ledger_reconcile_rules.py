"""test_ledger_reconcile_rules.py -- ledger_reconcile 对账规则回归(合成输入)。

覆盖 verify/ledger_reconcile.py 的历史假判定修复点与新契约(V-1~V-4,
2026-09-30 批次 E;列车口径 + fail-closed 入口门禁 2026-10-02 随 W 仓
列车感知口径移植同步):--cpp-log 必填与存在性自检、R2b 批级 watch 上界
[0, node_count]、R7a2 批命名空间残差剥离、数据源缺失/残缺一律 exit 2
fail-closed(原 V-4 的 SKIP 守卫随列车口径移植退役——digests/completed_
nodes/审计行缺失不再降级 SKIP,入口即拒绝)。全部使用 /tmp 级合成 run
目录(单请求、秒级),不依赖真实仿真产物。
"""

import copy
import json
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.verify import ledger_reconcile  # noqa: E402


CPP_COUNTERS = ("[online] service counters: accepted=112 completed=1 "
                "active=0 pending_alarm=0")
CPP_AUDIT = ("[online] phase-5 commit counters: graph_batch_count=2 "
             "single_node_bridge_count=0 total_nodes=3 "
             "avg_nodes_per_batch=1.5 max_nodes_per_batch=2 "
             "total_watches=3 total_assignments=2 total_kv_actions=1 "
             "total_future_alarms=0\n"
             "[online] phase-4 end audit: watch_registry_size=0 "
             "watch_stale=0 completed_facts_residual=0 "
             "affected_ranks_residual=0")

# V-2 形状:watch_count(2) 超过 ranks 数(1)但不超过批节点数(2)。
DIGESTS = [
    {"delivery_sequence": 0, "tick": 30, "watch_count": 2,
     "ranks": [0], "node_count": 2},
    {"delivery_sequence": 1, "tick": 40, "watch_count": 1,
     "ranks": [1], "node_count": 1},
]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")


def _build_run(root, *, with_digests=True, audit_lines=True, digests=None,
               batch_train_injected=True, final_barrier_nodes=1,
               with_train_ledger=True):
    bridge = os.path.join(root, "bridge")
    os.makedirs(bridge, exist_ok=True)
    with open(os.path.join(root, "manifest.json"), "w",
              encoding="utf-8") as out:
        json.dump({"requests": [{"request_id": "r0"}]}, out)
    _write_jsonl(os.path.join(bridge, "ledger.jsonl"), [{
        "request_id": "r0",
        "layers": {"completed_unreconciled": {
            "admitted_tick": 10, "first_commit_tick": 20,
            "completed_tick": 40,
            "stages": [{"stage": "prefill"}, {"stage": "decode"}]}},
    }])
    completed_groups = [
        {"request_id": "r0", "stage": "prefill", "generation": 0,
         "node_count": 1},
        {"request_id": "r0", "stage": "decode", "generation": 1,
         "node_count": 1},
        {"request_id": "r0", "stage": "", "generation": 1, "node_count": 1},
        {"request_id": "batch_train_t0", "stage": "prefill",
         "generation": 0, "node_count": 1},
    ]
    _write_jsonl(os.path.join(bridge, "request_journal.jsonl"), [{
        "schema_version": 1, "delivery_sequence": 0, "delivery_epoch": 0,
        "tick": 40, "deferred_from_tick": 0, "reasons": ["ARRIVAL"],
        "arrivals": [{"request_id": "r0", "session_id": "s0",
                      "turn_index": 0, "prefill_length": 16,
                      "decode_length": 4,
                      "inter_request_interval_ns": 0,
                      "arrival_world_ns": 10, "ingress_seq": 0,
                      "queue_index": 0}],
        "completed_groups": completed_groups,
        "completed_nodes": [
            {"request_id": "r0", "stage": "prefill", "rank": 0,
             "node_id": 0, "terminal_status": 0},
            {"request_id": "r0", "stage": "decode", "rank": 0,
             "node_id": 1, "terminal_status": 0}],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": 0, "tick": 40, "kind": ""},
        "snapshot": {},
        "ledger_summary": {"injected_unfinished": []},
    }])
    _write_jsonl(os.path.join(bridge, "online_decision_log.jsonl"), [
        {"seq": 0, "kind": "prefill", "request_id": "r0", "tick": 20,
         "decision": {"history_action": "LOCAL_HIT"}},
        {"seq": 1, "kind": "decode", "request_id": "r0", "tick": 30,
         "decision": {"kv_instance_after_completion": None}},
        {"seq": 2, "kind": "completion", "request_id": "r0", "tick": 40,
         "decision": {"kv_instance_after_completion": 3}},
    ])
    injected = [{
        "rank": 0, "node_count": 7, "in_flight_node_count": 7,
        "free_node_count": 0,
        "per_request": [{"request_id": "batch_train_t0", "stage": "prefill",
                         "node_count": 7, "compute_ops": 3,
                         "estimated_remaining_ns": 9, "comm_bytes": 4}],
    }] if batch_train_injected else []
    _write_jsonl(os.path.join(bridge, "sensing_query_log.jsonl"), [
        {"delivery_sequence": 0, "tick": 30,
         "admitted_not_injected_queued": {"admitted_count": 0, "detail": []},
         "injected_unfinished": injected,
         "ready": {"ready_count": 0, "detail": []},
         "issued": {"issued_count": 0, "detail": []}},
        {"delivery_sequence": 1, "tick": 40,
         "admitted_not_injected_queued": {"admitted_count": 0, "detail": []},
         "injected_unfinished": [{
             "rank": 0, "node_count": final_barrier_nodes,
             "in_flight_node_count": final_barrier_nodes,
             "free_node_count": 0,
             "per_request": [{"request_id": "r0", "stage": "decode",
                              "node_count": final_barrier_nodes,
                              "compute_ops": 0,
                              "estimated_remaining_ns": 0,
                              "comm_bytes": 1}]}],
         "ready": {"ready_count": 0, "detail": []},
         "issued": {"issued_count": 0, "detail": []}},
    ])
    # 列车感知口径(2026-10-02 移植):train_ledger.jsonl 是 R7a1/R8b 的
    # 必填数据源,缺失即入口 fail-closed;空账本合法(零 decode 工作),
    # 列车覆盖退化为纯请求级键核对。
    if with_train_ledger:
        _write_jsonl(os.path.join(bridge, "train_ledger.jsonl"), [])
    if with_digests:
        _write_jsonl(os.path.join(bridge, "graph_batch_digests.jsonl"),
                     DIGESTS if digests is None else digests)
    lines = [CPP_COUNTERS]
    if audit_lines:
        lines.append(CPP_AUDIT)
    with open(os.path.join(root, "cpp.log"), "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    return root


def _reconcile(root, cpp_log="cpp.log"):
    argv = [
        "--bridge-dir", os.path.join(root, "bridge"),
        "--manifest", os.path.join(root, "manifest.json"),
        "--cpp-log", os.path.join(root, cpp_log),
        "--expected-requests", "1",
    ]
    return ledger_reconcile.main(argv)


class LedgerReconcileRulesTest(unittest.TestCase):
    """合成 run 上的对账规则回归(每例毫秒级,无仿真)。"""

    def setUp(self):
        import tempfile
        self._root = tempfile.mkdtemp(prefix="ledger_reconcile_test.")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._root, ignore_errors=True)

    def test_cpp_log_required_arg(self):
        # V-1:--cpp-log 缺参由 argparse 拒绝(SystemExit 2),不再空洞对账。
        argv = ["--bridge-dir", os.path.join(self._root, "bridge"),
                "--manifest", os.path.join(self._root, "manifest.json")]
        with self.assertRaises(SystemExit) as ctx:
            ledger_reconcile.main(argv)
        self.assertEqual(ctx.exception.code, 2)

    def test_cpp_log_nonexistent_path_fails_closed(self):
        # V-1:路径给出但文件不存在(含 .gz 回退后)→ 退出码 2,不产报告。
        root = _build_run(self._root)
        self.assertEqual(_reconcile(root, cpp_log="missing.log"), 2)

    def test_full_run_balanced_with_v2_v3_shapes(self):
        # V-2:watch_count(2) > len(ranks)(1) 但 <= node_count(2) 不再误
        # FAIL;V-3:batch_train_ 列车组残差(节点数 7)被剥离,R7a2 不再
        # 误 FAIL;V-1:R0d/R0e/R2c/R3a/R3c 以 cpp.log 计数器真实执行。
        root = _build_run(self._root)
        self.assertEqual(_reconcile(root), 0)

    def test_r2b_bound_still_enforced(self):
        # V-2 负例:watch_count 超过批节点总数仍判失配。
        digests = copy.deepcopy(DIGESTS)
        digests[0]["watch_count"] = 3
        root = _build_run(self._root, digests=digests)
        self.assertEqual(_reconcile(root), 1)

    def test_r7a2_real_non_barrier_residual_still_fails(self):
        # V-3 负例:真实请求的非 barrier 残差仍判失配(剥离不放过真失配)。
        root = _build_run(self._root, final_barrier_nodes=2)
        self.assertEqual(_reconcile(root), 1)

    def test_missing_digests_fails_closed(self):
        # 列车口径移植后的 V-4 契约:digests 缺失不再 SKIP 降级,入口
        # fail-closed 退出 2(消息注明所需 SH_GRAPH_DIGESTS=1)。
        root = _build_run(self._root, with_digests=False)
        self.assertEqual(_reconcile(root), 2)

    def test_missing_audit_lines_fails_closed(self):
        # P2-19 补口:cpp.log 在场但缺 phase-5/4 审计行(R0d/R0e/R2c/R3a/
        # R3c 将空洞通过)→ 入口 fail-closed 退出 2,不产报告。
        root = _build_run(self._root, audit_lines=False)
        self.assertEqual(_reconcile(root), 2)

    def test_missing_train_ledger_fails_closed(self):
        # 列车感知口径:train_ledger.jsonl 缺失 → R7a1/R8b 无成员时间线
        # 数据源,入口 fail-closed 退出 2。
        root = _build_run(self._root, with_train_ledger=False)
        self.assertEqual(_reconcile(root), 2)


if __name__ == "__main__":
    unittest.main()
