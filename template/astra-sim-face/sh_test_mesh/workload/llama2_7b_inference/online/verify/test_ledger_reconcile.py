#!/usr/bin/env python3
"""test_ledger_reconcile.py -- ledger_reconcile 合成数据单测。

2026-09-30 补齐:该工具此前无任何单测(冒烟失败文档 F1 登记项),窗口
口径修正(R7 反向包含 / R5a end_barrier 按名锚定 / R7a3 列车台账交叉)
无法用合成数据裁决"门错位还是摘要错位"。本测试用最小合成 bridge 目录
直接驱动 main(),覆盖:

  1. 平衡基线:全部 R 门 PASS、exit 0——响应内列车键下同时挂列车体
     激活 all_reduce(512 KiB,名字不带 _end_barrier 后缀)与 1 字节
     end barrier,末边界在途残差 1 字节。旧 R5a 求和锚(524288+1)在该
     基线上恒假 FAIL,若口径回退本测试即红(F3 回归锚);
  2. R5a 字节失配:残差 comm_bytes != 已提交 end-barrier 字节 → exit 1;
  3. R7a1 反向包含:真实请求出现在 C++ injected 而同边界 Python issued
     无 → exit 1;
  4. R7a3 列车台账:only-C++ 窗口出现无台账行的列车键 → exit 1;
  5. R7a3 时序:列车键首现 tick 早于台账发射 tick → exit 1;
  6. R7a3 归一化:SH_FIRST_TOKEN_SPLIT=1 首步唤醒键剥离 _first_step
     后对上台账行 → exit 0(2026-10-01 修复锚:修复前恒假 FAIL);
  7. R0d fail-closed:--cpp-log 显式给出但计数行不可解析 → exit 1
     (2026-10-01 修复锚:修复前静默跳过、假绿 exit 0)。

合成数据形状与真实产物同构(request_journal.jsonl / response_*.json /
sensing_query_log.jsonl / ledger.jsonl / graph_batch_digests.jsonl /
train_ledger.jsonl);cpp.log 缺席时 R0d/R0e/R2b 双和/R2c 按 SKIP 处理,
不参与判定。
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONLINE_DIR = os.path.dirname(_HERE)
for _p in (_HERE, _ONLINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ledger_reconcile  # noqa: E402


def _sensing_row(seq, tick, *, issued, groups, rank_node=1,
                 in_flight=1, free=0):
    """一条 sensing 查询行;groups = [(request_id, stage, node_count,
    compute_ops, comm_bytes, estimated_remaining_ns, generation)]。"""
    per_request = [
        {
            "request_id": request_id,
            "stage": stage,
            "generation": generation,
            "node_count": node_count,
            "compute_ops": compute_ops,
            "comm_bytes": comm_bytes,
            "estimated_remaining_ns": remaining,
        }
        for (request_id, stage, node_count, compute_ops, comm_bytes,
             remaining, generation) in groups
    ]
    return {
        "delivery_sequence": seq,
        "tick": tick,
        "issued": {
            "issued_count": len(issued),
            "detail": [
                {"request_id": request_id, "stage": "decode",
                 "issued_tick": tick, "instance_index": 0}
                for request_id in issued
            ],
        },
        "ready": {"ready_count": 0, "detail": []},
        "admitted_not_injected_queued": {
            "admitted_count": 0, "committed_count": 0,
            "not_injected_count": 0, "by_queue": {}, "detail": [],
        },
        "injected_unfinished": [
            {
                "rank": 0,
                "node_count": rank_node,
                "in_flight_node_count": in_flight,
                "free_node_count": free,
                "compute_ops": 0,
                "comm_bytes": 0,
                "in_flight_gpu_ops": 0,
                "estimated_remaining_ns": 0,
                "per_request": per_request,
            },
        ],
    }


def _write_bridge(root, *, residual_bytes=1, extra_row2_groups=(),
                  train_ledger_rows=None, train_first_tick=1500,
                  omit_train_ledger=False):
    """最小合成 bridge 目录(平衡基线 + 可注入的单点违例)。"""
    bridge = os.path.join(root, "bridge")
    os.makedirs(bridge)
    manifest_path = os.path.join(root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as out:
        json.dump({"requests": [{"request_id": "req_0"}]}, out)
    # C++ 完成事实:一条交付携带 prefill/decode/request-complete。
    with open(os.path.join(bridge, "request_journal.jsonl"), "w",
              encoding="utf-8") as out:
        out.write(json.dumps({
            "delivery_sequence": 0,
            "tick": 3000,
            "completed_groups": [
                {"request_id": "req_0", "stage": "prefill",
                 "generation": 0, "node_count": 1},
                {"request_id": "req_0", "stage": "decode",
                 "generation": 1, "node_count": 1},
                {"request_id": "req_0", "stage": "", "generation": 1},
            ],
        }) + "\n")
    # committed 侧:真实请求 prefill/decode 节点 + 列车键下 512 KiB 激活
    # all_reduce(非 _end_barrier 名字)与 1 字节 end barrier。R5a 修正后
    # 只有 end barrier 进入参照字节;求和口径(旧)会把 524288 混进来。
    nodes = [
        {"rank": 0, "id": 1, "name": "req_0_prefill", "request_id": "req_0",
         "stage": "prefill", "generation": 0, "coll": None},
        {"rank": 0, "id": 2, "name": "req_0_decode", "request_id": "req_0",
         "stage": "decode", "generation": 1, "coll": None},
        {"rank": 0, "id": 3, "name": "batch_train_t1_body_ar",
         "request_id": "batch_train_t1", "stage": "decode",
         "generation": 1,
         "coll": {"comm_type": 0, "bytes": 524288, "priority": 0}},
        {"rank": 0, "id": 4, "name": "batch_train_t1_end_barrier",
         "request_id": "batch_train_t1", "stage": "decode",
         "generation": 1,
         "coll": {"comm_type": 0, "bytes": 1, "priority": 0}},
    ]
    watches = [
        {"request_id": "req_0", "stage": "prefill", "generation": 0,
         "members": {"0": 1}},
        {"request_id": "req_0", "stage": "decode", "generation": 1,
         "members": {"0": 2}},
    ]
    kv_actions = [
        {"trigger_request_id": "req_0", "phase": "admission",
         "event_type": "retain", "planner_time_ns": 100},
        {"trigger_request_id": "req_0", "phase": "completion",
         "event_type": "retain_complete", "planner_time_ns": 3000},
    ]
    with open(os.path.join(bridge, "response_1.json"), "w",
              encoding="utf-8") as out:
        json.dump({"nodes": nodes, "watches": watches,
                   "kv_actions": kv_actions, "assignments": []}, out)
    with open(os.path.join(bridge, "graph_batch_digests.jsonl"), "w",
              encoding="utf-8") as out:
        out.write(json.dumps({
            "delivery_sequence": 1, "node_count": len(nodes),
            "watch_count": len(watches), "ranks": [0]}) + "\n")
    with open(os.path.join(bridge, "ledger.jsonl"), "w",
              encoding="utf-8") as out:
        out.write(json.dumps({
            "request_id": "req_0",
            "layers": {"completed_unreconciled": {
                "completed_tick": 3000,
                "admitted_tick": 100,
                "first_commit_tick": 200,
                "stages": [],
            }},
        }) + "\n")
    rows = [
        _sensing_row(1, 1000, issued=["req_0"],
                     groups=[("req_0", "decode", 1, 5, 0, 100, 1)]),
        _sensing_row(2, 2000, issued=[],
                     groups=[("batch_train_t1", "decode", 1, 0,
                              residual_bytes, 0, 1)] + list(extra_row2_groups)),
    ]
    with open(os.path.join(bridge, "sensing_query_log.jsonl"), "w",
              encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    if not omit_train_ledger:
        if train_ledger_rows is None:
            train_ledger_rows = [{
                "train_id": "batch_train_t1", "instance_index": 0,
                "tick": train_first_tick, "iterations": 1,
                "member_count": 1, "member_iterations": 1, "joiners": [],
                "drains": [], "exits": [], "prefill_chunks": 0,
                "pass_spans": 1,
            }]
        with open(os.path.join(bridge, "train_ledger.jsonl"), "w",
                  encoding="utf-8") as out:
            for row in train_ledger_rows:
                out.write(json.dumps(row) + "\n")
    return bridge, manifest_path


def _run_reconcile(bridge, manifest_path):
    """驱动 main() 并返回 (exit_code, stdout 文本)。"""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = ledger_reconcile.main([
            "--bridge-dir", bridge,
            "--manifest", manifest_path,
            "--expected-requests", "1",
            "--expected-accepted-sessions", "1",
        ])
    return code, stdout.getvalue()


class LedgerReconcileSyntheticTest(unittest.TestCase):

    def test_balanced_baseline_blocks_old_sum_caliber(self):
        """平衡基线:列车体 AR + 1 字节 end barrier 共存于同一键,残差
        1 字节对平(旧求和锚 524289 会在此基线上假 FAIL)。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(root)
            code, stdout = _run_reconcile(bridge, manifest_path)
        self.assertEqual(code, 0, stdout)
        self.assertIn("对平 (balanced)", stdout)
        self.assertNotIn("FAIL", stdout)

    def test_r5a_residual_bytes_mismatch(self):
        """残差 comm_bytes 与已提交 end-barrier 字节失配 → R5a 失败。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(root, residual_bytes=2)
            code, stdout = _run_reconcile(bridge, manifest_path)
        self.assertEqual(code, 1, stdout)
        self.assertIn("FAIL: R5a", stdout)
        self.assertIn("expected_barrier_bytes", stdout)

    def test_r7a1_cpp_outside_python_issued(self):
        """真实请求出现在 only-C++ 窗口(同边界 Python issued 无)→
        R7a1 反向包含失败。req_x 的 committed 节点补入响应避免 R5b 噪声。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(
                root,
                extra_row2_groups=[("req_x", "decode", 1, 0, 1, 0, 1)])
            with open(os.path.join(bridge, "response_1.json"),
                      encoding="utf-8") as source:
                response = json.load(source)
            response["nodes"].append({
                "rank": 0, "id": 5, "name": "req_x_decode",
                "request_id": "req_x", "stage": "decode", "generation": 1,
                "coll": None})
            with open(os.path.join(bridge, "response_1.json"), "w",
                      encoding="utf-8") as out:
                json.dump(response, out)
            code, stdout = _run_reconcile(bridge, manifest_path)
        self.assertEqual(code, 1, stdout)
        self.assertIn("FAIL: R7a1", stdout)
        self.assertIn("req_x", stdout)

    def test_r7a3_train_key_without_ledger_row(self):
        """only-C++ 窗口的列车键无 train_ledger 行 → R7a3 失败。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(
                root, omit_train_ledger=True)
            code, stdout = _run_reconcile(bridge, manifest_path)
        self.assertEqual(code, 1, stdout)
        self.assertIn("FAIL: R7a3", stdout)
        self.assertIn("batch_train_t1", stdout)

    def test_r7a3_train_first_seen_before_emission_tick(self):
        """列车键首现 tick 早于台账发射 tick → R7a3 失败。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(
                root, train_first_tick=2500)
            code, stdout = _run_reconcile(bridge, manifest_path)
        self.assertEqual(code, 1, stdout)
        self.assertIn("FAIL: R7a3", stdout)
        self.assertIn("首现 tick", stdout)

    def test_r7a3_first_step_wakeup_key_normalizes(self):
        """SH_FIRST_TOKEN_SPLIT=1 的首步唤醒键("<train_id>_first_step",
        生产形状:stage=prefill、gen=0,见 graph_batch_builder 的
        emit_train_first_step)出现在 only-C++ 窗口、裸 train_id 有台账行
        且首现 tick == 发射 tick → R7a3 归一化后 PASS(修复前该形状恒假
        FAIL "无 train_ledger 行")。唤醒标记节点同步补入 committed 响应
        (R5b 注入峰值上限需要对得上账)。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(
                root,
                train_first_tick=1000,
                extra_row2_groups=[],
            )
            # row 1(tick 1000)追加唤醒键组:首现 tick == 台账发射 tick。
            query_path = os.path.join(bridge, "sensing_query_log.jsonl")
            with open(query_path, encoding="utf-8") as source:
                rows = [json.loads(line) for line in source if line.strip()]
            rows[0]["injected_unfinished"][0]["per_request"].append({
                "request_id": "batch_train_t1_first_step", "stage": "prefill",
                "generation": 0, "node_count": 1, "compute_ops": 0,
                "comm_bytes": 0, "estimated_remaining_ns": 0,
            })
            with open(query_path, "w", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row) + "\n")
            # committed 侧补唤醒标记节点(digest 行数同步)。
            response_path = os.path.join(bridge, "response_1.json")
            with open(response_path, encoding="utf-8") as source:
                response = json.load(source)
            response["nodes"].append({
                "rank": 0, "id": 5,
                "name": "batch_train_t1_first_step_wakeup",
                "request_id": "batch_train_t1_first_step",
                "stage": "prefill", "generation": 0, "coll": None})
            with open(response_path, "w", encoding="utf-8") as out:
                json.dump(response, out)
            digest_path = os.path.join(bridge, "graph_batch_digests.jsonl")
            with open(digest_path, encoding="utf-8") as source:
                digests = [json.loads(line) for line in source if line.strip()]
            digests[0]["node_count"] = len(response["nodes"])
            with open(digest_path, "w", encoding="utf-8") as out:
                for row in digests:
                    out.write(json.dumps(row) + "\n")
            code, stdout = _run_reconcile(bridge, manifest_path)
        self.assertEqual(code, 0, stdout)
        self.assertNotIn("FAIL", stdout)

    def test_report_rendering_writes_nonempty_markdown(self):
        """--report 全门渲染回归锚(2026-10-01 修复:R2b/R4 行曾引用未传入
        _render_report 的 main() 局部名,任何带 --report 的调用 NameError
        崩溃、留下 0 字节报告——七个既有用例都不传 --report,完全掩盖)。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(root)
            report_path = os.path.join(root, "report.md")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = ledger_reconcile.main([
                    "--bridge-dir", bridge,
                    "--manifest", manifest_path,
                    "--expected-requests", "1",
                    "--expected-accepted-sessions", "1",
                    "--report", report_path,
                ])
            self.assertEqual(code, 0, stdout.getvalue())
            self.assertTrue(os.path.isfile(report_path))
            with open(report_path, encoding="utf-8") as source:
                report = source.read()
        self.assertIn("对平(balanced)", report)
        self.assertIn("| R2b", report)
        self.assertIn("| R4", report)
        self.assertNotIn("None", report.split("| R2b")[1].split("|")[1])

    def test_cpp_log_given_but_counters_unparsable_fails_closed(self):
        """--cpp-log 显式给出但计数行不可解析 → R0d fail-closed(修复前
        与"未传 --cpp-log"不可区分地静默跳过,截断日志的崩溃现场会输出
        假绿 exit 0)。"""
        with tempfile.TemporaryDirectory() as root:
            bridge, manifest_path = _write_bridge(root)
            log_path = os.path.join(root, "cpp.log")
            with open(log_path, "w", encoding="utf-8") as out:
                out.write("[online] truncated tail without counter lines\n")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = ledger_reconcile.main([
                    "--bridge-dir", bridge,
                    "--manifest", manifest_path,
                    "--expected-requests", "1",
                    "--expected-accepted-sessions", "1",
                    "--cpp-log", log_path,
                ])
        self.assertEqual(code, 1, stdout.getvalue())
        self.assertIn("FAIL: R0d", stdout.getvalue())
        self.assertIn("service counters", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
