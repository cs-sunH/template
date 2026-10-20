#!/usr/bin/env python3
"""test_d1_sched_fixes.py -- 批次 D1(§3.5 调度核心/桥接层)行为修复钉子。

钉住的行为面(审查报告 §3.5,SCHED-* 为本批条目号):
  - on_commit_ack 消费 ack 的 success 保留字段:显式 false = 对端声称未
    提交,fail-closed raise;缺席/true 放行(兼容不携带该字段的夹具 ack)
    (SCHED-4);
  - verify_run_end 感知分支改断言 injected_unfinished 记录数 ==
    delivery_count:零交付感知运行(如 0-request 冒烟)合法通过,不再被
    "no injected-unfinished history" 误判失败(SCHED-5);
  - verify_run_end 首断言区分"桥 EOF 中断(C++ 中途停止 ack,尾部交付未
    确认)"与真正的 ack 记账失步——诊断文案可区分(SCHED-6);
  - build_static_pd_mapping 空 Prefill 拓扑构造期 fail-closed(此前静默
    返回空映射,消费方 route_for_prefill 才 KeyError)(SCHED-8);
  - decision_bridge 重复 request seq 幂等忽略方向:_handle_request 尊重
    validate_new 的重复判定,不二次调用 handler/不重复记账(SCHED-9);
  - mark_complete 不再携带恒空逐出管道(返回 None;completion 决策行的
    "completion_evictions": [] 字面保留,决策行字节不变)(SCHED-9)。

frontier/busy 失步 fail-closed(SCHED-3)与 decode 等待准入死防御删除
(SCHED-1)的钉子在 test_train_machinery.py(同批改造)。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest -q test_d1_sched_fixes.py   (或 unittest 同路径)
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

_ONLINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "online")
_WORKLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.decision_bridge import BridgeServer  # noqa: E402
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402
from wsc_llm_scheduler import (  # noqa: E402  (红线:只读 import)
    DECODE_ROLE,
    WscLlmHardware,
    WscLlmInstanceSpec,
    build_instances,
    build_static_pd_mapping,
)


def _hardware():
    return WscLlmHardware(
        mesh_rows=1, mesh_cols=6, local_hbm_capacity_bytes=160 << 30,
        local_hbm_bandwidth_gbps=273.0, d2d_bandwidth_gbps=100.0,
        peak_perf_tflops=43.5, d2d_latency_ns=1, local_hbm_latency_ns=20)


def _bare_base(**fields):
    """on_commit_ack / verify_run_end 直接可驱动的最小基类实例。

    生产构造器依赖 manifest/config 全家桶;两条被测路径只触协议簿记
    字段,__new__ + 显式赋值即得精确可控状态(与 test_train_machinery
    的 _bare_scheduler 同款手法)。"""
    s = OnlineSchedulerBase.__new__(OnlineSchedulerBase)
    s.sensing_enabled = False
    s.expected_request_count = 0
    s.unseen_request_ids = set()
    s.in_flight = {}
    s.arrived_request_count = 0
    s.completed_request_count = 0
    s.delivery_count = 0
    s.ack_count = 0
    s.last_applied_sequence = -1
    s._acked_through = -1
    s._acked_out_of_order = set()
    s._delivery_reply_cache = None
    s._batch = None
    s._emitted_by_delivery = {}
    s._provisional_kv_actions = {}
    s.committed_kv_action_batches = 0
    s.ledger_admitted = {}
    s.ledger_committed = {}
    s.ledger_issued = {}
    s.ledger_completed_unreconciled_count = 0
    s.injected_unfinished_record_count = 0
    for name, value in fields.items():
        setattr(s, name, value)
    return s


class AckSuccessFieldTests(unittest.TestCase):
    """SCHED-4:ack 的 success 保留字段被消费(显式 false fail-closed)。"""

    def _ack(self, seq, success=...):
        ack = {
            "schema_version": 1,
            "delivery_sequence": seq,
            "batch_id": seq,
        }
        if success is not ...:
            ack["success"] = success
        return ack

    def _ackable(self):
        return _bare_base(
            last_applied_sequence=0,
            _emitted_by_delivery={0: {"tick": 0, "requests": []}},
        )

    def test_ack_success_false_fails_closed(self):
        s = self._ackable()
        with self.assertRaises(ValueError) as caught:
            s.on_commit_ack(self._ack(0, success=False))
        self.assertIn("success=false", str(caught.exception))
        # fail-closed 前不落任何记账(ack 计数/层转移零变化)。
        self.assertEqual(s.ack_count, 0)
        self.assertIn(0, s._emitted_by_delivery)

    def test_ack_success_true_and_absent_pass(self):
        for ack in (self._ack(0, success=True), self._ack(0)):
            s = self._ackable()
            s.on_commit_ack(ack)
            self.assertEqual(s.ack_count, 1)
            self.assertNotIn(0, s._emitted_by_delivery)
            self.assertEqual(s._acked_through, 0)


class VerifyRunEndSensingTests(unittest.TestCase):
    """SCHED-5/SCHED-6:verify_run_end 感知分支与 EOF 诊断。"""

    def _sensing(self, **fields):
        return _bare_base(sensing_enabled=True, **fields)

    def test_zero_delivery_sensing_run_passes(self):
        """合法零交付感知运行(0-request 冒烟):不再被
        "no injected-unfinished history recorded" 误判失败。"""
        s = self._sensing()  # 全零态:0 交付 0 记录
        s.verify_run_end()   # 不得抛

    def test_injected_history_must_match_deliveries(self):
        """记录数与已应用交付数失衡 = 感知摘要链断裂,fail-closed。"""
        s = self._sensing(
            delivery_count=2, ack_count=2, last_applied_sequence=1,
            _acked_through=1,
            injected_unfinished_record_count=1)
        with self.assertRaises(RuntimeError) as caught:
            s.verify_run_end()
        self.assertIn("injected-unfinished", str(caught.exception))

    def test_eof_gap_diagnostic_names_bridge_eof(self):
        """尾部交付未确认 = ack 流中途断(C++ EOF 中止)——诊断必须可
        与"正常结束"区分,不再误报为记账失步的 protocol mismatch。"""
        s = self._sensing(
            delivery_count=2, ack_count=1, last_applied_sequence=1,
            _acked_through=0)
        with self.assertRaises(RuntimeError) as caught:
            s.verify_run_end()
        message = str(caught.exception)
        self.assertIn("EOF", message)
        self.assertIn("mid-run", message)


class BuildStaticPdMappingBoundaryTests(unittest.TestCase):
    """SCHED-8:空 Prefill 拓扑构造期 fail-closed。

    唯一生产拓扑构造器 build_instances 已强制角色完备("requires at least
    one Prefill and one Decode instance",wsc_llm_scheduler.py),故生产
    不可达;本钉覆盖"手工 WscLlmTopology 直接调 mapping"的报告边界缺口
    (直接绕过 build_instances 的调用方此前得到静默空映射)。"""

    @staticmethod
    def _hand_topology(phase_role):
        from wsc_llm_scheduler import WscLlmInstance, WscLlmTopology
        return WscLlmTopology(
            hardware=_hardware(),
            instances=(WscLlmInstance(
                index=0, ranks=(0,), phase_role=phase_role,
                wafer_center_manhattan_distance=0),),
            adjacency=((),),
        )

    def test_empty_prefill_topology_fails_closed(self):
        topology = self._hand_topology(DECODE_ROLE)
        self.assertEqual(topology.indices_for_role("prefill"), ())
        with self.assertRaises(ValueError) as caught:
            build_static_pd_mapping(topology)
        self.assertIn("Prefill", str(caught.exception))

    def test_decode_empty_topology_also_fails_closed(self):
        """空 Decode 拓扑不得静默(min 对空序列的裸 ValueError 也是
        fail-closed;此处钉其存在性防回归)。"""
        topology = self._hand_topology("prefill")
        with self.assertRaises(ValueError):
            build_static_pd_mapping(topology)


class BridgeDuplicateSkipTests(unittest.TestCase):
    """SCHED-9:重复 request seq 幂等忽略(validate_new 返回值被尊重);
    probe 扫描行为不变(_new_files 仍按 next_seq 起列新文件)。"""

    def setUp(self):
        self._bridge = tempfile.mkdtemp(prefix="d1_bridge_dup.")
        self.server = BridgeServer(self._bridge)

    def tearDown(self):
        self.server._close_audit_streams()
        self.server.discard_per_request_stats()
        shutil.rmtree(self._bridge, ignore_errors=True)

    def _write_request(self, seq):
        with open(os.path.join(
                self._bridge, "request_{}.json".format(seq)),
                "w", encoding="utf-8") as target:
            json.dump(
                {"schema_version": 1, "delivery_sequence": seq},
                target, separators=(",", ":"))

    def _journal_rows(self):
        self.server._close_audit_streams()
        path = os.path.join(self._bridge, "request_journal.jsonl")
        with open(path, "r", encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    def test_duplicate_request_seq_is_skipped_not_reprocessed(self):
        calls = []

        def handler(request):
            calls.append(request["delivery_sequence"])
            return {"batch_id": request["delivery_sequence"],
                    "source_delivery_sequence":
                        request["delivery_sequence"]}

        self._write_request(0)
        self.server._handle_request(0, handler)
        # 重复投递同一 seq(协议下不可达;真到达必须幂等忽略,不得二次
        # 调 handler/二次记账)。
        self.server._handle_request(0, handler)
        self.assertEqual(calls, [0])
        self.assertEqual(len(self._journal_rows()), 1)
        self.assertEqual(self.server._request_sequences.next_seq, 1)

    def test_probe_listing_unchanged(self):
        self._write_request(0)
        self._write_request(1)
        reqs, acks = self.server._new_files()
        self.assertEqual(reqs, [0, 1])
        self.assertEqual(acks, [])
        self.server._handle_request(
            0, lambda request: {"batch_id": 0,
                                "source_delivery_sequence": 0})
        reqs, _ = self.server._new_files()
        self.assertEqual(reqs, [1])


class MarkCompleteNoEvictionPipeTests(unittest.TestCase):
    """SCHED-9:mark_complete 恒空逐出管道已删(返回 None;completion
    决策行的 "completion_evictions": [] 字面保留,行字节不变)。"""

    def test_mark_complete_has_no_eviction_pipe(self):
        import inspect
        from session_kv_manager import SessionKVCacheManager
        source = inspect.getsource(SessionKVCacheManager.mark_complete)
        self.assertNotIn("return ()", source)
        self.assertIn("-> None", source)


if __name__ == "__main__":
    unittest.main()
