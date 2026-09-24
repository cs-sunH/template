#!/usr/bin/env python3
"""test_joint_arrival_contract.py -- C3b（2026-09-22 到达合同迁移）零后端
回归：下一轮到达锚 merge_done → 响应完成（service_done）+ 到达后数据
依赖门控（保守 merge_done 整体口径）。

被测合同（实验思路 §三.4 / 设计方案 §2.3 / 设计文档 §2.4；F11 到达锚
独立条款——预测终点不动）：

  a(s, k+1) = f(s, k) + z(s, k)

  f(s,k) = 上一轮向外完成响应的时刻（service_done——本仓事件侧 =
  REQUEST_COMPLETE 交付 tick，即 _complete_requests 的 tick）；z = 工具/
  用户等待时间（inter_request_interval_ns）。到达 ≠ 就绪：到达后的数据
  准备仍受真实块就绪与合并依赖门控（图侧 interval gate 前递锚 merge 尾
  标记；**保守口径 = merge_done 整体门控，块级门控留后续，不冒称已块级**）。

覆盖：
  1. 到达 alarm 重排——有/无 merge 流同锚 service_done + interval
     （合同统一适用：到达锚与 merge 流存在性解耦）；
  2. 门控状态机——到达 ≠ 就绪（到达被调度器接受时 merge watch 仍在册）、
     门控解除事件驱动（merge watch 交付 → 流注销 + merge_done 披露行，
     不再排 alarm）、重复/未知 watch 交付 fail-closed；
  3. N2 场景回归（合成列车构造）——thinktime < merge 时长时 merge 成本
     以"下一轮数据等待"形式留在闭环内：既不逃出会话 E2E（到达不被 merge
     推迟到 merge_done + interval 旧锚），也不消失（等待窗口
     [arrival, merge_done] > 0，门控在 merge_done 事件解除）；
  4. 事件侧 service_done 与 merge_done 分别可观测（决策日志 completion
     行 tick = service_done、merge_done 行 tick = 物理 merge_done——
     F11 分报配套 / C12 G5 缺口路由承接）；
  5. 图侧保守门控锚钉（GraphBatchBuilder 只读驱动）——下一轮 interval
     gate 的前递依赖 = merge 尾标记节点（有 merge 流时）；无 merge 流时
     = seg2 块末（不引入额外等待）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_arrival_contract.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from collections import deque
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from face_scheduler import KVTransfer  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineRequestRuntime,
)

SESSION = "session_arrival_0"
R0 = f"{SESSION}_request_0"   # turn 0（本轮完成者）
R1 = f"{SESSION}_request_1"   # turn 1（下一轮，thinktime z 自 R0 响应完成起算）

# N2 合成列车参数（thinktime < merge 时长）：
SERVICE_DONE_NS = 1_000_000   # T：R0 的 REQUEST_COMPLETE 交付 tick
THINKTIME_NS = 500_000        # z：下一轮到达 = T + z
MERGE_DURATION_NS = 2_000_000  # M：merge 传输物理时长（T → T+M 交付）
MERGE_DONE_NS = SERVICE_DONE_NS + MERGE_DURATION_NS
ARRIVAL_NS = SERVICE_DONE_NS + THINKTIME_NS          # 新合同到达 = T + z
OLD_CONTRACT_ARRIVAL_NS = MERGE_DONE_NS + THINKTIME_NS  # 旧合同（merge_done+z）


def _record(request_id, turn_index, queue_index):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": turn_index,
        "queue_index": queue_index,
        "prefill_length": 512,
        "decode_length": 8,
        "history_tokens_before": 0 if turn_index == 0 else 512,
        "prefill_context_tokens": 512,
        "final_context_tokens": 520,
    }


def _runtime(request_id, turn_index, queue_index):
    runtime = _OnlineRequestRuntime(_record(request_id, turn_index,
                                            queue_index), 512)
    runtime.prefill_instance_index = 0
    runtime.decode_instance_index = 0
    return runtime


class _RecordingFlows:
    """R15 在途流登记表的记录替身：只断言注销侧（release_owner）。"""

    def __init__(self):
        self.released = []

    def release_owner(self, owner):
        self.released.append(owner)


class _KvManager:
    """merge_back 可注入合成 merge 流的最小账本替身。"""

    def __init__(self, merge_transfers=()):
        self._sessions = {}
        self._merge_transfers = tuple(merge_transfers)
        self.calls = []
        # 对齐真 KVCacheManager __init__ 初值（F6 销账：软门已删，替身
        # 漏设 = AttributeError；真 manager 每次 merge_back 后覆盖）。
        self.last_merge_outcome = None

    def merge_back(self, **kwargs):
        self.calls.append(("merge_back", kwargs))
        return self._merge_transfers

    def kv_delta_find(self, trigger_request_id):
        # 对齐真 KVCacheManager 接口（F6 销账：C14 闭合门直达调用，替身
        # 漏接口 = AttributeError）。本替身 merge_back 恒"结算"——闭合
        # 门语义 = watch 交付时 journal 行在案，返回合成结算行。
        return {"trigger_request_id": trigger_request_id}

    def observe_completed_input(self, *args, **kwargs):
        pass

    def mark_complete(self, session_id, completion_ns,
                      next_request_type=None):
        self.calls.append(("mark_complete", session_id, completion_ns))

    def session_snapshot(self, session_id):
        return SimpleNamespace(location="local_hbm", instance_index=0)

    def retire_terminal_session(self, *args, **kwargs):
        pass


class _Graph:
    """emit_completion_batch 的记录替身：按 plan 的 merge 流返回
    merge_done_members（真实构图器行为：有 merge 流即有尾标记节点）。"""

    def __init__(self):
        self.synced = []
        self.completion_plans = []

    def sync_pending_history_after_evictions(self, transfers):
        self.synced.append(transfers)

    def emit_completion_batch(self, plan):
        self.completion_plans.append(plan)
        has_merge = bool(plan.get("merge_transfers"))
        return {
            "merge_done_members": {2: 101, 3: 102} if has_merge else {},
            "has_merge": has_merge,
        }


def _merge_transfer_stub():
    # 调度器侧只需"merge 流非空"这一事实；完成行的 _transfer_summary
    # 消费完整字段，故用真实 KVTransfer（零字节零 shard 的 local_hit 形
    # ——节点/计价语义不在调度器侧钉子范围，图侧结构另有
    # IntervalGateAnchorTest 直接驱动真实构图器）。
    return KVTransfer(
        kind="local_hit",
        phase="completion",
        reason="c3b_scheduler_stub",
        session_id=SESSION,
        trigger_request_id=R0,
        source_instance_index=1,
        target_instance_index=0,
        total_bytes=0,
        shards=(),
        model_layers=2,
        layer_start=0,
        layer_end=2,
        resident_prefix_layers_before=0,
        resident_prefix_layers_after=2,
    )


def _bare_scheduler(merge_transfers=()):
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(inter_request_interval_ns=None,
                        next_trigger_type=None),
        SimpleNamespace(inter_request_interval_ns=THINKTIME_NS,
                        next_request_type="human"),
    ])
    scheduler.completed_requests = 0
    scheduler.kv_manager = _KvManager(merge_transfers)
    scheduler.graph = _Graph()
    scheduler._joint_horizon = SimpleNamespace(
        observe_completed=lambda *args, **kwargs: None)
    scheduler._kv_ledger_epoch = 0
    scheduler._stalled_by_instance = {}
    scheduler._joint_flows = _RecordingFlows()
    scheduler._pool_ports = _RecordingFlows()
    # 对齐 __init__ 初值（F6 销账：_release_transfer_flows 直达
    # _hbm_ports.release_owner，替身漏设 = AttributeError）。完成路径
    # 注销断言走 _RecordingFlows 同款记录替身。
    scheduler._hbm_ports = _RecordingFlows()
    scheduler._quota_tracker = None  # off 档 __init__ 初值（F6 销账）
    scheduler._batch = {"future_alarms": [], "watches": []}
    scheduler._pending_merge_alarms = {}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    r0 = _runtime(R0, 0, 0)
    r1 = _runtime(R1, 1, 1)
    scheduler.runtimes = [r0, r1]
    scheduler.runtime_by_request_id = {R0: r0, R1: r1}
    scheduler._runtime_index = {R0: 0, R1: 1}
    scheduler.next_request = {R0: r1, R1: None}
    # 到达批（N2 场景的"到达早于 merge_done"半边）。
    scheduler.arrival_heap = []
    scheduler._sequence = 0
    scheduler._arrived_request_count = 0
    scheduler.pending_admissions = deque()
    scheduler._retry = False
    return scheduler


def _alarms(scheduler):
    return scheduler._batch["future_alarms"]


def _rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


# ----------------------------------------------------- 到达 alarm 重排 --

class ArrivalAnchorTest(unittest.TestCase):
    """步骤 1：下一轮 alarm 锚 = service_done + interval（有/无 merge 同锚）。"""

    def test_alarm_anchored_at_service_done_with_merge(self):
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        alarms = _alarms(scheduler)
        self.assertEqual(len(alarms), 1)
        # 到达 = service_done + thinktime（不是 merge_done + interval）。
        self.assertEqual(alarms[0]["arrival_world_ns"], ARRIVAL_NS)
        self.assertNotEqual(alarms[0]["arrival_world_ns"],
                            OLD_CONTRACT_ARRIVAL_NS)
        self.assertEqual(alarms[0]["envelope"]["request_id"], R1)
        self.assertEqual(
            alarms[0]["envelope"]["inter_request_interval_ns"],
            THINKTIME_NS)
        # merge watch 仍注册（流注销 + 披露义务），但其条目不再携带
        # interval/envelope（alarm 已与 merge 解耦）。
        watch_id = "batch_train_merge_" + R0
        self.assertIn(watch_id, scheduler._pending_merge_alarms)
        pending = scheduler._pending_merge_alarms[watch_id]
        self.assertEqual(pending["request_id"], R0)
        self.assertEqual(pending["next_arrival_world_ns"], ARRIVAL_NS)
        self.assertNotIn("interval", pending)
        self.assertNotIn("envelope", pending)

    def test_alarm_anchored_at_service_done_without_merge(self):
        """无 merge 流（stay/recompute@home 本地提交）同锚——合同统一适用。"""
        scheduler = _bare_scheduler()
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        alarms = _alarms(scheduler)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["arrival_world_ns"], ARRIVAL_NS)
        # 无 merge 流：无 watch 注册。
        self.assertEqual(scheduler._pending_merge_alarms, {})
        self.assertEqual(scheduler._batch["watches"], [])

    def test_terminal_turn_schedules_no_alarm(self):
        """终轮（无 following）：无到达 alarm；K2（P1-②，2026-09-23
        外部审计）后 merge watch **照常注册**（旧边界"终轮不注册"使
        胜者侧 quota 预留滞留到 verify_run_end fail-closed、终轮 merge
        流不进 R15 登记）——pending 记录 following 三字段 None、不排
        到达 alarm，watch 交付走 _on_merge_done 闭合。"""
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        scheduler.next_request[R0] = None
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        self.assertEqual(_alarms(scheduler), [])
        pending = scheduler._pending_merge_alarms
        self.assertEqual(len(pending), 1)
        record = next(iter(pending.values()))
        self.assertIsNone(record["following_request_id"])
        self.assertIsNone(record["next_arrival_world_ns"])
        self.assertEqual(record["request_id"], R0)
        self.assertEqual(
            scheduler.kv_manager.calls.count(
                ("mark_complete", SESSION, SERVICE_DONE_NS)), 1)


# ------------------------------------------------- 门控状态机（步骤 2） --

class GateStateMachineTest(unittest.TestCase):
    """到达 ≠ 就绪；门控解除事件驱动；watch 通道 fail-closed。"""

    def test_arrival_accepted_while_gate_pending(self):
        """到达（T+z，早于 merge_done）被调度器接受——到达 ≠ 就绪：
        到达登记与 merge watch 在册并存，数据就绪由图侧门控承载。"""
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        self.assertEqual(len(_alarms(scheduler)), 1)
        # C++ 在 ARRIVAL_NS 交付到达事件。
        scheduler._push_arrival({"request_id": R1}, ARRIVAL_NS)
        scheduler._drain_arrival_heap(ARRIVAL_NS)
        r1 = scheduler.runtime_by_request_id[R1]
        self.assertEqual(r1.estimated_arrival_ns, ARRIVAL_NS)
        self.assertIn(r1, scheduler.pending_admissions)
        # merge watch 仍在册：就绪未解除（门控等待真实 merge 尾标记交付）。
        self.assertIn("batch_train_merge_" + R0,
                      scheduler._pending_merge_alarms)

    def test_gate_release_is_event_driven_and_schedules_no_alarm(self):
        """merge watch 交付（T+M）：流注销 + merge_done 披露行；**不再排
        下一轮 alarm**（到达已在响应完成点排定）——alarm 数恒 1。"""
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        before = len(_alarms(scheduler))
        scheduler._on_merge_done("batch_train_merge_" + R0, MERGE_DONE_NS)
        self.assertEqual(len(_alarms(scheduler)), before)   # 无新 alarm
        self.assertEqual(scheduler._pending_merge_alarms, {})
        self.assertIn(R0 + "#merge", scheduler._joint_flows.released)
        self.assertIn(R0 + "#merge", scheduler._pool_ports.released)

    def test_duplicate_and_unknown_watch_fail_closed(self):
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        watch_id = "batch_train_merge_" + R0
        scheduler._on_merge_done(watch_id, MERGE_DONE_NS)
        with self.assertRaisesRegex(RuntimeError, "without a pending entry"):
            scheduler._on_merge_done(watch_id, MERGE_DONE_NS + 1)
        with self.assertRaisesRegex(RuntimeError, "without a pending entry"):
            scheduler._on_merge_done("batch_train_merge_unknown",
                                     MERGE_DONE_NS)


# --------------------------------------------- N2 场景回归（步骤 3） --

class N2ThinktimeShorterThanMergeTest(unittest.TestCase):
    """合成列车构造：thinktime(z) < merge 时长(M)——merge 成本必须以
    "下一轮数据等待"形式出现在闭环内（2026-09-14 N2 修复关切在新合同下
    的承接）。"""

    def test_merge_cost_stays_as_next_round_data_wait(self):
        self.assertLess(THINKTIME_NS, MERGE_DURATION_NS)  # N2 构造前提
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        scheduler._complete_requests([R0], SERVICE_DONE_NS)

        # (1) 不逃出：到达 = T + z（旧合同 merge_done + z 会把 merge 成本
        # 推进到达时刻之外——本断言钉死到达锚）。
        alarms = _alarms(scheduler)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["arrival_world_ns"], ARRIVAL_NS)

        # (2) 到达早于 merge_done：等待窗口存在（数据就绪未到）。
        scheduler._push_arrival({"request_id": R1}, ARRIVAL_NS)
        scheduler._drain_arrival_heap(ARRIVAL_NS)
        self.assertEqual(
            scheduler.runtime_by_request_id[R1].estimated_arrival_ns,
            ARRIVAL_NS)

        # (3) 不消失：门控在 merge_done 事件解除（T+M），等待窗口
        # [arrival, merge_done] = M - z > 0 落在下一轮 E2E 内（下一轮
        # prefill 经图侧 interval gate 前递锚 merge 尾标记，物理起点
        # ≥ merge_done——见 IntervalGateAnchorTest）。
        scheduler._on_merge_done("batch_train_merge_" + R0, MERGE_DONE_NS)
        merge_rows = _rows(scheduler, "merge_done")
        self.assertEqual(len(merge_rows), 1)
        row = merge_rows[0]
        self.assertEqual(row["tick"], MERGE_DONE_NS)
        self.assertEqual(row["request_id"], R0)
        decision = row["decision"]
        self.assertEqual(decision["merge_done_ns"], MERGE_DONE_NS)
        self.assertEqual(decision["next_turn_request_id"], R1)
        self.assertEqual(decision["next_arrival_world_ns"], ARRIVAL_NS)
        # N2 可观测标志：到达早于 merge_done（等待窗口存在且被披露）。
        self.assertTrue(decision["next_turn_arrived_before_merge_done"])
        expected_wait = MERGE_DONE_NS - ARRIVAL_NS   # = M - z
        self.assertGreater(expected_wait, 0)

    def test_service_done_and_merge_done_separately_observable(self):
        """F11 分报配套（C12 G5 缺口路由承接）：completion 行 tick =
        service_done（REQUEST_COMPLETE 事件侧），merge_done 行 tick =
        物理 merge_done——两事件行分别可观测且 tick 不同。"""
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        scheduler._on_merge_done("batch_train_merge_" + R0, MERGE_DONE_NS)
        completion_rows = _rows(scheduler, "completion")
        merge_rows = _rows(scheduler, "merge_done")
        self.assertEqual(len(completion_rows), 1)
        self.assertEqual(len(merge_rows), 1)
        self.assertEqual(completion_rows[0]["tick"], SERVICE_DONE_NS)
        self.assertEqual(merge_rows[0]["tick"], MERGE_DONE_NS)
        self.assertNotEqual(completion_rows[0]["tick"], merge_rows[0]["tick"])

    def test_thinktime_longer_than_merge_hides_wait(self):
        """对照半边：z ≥ M 时 merge_done ≤ 到达，门控无残余等待（合并成本
        被思考时间完全掩盖——新合同下这是合法重叠，不是消失）。"""
        scheduler = _bare_scheduler(merge_transfers=(_merge_transfer_stub(),))
        # 用更长的 thinktime 复用同一完成路径：改 r1 的 interval 来源。
        scheduler.config.request_queue[1] = SimpleNamespace(
            inter_request_interval_ns=MERGE_DURATION_NS * 2,
            next_request_type="human")
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        arrival = SERVICE_DONE_NS + MERGE_DURATION_NS * 2
        self.assertEqual(_alarms(scheduler)[0]["arrival_world_ns"], arrival)
        scheduler._on_merge_done("batch_train_merge_" + R0, MERGE_DONE_NS)
        row = _rows(scheduler, "merge_done")[0]
        # 到达晚于 merge_done：无数据等待窗口。
        self.assertFalse(row["decision"]["next_turn_arrived_before_merge_done"])


# --------------------------- 图侧保守门控锚钉（真实构图器，只读驱动） --

class IntervalGateAnchorTest(unittest.TestCase):
    """到达后数据依赖门控的图侧锚：有 merge 流时下一轮 interval gate 的
    前递依赖 = merge 尾标记节点（保守 merge_done 整体口径）；无 merge 流
    时 = seg2 块末（响应完成，不引入额外等待）。"""

    @staticmethod
    def _config():
        return SimpleNamespace(
            npus_count=4,
            remote_operand_loads=False,
            trace_granularity="request_aggregated",
            inference_groups=[
                SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
                SimpleNamespace(ranks=(2, 3), pg_name="tp_decode"),
            ],
            layers=2,
            hidden_size=64,
            ffn_size=128,
            vocab_size=256,
            bytes_per_elem=2,
            num_heads=8,
            mlp_variant="gelu",
            request_queue=[
                SimpleNamespace(session_arrival_time_ns=0,
                                inter_request_interval_ns=None),
                SimpleNamespace(session_arrival_time_ns=None,
                                inter_request_interval_ns=1000),
            ],
        )

    @staticmethod
    def _merge_transfer():
        # local_hit 形零节点载荷：本测试只考察"merge 尾标记节点存在 +
        # interval gate 锚在其上"的结构，字节级传输语义不在钉子范围。
        return KVTransfer(
            kind="local_hit",
            phase="completion",
            reason="c3b_anchor_test",
            session_id=SESSION,
            trigger_request_id=R0,
            source_instance_index=1,
            target_instance_index=0,
            total_bytes=0,
            shards=(),
            model_layers=2,
            layer_start=0,
            layer_end=2,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=2,
        )

    @staticmethod
    def _completion_plan(merge_transfers):
        return {
            "request_id": R0,
            "session_id": SESSION,
            "turn_index": 0,
            "queue_index": 0,
            "decode_instance_index": 1,
            "completion_evictions": [],
            "kv_location_after_completion": "local_hbm",
            "merge_transfers": list(merge_transfers),
        }

    @staticmethod
    def _emit(builder, plan):
        builder.set_next_plan({R0: R1, R1: None})
        builder.set_plan_resolver(lambda request_id: {
            "request_id": request_id, "queue_index": 1, "hbm_wait_ns": 0})
        builder._block_ends[R0] = {"seg2": {2: 0, 3: 0}}
        builder.begin_batch()
        return builder.emit_completion_batch(plan)

    def test_gate_anchored_on_merge_done_markers(self):
        builder = GraphBatchBuilder(self._config())
        result = self._emit(builder, self._completion_plan(
            [self._merge_transfer()]))
        members = result["merge_done_members"]
        self.assertEqual(sorted(members), [2, 3])   # 每 decode rank 一标记
        gate = builder.pending_history[R1]
        edges = builder.batch["parent_edges"]
        for timer_id, (rank, member_id) in zip(
                gate.timer_gates, sorted(members.items())):
            # 下一轮 gate 的前递依赖 = 本 rank 的 merge 尾标记：到达后
            # 数据准备（经 gate armed 的历史迁移/恢复链）等待真实
            # merge_done——保守整体门控的结构钉。
            self.assertIn(
                (member_id, timer_id),
                {(edge["from"], edge["to"]) for edge in edges
                 if edge["rank"] == rank})

    def test_gate_anchored_on_decode_block_ends_without_merge(self):
        builder = GraphBatchBuilder(self._config())
        # seg2 块末账本条目在完成发射中被消费——先取快照再发射。
        builder._block_ends[R0] = {"seg2": {2: 0, 3: 0}}
        seg2 = dict(builder._block_ends[R0]["seg2"])
        result = self._emit(builder, self._completion_plan(()))
        self.assertEqual(result["merge_done_members"], {})
        gate = builder.pending_history[R1]
        edges = builder.batch["parent_edges"]
        for timer_id, (rank, _) in zip(
                gate.timer_gates, sorted(seg2.items())):
            self.assertIn(
                (seg2[rank], timer_id),
                {(edge["from"], edge["to"]) for edge in edges
                 if edge["rank"] == rank})


if __name__ == "__main__":
    unittest.main()
