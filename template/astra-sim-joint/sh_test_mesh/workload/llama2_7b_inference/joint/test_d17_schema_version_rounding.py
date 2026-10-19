#!/usr/bin/env python3
"""test_d17_schema_version_rounding.py -- D17 schema/取整/版本钉测。

验收（joint修改文档 D17 / v2.0 §18/D17/§19.2）：
  1. **传输取整**：传输函数带宽服务项向上取整（``int(bytes/rate)``
     向下截断废除）——需要等待的服务不向下少报；零字节无事务严格
     为 0（数据腿恒 0；F1"shard 在场即事务发生"的启动固定成本契约
     不变）；恰整除值不漂移（ceil 零漂移锚）。
  2. **版本身份**：``model_semantics_version`` 常量随 run manifest
     （``manifest_dict`` 键）落盘；语义改变必须升级，旧实验不自动
     代表新模型。
  3. **决策 schema**：``ActionCostBreakdown`` 新增字段（决策阶段/
     纪元、两阶段实例/动作配对、继续标记、剩余前向、预测里程碑
     ``predicted_*``、区间布局、继承义务 ID、分阶段峰值空间、暂存
     峰值、近似状态）在 P/D 两入口逐候选填充；直接构造夹具零漂移。
  4. **真实与预测分列**：schema 无任何 measured/actual 真值字段；
     预测里程碑带 ``predicted_`` 前缀且序有序（next ≤ service ≤
     merge = cost 终点）。
  5. **纯披露通道**：``decision_epoch`` / ``inherited_obligation_ids``
     透传（入口 → 选择器 → breakdown）不改变任何候选代价与选择；
     决策日志反序列化（嵌套元组 → list）JSON 可重建。
  6. **近似状态**：深缺口 ``capacity_wait_uncertain`` 结构化披露
     （credit 腿单步口径已随远读退役删除）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d17_schema_version_rounding.py   （或 pytest）
"""
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import (  # noqa: E402
    MODEL_SEMANTICS_VERSION,
    parse_joint_config,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_STAY,
    ActionCostBreakdown,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
    RequestView,
    ServiceFactors,
    SessionKVView,
    _approximation_flags,
    _pool_transfer_ns_shards,
    _session_interval_layout,
    _shard_stream_ns_shards,
    _transfer_ns,
    _transfer_ns_shards,
)
from joint.joint_scheduler import select_decode_target_and_action  # noqa: E402

KV_TOKEN_BYTES = 128


def _route2(source, target):
    return ((source, target), 1)


def _rates(noc=10.0, hbm=100.0, pool=5.0, lat=10, pool_latency=100):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=pool, local_hbm_gbps=hbm,
        d2d_latency_ns=lat, pool_latency_ns=pool_latency)


def _model(loads, rates=None):
    return JointCostModel(
        rates=rates or _rates(), loads=loads,
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route2)


def _load(instance_index, running=0):
    return InstanceLoadView(
        instance_index=instance_index, queued_task_load_ns=0,
        running_task_load_ns=running, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10 ** 9, 10 ** 9),
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


def _loads(running=(0, 0)):
    return {index: _load(index, running=running[index])
            for index in range(2)}


def _loads3(running=(0, 0, 0)):
    return {index: _load(index, running=running[index])
            for index in range(3)}


def _session(resident=1, history_tokens=100, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm"):
    return SessionKVView(
        session_id="s", home_instance=1, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=4, history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25)):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


def _complete_boundary(**overrides):
    values = dict(
        session_id="s", home_instance=0, p_instance_index=0,
        p_action="stay", history_tokens=100,
        working_bytes_by_tp_rank=(KV_TOKEN_BYTES * 100,) * 2,
        decode_steps=8, kv_token_bytes=KV_TOKEN_BYTES)
    values.update(overrides)
    return PDBoundaryView(**values)


# ======================================================== 1. 传输取整 ==


class TransferCeilingTest(unittest.TestCase):
    """D17：带宽服务项向上取整；零字节无事务严格为 0。"""

    def test_transfer_ns_ceils_non_divisible_service(self):
        rates = _rates(noc=10.0, lat=0, pool_latency=0)
        # 1001 B @ 10 B/ns = 100.1 ns：旧 int() = 100（少报），D17
        # 向上取整 = 101。
        self.assertEqual(_transfer_ns(
            total_bytes=1001, path_hops=0, divisor=1, rates=rates,
            per_hop_latency_ns=0, startup_ns=0), 101)

    def test_transfer_ns_exact_division_undrifted(self):
        rates = _rates(noc=10.0, lat=0, pool_latency=0)
        # 恰整除（1000/10 = 100）：ceil 零漂移锚。
        self.assertEqual(_transfer_ns(
            total_bytes=1000, path_hops=0, divisor=1, rates=rates,
            per_hop_latency_ns=0, startup_ns=0), 100)

    def test_transfer_ns_zero_bytes_strictly_zero(self):
        rates = _rates(noc=10.0, lat=0, pool_latency=0)
        # 零字节无事务：带宽项严格为 0（startup/hops 恒 0 时全 0）。
        self.assertEqual(_transfer_ns(
            total_bytes=0, path_hops=0, divisor=1, rates=rates,
            per_hop_latency_ns=0, startup_ns=0), 0)

    def test_shard_legs_zero_bytes_strictly_zero(self):
        rates = _rates()
        # 零字节 shard：三腿服务恒 0（无流即无服务时间）；在场 shard
        # 的启动固定成本照 F1 契约由调用方独立承载。
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(0,),
            include_self=True, kind="copy")
        self.assertEqual(wall, 0)
        self.assertEqual(_shard_stream_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(0,),
            include_self=True, kind="copy"), 0)

    def test_shard_transfer_ceils_and_keeps_hop_separate(self):
        rates = _rates(noc=10.0, hbm=1e9, lat=10)
        # 1001 B @ 10 B/ns：hop 10 + ⌈100.1⌉ = 111（旧 int = 110）。
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(1001,),
            include_self=True, kind="copy")
        self.assertEqual(wall, 10 + 101)
        # 恰整除不漂移：1000 B → 10 + 100。
        self.assertEqual(_transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(1000,),
            include_self=True, kind="copy"), 10 + 100)

    def test_stream_segment_ceils(self):
        rates = _rates(noc=10.0, hbm=1e9, lat=10)
        # 纯流送段：3344 B @ 10 B/ns = 334.4 → ⌈⌉ = 335。
        self.assertEqual(_shard_stream_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(3344,),
            include_self=True, kind="copy"), 335)

    def test_pool_transfer_ceils_and_zero_bytes_strictly_zero(self):
        rates = _rates(noc=10.0, hbm=100.0, pool=5.0, lat=10,
                       pool_latency=100)
        # 池恢复 7 B/rank：时延 100+10 = 110；腿 = max(7/5, 7/10,
        # 7/100) = 1.4 → ⌈⌉ = 2 ⇒ 112（旧 int(1.4)=1 → 111）。
        wall = _pool_transfer_ns_shards(
            rates, LinkFlowRegistry(), None, kind="pool_restore",
            edge_ranks_by_rank=(0,), paths_by_rank=((0, 1),),
            bytes_by_rank=(7,))
        self.assertEqual(wall, 110 + 2)
        # 全零字节向量：无事务严格为 0（不收池时延）。
        self.assertEqual(_pool_transfer_ns_shards(
            rates, LinkFlowRegistry(), None, kind="pool_restore",
            edge_ranks_by_rank=(0,), paths_by_rank=((0, 1),),
            bytes_by_rank=(0,)), 0)


# ======================================================== 2. 版本身份 ==


class ModelSemanticsVersionTest(unittest.TestCase):

    def test_version_registered_in_fixed_config_manifest(self):
        self.assertIsInstance(MODEL_SEMANTICS_VERSION, str)
        self.assertTrue(MODEL_SEMANTICS_VERSION)
        manifest = parse_joint_config({}).manifest_dict()
        self.assertEqual(manifest["model_semantics_version"],
                         MODEL_SEMANTICS_VERSION)

    def test_breakdown_default_version_matches_constant(self):
        # 直接构造夹具零漂移：缺省版本 = 常量（生产路径同源填充）。
        breakdown = ActionCostBreakdown(
            target_wait_ns=0, history_prep_ns=0, eviction_wait_ns=0,
            compute_ns=0, merge_ns=0,
            contention_divisor=1, hops=0)
        self.assertEqual(breakdown.model_semantics_version,
                         MODEL_SEMANTICS_VERSION)
        self.assertEqual(breakdown.decision_stage, "P")
        self.assertEqual(breakdown.decision_epoch, 0)
        self.assertIsNone(breakdown.predicted_merge_done)
        self.assertFalse(breakdown.is_continuation)


# ==================================================== 3. P 入口 schema ==


class PEntrySchemaTest(unittest.TestCase):

    def test_stay_candidate_schema_population(self):
        model = _model(_loads())
        candidate = model.estimate_P_with_continuation(
            session=_session(resident=1), request=_request(),
            instance_index=1, action=ACTION_STAY)
        self.assertTrue(candidate.applicable)
        b = candidate.breakdown
        self.assertEqual(b.model_semantics_version,
                         MODEL_SEMANTICS_VERSION)
        self.assertEqual(b.decision_stage, "P")
        self.assertEqual(b.decision_epoch, 0)
        self.assertEqual(b.prefill_instance, 1)
        self.assertEqual(b.decode_instance, 1)
        self.assertEqual(b.prefill_action, ACTION_STAY)
        # 默认延续 π0：本地接续（stay）。
        self.assertEqual(b.decode_action, ACTION_STAY)
        self.assertFalse(b.is_continuation)
        self.assertEqual(b.estimated_remaining_decode_steps, 10)
        # 预测里程碑：首 token = prefill 首前向；service ≤ merge = cost。
        self.assertIsInstance(b.predicted_next_token_done, int)
        self.assertLessEqual(b.predicted_next_token_done,
                             b.predicted_service_done)
        self.assertEqual(b.predicted_merge_done, candidate.cost_ns)
        self.assertEqual(b.predicted_service_done,
                         candidate.cost_ns - b.merge_ns)
        # 分阶段峰值空间：四阶段 × 逐 rank；decode/merge 同末态。
        self.assertEqual(len(b.phase_space_bytes), 4)
        for phase in b.phase_space_bytes:
            self.assertEqual(len(phase), 2)
        self.assertEqual(b.phase_space_bytes[2], b.phase_space_bytes[3])
        self.assertEqual(b.inherited_obligation_ids, ())
        self.assertEqual(
            b.approximations,
            ("aggregate_service_equivalent_hbm_profile",
             "profile_latency_unavailable"))

    def test_p_entry_interval_layout_describes_decision_state(self):
        model = _model(_loads())
        session = _session(
            resident=1, history_bytes=(1000, 1000), missing=(240, 240),
            location="partial_hbm_remote")
        candidate = model.estimate_P_with_continuation(
            session=session, request=_request(input_tokens=50,
                                              input_bytes=(25, 25)),
            instance_index=1, action=ACTION_STAY)
        layout = _session_interval_layout(
            session, _request(input_tokens=50, input_bytes=(25, 25)))
        by_replica = {entry[0]: entry for entry in layout}
        # 历史驻留区间（holder = resident）+ 缺失后缀（池侧）+ 待算增量。
        self.assertEqual(by_replica["decision_history"][2], 1)
        self.assertEqual(by_replica["decision_history"][4:6], (0, 100))
        self.assertIsNone(by_replica["decision_missing"][2])
        self.assertEqual(by_replica["decision_missing"][6], (240, 240))
        self.assertIsNone(by_replica["decision_increment"][2])
        self.assertEqual(by_replica["decision_increment"][4:6], (100, 150))
        self.assertEqual(candidate.breakdown.interval_layout, layout)

    def test_deep_gap_marks_capacity_wait_uncertain(self):
        tight = InstanceLoadView(
            instance_index=1, queued_task_load_ns=0,
            running_task_load_ns=10 ** 7, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(8, 8),
            reclaimable_bytes_by_tp_rank=(0, 0))
        model = _model({0: _load(0), 1: tight})
        candidate = model.estimate_P_with_continuation(
            session=_session(resident=1), request=_request(),
            instance_index=1, action=ACTION_STAY)
        self.assertIn("capacity_wait_uncertain",
                      candidate.breakdown.approximations)
        self.assertIn(
            "capacity_wait_uncertain",
            _approximation_flags(candidate.breakdown.notes))
        self.assertIn(
            "aggregate_service_equivalent_hbm_profile",
            candidate.breakdown.approximations)
        self.assertIn(
            "profile_latency_unavailable",
            candidate.breakdown.approximations)


# ==================================================== 4. D 入口 schema ==


class DBoundarySchemaTest(unittest.TestCase):

    def test_continuation_schema_population(self):
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY,
            decision_epoch=7, inherited_obligation_ids=("ob-a", "ob-b"))
        self.assertTrue(candidate.is_continuation)
        b = candidate.breakdown
        self.assertEqual(b.decision_stage, "D")
        self.assertEqual(b.decision_epoch, 7)
        self.assertEqual(b.prefill_instance, 0)
        self.assertEqual(b.decode_instance, 0)
        self.assertEqual(b.prefill_action, "stay")
        self.assertEqual(b.decode_action, ACTION_STAY)
        self.assertTrue(b.is_continuation)
        self.assertEqual(b.estimated_remaining_decode_steps, 8)
        # 预测里程碑（t_B 相对轴）：零等待下 next = 单步服务
        # （8 步标量率 2.0 → 单步 2）；merge 0（home == j）⇒
        # service = merge_done = cost。
        self.assertEqual(b.predicted_next_token_done, 2)
        self.assertEqual(b.predicted_merge_done, candidate.cost_ns)
        self.assertEqual(b.predicted_service_done,
                         candidate.cost_ns - b.merge_ns)
        # 区间布局 = 完整型单 context 区间（holder = i_P）。
        self.assertEqual(len(b.interval_layout), 1)
        replica_id, role, holder, _authority, start, end, _bytes = (
            b.interval_layout[0])
        self.assertEqual(role, "context")
        self.assertEqual(holder, 0)
        self.assertEqual((start, end), (0, 100))
        self.assertEqual(b.phase_space_bytes[2],
                         b.phase_space_bytes[3])

    def test_disclosure_channels_do_not_change_pricing(self):
        # W3 升级（§7.2，v4-projection 世代）：``inherited_obligation_ids``
        # 自"纯披露"升级为**实际依赖**——``obligation_resolver`` 在场时
        # 未完 P 义务解析为图节点参竞争（延迟候选/门控 settle）、已完
        # P 义务零腿（不重付沉没费用）；resolver 缺席（无 ledger）按
        # LoadsWorldSource 降级：ID 照常披露、代价不变、登记
        # approximation（不虚构义务腿）。epoch 恒纯披露。
        boundary = _complete_boundary()
        without = [
            model_cost for model_cost in (
                _model(_loads3()).estimate_D_remaining(
                    boundary=boundary, instance_index=index, action=action).cost_ns
                for index in range(3) for action in
                ("stay", "copy"))]
        with_ids = [
            model_cost for model_cost in (
                _model(_loads3()).estimate_D_remaining(
                    boundary=boundary, instance_index=index, action=action,
                    decision_epoch=3,
                    inherited_obligation_ids=("x", "y")).cost_ns
                for index in range(3) for action in
                ("stay", "copy"))]
        self.assertEqual(without, with_ids)
        # 降级登记：resolver 缺席时逐 ID 披露 unresolved 近似。
        degraded = _model(_loads3()).estimate_D_remaining(
            boundary=boundary, instance_index=1, action="copy",
            inherited_obligation_ids=("x",))
        self.assertIn("inherited_obligation_unresolved:x",
                      degraded.breakdown.approximations)
        self.assertEqual(degraded.breakdown.inherited_obligation_ids, ("x",))
        # 真实依赖语义：未完义务参竞争（settle 被门控/资源被占）⇒ 代价
        # 严格上升；已完义务零腿 ⇒ 与无义务基线逐位一致。
        from joint.action_work_graph import ObligationView
        loads = _loads3()

        def _resolver(done: bool):
            table = {
                "ob-live": ObligationView(
                    obligation_id="ob-live", kind="committed_transfer",
                    source_instance=0, target_instance=1,
                    remaining_bytes_by_tp_rank=(4000, 4000),
                    gate_settle=True, done=done),
            }
            return table.get

        base = _model(loads).estimate_D_remaining(
            boundary=boundary, instance_index=1, action="copy").cost_ns
        live_model = _model(loads)
        live_model.obligation_resolver = _resolver(done=False)
        live = live_model.estimate_D_remaining(
            boundary=boundary, instance_index=1, action="copy",
            inherited_obligation_ids=("ob-live",)).cost_ns
        done_model = _model(loads)
        done_model.obligation_resolver = _resolver(done=True)
        settled = done_model.estimate_D_remaining(
            boundary=boundary, instance_index=1, action="copy",
            inherited_obligation_ids=("ob-live",)).cost_ns
        self.assertGreater(live, base)      # 未完 P 义务参竞争
        self.assertEqual(settled, base)     # 已完 P 零腿（沉没不重付）


class SelectorDisclosureThreadingTest(unittest.TestCase):
    """select_decode_target_and_action 披露通道透传（D17）。"""

    def test_threading_reaches_every_breakdown_and_diagnostics(self):
        model = _model(_loads3())
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary(),
            decision_epoch=5, inherited_obligation_ids=("ob-1", "ob-2"))
        self.assertEqual(record.diagnostics["decision_stage"], "D")
        self.assertEqual(record.diagnostics["decision_epoch"], 5)
        for candidate in record.candidates:
            if candidate.breakdown is None:
                continue
            self.assertEqual(candidate.breakdown.decision_epoch, 5)
            self.assertEqual(candidate.breakdown.inherited_obligation_ids,
                             ("ob-1", "ob-2"))
            self.assertEqual(candidate.breakdown.model_semantics_version,
                             MODEL_SEMANTICS_VERSION)

    def test_threading_does_not_change_selection(self):
        def _select(**kwargs):
            return select_decode_target_and_action(
                mode="joint", cost_model=_model(_loads3()),
                boundary=_complete_boundary(),
                **kwargs)
        plain = _select()
        threaded = _select(decision_epoch=5,
                           inherited_obligation_ids=("ob-1",))
        self.assertEqual(plain.chosen.instance_index,
                         threaded.chosen.instance_index)
        self.assertEqual(plain.chosen.action, threaded.chosen.action)
        self.assertEqual(plain.chosen.cost_ns, threaded.chosen.cost_ns)


# ==================================================== 5. 日志可重建 ==


class DecisionLogReconstructionTest(unittest.TestCase):
    """验收：可从日志重建一次候选计算的公式版本、输入来源、字节/空间
    与关键等待——breakdown 全字段序列化（嵌套元组 → list）后 JSON
    round-trip；schema 无实测真值字段（预测/实测严格分列）。"""

    def test_breakdown_log_dict_json_round_trip(self):
        from online.sh30_online_scheduler import _JOINT_BREAKDOWN_LOG_FIELDS
        from online.sh30_online_scheduler import Sh30OnlineScheduler

        model = _model(_loads3())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY,
            decision_epoch=2, inherited_obligation_ids=("ob-9",))
        payload = Sh30OnlineScheduler._joint_breakdown_log_dict(
            None, candidate.breakdown)
        self.assertEqual(sorted(payload),
                         sorted(_JOINT_BREAKDOWN_LOG_FIELDS))
        # JSON round-trip（嵌套元组全部转 list，可解析重建）。
        reparsed = json.loads(json.dumps(payload))
        self.assertEqual(reparsed, json.loads(json.dumps(payload)))
        self.assertIsInstance(reparsed["interval_layout"], list)
        self.assertIsInstance(reparsed["interval_layout"][0], list)
        self.assertEqual(reparsed["interval_layout"],
                         json.loads(json.dumps(
                             candidate.breakdown.interval_layout)))
        self.assertEqual(reparsed["inherited_obligation_ids"], ["ob-9"])
        self.assertEqual(reparsed["model_semantics_version"],
                         MODEL_SEMANTICS_VERSION)
        # 关键等待/里程碑可重建（决策相对轴整数 ns）。
        for key in ("predicted_next_token_done", "predicted_service_done",
                    "predicted_merge_done", "target_wait_ns",
                    "eviction_wait_ns", "merge_ns"):
            self.assertIsInstance(reparsed[key], int)
        # 字节/空间可重建。
        self.assertEqual(len(reparsed["phase_space_bytes"]), 4)

    def test_schema_has_no_measured_truth_fields(self):
        # 真实时间与预测字段严格分列：schema 层无 measured/actual 真值
        # 字段；预测里程碑恰为 predicted_ 三键（非执行候选真值 NA 由决
        # 策行 measured 槽位承载，恒 None——见 online/test_joint_decision_
        # schema.py 全路径钉测）。
        declared = tuple(ActionCostBreakdown.__dataclass_fields__)
        for name in declared:
            self.assertFalse(name.startswith("measured"), name)
            self.assertFalse(name.startswith("actual"), name)
        self.assertEqual(
            {name for name in declared if name.startswith("predicted_")},
            {"predicted_next_token_done", "predicted_service_done",
             "predicted_merge_done"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
