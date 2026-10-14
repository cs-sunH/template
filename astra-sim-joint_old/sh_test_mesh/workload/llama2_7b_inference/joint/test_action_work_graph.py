#!/usr/bin/env python3
"""test_action_work_graph.py -- W3（v4-projection 世代）四动作预测工作图
构造的零后端单测（joint/action_work_graph.py）。

覆盖（《joint仓库当下+历史修改方案》§5.2/§6.2/§6.3/§7.1/§7.2/§7.4）：

* **四动作节点集**（§6.2 表）：stay（本地零传输/PARTIAL 池后缀恢复腿）、
  copy（缺失区间 NoC chunk 腿 + 池后缀/整份池恢复；不重传已有副本）、
  remote-read（fill/consume 腿族 + 2 槽 buffer 门 + 首块发射约束）、
  recompute（无传输腿，重建进计算段）；
* **A/B/C 例字节核账**（§6.3 表）：remote_effective = n̂×Σmissing、
  copy_effective = Σmissing（有效载荷核账非完成时间公式）；
  ``decode_context_weighted_tokens`` 式(18b) 守恒闭式保留；
* **参考接续**（§5.2/§7.1）：P copy/recompute 已物化不重搬（decode 腿
  本地、无第二次搬运腿）、P remote 留下的源端访问仍计入（decode
  fill/consume）；**不提交真实图**（图仅预测副本——构造无任何注册表/
  账本副作用）；
* **控制/阶段延迟节点**（§7.1）：hardware profile 的 HBM startup 只附着
  对应首个 compute leg，controller latency 由每候选独立 post leg 承担；
  缺省/aggregate profile 的相位折叠状态显式披露；
* **distributed 显式拒绝**（§7.4：predict/settle 不静默）；
* **WorldSource**：LoadsWorldSource（注册表计数 → unknown-ETA in-service
  背景腿，progress_source=inferred）/ LedgerWorldSource（占位
  fail-closed——不静默空背景）；
* **§7.2 继承义务**：未完 P 义务参竞争（延迟 settle）、已完 P 零腿。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_action_work_graph.py（或 pytest）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.action_work_graph import (  # noqa: E402
    ActionWorkGraphError,
    LedgerWorldSource,
    LoadsWorldSource,
    ObligationView,
    build_d_candidate,
    build_p_candidate,
    hbm_resource_key,
    link_resource_key,
    pool_resource_key,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_RECOMPUTE,
    ACTION_REMOTE,
    ACTION_STAY,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
    RequestView,
    ServiceFactors,
    SessionKVView,
    decode_context_weighted_tokens,
)
from joint.prediction_config import (  # noqa: E402
    FrozenDiagnosticSpec,
    JointPredictionConfig,
    PredictionConfigError,
    PredictionParams,
    SETTLEMENT_DISTRIBUTED,
)
from joint.resource_projection import (  # noqa: E402
    GateSpec,
    ResourceSpec,
    TASK_KIND_COMPUTE,
    TASK_KIND_FIXED,
    TASK_KIND_STREAM,
)

KV_TOKEN_BYTES = 128


def _rates(**kw):
    base = dict(noc_link_gbps=10.0, pool_port_gbps=5.0,
                local_hbm_gbps=100.0, d2d_latency_ns=10,
                pool_latency_ns=100)
    base.update(kw)
    return JointHardwareRates.from_gbps(**base)


def _load(index, queued=0, running=0, active=0,
          remaining=(10 ** 9, 10 ** 9)):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=queued,
        running_task_load_ns=running, active_decode_task_load_ns=active,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


def _loads3():
    return {index: _load(index) for index in range(3)}


def _session(location="local_hbm", home=0, resident=0,
             hist=(1000, 1000), miss=(0, 0), prefix=4, tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=hist, missing_bytes_by_tp_rank=miss)


def _request(inp=50, decode=10, ibytes=(25, 25)):
    return RequestView(
        request_id="r", session_id="s", input_tokens=inp,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=ibytes)


def _route(source, target):
    return ((source, target), 1)


def _disjoint(source, target):
    return ((2 * source, 2 * target),
            (2 * source + 1, 2 * target + 1))


def _model(loads=None, *, rt=None, hbm_registry=None, link_registry=None,
           service_factors=None):
    return JointCostModel(
        rates=rt if rt is not None else _rates(),
        loads=loads if loads is not None else _loads3(),
        flow_registry=link_registry if link_registry is not None
        else LinkFlowRegistry(),
        service_factors=(service_factors if service_factors is not None
                         else ServiceFactors()),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        route_paths_fn=_disjoint,
        hbm_port_registry=hbm_registry)


def _p_graph(model, action, *, session=None, request=None, config=None):
    return build_p_candidate(
        model,
        session=session if session is not None else _session(),
        request=request if request is not None else _request(),
        instance_index=1, action=action,
        prediction_config=config if config is not None
        else JointPredictionConfig())


def _boundary(kind="complete"):
    values = dict(
        session_id="s", home_instance=0, p_instance_index=0,
        p_action="stay", history_tokens=100,
        working_bytes_by_tp_rank=(KV_TOKEN_BYTES * 100,) * 2,
        base_bytes_by_tp_rank=(0, 0), decode_steps=8,
        base_tokens=100, kv_token_bytes=KV_TOKEN_BYTES)
    if kind == "increment":
        values.update(
            home_instance=2, p_action="remote-read",
            working_bytes_by_tp_rank=(KV_TOKEN_BYTES * 40,) * 2,
            base_bytes_by_tp_rank=(KV_TOKEN_BYTES * 100,) * 2)
    return PDBoundaryView(**values)


def _d_graph(model, action, *, boundary=None, config=None, **kwargs):
    return build_d_candidate(
        model,
        boundary=boundary if boundary is not None else _boundary(),
        instance_index=1, action=action,
        prediction_config=config if config is not None
        else JointPredictionConfig(), **kwargs)


def _leg_kinds(graph):
    return sorted({leg.kind for leg in graph.work.legs})


def _tags(graph):
    return [leg.leg_id.split("#", 1)[1] for leg in graph.work.legs]


# ==================================================== 四动作节点集 ==


class ActionNodeSetTest(unittest.TestCase):
    """§6.2 表：逐动作必要内容（图形态）与不允许的简化。"""

    def test_stay_local_has_no_transfer_legs(self):
        graph = _p_graph(_model(), ACTION_STAY,
                         session=_session(home=1, resident=1))
        self.assertEqual(
            [leg for leg in graph.work.legs
             if leg.kind == TASK_KIND_STREAM], [])
        # TP barrier 在场（逐层同步）。
        self.assertTrue(graph.barriers)

    def test_stay_partial_has_pool_restore_legs(self):
        graph = _p_graph(
            _model(), ACTION_STAY,
            session=_session(location="partial_hbm_remote", home=1,
                             resident=1, miss=(1000, 1000), prefix=2))
        pool_legs = [
            leg for leg in graph.work.legs
            if leg.kind == TASK_KIND_STREAM
            and any(key.startswith("pool:") for key in leg.route)]
        self.assertEqual(len(pool_legs), 2)   # 每 rank 一腿（式 7 资源集）
        for leg in pool_legs:
            self.assertEqual(leg.latency_ns,
                             _rates().pool_latency_ns)  # 池时延（本地退化无跳）
            self.assertTrue(any(key.startswith("hbm:")
                                for key in leg.route))  # 目标写腿在场

    def test_copy_prefix_chunks_and_pool_suffix(self):
        graph = _p_graph(
            _model(), ACTION_COPY,
            session=_session(location="partial_hbm_remote", home=0,
                             resident=0, hist=(1000, 1000),
                             miss=(1000, 1000), prefix=2))
        chunk_legs = [
            leg for leg in graph.work.legs
            if leg.kind == TASK_KIND_STREAM
            and any(key.startswith("link:") for key in leg.route)]
        pool_legs = [
            leg for leg in graph.work.legs
            if leg.kind == TASK_KIND_STREAM
            and any(key.startswith("pool:") for key in leg.route)]
        self.assertEqual(len(chunk_legs), 2)  # NoC 前缀：每 rank 一腿
        self.assertEqual(len(pool_legs), 2)   # 池后缀：每 rank 一腿
        # 通道内串行链：chunk 腿依赖 space marker（决策时刻发射）。
        for leg in chunk_legs:
            self.assertTrue(leg.deps)

    def test_remote_read_has_double_buffer_legs(self):
        graph = _p_graph(_model(), ACTION_REMOTE)
        fills = [leg for leg in graph.work.legs if "#F" in leg.leg_id]
        consumes = [leg for leg in graph.work.legs if "#C" in leg.leg_id]
        self.assertTrue(fills)
        self.assertEqual(len(fills), len(consumes))
        # 2 槽 buffer 准入门（≤2 块暂存恒成立——D09 锚）。
        buffer_gates = [g for g in graph.gates if g.kind == "buffer"]
        self.assertTrue(buffer_gates)
        for gate in buffer_gates:
            self.assertEqual(gate.capacity_units, 2)
        # fill 取槽 / consume 释放（慢消费不无限泵出）。
        self.assertTrue(all(leg.acquires for leg in fills))
        self.assertTrue(all(leg.releases for leg in consumes))
        # 首块发射约束：fill[1] 依赖 fill[0]（A_1 = max(P, T_1)）。
        by_rank = {}
        for leg in fills:
            rank = leg.leg_id.split("#R")[1][0]
            by_rank.setdefault(rank, []).append(leg)
        for legs in by_rank.values():
            ordered = sorted(legs, key=lambda l: l.leg_id)
            if len(ordered) > 1:
                self.assertIn(ordered[0].leg_id, ordered[1].deps)

    def test_recompute_has_no_transfer_legs(self):
        graph = _p_graph(_model(), ACTION_RECOMPUTE)
        self.assertEqual(
            [leg for leg in graph.work.legs
             if leg.kind == TASK_KIND_STREAM], [])
        # 重建进计算段（rebuild_context_tokens 披露）。
        self.assertIn("recompute_history_in_compute", graph.audits["notes"])


# ================================================ A/B/C 例字节核账 ==


class ABCByteAccountingTest(unittest.TestCase):
    """§6.3 表：remote/copy 有效载荷核账 + 式(18b) 守恒闭式。"""

    def test_remote_effective_is_steps_times_missing_union(self):
        # A 例（D 实例 = A）：remote 的外部片上来源 = B 的输入 KV；
        # remote_effective = n̂ × Σmissing（每 D 步读一遍远端基础）。
        graph = _d_graph(_model(), ACTION_REMOTE,
                         boundary=_boundary("complete"))
        base_total = sum(_boundary("complete").working_bytes_by_tp_rank)
        self.assertEqual(graph.audits["remote_effective_bytes"],
                         8 * base_total)

    def test_copy_effective_is_missing_union_once(self):
        # B 例（D 实例 = B）：copy 的新增持久字节 = A→B 的缺失区间
        # （一次性搬运，Σmissing——非 n̂ 倍）。
        graph = _d_graph(_model(), ACTION_COPY)
        base_total = sum(_boundary("complete").working_bytes_by_tp_rank)
        self.assertEqual(graph.audits["copy_effective_bytes"], base_total)
        remote = _d_graph(_model(), ACTION_REMOTE)
        self.assertEqual(
            remote.audits["remote_effective_bytes"],
            8 * graph.audits["copy_effective_bytes"])

    def test_multi_source_c_reads_both_real_sources(self):
        # C 例（D 实例 = C）：增量型两真实来源（A 历史 + B 输入），
        # 逐区间逐遍读、不漏任一来源。
        graph = _d_graph(_model(), ACTION_REMOTE,
                         boundary=_boundary("increment"))
        notes = graph.audits["notes"]
        self.assertIn("boundary_remote_sources=0,2", notes)
        union = (KV_TOKEN_BYTES * 100 + KV_TOKEN_BYTES * 40) * 2
        self.assertEqual(
            int(next(n for n in notes
                     if n.startswith("boundary_remote_union_bytes="))
                .split("=")[1]),
            union)

    def test_decode_context_weighted_tokens_conservation(self):
        # 式(18b) 整数闭式（守恒锚，不经任何折叠/事件近似）。
        for base, steps in ((150, 10), (100, 8), (7, 1)):
            self.assertEqual(
                decode_context_weighted_tokens(base, steps),
                steps * base + steps * (steps + 1) // 2)


# ==================================================== 参考接续 ==


class ReferenceContinuationTest(unittest.TestCase):
    """§5.2/§7.1：P 参考接续（预测副本，不提交真实图）。"""

    def test_copy_materialized_not_re_moved(self):
        # P=copy：历史已搬目标 ⇒ decode 腿为本地 compute/HBM 腿（无第二次
        # 搬运/读流腿）。
        graph = _p_graph(_model(), ACTION_COPY)
        decode_legs = [leg for leg in graph.work.legs
                       if "#dec#" in leg.leg_id]
        self.assertTrue(decode_legs)
        self.assertTrue(all(
            leg.kind == TASK_KIND_COMPUTE for leg in decode_legs))
        self.assertTrue(all(leg.route and leg.bytes_amount > 0
                            for leg in decode_legs))
        self.assertFalse(any(
            "#F" in leg.leg_id for leg in graph.work.legs))

    def test_remote_continuation_keeps_source_access(self):
        # P=remote：decode 延续远读同一区间（credit fill/consume 腿
        # 在场——"P remote 留下的多源访问仍计入参考 D"）。
        graph = _p_graph(_model(), ACTION_REMOTE)
        self.assertTrue(any(
            "#dcstream" in leg.leg_id for leg in graph.work.legs))

    def test_graph_construction_has_no_registry_side_effect(self):
        registry = LinkFlowRegistry()
        model = _model(link_registry=registry)
        _p_graph(model, ACTION_COPY)
        _d_graph(model, ACTION_REMOTE)
        self.assertEqual(registry.snapshot(), {})   # 不提交真实图


class ComputeProfileConservationTest(unittest.TestCase):
    """Profile bytes stay physical while efficiency changes FLOP floors."""

    def test_service_factor_does_not_scale_profile_bytes(self):
        model = _model(service_factors=ServiceFactors(
            prefill_factor=2.0, decode_factor=3.0))
        model.prefill_pure_compute_ns_fn = lambda _tokens, _history: 10
        model.prefill_hbm_bytes_fn = lambda _tokens, _history: 100
        model.decode_pure_compute_ns_fn = lambda _context: 7
        model.decode_hbm_bytes_fn = lambda _context: 70
        graph = _p_graph(model, ACTION_STAY)
        prefill = [leg for leg in graph.work.legs
                   if leg.kind == TASK_KIND_COMPUTE
                   and leg.group == "prefill"]
        decode = [leg for leg in graph.work.legs
                  if leg.kind == TASK_KIND_COMPUTE
                  and leg.group == "decode"]
        self.assertEqual(sum(leg.bytes_amount for leg in prefill), 200)
        self.assertEqual(sum(leg.bytes_amount for leg in decode), 1400)
        self.assertEqual(sum(int(leg.base_ns) for leg in prefill), 40)
        self.assertEqual(sum(int(leg.base_ns) for leg in decode), 420)

    def test_fixed_latency_changes_prediction_without_changing_work_profile(self):
        """Hardware fixed legs must affect an isolated prediction.

        Keep the pure FLOP and HBM callbacks identical while changing only
        the phase latency profile.  The graph should charge the startup once
        on the first compute event and the controller as one folded fixed
        phase leg, so the projection cost moves without inventing bytes.
        """
        model = _model()
        model.prefill_pure_compute_ns_fn = lambda _tokens, _history: 100
        model.prefill_hbm_bytes_fn = lambda _tokens, _history: 100
        model.decode_pure_compute_ns_fn = lambda _context: 100
        model.decode_hbm_bytes_fn = lambda _context: 100
        model.prefill_latency_profile_fn = lambda _tokens, _history: (
            0, 0)
        model.decode_latency_profile_fn = lambda _context: (0, 0)
        session = _session(home=1, resident=1)
        request = _request()
        base = model.estimate_P_with_continuation(
            session=session, request=request, instance_index=1,
            action=ACTION_STAY, remote_enabled=True)
        base_graph = _p_graph(model, ACTION_STAY,
                              session=session, request=request)
        base_work = tuple(
            (leg.base_ns, leg.bytes_amount)
            for leg in base_graph.work.legs
            if leg.kind == TASK_KIND_COMPUTE)

        model.prefill_latency_profile_fn = lambda _tokens, _history: (
            11, 13)
        model.decode_latency_profile_fn = lambda _context: (17, 19)
        delayed = model.estimate_P_with_continuation(
            session=session, request=request, instance_index=1,
            action=ACTION_STAY, remote_enabled=True)
        delayed_graph = _p_graph(model, ACTION_STAY,
                                 session=session, request=request)
        delayed_work = tuple(
            (leg.base_ns, leg.bytes_amount)
            for leg in delayed_graph.work.legs
            if leg.kind == TASK_KIND_COMPUTE)

        self.assertEqual(base_work, delayed_work)
        self.assertGreater(delayed.cost_ns, base.cost_ns)
        self.assertIn("aggregate_phase_latency_folded",
                      delayed_graph.approximations)
        startup = [
            leg for leg in delayed_graph.work.legs
            if leg.kind == TASK_KIND_COMPUTE and leg.group == "prefill"
        ]
        self.assertEqual(max(leg.latency_ns for leg in startup), 11)
        self.assertEqual(sum(leg.latency_ns for leg in startup), 2 * 11)
        controller = next(
            leg for leg in delayed_graph.work.legs
            if leg.leg_id.endswith("#controller"))
        self.assertEqual(controller.duration_ns, 13 + 10 * 19)


# ==================================================== 控制延迟节点 ==


class ControlLatencyNodeTest(unittest.TestCase):
    """§7.1：目标系统控制延迟由所有 D 候选承担。"""

    def _ctrl_leg(self, graph):
        return next(leg for leg in graph.work.legs
                    if leg.leg_id.endswith("#ctrl"))

    def test_nonzero_control_latency_is_rejected(self):
        # 执行侧尚未实现延迟提交事件，配置门拒绝非零值；登记标记不能
        # 把预测侧延迟伪装成已兑现的执行语义。
        with self.assertRaises(PredictionConfigError):
            PredictionParams(control_latency_ns=500)

    def test_zero_control_latency_is_borne_by_all_d_candidates(self):
        # 零控制延迟仍在所有 D 候选中以显式零节点披露。
        config = JointPredictionConfig(params=PredictionParams())
        for action in (ACTION_STAY, ACTION_COPY, ACTION_REMOTE,
                       ACTION_RECOMPUTE):
            graph = _d_graph(_model(), action, config=config,
                             boundary=_boundary())
            leg = self._ctrl_leg(graph)
            self.assertEqual(leg.duration_ns, 0.0)
            join = next(l for l in graph.work.legs
                        if l.leg_id.endswith("#trainjoin"))
            self.assertIn(leg.leg_id, join.deps)

    def test_zero_control_latency_disclosed(self):
        graph = _d_graph(_model(), ACTION_STAY)
        self.assertEqual(graph.audits["control_latency_ns"], 0)
        # assumed-zero 也披露（§9.2 末行）——manifest 侧由
        # prediction_config.params_manifest 承担（control_latency_source）。
        params = PredictionParams()
        self.assertEqual(
            params.params_manifest()["control_latency_source"],
            "assumed-zero")


# ============================================ distributed 结算（W9 §7.4） ==


class DistributedSettlementPricingTest(unittest.TestCase):
    """W9（§7.4）：distributed = 目录提交型结算——预测侧 settle 腿零载荷
    （无物理 merge 腿；settle_done = service_done 同刻闭合），consolidate
    路径不受影响。旧"distributed 显式拒绝"（W3/W5 双侧同拒）随 W9 交付
    退役——直接替换，无并存开关。"""

    def test_p_graph_distributed_skips_merge_legs(self):
        config = JointPredictionConfig(
            settlement_policy=SETTLEMENT_DISTRIBUTED)
        graph = _p_graph(_model(), ACTION_STAY, config=config)
        notes = ";".join(graph.audits.get("notes", ()))
        self.assertIn(
            "settlement=distributed:directory-commit-no-merge-payload",
            notes)
        # 零 merge 载荷：predicted merge audit 恒 0。
        self.assertEqual(graph.audits.get("merge_ns", 0), 0)

    def test_d_graph_distributed_skips_merge_legs(self):
        config = JointPredictionConfig(
            settlement_policy=SETTLEMENT_DISTRIBUTED)
        graph = _d_graph(_model(), ACTION_STAY, config=config)
        notes = ";".join(graph.audits.get("notes", ()))
        self.assertIn(
            "settlement=distributed:directory-commit-no-merge-payload",
            notes)
        self.assertEqual(graph.audits.get("merge_ns", 0), 0)

    def test_consolidate_default_unchanged(self):
        # 缺省 consolidate：distributed 注记缺席（零漂移断言）。
        graph = _p_graph(_model(), ACTION_STAY, config=None)
        notes = ";".join(graph.audits.get("notes", ()))
        self.assertNotIn(
            "settlement=distributed:directory-commit-no-merge-payload",
            notes)


# ==================================================== WorldSource ==


class WorldSourceTest(unittest.TestCase):
    """背景世界源：LoadsWorldSource inferred 背景 / Ledger 占位。"""

    def test_loads_world_source_builds_background_from_registries(self):
        registry = HbmPortFlowRegistry()
        registry.attach_active_decode_provider(
            lambda port: 2 if port in (0, 1) else 0)
        links = LinkFlowRegistry()
        links.register(2, 3)
        model = _model(hbm_registry=registry, link_registry=links)
        world = LoadsWorldSource().build_context(model)
        self.assertEqual(world.progress_source, "inferred")
        works = {work.work_id for work in world.background_works}
        self.assertIn("bg:hbm:0", works)
        self.assertIn("bg:hbm:1", works)
        self.assertIn("bg:link:2->3", works)
        # 每在册计数一条 unknown-ETA in-service 腿（剩余量声明在案）。
        hbm_work = next(w for w in world.background_works
                        if w.work_id == "bg:hbm:0")
        self.assertEqual(len(hbm_work.legs), 2)
        for leg in hbm_work.legs:
            self.assertTrue(leg.in_service)
            self.assertGreater(leg.remaining_bytes, 0)
        self.assertIn("hbm_registry_counts_inferred_unknown_eta",
                      world.approximations)

    def test_world_efficiency_reaches_candidate_projection_input(self):
        model = _model(service_factors=ServiceFactors(transfer_factor=2.0))
        world = LoadsWorldSource().build_context(model)
        graph = _p_graph(
            model, ACTION_COPY,
            session=_session(
                location="local_hbm", home=0, resident=0,
                hist=(128, 128), miss=(0, 0), prefix=2),
            request=_request())
        projected = world.projection_input(graph)
        self.assertAlmostEqual(
            projected.efficiency.eta_of("link:0->1"), 0.5)
        self.assertEqual(projected.efficiency.eta_of("hbm:0"), 1.0)
        self.assertEqual(projected.efficiency.gamma, {})
        # ``projection_input`` carries the same efficiency through all
        # forecast/background additions; candidate placement cannot drop it.
        self.assertIs(projected.efficiency, world.efficiency)

    def test_ledger_world_source_fails_closed(self):
        # W5：占位已实装——无账本的裸构造仍显式失败（fail-closed 语义
        # 不变，文案换为"缺账本"；带账本的投影行为在
        # test_ledger_world_source.py）。
        model = _model()
        with self.assertRaises(ActionWorkGraphError) as ctx:
            LedgerWorldSource().build_context(model)
        self.assertIn("requires the causal work ledger",
                      str(ctx.exception))
        self.assertIn("refusing to price", str(ctx.exception))

    def test_jcm_default_world_source_is_loads(self):
        model = _model()
        world = model._world_context()
        self.assertEqual(world.progress_source, "inferred")
        # 决策级共享（同一 JCM 两次取用同对象——§4.4 只读背景结构）。
        self.assertIs(world, model._world_context())


# ==================================================== §7.2 义务 ==


class InheritedObligationTest(unittest.TestCase):
    """§7.2：inherited_obligation_ids 自披露升级为实际依赖。"""

    @staticmethod
    def _resolver(done):
        table = {
            # 大剩余量（长未完传输）：跨越候选 decode 段、在 settle
            # 时刻仍在飞（4000 B 会在 decode 段内早早完成——测不出门控）。
            "ob-live": ObligationView(
                obligation_id="ob-live", kind="committed_transfer",
                source_instance=2, target_instance=1,
                remaining_bytes_by_tp_rank=(3_000_000, 3_000_000),
                gate_settle=True, done=done),
        }
        return table.get

    def test_unfinished_obligation_competes_and_gates_settle(self):
        model = _model()
        base = model.estimate_D_remaining(
            boundary=_boundary(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        model.obligation_resolver = self._resolver(done=False)
        live = model.estimate_D_remaining(
            boundary=_boundary(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True,
            inherited_obligation_ids=("ob-live",))
        self.assertGreater(live.cost_ns, base.cost_ns)
        graph = build_d_candidate(
            model, boundary=_boundary(), instance_index=1,
            action=ACTION_REMOTE,
            inherited_obligation_ids=("ob-live",),
            obligation_resolver=self._resolver(done=False),
            prediction_config=JointPredictionConfig())
        self.assertEqual([w.work_id for w in
                          graph.extra_background_works],
                         ["obl:ob-live"])

    def test_finished_obligation_charges_nothing(self):
        # 已完 P 义务零腿（沉没费用不重付——D18 红线在义务面的延伸）。
        model = _model()
        base = model.estimate_D_remaining(
            boundary=_boundary(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        model.obligation_resolver = self._resolver(done=True)
        settled = model.estimate_D_remaining(
            boundary=_boundary(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True,
            inherited_obligation_ids=("ob-live",))
        self.assertEqual(settled.cost_ns, base.cost_ns)

    def test_missing_resolver_degrades_with_disclosure(self):
        # 无 ledger：ID 披露 + approximation（不虚构义务腿）。
        model = _model()
        candidate = model.estimate_D_remaining(
            boundary=_boundary(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True,
            inherited_obligation_ids=("ob-x",))
        self.assertEqual(
            candidate.breakdown.inherited_obligation_ids, ("ob-x",))
        self.assertIn("inherited_obligation_unresolved:ob-x",
                      candidate.breakdown.approximations)


if __name__ == "__main__":
    unittest.main()
