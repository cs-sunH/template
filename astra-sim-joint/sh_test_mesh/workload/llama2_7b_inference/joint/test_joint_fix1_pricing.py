#!/usr/bin/env python3
"""test_joint_fix1_pricing.py -- F1 修复卡（2026-09-22，§4.3 补遗
A7'/A8'/A9'/A10' + quota_deferred 安全网钉测 + _transfer_ns_shards 空
shard 契约）的零后端单测。

覆盖（按补遗逐项）：

1. **D10 recompute 语义重定义**（v2.0 §6.2，2026-09-26；
   joint_cost_model.py ``rebuild_context_tokens`` 取代旧 R13
   ``_recompute_missing_tokens``）：重建基数 = 整份已保存上下文 H、
   缺失层比例折算 ceil(H×missing_layers/L) 取消（旧审计锚
   L=32/prefix=16/H=1000 ⇒ 500 与取整边界 501 按新语义作废重写）；
   LOCAL/PARTIAL/REMOTE × 驻留/异地四组 compute_ns 同值（空间复用不
   免 FLOP）、rebuild 阶段字段（rebuild_context_tokens / stage=
   prefill_replay / suppress_sampling=output=False）随 notes 披露。
2. **A8' transfer_factor EWMA 接线**：SH 侧 ``_ingest_link_telemetry``
   逐窗口喂 ``observe_transfer_from_link_window``——updates>0、
   factor≥1.0（拥胀方向）、同 tick 多链路 Σactual/Σbase 单条更新、
   名义速率 = noc_link_bytes_per_ns、软门替身不炸、决策构造（JCM
   __post_init__ flush）后因子到达代价模型。
3. **A9' 遥测速率过期**：tick 包在场核对——忙→转闲（包缺席）⇒
   rates 条目删除、divisor_effective 回落注册表值；缺席一拍后重现 ⇒
   键重建（新速率非陈旧值）；空数组包 = 全闲 ⇒ 全部过期；键缺席包
   （旗标关）不做过期；整型键透传形态同语义。
4. **A10'(a) 零速率样本丢弃 + 披露**：served==0 ∧ active>0 ⇒ 不写
   rates、不喂因子、丢弃计数（决策日志遥测块字段
   telemetry_zero_rate_samples_dropped）、下游视图构造不崩。
5. **A10'(b) 配额入册失败防御回滚**：失败注入（_quota_enroll_
   admission → False）走 _try_admit_request 防御分支 ⇒ rid 历史迁移
   登记 + rid#readplan 预登记 + preplan 全部回滚、注册表归零。
   A11' 增补：预约处置两态（真实 KVCacheManager）——未物化释放、
   已物化 fail-closed（无 un-prepare 逆路径，假回队会在重试的重复
   预约上崩溃）。
6. **quota_deferred 安全网钉测**（C9 冻结语义，不改生产逻辑）：全拒
   verdicts（合成——生产中 recompute 恒可行属设计结果，deferred 机器
   是安全网）→ _quota_filter_candidates 回队 → _admit_waiting_requests
   单份回队（无丢请求、无重复回队）→ 键未变门跳过（不重评估）→
   信用释放 bump 配额代数 ⇒ 键变 → 唤醒（再评估发生、二次 defer 折叠
   attempt_count=2）。
7. **_transfer_ns_shards 空 shard 契约**（F1 小项）：空 shard 序列 ⇒
   返回 0 且 startup_ns 不计入；在场 shard（含零字节）⇒ startup 计入
   ——docstring 钉字的行为锚。
8. **D10 recompute span 基恒 0**（v2.0 §6.2，取代 O1/A19'(a) 驻留
   二分）：``prefill_task_load_ns_fn`` 的 history 实参对 recompute 恒
   0（完整 H+U 重放自 0 物化工作副本，四组同口径）；重算量 = input +
   整份 H；prefix=L 时 recompute 不再 ≡ stay（作废锚正向钉）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_fix1_pricing.py   （或 pytest 同路径）
"""
import dataclasses
import os
import sys
import unittest
from collections import deque
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_ONLINE_DIR = os.path.join(_PARENT, "online")
for _p in (_HERE, _PARENT, _ONLINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    KVTransfer,
    KVTransferShard,
    build_instances,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_config import JointMechanismConfig  # noqa: E402
from joint.capacity_summary import CapacitySummary  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
    TelemetryLinkFlowView,
    _transfer_ns_shards,
)
from joint.joint_scheduler import SelectionRecord  # noqa: E402
from joint.link_quota import (  # noqa: E402
    LinkQuotaTracker,
    QuotaVerdict,
)
from online import sh30_online_scheduler as sh30  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    Sh30OnlineScheduler,
)


# ============================================================ 夹具构造 ==

def _rates(noc=10.0, hbm=100.0, lat=10, pool=5.0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=pool, local_hbm_gbps=hbm,
        d2d_latency_ns=lat, pool_latency_ns=100)


def _load(index=0, total=0):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=0, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25)):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


def _route(source, target):
    return ((source, target), 1)


def _disjoint_paths(source, target):
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


def _jcm(model_layers=4, tp=2):
    """JCM 直测夹具（空注册表、因子 1.0 冷启动、tp=2）。"""
    return JointCostModel(
        rates=_rates(),
        loads={0: _load(0), 1: _load(1)},
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=model_layers, instance_tp_size=tp,
        route_fn=_route, route_paths_fn=_disjoint_paths)


# ---------------------------------------------- SH 替身（__new__ 范式） --

def _hardware():
    return FaceHardware(
        mesh_rows=2,
        mesh_cols=4,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )


def _face_model():
    return FaceModel(
        layers=4,
        hidden_size=64,
        ffn_size=128,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )


def _topology():
    # 2×4 mesh 四实例（tp=2）：实例 0 = ranks (0,1)、实例 1 = ranks
    # (2,3)（XY 路由 1→0 的 shard 路径 (2,1,0)/(3,2,1) 共享链路
    # (2,1)）；实例 2/3 = (4,5)/(6,7)。LinkId 端点枚举序与
    # test_joint_quota_integration.TelemetryEndpointKeyTest 同款。
    return build_instances(
        _hardware(),
        (FaceInstanceSpec("g0", "pg0", (0, 1)),
         FaceInstanceSpec("g1", "pg1", (2, 3)),
         FaceInstanceSpec("g2", "pg2", (4, 5)),
         FaceInstanceSpec("g3", "pg3", (6, 7))),
        require_equal_size=True)


def _make_runtime(request_id, *, session_id="s1"):
    return _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": 50,
        "decode_length": 16,
        "history_tokens_before": 100,
        "prefill_context_tokens": 150,
        "final_context_tokens": 166,
    })


def _scheduler(*, quota_mode="static"):
    """__new__ 范式（C8/C14/F1 同款）：真实注册表 + 真实配额 tracker +
    决策路径所需的最小属性面。B_link = 200 B/ns（A8' 名义速率锚）。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.hardware = _hardware()
    scheduler.model = _face_model()
    scheduler.topology = _topology()
    scheduler._joint_rates = _rates(noc=200.0, hbm=100.0, lat=0, pool=1.0)
    scheduler._joint_factors = ServiceFactors()
    scheduler._decode_task_load_cache = {}
    # 对齐 __init__ 初值（F6 销账：_task_load_cache_capacity 软门已删，
    # 替身漏设 = AttributeError）。
    scheduler._task_load_cache_capacity = sh30._TASK_LOAD_CACHE_CAPACITY
    scheduler._prefill_task_cache = {}
    scheduler.instances = [_OnlineInstanceState(index=i)
                           for i in range(4)]
    scheduler._rank_to_instance = {
        rank: instance.index
        for instance in scheduler.topology.instances
        for rank in instance.ranks}
    scheduler.runtime_by_request_id = {}
    scheduler.kv_manager = SimpleNamespace(
        tp_degree=2,
        # D04：JCM 构造注入 nearest_edge_fn——替身把池端口钉在每 rank
        # 本地（pool 腿退化口径：无 NoC 段；本文件用例不依赖池腿）。
        nearest_edge=lambda rank: rank,
        _sessions={
            "s1": SimpleNamespace(
                working_kind=None, context_tokens=100,
                base_history_tokens=100,
                base_resident_prefix_layers=4),
        },
        _reservations={},
        release_request_capacity_reservation=lambda rid: None,
        request_hbm_eventually_feasible_instances=lambda **kw: (
            [True, True, True, True]),
        _effective_remaining_by_tp_rank=lambda index: (10**9, 10**9),
        _instance_reclaimable_capacity_by_tp_rank=lambda index: (0, 0),
        # D11：容量驱逐摘要/容量账替身（零桶 = 快照时刻无合法 victim；
        # 逐 rank 台账 (C, W, O, Z) 与 remaining=1e9 同源核对守恒）。
        capacity_summary_snapshot=lambda: CapacitySummary(instances={}),
        capacity_ledger_by_tp_rank=lambda index: (
            (2 * 10**9, 0, 0, 10**9), (2 * 10**9, 0, 0, 10**9)),
    )
    scheduler.joint_config = JointMechanismConfig(
        category_mode="typed", scheduler_mode="joint",
        layer_policy="adaptive",
        quota_mode=quota_mode)
    scheduler._kv_ledger_epoch = 0
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._admit_attempt_epoch = {}
    scheduler._admission_failure_state = {}
    scheduler._admit_gate_verify = False
    scheduler.pending_admissions = deque()
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._hbm_ports.attach_active_decode_provider(
        scheduler._hbm_active_decode_streams)
    scheduler._instance_edge_ports = {}
    scheduler._train_max_iter = 8
    scheduler._quota_tracker = (
        LinkQuotaTracker(
            mode=quota_mode,
            noc_link_bytes_per_ns=(
                scheduler._joint_rates.noc_link_bytes_per_ns),
            local_hbm_bytes_per_ns=(
                scheduler._joint_rates.local_hbm_bytes_per_ns),
            delta_adm_ns=0,
        ) if quota_mode != "off" else None)
    scheduler._quota_enrolled = {}
    scheduler._quota_merge_reserves = {}
    scheduler._quota_merge_reserves_created = 0
    scheduler._quota_decode_owner_seq = {}
    scheduler._joint_action_selection_counts = {
        "stay": 0, "copy": 0,
        "recompute_elected": 0,
        "recompute_forced_no_history": 0,
        "recompute_forced_action_mask": 0,
        "recompute_forced_quota_deferred": 0,
        "recompute_forced_evicted_permanent": 0,
    }
    scheduler._quota_deferred_wait_counts = {
        "capacity": 0, "quota_link": 0, "quota_port": 0}
    scheduler._quota_deferred_dwell_ns = []
    scheduler._admission_decision_wall_ns_total = 0
    scheduler._admission_decision_wall_ns_max = 0
    scheduler._admission_decision_count = 0
    scheduler._quota_verdict_wall_ns_total = 0
    scheduler._quota_verdict_candidate_checks = 0
    scheduler._quota_admit_events = 0
    scheduler._quota_release_events = 0
    scheduler._quota_aimd_action_counts = {}
    # C8 遥测状态（__init__ 同款；A8'/A9'/A10'(a) 全量在场）。
    scheduler._link_telemetry_rates = {}
    # M1：C8 遥测状态 __init__ 同款（时间加权活跃流数）。
    scheduler._link_telemetry_flow_counts = {}
    # ACC（验收纪律第 3 类，v2.0 §19.3）：预测器开销账本 __init__ 同款
    # 初值（F6 销账口径：决策/摘要记账直访——替身漏设 = AttributeError）。
    from joint.predictor_cost_accounting import PredictorCostAccounting
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler.p_chunk = 512  # N4：_joint_prefill_total_load_ns 同形切分（F6 替身补设）
    scheduler._link_telemetry_epoch_count = 0
    scheduler._link_telemetry_sample_count = 0
    scheduler._telemetry_absent_seen = False
    scheduler._telemetry_window_broken = False
    scheduler._telemetry_window_end_ns = 0
    scheduler._telemetry_last_tick_ns = 0
    scheduler._telemetry_zero_rate_dropped = 0
    scheduler._telemetry_link_id_map = None
    # log_decision 基类契约（__new__ 替身手工置初值）。
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    return scheduler


def _sample(link_id, served, active, start, end):
    return {"link_id": link_id, "served_bytes": served,
            "active_ns": active, "window_start_ns": start,
            "window_end_ns": end}


def _decision_rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


def _telemetry_view(scheduler):
    """当前 rates 快照 → TelemetryLinkFlowView（JCM 构造同式）。"""
    return scheduler._joint_flows.with_effective_rates(
        dict(scheduler._link_telemetry_rates),
        link_capacity_bytes_per_ns=(
            scheduler._joint_rates.noc_link_bytes_per_ns))


# ==================================== 1. D10 recompute 语义重定义 ==


class D10RebuildContextTokensTest(unittest.TestCase):
    """D10（v2.0 §6.2，2026-09-26）：``rebuild_context_tokens`` 取代旧
    R13 ``_recompute_missing_tokens``——重建基数 = 整份已保存上下文 H，
    **缺失层比例折算 ceil(H×missing_layers/L) 取消**（KV 非任意层激活
    检查点，v2.0 C05/A2）；LOCAL/PARTIAL/REMOTE × 驻留/异地四组同基数
    （重放不依赖旧 KV）。"""

    def test_full_history_rebuild_in_all_four_groups(self):
        # 旧折算回归值按新语义重写：旧锚 L=32/prefix=16/H=1000 的驻留
        # 折算 500 与取整边界 501 全部作废——四组恒整份 1000。
        model = _jcm(model_layers=32)
        history = 1000
        groups = {
            # (home, resident, location, prefix) × 驻留/异地
            "local_resident": (0, 0, "local_hbm", 32),
            "partial_resident": (0, 0, "partial_hbm_remote", 16),
            "local_away": (1, 1, "local_hbm", 32),
            "partial_away": (1, 1, "partial_hbm_remote", 16),
            "remote_base": (0, 0, "remote_memory", 32),
        }
        for name, (home, resident, location, prefix) in groups.items():
            session = _session(
                home=home, resident=resident, location=location,
                prefix=prefix, history_tokens=history,
                history_bytes=(500, 500), missing=(500, 500))
            # 重建基数与目标 instance 无关（旧接口的 instance_index
            # 参数随驻留二分一并删除）。
            self.assertEqual(
                model.rebuild_context_tokens(session), history,
                msg=f"rebuild base must be full H for {name}")

    def test_prefix_does_not_prorate(self):
        # prefix ∈ {0, 16, 32}（L=32）同会话同基数——层驻留状态只影响
        # 空间足迹（空间复用），不影响重建 FLOP。
        model = _jcm(model_layers=32)
        for prefix in (0, 16, 32):
            session = _session(
                location="partial_hbm_remote", prefix=prefix,
                history_tokens=1000,
                history_bytes=(500, 500), missing=(500, 500))
            self.assertEqual(model.rebuild_context_tokens(session), 1000)

    def test_zero_history_rebuilds_nothing(self):
        model = _jcm(model_layers=4)
        session = _session(location="local_hbm", prefix=4,
                           history_tokens=0)
        self.assertEqual(model.rebuild_context_tokens(session), 0)

    def test_estimate_p_with_continuation_compute_identical_across_four_groups(self):
        # 决策级（验收）：四组 token 序列与算子工作解释一致——重算量 =
        # input + 整份 H（50+100），span 基 0；compute_ns 同值（空间
        # 复用不免 FLOP，驻留/异地差异只在空间/合并腿，不在计算段）。
        # prefill=1.0 ns/token、decode 10×2.0 ⇒ compute = 150 + 20 = 170。
        model = _jcm(model_layers=4)
        request = _request(input_tokens=50, decode=10)
        groups = {
            "local_resident": _session(
                home=0, resident=0, location="local_hbm", prefix=4),
            "partial_resident": _session(
                home=0, resident=0, location="partial_hbm_remote",
                prefix=2, history_bytes=(500, 500), missing=(500, 500)),
            "local_away": _session(
                home=1, resident=1, location="local_hbm", prefix=4),
            "partial_away": _session(
                home=1, resident=1, location="partial_hbm_remote",
                prefix=2, history_bytes=(500, 500), missing=(500, 500)),
            "remote_base": _session(
                home=0, resident=0, location="remote_memory", prefix=4),
        }
        seen_compute = {}
        for name, session in groups.items():
            candidate = model.estimate_P_with_continuation(
                session=session, request=request, instance_index=0,
                action="recompute")
            self.assertTrue(candidate.applicable, msg=name)
            seen_compute[name] = candidate.breakdown.compute_ns
            # token 序列守恒：重建（H=100）+ 本轮输入（U=50）各恰一次
            # （不把旧 U 再算成新 prefill），阶段字段随 notes 披露。
            self.assertIn("rebuild_context_tokens=100",
                          candidate.breakdown.notes, msg=name)
            self.assertIn("rebuild_stage=prefill_replay",
                          candidate.breakdown.notes, msg=name)
            self.assertIn("rebuild_suppress_sampling=False",
                          candidate.breakdown.notes, msg=name)
            self.assertIn("rebuild_suppress_output=False",
                          candidate.breakdown.notes, msg=name)
        self.assertEqual(
            set(seen_compute.values()), {(50 + 100) + 20})


class D10RecomputeSpanBaseZeroTest(unittest.TestCase):
    """D10（v2.0 §6.2，取代 O1/A19'(a) 驻留二分）：recompute 的
    ``prefill_task_load_ns_fn`` history 实参恒 0——重放自已保存 token
    物化工作副本、不依赖旧 KV（LOCAL/PARTIAL/REMOTE × 驻留/异地四组
    同口径）；重算量 = input + 整份 H（缺失层折算取消）。

    旧语义作废锚：@驻留基=H、prefix=L 时 recompute ≡ stay 逐位同值
    （R13/O1 口径）——新语义下 recompute 恒付整份重放（空间复用不免
    FLOP），与 stay 的 (input, H) 实参恒不同。
    """

    @staticmethod
    def _shaped_fn(p_chunk=16):
        """N4 生产同形简化：chunk 切分 + 累计 context 逐 chunk 求和
        （每 chunk load = chunk×(context+10)——固定项使线性外推可分辨），
        同时记录 (tokens, history) 实参供对拍断言。D06 契约：接受
        ``hbm_share_divisor`` 关键字（本夹具值不随份额缩放；份额实参
        逐次记录于 ``fn.shares`` 供份额钉测）。"""
        calls = []
        shares = []

        def fn(input_tokens, history_tokens, hbm_share_divisor=1):
            calls.append((input_tokens, history_tokens))
            shares.append(hbm_share_divisor)
            total = 0
            completed = 0
            while completed < input_tokens:
                chunk = min(p_chunk, input_tokens - completed)
                total += chunk * (history_tokens + completed + chunk + 10)
                completed += chunk
            return total

        fn.calls = calls
        fn.shares = shares
        return fn

    @staticmethod
    def _model_with_fn(fn, model_layers=4):
        return dataclasses.replace(
            _jcm(model_layers=model_layers),
            prefill_task_load_ns_fn=fn)

    def test_all_four_groups_span_base_zero_full_replay(self):
        # 四组统一：fn 实参 = (input + H, 0)——tokens 含整份 H（重放），
        # history 基 0（工作副本自 0 物化，attention 随 chunk 累计）。
        request = _request(input_tokens=50, decode=10)
        groups = {
            "local_resident": lambda: _session(
                location="local_hbm", prefix=4, history_tokens=100),
            "partial_resident": lambda: _session(
                location="partial_hbm_remote", prefix=2,
                history_bytes=(500, 500), missing=(500, 500),
                history_tokens=100),
            "local_away": lambda: _session(
                home=1, resident=1, location="local_hbm", prefix=4,
                history_tokens=100),
            "remote_base": lambda: _session(
                home=0, resident=0, location="remote_memory", prefix=4,
                history_tokens=100),
        }
        for name, make_session in groups.items():
            fn = self._shaped_fn()
            model = self._model_with_fn(fn)
            candidate = model.estimate_P_with_continuation(
                session=make_session(), request=request, instance_index=0,
                action="recompute")
            self.assertTrue(candidate.applicable, msg=name)
            self.assertEqual(fn.calls, [(50 + 100, 0)], msg=name)

    def test_full_layer_resident_recompute_no_longer_equals_stay(self):
        # 作废锚正向钉：prefix=L 驻留（旧语义 recompute ≡ stay）下，
        # recompute 实参 (150, 0) ≠ stay 实参 (50, 100)，且 recompute
        # 计算段更贵（整份重放 FLOP，空间复用不免）。
        fn = self._shaped_fn()
        model = self._model_with_fn(fn)
        session = _session(location="local_hbm", prefix=4,
                           history_tokens=100)
        request = _request(input_tokens=50, decode=10)
        recompute = model.estimate_P_with_continuation(
            session=session, request=request, instance_index=0,
            action="recompute")
        recompute_call = fn.calls[-1]
        stay = model.estimate_P_with_continuation(
            session=session, request=request, instance_index=0,
            action="stay")
        stay_call = fn.calls[-1]
        self.assertEqual(recompute_call, (50 + 100, 0))
        self.assertEqual(stay_call, (50, 100))
        # compute = fn(tokens, history)×1.0 + 10×2.0（因子冷启动 1.0）：
        # recompute 付整份重放（基 0、tokens 150）> stay（基 H、50）。
        expected_recompute = self._shaped_fn()(150, 0) + 20
        expected_stay = self._shaped_fn()(50, 100) + 20
        self.assertEqual(recompute.breakdown.compute_ns, expected_recompute)
        self.assertEqual(stay.breakdown.compute_ns, expected_stay)
        self.assertGreater(
            recompute.breakdown.compute_ns, stay.breakdown.compute_ns)

    def test_away_and_remote_recompute_history_base_zero(self):
        # 回归锚（旧 O1 异地/REMOTE 腿保留）：副本 0 基不动；重算量自
        # 旧"整份"口径不变（改变的是驻留腿并入同口径）。
        fn = self._shaped_fn()
        model = self._model_with_fn(fn)
        request = _request(input_tokens=50, decode=10)
        away = _session(home=1, resident=1, location="local_hbm",
                        prefix=4, history_tokens=100)
        model.estimate_P_with_continuation(
            session=away, request=request, instance_index=0,
            action="recompute")
        self.assertEqual(fn.calls[-1], (50 + 100, 0))
        remote = _session(home=0, resident=0, location="remote_memory",
                          prefix=4, history_tokens=100)
        model.estimate_P_with_continuation(
            session=remote, request=request, instance_index=0,
            action="recompute")
        self.assertEqual(fn.calls[-1], (50 + 100, 0))


# ======================================== 2. A8' transfer_factor 接线 ==


class A8TransferFactorWiringTest(unittest.TestCase):
    """_ingest_link_telemetry → observe_transfer_from_link_window。"""

    def test_congested_window_updates_factor_ge_1(self):
        # link 5 ⇒ (3,2)：served 1000 B / active 500 ns ⇒ rate 2 B/ns；
        # 名义 200 B/ns ⇒ base = int(1000/200) = 5、actual = 500。
        # D14 固定 1/16 EWMA 平滑效率 η（首样本同规则）：
        # η_sample = 5/500 = 0.01，η = (15+0.01)/16，因子 = 16/15.01。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000)]})
        factors = scheduler._joint_factors
        self.assertEqual(
            factors.updates.get("transfer_factor", 0), 0)  # 决策前 flush
        factors.flush()  # 决策时刻纪律（P3：读因子前 flush）
        self.assertEqual(factors.updates["transfer_factor"], 1)
        self.assertGreaterEqual(factors.transfer_factor, 1.0)
        self.assertAlmostEqual(factors.transfer_factor, 16.0 / 15.01)

    def test_factor_reaches_cost_model_at_decision_time(self):
        # JCM 构造（__post_init__ flush）后 service_factors 同一对象、
        # 因子值可见——决策链端到端。
        scheduler = _scheduler()
        scheduler._task_load_snapshot = (
            lambda state, now_ns: SimpleNamespace(
                queued_prefill_task_load_ns=0,
                running_prefill_task_load_ns=0,
                active_decode_task_load_ns=0,
                ordering_key=0))
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000)]})
        model = scheduler._joint_cost_model(0)
        self.assertIs(model.service_factors, scheduler._joint_factors)
        self.assertGreaterEqual(model.service_factors.transfer_factor, 1.0)
        self.assertEqual(
            model.service_factors.updates["transfer_factor"], 1)

    def test_same_tick_links_merge_into_single_update(self):
        # 同 tick 两链路经 _record 时刻汇总（η_sample = Σbase/Σactual）
        # 为一条更新。base_ns 逐链路**向上取整**（D17 修复轮二：
        # observe_transfer_from_link_window 的 int() 向下截断改 ceil）：
        # link 5 base = ceil(1000/200) = 5、link 7 = ceil(500/200) = 3
        # ⇒ η_sample = 8/1000；D14 固定 1/16 EWMA
        # 平滑效率（首样本同规则）：η = (15+0.008)/16，因子 = 16/15.008。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000),
                _sample(7, 500, 500, 0, 1000)]})
        factors = scheduler._joint_factors
        factors.flush()
        self.assertEqual(factors.updates["transfer_factor"], 1)
        self.assertAlmostEqual(
            factors.transfer_factor, 16.0 / 15.008)

    def test_nominal_rate_window_gives_exactly_one(self):
        # 实测 = 名义（200 B/ns：served 1000 / active 5）⇒ 样本比 1.0。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 5, 0, 1000)]})
        scheduler._joint_factors.flush()
        self.assertEqual(scheduler._joint_factors.transfer_factor, 1.0)

    def test_stub_without_factors_fails_loud(self):
        # F6 销账：_joint_factors 的 getattr 软门已删——__new__ 替身漏设
        # 该属性时 ingest 必须 AttributeError（fail-loud，因子喂入静默
        # 丢失不再可能；A12'：喂入先于 rates 落账——异常时速率条目
        # 不落账）。断言锚定缺失属性名（别的属性先炸不许冒名通过）。
        scheduler = _scheduler()
        del scheduler._joint_factors
        with self.assertRaisesRegex(AttributeError, "_joint_factors"):
            scheduler._ingest_link_telemetry({
                "tick": 1000, "link_telemetry": [
                    _sample(5, 1000, 500, 0, 1000)]})
        self.assertEqual(scheduler._link_telemetry_rates, {})

    def test_subnominal_window_anomalous_skipped(self):
        # D14（v2.0 §9.2）：亚名义样本（实测快于名义：active < served/
        # 名义速率，ratio<1 ⟺ η_sample>1 出 0<η≤1 有效窗口）记录异常并
        # 跳过——不钳位、不更新（原 A12' 钳位 1.0 属"无声截断"，退役）。
        # link 5：served 1000 / active 4 ⇒ base=int(1000/200)=5、
        # ratio=4/5=0.8 → 异常跳过。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 4, 0, 1000)]})
        factors = scheduler._joint_factors
        factors.flush()
        self.assertEqual(factors.transfer_factor, 1.0)
        self.assertEqual(factors.anomalous.get("transfer_factor"), 1)
        self.assertNotIn("transfer_factor", factors.updates)
        self.assertNotIn("transfer_factor", factors.rejected)
        # 随后拥胀样本正常进入固定 1/16 EWMA 平滑效率（自 η=1 出发，
        # 跳过的样本不留残迹）：η_sample = 5/500 = 0.01，
        # η = (15+0.01)/16，因子 = 16/15.01。
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": [
                _sample(5, 1000, 500, 1000, 2000)]})
        factors.flush()
        self.assertAlmostEqual(factors.transfer_factor, 16.0 / 15.01)
        self.assertEqual(factors.anomalous.get("transfer_factor"), 1)
        self.assertEqual(factors.updates["transfer_factor"], 1)
        self.assertEqual(
            factors.as_dict()["anomalous"]["transfer_factor"], 1)


# ============================================ 3. A9' 遥测速率过期 ==


class A9TelemetryRateExpiryTest(unittest.TestCase):
    """tick 包在场核对：缺席即失效（速率条目过期删除）。D13：速率-
    only 窗口 = 流数显式未知——不产生除数腿，除数退注册表在册流近
    似（旧 capacity/速率冒充流数口径废除）。"""

    def test_busy_to_idle_falls_back_to_registered_divisor(self):
        # 忙窗：rate 2 B/ns、容量 200 —— D13 起该速率样本不再换算成
        # 除数 100：流数未知 ⇒ 在册流近似（背景 1 流 ⇒ 1）。
        scheduler = _scheduler()
        registry = scheduler._joint_flows
        registry.register(3, 2)  # 背景 1 流 ⇒ 注册表除数 1
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000)]})
        view = _telemetry_view(scheduler)
        self.assertEqual(view.divisor_effective((3, 2)), 1.0)
        self.assertIsNone(view.effective_divisor((3, 2)))
        self.assertEqual(view.disclosure()["unknown_flow_count_links"], 1)
        # 转闲：下一包省略该链路（C6 契约）⇒ 条目删除、除数维持注册值。
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": [
                _sample(14, 500, 500, 1000, 2000)]})
        self.assertNotIn((3, 2), scheduler._link_telemetry_rates)
        self.assertEqual(
            _telemetry_view(scheduler).divisor_effective((3, 2)), 1.0)

    def test_reappear_after_absence_rebuilds_key(self):
        # 缺席一拍后重现 ⇒ 键重建、值为新窗实测（非陈旧 2.0）。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000)]})
        self.assertEqual(
            scheduler._link_telemetry_rates[(3, 2)], 2.0)
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": [
                _sample(14, 500, 500, 1000, 2000)]})
        self.assertNotIn((3, 2), scheduler._link_telemetry_rates)
        scheduler._ingest_link_telemetry({
            "tick": 3000, "link_telemetry": [
                _sample(5, 3000, 500, 2000, 3000)]})
        self.assertEqual(
            scheduler._link_telemetry_rates[(3, 2)], 6.0)

    def test_empty_packet_expires_all(self):
        # 空数组包 = 全闲 epoch（observer 开、全部链路省略）⇒ 全部过期。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000),
                _sample(7, 500, 500, 0, 1000)]})
        self.assertEqual(len(scheduler._link_telemetry_rates), 2)
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": []})
        self.assertEqual(scheduler._link_telemetry_rates, {})

    def test_absent_key_packet_expires_cached_rates(self):
        # A12'：遥测面整体缺席（旗标关/桥断供）⇒ 缓存实测速率一并失
        # 效（与 A9' 单链路缺席失效同语义——防陈旧速率在遥测停发后永
        # 久驻留；旧行为"键缺席条目保留"系防御缺口，本测试改钉新契约）。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000)]})
        self.assertEqual(
            scheduler._link_telemetry_rates, {(3, 2): 2.0})
        scheduler._ingest_link_telemetry({"tick": 2000})
        self.assertEqual(scheduler._link_telemetry_rates, {})

    def test_zero_active_sample_keeps_link_present(self):
        # 全零样本（active=0）在包内 = 链路在场（包为其作保）——旧速率
        # 保留、不因丢弃而缺席删除（A10' 只丢 served==0∧active>0 伪影）。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000)]})
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": [
                _sample(5, 0, 0, 1000, 2000)]})
        self.assertEqual(
            scheduler._link_telemetry_rates, {(3, 2): 2.0})

    def test_int_key_passthrough_form_expires_too(self):
        # F6 销账：hasattr 透传软门已删——整型键形态改由替身显式预置
        # 恒等映射 {i: i}（等价旧退化路径，语义自担，不再依赖属性缺席
        # 静默透传）。过期语义同款：缺席即删除。
        scheduler = _scheduler()
        scheduler._telemetry_link_id_map = {5: 5, 7: 7}
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000),
                _sample(7, 500, 500, 0, 1000)]})
        self.assertEqual(
            scheduler._link_telemetry_rates, {5: 2.0, 7: 1.0})
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": [
                _sample(7, 500, 500, 1000, 2000)]})
        self.assertEqual(scheduler._link_telemetry_rates, {7: 1.0})


# ================================ 4. A10'(a) 零速率样本丢弃 + 披露 ==


class A10aZeroRateSampleGuardTest(unittest.TestCase):
    """served==0 ∧ active>0 = 有活动无载荷字节的有效速率零窗口
    （A12' 定性订正——uint64 差分计量无"整数截断伪影"）：丢弃样本 +
    计数，并同窗失效缓存实测速率（A9' 在场核对不剪除在场键）。"""

    def test_zero_rate_window_expires_cached_rate(self):
        # A12'：零速率窗在包内在场——A9' 在场核对不会剪除它，丢弃
        # 样本时须同窗失效缓存实测速率（否则陈旧高速率驻留、NoC 计
        # 价除数欠计拥塞）；下一 served>0 窗即重建（新值非陈旧值）。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 1000, 500, 0, 1000)]})
        self.assertEqual(scheduler._link_telemetry_rates[(3, 2)], 2.0)
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": [
                _sample(5, 0, 400, 1000, 2000)]})
        self.assertNotIn((3, 2), scheduler._link_telemetry_rates)
        self.assertEqual(
            _telemetry_view(scheduler).divisor_effective((3, 2)), 1.0)
        self.assertEqual(scheduler._telemetry_zero_rate_dropped, 1)
        scheduler._ingest_link_telemetry({
            "tick": 3000, "link_telemetry": [
                _sample(5, 3000, 500, 2000, 3000)]})
        self.assertEqual(scheduler._link_telemetry_rates[(3, 2)], 6.0)
        self.assertEqual(scheduler._telemetry_zero_rate_dropped, 1)

    def test_zero_rate_dropped_counted_and_disclosed(self):
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": [
                _sample(5, 0, 400, 0, 1000),       # 零速率窗样本
                _sample(7, 500, 500, 0, 1000)]})    # 正常样本
        # rates 不含零速率窗键；正常键照写。
        self.assertEqual(
            scheduler._link_telemetry_rates, {(5, 4): 1.0})
        # 样本证据计数照涨（覆盖证据不受丢弃影响）。
        self.assertEqual(scheduler._link_telemetry_sample_count, 2)
        # 丢弃计数 + 决策日志遥测块披露字段。
        self.assertEqual(scheduler._telemetry_zero_rate_dropped, 1)
        block = scheduler._telemetry_coverage_decision()
        self.assertEqual(
            block["telemetry_zero_rate_samples_dropped"], 1)
        self.assertEqual(block["telemetry_rate_entries"], 1)
        self.assertTrue(block["collective_coverage"])
        # 因子只收正常样本（零速率窗样本在喂入侧同点丢弃——不进
        # updates 也不进 rejected 双通道）。
        scheduler._joint_factors.flush()
        self.assertEqual(
            scheduler._joint_factors.updates.get("transfer_factor"), 1)
        self.assertNotIn(
            "transfer_factor", scheduler._joint_factors.rejected)
        # 下游视图构造不崩（写入 0.0 会让 JointCostError 崩在此处）；
        # 正常样本键 (5,4) 有速率无流数——D13：显式未知（除数退注册
        # 表在册流近似，不再 capacity/速率 换算 200）；零速率窗键
        # (3,2) 无测量（None——注册表除数即有效除数）。
        view = _telemetry_view(scheduler)
        self.assertEqual(view.divisor_effective((5, 4)), 1.0)
        self.assertIsNone(view.effective_divisor((5, 4)))
        self.assertIsNone(view.measured_effective_rate((3, 2)))

    def test_coverage_block_zero_when_no_artifacts(self):
        # 无零速率窗样本（或遥测未开）⇒ 披露字段 0。
        scheduler = _scheduler()
        block = scheduler._telemetry_coverage_decision()
        self.assertEqual(
            block["telemetry_zero_rate_samples_dropped"], 0)
        self.assertFalse(block["collective_coverage"])

    def test_positive_bytes_zero_active_fails_closed(self):
        # A14'（H5，2026-09-22 第三轮复审）：served>0 ∧ active==0 违反
        # uint64 差分契约（有载荷必有活跃时间）——fail-closed，与未知
        # link_id 教义对齐。旧行为（静默跳过）下该样本以在场键豁免
        # A9' 剪除、陈旧速率钉驻——本入口唯一病态样本静默通道，封死。
        scheduler = _scheduler()
        with self.assertRaisesRegex(
                ValueError, "differential contract"):
            scheduler._ingest_link_telemetry({
                "tick": 1000, "link_telemetry": [
                    _sample(5, 10, 0, 0, 1000)]})
        self.assertEqual(scheduler._link_telemetry_rates, {})
        self.assertEqual(scheduler._telemetry_zero_rate_dropped, 0)


# ============================ 5. A10'(b) 配额入册失败防御回滚 ==


class A10bQuotaEnrollFailureRollbackTest(unittest.TestCase):
    """失败注入：_quota_enroll_admission → False ⇒ 承诺全回滚、归零。"""

    @staticmethod
    def _admission_transfer(rid):
        # noc_migrate 逐出支链（reserve 返回值）：1 shard 路径 (0,1)。
        return KVTransfer(
            kind="noc_migrate", phase="admission", reason="test",
            session_id="s1", trigger_request_id=rid,
            source_instance_index=0, target_instance_index=0,
            total_bytes=8,
            shards=(KVTransferShard(
                source_rank=0, target_rank=1, edge_rank=None, bytes=8,
                noc_path=(0, 1), layer_start=0, layer_end=4),),
            model_layers=4, layer_start=0, layer_end=4,
            resident_prefix_layers_before=4,
            resident_prefix_layers_after=4)

    def _drive(self, *, enroll_returns):
        """装配 _try_admit_request 全链（真实事务段 + 真实登记），注入
        配额入册失败；返回 (scheduler, runtime, captured)。"""
        scheduler = _scheduler()
        scheduler._task_load_snapshot = (
            lambda state, now_ns: SimpleNamespace(
                queued_prefill_task_load_ns=0,
                running_prefill_task_load_ns=0,
                active_decode_task_load_ns=0,
                ordering_key=0))
        scheduler._joint_horizon = SimpleNamespace(
            estimate=lambda sid: (8, "session_online_mean"))
        scheduler._joint_session_view = lambda sid: SessionKVView(
            session_id=sid, home_instance=1, resident_instance=1,
            location="local_hbm", history_tokens=100,
            resident_prefix_layers=4,
            history_bytes_by_tp_rank=(400, 400),
            missing_bytes_by_tp_rank=(0, 0))
        scheduler.graph = SimpleNamespace(
            sync_pending_history_after_evictions=lambda evictions: None)
        runtime = _make_runtime("rX")
        runtime.estimated_arrival_ns = 0
        eviction = self._admission_transfer("rX")
        scheduler.kv_manager.reserve_request_capacity = (
            lambda **kw: (eviction,))
        scheduler.kv_manager.prepare_prefill = (
            lambda **kw: (None, (), ()))
        # 选中 copy@3（源 1 → 目标 3，跨实例主流程）：模块级
        # select_instance_and_action 打桩（场景设定——被测机器 = 防御
        # 分支回滚，非 argmin）。判据面（_quota_filter_candidates）在
        # 空注册表起步，链路门放行——事务段/入册照常走到防御分支。
        from joint.joint_cost_model import ActionCandidate
        fake_record = SelectionRecord(
            mode="joint",
            chosen=ActionCandidate(
                instance_index=3, action="copy", applicable=True,
                inapplicable_reason=None, cost_ns=1000),
            candidates=(
                ActionCandidate(
                    instance_index=3, action="stay", applicable=False,
                    inapplicable_reason="history not resident at target",
                    cost_ns=None),
                ActionCandidate(
                    instance_index=3, action="recompute", applicable=False,
                    inapplicable_reason="synthetic scene", cost_ns=None),
                ActionCandidate(
                    instance_index=3, action="copy",
                    applicable=True, inapplicable_reason=None,
                    cost_ns=1000),
            ),
            instance_rule_note="synthetic copy scene")
        original_select = sh30.select_instance_and_action
        sh30.select_instance_and_action = lambda **kw: fake_record
        captured = {}

        def fake_admit_flow(owner, rid, **kwargs):
            # 主流程入册翻转注入（O2 真异常形态：判据 PASS → 入册拒）。
            # 翻转前逐出支链（事务成功段 rid#evict）已真实在册——
            # captured 锚证明回滚真的清了东西，而非空放。
            captured["flows"] = scheduler._joint_flows.snapshot()
            captured["hbm_leaks"] = dict(
                scheduler._hbm_ports.leaked_owners())
            return False

        original_enroll_flow = scheduler._quota_admit_flow
        scheduler._quota_admit_flow = fake_admit_flow
        try:
            admitted = scheduler._try_admit_request(runtime, 500)
        finally:
            sh30.select_instance_and_action = original_select
            scheduler._quota_admit_flow = original_enroll_flow
        return scheduler, runtime, captured, admitted

    def test_failure_rolls_back_registrations_to_zero(self):
        scheduler, runtime, captured, admitted = self._drive(
            enroll_returns=False)
        # 防御分支生效：准入失败、按 quota_deferred 回队。
        self.assertFalse(admitted)
        self.assertEqual(
            scheduler._quota_deferred_wait_counts["quota_link"], 1)
        self.assertFalse(scheduler.pending_admissions)  # 调用方回队职责
        # 回滚前非空（证明回滚真的清了东西，而非空放）：事务成功段的
        # 逐出支链流（rid#evict）在册。
        self.assertTrue(captured["flows"])
        self.assertIn("rX#evict", captured["hbm_leaks"])
        # 注册表归零：链路流 / 池端口 / HBM 端口。
        self.assertEqual(scheduler._joint_flows.snapshot(), {})
        self.assertEqual(scheduler._pool_ports._counts, {})
        self.assertEqual(scheduler._hbm_ports.snapshot(), {})
        self.assertEqual(scheduler._hbm_ports.leaked_owners(), {})
        # 失败日志落地（quota_deferred 类；A14'/H6：failure_class 带
        # quota_link 后缀 ⇒ 决策日志 wait_reason 与
        # _quota_deferred_wait_counts 计数键同源）。
        rows = _decision_rows(scheduler, "joint_admission_failed")
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["decision"]["failure_class"],
            "quota_deferred_quota_link")
        self.assertEqual(
            rows[0]["decision"]["wait_reason"], "quota_link")


class A10bReservationDisposalTest(unittest.TestCase):
    """A11'：配额入册失败防御分支的预约处置两态（真实 KVCacheManager）。

    旧测试把 reserve/prepare 打桩致 ``_reservations`` 恒空、旧守卫走
    空放——"回滚归零"在真空里通过（审计 P1：真实事务段 prepare 成功
    ⇒ 预约恒部分物化 ⇒ 旧 ``_release_orphan_reservation`` 的守卫先炸，
    回滚不可达）。本类用真实预约态钉住新契约：未物化（extra 全零）⇒
    释放归零 + 放行回队；已物化 ⇒ fail-closed（预约无 un-prepare 逆
    路径，假回队会在重试的重复预约 ValueError 上崩溃）。"""

    @staticmethod
    def _real_manager():
        scheduler = _scheduler()
        model = _face_model()
        real = KVCacheManager(scheduler.topology, model)
        scheduler.kv_manager = real
        # 会话在实例 0 完成驻留 100 tokens（extra = final − local 的
        # local 侧来源）。
        real.prepare_prefill(
            session_id="s1", target_instance_index=0,
            history_tokens=0, trigger_request_id="seed")
        real.expand_prefill(
            session_id="s1", instance_index=0, context_tokens=100,
            trigger_request_id="seed")
        real.mark_complete("s1", 100)
        return scheduler, real

    def test_unmaterialized_reservation_released(self):
        # 未物化：final_context == 当前驻留（输入增量 0 ⇒ extra 全零）
        # ⇒ 释放 + 纪元 bump，调用方可照常 quota_deferred 回队。
        scheduler, real = self._real_manager()
        real.reserve_request_capacity(
            request_id="rD", session_id="s1", instance_index=0,
            final_context_tokens=100, action="stay")
        self.assertIn("rD", real._reservations)
        epoch_before = scheduler._kv_ledger_epoch
        scheduler._release_reserved_admission_or_fail("rD")
        self.assertNotIn("rD", real._reservations)
        self.assertEqual(scheduler._kv_ledger_epoch, epoch_before + 1)

    def test_materialized_reservation_fails_closed(self):
        # 已物化：final = 100 + 50 输入增量 ⇒ extra>0（prepare 已推进
        # 会话账目的正常在册形态）⇒ RuntimeError 带全上下文，预约不释
        # 放（静默释放违反 R2 报警线纪律）。
        scheduler, real = self._real_manager()
        real.reserve_request_capacity(
            request_id="rE", session_id="s1", instance_index=0,
            final_context_tokens=150, action="stay")
        self.assertIn("rE", real._reservations)
        with self.assertRaisesRegex(
                RuntimeError, "materialized admission transaction"):
            scheduler._release_reserved_admission_or_fail("rE")
        self.assertIn("rE", real._reservations)

    def test_absent_reservation_no_op(self):
        # 无预约（reserve 失败早退等形态）⇒ 空放，不炸。
        scheduler, real = self._real_manager()
        scheduler._release_reserved_admission_or_fail("rG")
        self.assertEqual(real._reservations, {})


# ============================== 6. quota_deferred 安全网钉测 ==


class QuotaDeferredSafetyNetPinTest(unittest.TestCase):
    """C9 冻结语义直接单测（不改生产逻辑）：全拒 ⇒ 回队闭环。

    生产中 recompute 恒可行（免链路动作不被配额裁）属设计结果，
    deferred 机器是安全网——本钉测以合成全拒 verdicts 驱动真实
    _quota_filter_candidates/_admit_waiting_requests/重试门。
    """

    def _full_stub(self):
        scheduler = _scheduler(quota_mode="static")
        scheduler._task_load_snapshot = (
            lambda state, now_ns: SimpleNamespace(
                queued_prefill_task_load_ns=0,
                running_prefill_task_load_ns=0,
                active_decode_task_load_ns=0,
                ordering_key=0))
        scheduler._joint_horizon = SimpleNamespace(
            estimate=lambda sid: (8, "session_online_mean"))
        scheduler._joint_session_view = lambda sid: SessionKVView(
            session_id=sid, home_instance=1, resident_instance=1,
            location="local_hbm", history_tokens=100,
            resident_prefix_layers=4,
            history_bytes_by_tp_rank=(400, 400),
            missing_bytes_by_tp_rank=(0, 0))
        scheduler.graph = SimpleNamespace(
            sync_pending_history_after_evictions=lambda evictions: None)
        return scheduler

    def test_all_rejected_loop_requeue_gate_and_wakeup(self):
        scheduler = self._full_stub()
        calls = {"verdicts": 0}

        def all_rejected(candidate, session_view, r_hat):
            calls["verdicts"] += 1
            return QuotaVerdict(
                admitted=False, resource_kind="link", remaining=0,
                wait_reason="quota_link",
                inapplicable_reason=(
                    "synthetic safety-net rejection: {}".format(
                        candidate.action)))

        scheduler._quota_candidate_verdict = all_rejected
        runtime = _make_runtime("rq")
        runtime.estimated_arrival_ns = 0
        scheduler.pending_admissions.append(runtime)
        scheduler.runtime_by_request_id["rq"] = runtime

        # pass 1：全拒 ⇒ deferred 回队（单份、无丢）。
        scheduler._admit_waiting_requests(100)
        self.assertEqual(len(scheduler.pending_admissions), 1)
        self.assertIs(scheduler.pending_admissions[0], runtime)
        self.assertEqual(
            scheduler._quota_deferred_wait_counts["quota_link"], 1)
        self.assertEqual(runtime.quota_deferred_since_ns, 100)
        last_key = scheduler._admit_attempt_epoch["rq"]
        self.assertIsNotNone(last_key)
        verdicts_after_pass1 = calls["verdicts"]
        self.assertGreater(verdicts_after_pass1, 0)

        # pass 2：键未变 ⇒ 门跳过（不重评估、仍单份、无重复回队）。
        self.assertEqual(
            last_key, scheduler._current_retry_key(last_key))
        scheduler._admit_waiting_requests(200)
        self.assertEqual(len(scheduler.pending_admissions), 1)
        self.assertIs(scheduler.pending_admissions[0], runtime)
        self.assertEqual(calls["verdicts"], verdicts_after_pass1)

        # 信用释放 bump 配额代数 ⇒ 键变 ⇒ 唤醒门重开。
        tracker = scheduler._quota_tracker
        verdict = tracker.admit_flow(
            owner="cred#x", flow_class="oneshot", links=((3, 9),),
            port_id=3, now_ns=250)
        self.assertTrue(verdict.admitted)
        self.assertEqual(
            last_key, scheduler._current_retry_key(last_key))
        tracker.release_flow("cred#x", now_ns=300)
        self.assertNotEqual(
            last_key, scheduler._current_retry_key(last_key))

        # pass 3：唤醒——再评估发生（verdict 计数前进）、二次 defer
        # 折叠 attempt_count=2、仍单份回队。
        scheduler._admit_waiting_requests(300)
        self.assertGreater(calls["verdicts"], verdicts_after_pass1)
        self.assertEqual(len(scheduler.pending_admissions), 1)
        self.assertIs(scheduler.pending_admissions[0], runtime)
        self.assertEqual(
            scheduler._quota_deferred_wait_counts["quota_link"], 2)
        wait_rows = _decision_rows(scheduler, "joint_admission_wait")
        self.assertEqual(len(wait_rows), 1)
        self.assertEqual(wait_rows[0]["decision"]["attempt_count"], 2)
        self.assertEqual(
            wait_rows[0]["decision"]["wait_reason"], "quota_link")

    def test_filter_deferred_record_shape(self):
        # _quota_filter_candidates 直测：全拒 ⇒ deferred 记录形态
        # （requeue=True、wait_reason、重试键尾位 = 配额代数）。
        scheduler = self._full_stub()
        scheduler._quota_candidate_verdict = (
            lambda candidate, session_view, r_hat: QuotaVerdict(
                admitted=False, resource_kind="port", remaining=0,
                wait_reason="quota_port",
                inapplicable_reason="synthetic"))
        from joint.joint_cost_model import ActionCandidate
        record = SelectionRecord(
            mode="joint",
            chosen=ActionCandidate(
                instance_index=3, action="copy", applicable=True,
                inapplicable_reason=None, cost_ns=100),
            candidates=(
                ActionCandidate(
                    instance_index=3, action="stay", applicable=False,
                    inapplicable_reason="scene", cost_ns=None),
                ActionCandidate(
                    instance_index=3, action="copy",
                    applicable=True, inapplicable_reason=None,
                    cost_ns=100),
            ),
            instance_rule_note="scene")
        session_view = scheduler._joint_session_view("s1")
        chosen, candidates, deferred = scheduler._quota_filter_candidates(
            record, session_view, "rq", 0)
        self.assertIsNone(chosen)
        record_deferred, r_hat = deferred
        self.assertTrue(record_deferred.requeue)
        self.assertEqual(record_deferred.wait_reason, "quota_port")
        self.assertEqual(
            record_deferred.retry_key[-1],
            scheduler._quota_tracker.quota_retry_key()[0])
        # 候选表全不适用化（cost/breakdown 清空）。
        self.assertTrue(all(
            not candidate.applicable for candidate in candidates))


# ============================== 7. _transfer_ns_shards 空 shard 契约 ==


class TransferNsShardsEmptyContractTest(unittest.TestCase):
    """空 shard 序列 ⇒ 0 且 startup 不计入（docstring 钉字的行为锚）。"""

    def test_empty_shard_sequence_returns_zero_dropping_startup(self):
        # 无 shard = 无传输事务：startup 属事务固定成本，不随不存在的
        # 流产生（契约登记见 _transfer_ns_shards docstring，F1 小项）。
        self.assertEqual(_transfer_ns_shards(
            _rates(), LinkFlowRegistry(), None,
            paths_by_rank=(), bytes_by_rank=(),
            include_self=True, kind="copy", startup_ns=12345), 0)

    def test_present_shard_keeps_startup_even_with_zero_bytes(self):
        # 对照：shard 在场（哪怕零字节、零跳路径）⇒ startup 计入——
        # 与"空序列丢弃 startup"的边界分界锚。
        self.assertEqual(_transfer_ns_shards(
            _rates(), LinkFlowRegistry(), None,
            paths_by_rank=((),), bytes_by_rank=(0,),
            include_self=True, kind="copy", startup_ns=12345), 12345)


if __name__ == "__main__":
    unittest.main()
