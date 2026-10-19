#!/usr/bin/env python3
"""test_decision_route_serialization.py -- 问题 4a（S3 路由序列化，
2026-09-05，方案 §5/T-E2）单测：sh_3.0 决策日志补齐五个 holder 的
shard 级序列化（prefill 行 history_transfer / history_evictions /
prefill_evictions；decode 行 prefill_decode_transfer / decode_evictions），
全部经 _transfer_summary 只读导出 shards[].noc_hops / noc_path——路由在
KVTransferShard 创建时由 deterministic_xy_route 算好，序列化零新算；
既有聚合 history_transfer_bytes 保留；核销置空沿用 completion M4 块
（不新增提前置空，每请求恰一条 decode 行）。对齐目标 = S1
（sh10_online_scheduler.py 姊妹仓的 shard 级序列化段）；hopbytes.py
collect_sh30 读取端前向兼容（旧 log 无新键 → 数值不变，0902 回归另证）。

覆盖：
  - prefill 行（真实 KVCacheManager 压力夹具）：REMOTE 会话恢复准入 ->
    history_transfer = remote_load（shards 自 edge rank 起、止于目标
    rank）；history_evictions = 逐出 KVTransfer（remote_store）逐条落盘；
    prefill_evictions = []（准入时点初始值，与 S1 同时序：expand_prefill
    在 drain 边界才发生）；聚合 history_transfer_bytes 保留。
  - decode 行（no_affinity drain 全链）：prefill_decode_transfer =
    noc_migrate 含手算 XY 路径（2x2 mesh 实例 (0,1)->(2,3)：路径 (0,2)/
    (1,3)，各 1 hop）；decode_evictions 列表落盘；不新增提前置空
    （发射后 holder 保留，completion M4 块才核销）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_decision_route_serialization.py   （或 pytest）
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    PREFILL_CHUNK_SIZE,
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
    deterministic_xy_route,
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
    _transfer_summary,
)
from joint.hbm_port_flow_registry import (  # noqa: E402  F6 销账替身对齐
    HbmPortFlowRegistry,
)


# ---------------------------------------------------------------------------
# 夹具（与 test_ablation_switch.py 同款小参数；调度器一律 __new__ 壳式
# 装配，只装被测路径属性；KV 一律真实 KVCacheManager）。
# ---------------------------------------------------------------------------

def _tiny_hardware(capacity_bytes=1_000_000_000):
    return FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )


def _tiny_model(layers=4):
    return FaceModel(
        layers=layers,
        hidden_size=16,
        ffn_size=32,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )


def _two_instance_topology(hardware):
    return build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )


class _GraphStub:
    def __init__(self):
        # （原 pipeline_identity_allocator 挂载已随 2026-10-14 远读退役
        # 删除：descriptor 协议整删，_pipeline_identity_provider 不复
        # 存在；替身仅保留决策行发射所需的最小壳面。）
        pass

    def emit_admission_batch(self, plan):
        return {"eviction_watches": []}

    def sync_pending_history_after_evictions(self, evictions):
        pass


def _decision_shell(kv_manager, *, hardware=None, model=None, topology=None,
                    no_affinity=False):
    """装配决策发射（_emit_admission / _emit_join_decision 经真
    log_decision）+ no_affinity drain 路径用到的全部属性。"""
    hardware = hardware or _tiny_hardware()
    model = model or _tiny_model()
    topology = topology or _two_instance_topology(hardware)
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.topology = topology
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.p_chunk = PREFILL_CHUNK_SIZE
    # N12（R15-4）甄别改写：标定常数 → 在线估计器（冷启动 10 同数值）。
    from joint.joint_cost_model import (
        CausalHorizonEstimator, LinkFlowRegistry,
    )
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_forward_steps=10)
    # R15：流登记表/池端口（drain 路径的登记/注销钩子）。
    from online.sh30_online_scheduler import _PoolPortRegistry
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._instance_edge_ports = {
        instance.index: tuple(sorted({
            kv_manager.nearest_edge(rank) for rank in instance.ranks}))
        for instance in topology.instances
    }
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError）。
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None  # off 档 __init__ 初值（F6 销账）
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._snapshot_verify = False
    # D18（2026-09-26）：drain 边界二次决策（_joint_boundary_view +
    # _joint_boundary_cost_model → _joint_cost_model）需 J 组件同源
    # 初值——缺设即 AttributeError（fail-loud，不静默退化）。
    from joint.joint_config import parse_joint_config
    from joint.joint_cost_model import JointHardwareRates, ServiceFactors
    scheduler.joint_config = parse_joint_config()
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=hardware.d2d_bandwidth_gbps,
        pool_port_gbps=hardware.d2d_bandwidth_gbps,
        local_hbm_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_latency_ns=int(hardware.d2d_latency_ns),
        pool_latency_ns=0,
    )
    scheduler._joint_factors = ServiceFactors()
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    # cfg2（§5.6"版本失效、失败重评等额外遍历须累计报告"，2026-09-30
    # 修复轮）：提交点版本失效/容量回退 = 该边界一次额外遍历，run 级
    # 累计披露（_joint_boundary_fallback_count 直访递增）——__init__
    # 同款初值（F6 销账口径：替身漏设 = AttributeError）。
    scheduler._joint_boundary_fallback_count = 0
    # ACC（验收纪律第 3 类，v2.0 §19.3）：预测器开销账本 __init__ 同款
    # 初值（F6 销账口径：_account_decision_wall / 边界决策 / 容量摘要
    # 记账直访——替身漏设 = AttributeError）。
    from joint.predictor_cost_accounting import PredictorCostAccounting
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler.kv_manager = kv_manager
    scheduler.instances = [
        _OnlineInstanceState(index=i)
        for i in range(len(topology.instances))
    ]
    scheduler.runtime_by_request_id = {}
    scheduler._kv_ledger_epoch = 0
    scheduler._ready_frontier = set()
    # SH30_ABLATION 消融旗标已退役（no_affinity 形参仅为保持调用签名，
    # 不再有任何行为效果——decode 实例由 D18 边界二次决策决定）。
    scheduler.graph = _GraphStub()
    scheduler._ledger_admit = lambda *args, **kwargs: None
    scheduler._batch = {"assignments": [], "delivery_sequence": 0}
    scheduler._emitted_by_delivery = {0: {"requests": []}}
    scheduler.ledger_issued = {}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    return scheduler


def _make_runtime(request_id, *, prefill_context_tokens=1024,
                  decode_length=8, history_tokens_before=0):
    return _OnlineRequestRuntime(
        {
            "request_id": request_id,
            "session_id": "%s_session" % request_id,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_length": (
                prefill_context_tokens - history_tokens_before),
            "decode_length": decode_length,
            "history_tokens_before": history_tokens_before,
            "prefill_context_tokens": prefill_context_tokens,
            "final_context_tokens": (
                prefill_context_tokens + decode_length),
        },
        PREFILL_CHUNK_SIZE,
    )


def _assert_shard_route(test, shard, hardware):
    """单个序列化 shard 的路由口径:字段集齐、noc_path 为 rank 序列,
    首 = source、末 = target,hops = len-1,与 deterministic_xy_route
    逐点一致。"""
    test.assertTrue(
        {"source_rank", "target_rank", "edge_rank", "bytes", "noc_hops",
         "noc_path"} <= set(shard))
    path = shard["noc_path"]
    test.assertIsInstance(path, list)
    test.assertTrue(path)
    test.assertEqual(path[0], shard["source_rank"])
    test.assertEqual(path[-1], shard["target_rank"])
    test.assertEqual(shard["noc_hops"], len(path) - 1)
    test.assertEqual(
        tuple(path),
        deterministic_xy_route(
            hardware, shard["source_rank"], shard["target_rank"]))


class PrefillDecisionRouteSerializationTest(unittest.TestCase):
    """prefill 决策行三键（问题 4a）：REMOTE 恢复 remote_load + 压力逐出
    remote_store 的 shards[].noc_path 落盘；prefill_evictions = 准入时点
    初始 [];聚合 history_transfer_bytes 保留。"""

    def test_remote_restore_row_carries_shard_routes(self):
        # 小容量单层夹具：oldest(10,t=10) / second(10,t=20) / active(5)
        # 同驻实例 0；L=1 无非空半层后缀，阶段 1 结构性无候选，首受害者
        # 整段直达 REMOTE 是主线行为（先例
        # test_passive_only_admission_pressure_evicts_exactly_enough）；
        # 本测试关注序列化而非逐出序。
        hardware = FaceHardware(
            mesh_rows=2,
            mesh_cols=2,
            local_hbm_capacity_bytes=80,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=2.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        model = FaceModel(
            layers=1, hidden_size=2, ffn_size=2, num_heads=2,
            vocab_size=2, bytes_per_elem=1, mlp_variant="gelu")
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(topology, model)

        def _seed(session_id, tokens, completion_ns=None):
            kv.prepare_prefill(
                session_id=session_id,
                target_instance_index=0,
                history_tokens=0,
                trigger_request_id="%s_seed" % session_id)
            kv.expand_prefill(
                session_id=session_id,
                instance_index=0,
                context_tokens=tokens,
                trigger_request_id="%s_seed" % session_id)
            if completion_ns is not None:
                kv.mark_complete(
                    session_id, completion_ns, next_request_type="human")

        _seed("oldest", 10, 10)
        _seed("second", 10, 20)
        _seed("active", 5)  # 在飞会话（无完成态）
        # 压力逐出（real _ensure_capacity）：oldest 整段 -> REMOTE。
        pressure_evictions = kv.expand_prefill(
            session_id="active", instance_index=0,
            context_tokens=12, trigger_request_id="grow")
        self.assertEqual(len(pressure_evictions), 1)
        self.assertEqual(
            kv.session_snapshot("oldest").location,
            KVCacheManager.REMOTE_MEMORY)
        # REMOTE 会话恢复准入：remote_load（自 edge rank 全层拉回）。
        _, _transfers, prepare_evictions = kv.prepare_prefill(
            session_id="oldest",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="restore",
            action="copy")
        history_transfer = _transfers[0]
        self.assertEqual(history_transfer.kind, "remote_load")
        self.assertTrue(history_transfer.shards)

        scheduler = _decision_shell(
            kv, hardware=hardware, model=model, topology=topology)
        runtime = _make_runtime(
            "restore", prefill_context_tokens=12,
            history_tokens_before=10)
        runtime.estimated_arrival_ns = 0
        runtime.prefill_instance_index = 0
        runtime.prefill_assignment_key = (0, 0, 0)
        runtime.prefill_affinity_reason = "remote_edge_load_balance"
        runtime.admission_time_ns = 100
        runtime.hbm_wait_ns = 100
        runtime.history_transfer = history_transfer
        # 真实准入路径形态：admission(reserve) + prepare 两源逐出拼接。
        runtime.history_evictions = (
            tuple(pressure_evictions) + tuple(prepare_evictions))
        runtime.prefill_evictions = ()  # 准入时点初始值（drain 才赋值）
        runtime.history_transfer_bytes = history_transfer.total_bytes

        scheduler._emit_admission(runtime, 100)
        self.assertEqual(len(scheduler.online_log_rows), 1)
        decision = scheduler.online_log_rows[0]["decision"]

        # 新键 1：history_transfer = remote_load 全摘要 + 逐 shard 路由。
        summary = decision["history_transfer"]
        self.assertEqual(summary["kind"], "remote_load")
        self.assertEqual(
            summary, _transfer_summary(history_transfer))
        self.assertTrue(summary["shards"])
        for shard in summary["shards"]:
            _assert_shard_route(self, shard, hardware)

        # 新键 2：history_evictions = 逐出 KVTransfer 逐条（含路由）。
        evictions = decision["history_evictions"]
        self.assertEqual(
            len(evictions),
            len(pressure_evictions) + len(prepare_evictions))
        for entry, transfer in zip(
                evictions,
                tuple(pressure_evictions) + tuple(prepare_evictions)):
            self.assertEqual(entry, _transfer_summary(transfer))
            for shard in entry["shards"]:
                _assert_shard_route(self, shard, hardware)

        # 新键 3：prefill_evictions = 准入时点初始（与 S1 同时序）。
        self.assertEqual(decision["prefill_evictions"], [])
        # 既有聚合键保留不动。
        self.assertEqual(
            decision["history_transfer_bytes"],
            history_transfer.total_bytes)


class DecodeDecisionRouteSerializationTest(unittest.TestCase):
    """decode 决策行两键（问题 4a）：D18 边界二次决策切换 decode 实例
    后的 prefill_decode_transfer = noc_migrate 含手算 XY 路径；
    decode_evictions 列表落盘；不新增提前置空。

    （D18 修订 2026-09-26：本用例曾钉"drain 不做跨实例迁移、
    prefill_decode_transfer 恒 local_hit"的旧同址协议——该协议已废除；
    强制同址取消后重载场景真实产生 noc_migrate 迁移腿，本名下的路由
    序列化断言自此落在真实跨实例全链上。）"""

    def test_noc_migrate_row_carries_shard_routes(self):
        hardware = _tiny_hardware()
        model = _tiny_model()
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        scheduler = _decision_shell(
            kv, hardware=hardware, model=model, topology=topology,
            no_affinity=True)
        # r0 已在实例 0 准入（预约+会话+qp），实例 0 另有排队负载 ->
        # D18 边界二次决策（式 1b/1c）把 decode 切换到轻载实例 1
        # （跨实例 noc_migrate 全链；预约换址 + 工作副本迁移）。
        runtime = _make_runtime("r0")
        scheduler.runtime_by_request_id["r0"] = runtime
        # joint（§3.1）：预约只覆盖已知 prefill 上下文（history+input），
        # decode 增量按实际进展因果增长——不再预约真实最终长度。
        kv.reserve_request_capacity(
            request_id="r0",
            session_id=runtime.session_id,
            instance_index=0,
            final_context_tokens=runtime.prefill_context_tokens,
        )
        kv.prepare_prefill(
            session_id=runtime.session_id,
            target_instance_index=0,
            history_tokens=0,
            trigger_request_id="r0",
        )
        kv.expand_prefill(
            session_id=runtime.session_id,
            instance_index=0,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id="r0",
        )
        runtime.prefill_instance_index = 0
        scheduler.instances[0].qp.append(runtime)
        runtime.queued_chunk_load_ns = 0  # drain 的 fail-closed 断言要求 0
        # D19 等待接口拆分（2026-09-26）：排队积压不再重压原地接续
        # （不重排队，见 §19.1 P→D 专项）——边界 argmin 翻转机制改用
        # **在服**负载（active decode 成员进原地接续等待基）。
        for seq in range(3):
            heavy = _make_runtime(f"active0_{seq}")
            scheduler.runtime_by_request_id[f"active0_{seq}"] = heavy
            scheduler.instances[0].active_decode.append(heavy)
        self.assertGreater(
            scheduler._task_load_snapshot(
                scheduler.instances[0], 1000).total_task_load_ns,
            scheduler._task_load_snapshot(
                scheduler.instances[1], 1000).total_task_load_ns)

        # D18：P_COMPUTE_DONE -> D_DECISION_PENDING -> D_COMMITTED，
        # 切换 decode 执行实例到 1（严格更低剩余成本才切换，式 1c）。
        # C4 完整动作面：本例被测机器 = 迁移行的 shard 路由披露，允许
        # 面掩成 {copy,stay} 保持迁移臂（copy 迁移执行面由 C4 专项套件
        # 覆盖；stay 放行 = 式 1c 回退腿保持合法——remote/recompute 已
        # 分别随 2026-10-14/2026-10-15 退役）。
        from joint.joint_config import parse_joint_config
        scheduler.joint_config = parse_joint_config(
            {"JOINT_ALLOWED_ACTIONS": "copy,stay"})
        scheduler._joint_mode = scheduler.joint_config.scheduler_mode
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index, 1)
        self.assertFalse(runtime.joint_decode_is_continuation)
        self.assertEqual(
            runtime.prefill_decode_transfer.kind, "noc_migrate")
        self.assertEqual(
            kv._sessions[runtime.session_id].instance_index, 1)

        decode_rows = [row for row in scheduler.online_log_rows
                       if row["kind"] == "decode"]
        self.assertEqual(len(decode_rows), 1)
        decision = decode_rows[0]["decision"]
        # decode 执行实例 1，prefill_decode_transfer = noc_migrate：
        # 逐 shard XY 路由随行序列化（_transfer_summary 全链）。
        self.assertEqual(decision["decode_instance_index"], 1)
        # D18 披露键：边界决策产物与阶段。修复轮三 F7：decision_stage
        # 词表四方对齐为 "P"/"D"（P 行、候选面 notes、ACC、probe 合成
        # 行同此）——状态机细节（D_COMMITTED）由 boundary_transaction/
        # stage_epoch 披露，不再进本词表位。
        self.assertEqual(decision["decision_stage"], "D")
        self.assertEqual(
            decision["joint_decode_action"],
            runtime.joint_decode_action)
        self.assertFalse(decision["is_continuation"])
        boundary = decision["boundary_decision"]
        self.assertEqual(boundary["chosen_instance_index"], 1)
        self.assertFalse(boundary["is_continuation"])
        self.assertFalse(boundary["fallback"])
        self.assertGreater(boundary["decode_steps"], 0)
        summary = decision["prefill_decode_transfer"]
        self.assertEqual(summary["kind"], "noc_migrate")
        self.assertEqual(summary, _transfer_summary(
            runtime.prefill_decode_transfer))
        for shard in summary["shards"]:
            _assert_shard_route(self, shard, hardware)

        # 新键 2：decode_evictions 列表落盘（空目标实例 -> 空列表，
        # 本夹具自然无逐出；列表类型本身即序列化契约）。
        self.assertEqual(decision["decode_evictions"], [])
        # 不新增提前置空：发射后 holder 保留，completion M4 块才核销。
        self.assertIsNotNone(runtime.prefill_decode_transfer)


if __name__ == "__main__":
    unittest.main()
