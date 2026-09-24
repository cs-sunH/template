#!/usr/bin/env python3
"""test_eviction_tail_watch_real_paths.py -- 逐出尾 watch 四条生产路径
真实路径钉测（2026-09-24，最小修改"流与配额改由实际 eviction_done 尾
watch 释放"的验证工程师钉测）。

与 test_eviction_side_branch_structure.py（图结构）与
test_joint_quota_integration.py（__new__ 替身骨架）互补：本文件用
**真实 GraphBatchBuilder** + **真实 Sh30OnlineScheduler 方法**（被测方法
零替身；骨架属性对齐 __init__ 初值，C11/F6 销账口径）驱动四条生产路径：

  1. admission history/prefill 逐出旁支（graph_batch_builder.py :2190/:2353
     watch_context；scheduler _emit_admission :5317 起、watch 注册段
     :5322-5352，owner = rid#evict / rid#evict#prefill）；
  2. decode joiner 旁支（graph_batch_builder.py :776-792
     decode_eviction_watch_id 透传；scheduler drain 时先登记
     （_on_prefill_drain :1679-1697 内联块，本文件以同款真实原语镜像——
     该块非独立方法）+ 构图时 _register_train_eviction_watches :2575
     转正）；
  3. 失败准入旁支（graph_batch_builder.py emit_eviction_side_branch
     :2003-2056；scheduler _emit_eviction_only_nodes :5128-5170，
     watch_id = batch_train_evict_<rid>_side_<seq>）；
  4. decode 增长旁支（scheduler :1675-1703 decode_evictions 段 +
     _emit_eviction_only_nodes）。

判据（任务书 (a)-(e)）：
  (a) 主链 drain/service_done 释放边界之后旁支流/配额/watch 仍在场
      （主链释放不提前注销 #evict/#evict#prefill/#flow 旁支 owner）；
  (b) 尾 watch 交付（_on_eviction_watch :2555-2573）仅释放本旁支的流与
      配额——同场其他 owner 与未交付旁支不受扰动；
  (c) run 尾守恒：_pending_eviction_watches 空（:5938-5945）+
      _assert_quota_tracker_ledgers_clean（:3606，:5890 调用）+
      _assert_no_flow_registry_leaks（:3637，:5894 调用，三注册表）+
      _quota_enrolled 空（:5875）；
  (d) 空 shard 分支不造 watch（graph 只对确实发出节点的旁支返回 watch，
      _register_eviction_watch docstring :2512-2513 声明的生产前提）；
  (e) 重复/错阶段回调拒绝（_register_eviction_watch :2506 的
      scheduled-twice / duplicate-owners / changed-owner / outside-batch /
      malformed / fired-before-scheduled / unknown / wrong-stage 分支）。

表覆盖披露：生产现行逐出支链转移恒 remote_store（进链路表 + 池端口
表）；HBM 端口表登记臂（_register_transfer_flows :2650-2655）只对
noc_migrate 生效——decode 增长用例以 API 合法的 noc_migrate 转移驱动
该臂，钉住三注册表借还配对非空转（登记臂本身 = 生产代码原路径）。

配额夹具披露：tracker 为真实 LinkQuotaTracker(static)，Q_init =
floor(B_link/B_HBM) = 10（F2 派生，无外部覆写）——取高 B_link 使夹具
共享 XY 路由不触末槽；配额入册降级（O2 披露）路径不在本测试钉测面。

运行：cd sh_test_mesh &&
      python3 -m pytest workload/llama2_7b_inference/online/test_eviction_tail_watch_real_paths.py -q
（零后端、无 C++ 仿真、总时长 << 30s。）
"""
import os
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    KVTransfer,
    KVTransferShard,
    build_instances,
    deterministic_xy_route,
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_config import JointMechanismConfig  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    JointHardwareRates,
    LinkFlowRegistry,
    SessionKVView,
)
from joint.link_quota import FLOW_ONESHOT, LinkQuotaTracker  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,
    Sh30OnlineScheduler,
)
from online.online_scheduler_base import STAGE_PREFILL  # noqa: E402

PREFILL_RANKS = (0, 1)
DECODE_RANKS = (2, 3)
LAYERS = 2
EDGE_OF_RANK = {0: 2, 1: 3, 2: 2, 3: 3}
SESSION_VICTIM = "session_victim"
SESSION_VICTIM_B = "session_victim_b"
RID = "session_new_request_0"
TICK_ADMIT = 1000
TICK_DELIVER = 2000


# ------------------------------------------------------------ 图侧夹具 --
def _make_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=PREFILL_RANKS, pg_name="tp_prefill"),
            SimpleNamespace(ranks=DECODE_RANKS, pg_name="tp_decode"),
        ],
        layers=LAYERS,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        remote_memory=SimpleNamespace(edge_npus=DECODE_RANKS),
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
    )


def _make_builder():
    builder = GraphBatchBuilder(_make_config())
    builder.begin_batch()
    return builder


# -------------------------------------------------------- 调度器侧夹具 --
def _hardware():
    return FaceHardware(
        mesh_rows=2,
        mesh_cols=4,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=1000.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )


def _model():
    return FaceModel(
        layers=LAYERS,
        hidden_size=64,
        ffn_size=128,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )


def _topology():
    return build_instances(
        _hardware(),
        (FaceInstanceSpec("g0", "pg0", (0, 1)),
         FaceInstanceSpec("g1", "pg1", (2, 3)),
         FaceInstanceSpec("g2", "pg2", (4, 5)),
         FaceInstanceSpec("g3", "pg3", (6, 7))),
        require_equal_size=True)


def _make_scheduler(graph):
    """__new__ 范式（C8/C11/C14 测试同款）：只装配被测路径用到的属性；
    流/端口注册表与配额 tracker 全部为生产真类。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.hardware = _hardware()
    scheduler.model = _model()
    scheduler.topology = _topology()
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=1000.0, pool_port_gbps=1.0,
        local_hbm_gbps=100.0, d2d_latency_ns=0, pool_latency_ns=0)
    scheduler._decode_task_load_cache = {}
    scheduler._prefill_task_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler.instances = [_OnlineInstanceState(index=i) for i in range(4)]
    scheduler._rank_to_instance = {
        rank: instance.index
        for instance in scheduler.topology.instances
        for rank in instance.ranks}
    scheduler.runtime_by_request_id = {}
    scheduler.kv_manager = SimpleNamespace(tp_degree=2, _sessions={})
    scheduler.joint_config = JointMechanismConfig(
        category_mode="typed", scheduler_mode="joint",
        layer_policy="adaptive", remote_actions="on", quota_mode="static")
    scheduler._kv_ledger_epoch = 0
    scheduler._admit_attempt_epoch = {}
    scheduler._admission_failure_state = {}
    scheduler._admit_gate_verify = False
    scheduler.pending_admissions = []
    scheduler.graph = graph
    # 逐出尾 watch 状态（__init__ 同款初值，sh30:790-791）。
    scheduler._pending_eviction_watches = {}
    scheduler._eviction_watch_seq = {}
    # R15 三注册表：生产真类（链路/池端口/HBM 端口）。
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._hbm_ports.attach_active_decode_provider(
        scheduler._hbm_active_decode_streams)
    # C11 配额（真实 tracker；static；Q_init=10 见模块头披露）。
    scheduler._quota_tracker = LinkQuotaTracker(
        mode="static",
        noc_link_bytes_per_ns=1000.0,
        local_hbm_bytes_per_ns=100.0,
        delta_adm_ns=0)
    scheduler._quota_enrolled = {}
    scheduler._quota_merge_reserves = {}
    scheduler._quota_merge_reserves_created = 0
    scheduler._quota_decode_owner_seq = {}
    scheduler._quota_oneshot_overflow_events = 0
    scheduler._quota_admit_events = 0
    scheduler._quota_release_events = 0
    scheduler._joint_action_selection_counts = {
        "stay": 0, "copy": 0, "remote-read": 0,
        "recompute_elected": 0,
        "recompute_forced_no_history": 0,
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
    scheduler._quota_aimd_action_counts = {}
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    scheduler.p_chunk = 512
    scheduler._link_telemetry_epoch_count = 0
    scheduler._link_telemetry_sample_count = 0
    scheduler._telemetry_absent_seen = False
    scheduler._telemetry_window_broken = False
    scheduler._telemetry_window_end_ns = 0
    scheduler._telemetry_last_tick_ns = 0
    scheduler._telemetry_link_id_map = None
    # log_decision / _note_emitted / _ledger_issue 基类契约（__new__ 手工
    # 置初值；ledger 层 = online_scheduler_base.__init__ 同款空账）。
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    scheduler._emitted_by_delivery = {}
    scheduler.ledger_admitted = {}
    scheduler.ledger_committed = {}
    scheduler.ledger_issued = {}
    return scheduler


def _open_batch(scheduler, seq, tick):
    """真实批次累加器（online_scheduler_base._start_batch :808-825 形态；
    delta schema 门不是本测试被测面，故按生产形态直建）。"""
    scheduler._batch = {
        "delivery_sequence": seq,
        "tick": tick,
        "reasons": ["arrivals"],
        "nodes": [],
        "parent_edges": [],
        "watches": [],
        "assignments": [],
        "kv_actions": [],
        "future_alarms": [],
    }
    scheduler._emitted_by_delivery[seq] = {"tick": tick, "requests": []}
    scheduler.graph.begin_batch()


# ------------------------------------------------------------ 转移夹具 --
def _remote_store_eviction(session_id, trigger_request_id, *,
                           source_instance_index, phase, reason):
    """生产同构的受害者整会话外迁：逐 TP rank 真实 XY 路由
    （face_scheduler._remote_store_transfer :3503-3555 同构——
    deterministic_xy_route 现算，路径长度 >= 2 ⇒ 链路表有真实条目）。"""
    source_ranks = (
        PREFILL_RANKS if source_instance_index == 0 else DECODE_RANKS)
    hardware = _hardware()
    shards = tuple(
        KVTransferShard(
            source_rank=rank,
            target_rank=EDGE_OF_RANK[rank],
            edge_rank=EDGE_OF_RANK[rank],
            bytes=1000,
            noc_path=deterministic_xy_route(
                hardware, rank, EDGE_OF_RANK[rank]),
            layer_start=0,
            layer_end=LAYERS)
        for rank in source_ranks)
    return KVTransfer(
        kind="remote_store",
        phase=phase,
        reason=reason,
        session_id=session_id,
        trigger_request_id=trigger_request_id,
        source_instance_index=source_instance_index,
        target_instance_index=None,
        total_bytes=1000 * len(shards),
        shards=shards,
        model_layers=LAYERS,
        layer_start=0,
        layer_end=LAYERS,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=0)


def _noc_migrate_eviction(session_id, trigger_request_id):
    """noc_migrate 转移（API 合法；仅用于驱动 _register_transfer_flows
    的 HBM 端口登记臂 :2650-2655 与 _quota_transfer_footprint 的端点端口
    判据 :2747-2753——生产现行逐出支链恒 remote_store，见模块头披露）。"""
    hardware = _hardware()
    shards = tuple(
        KVTransferShard(
            source_rank=src,
            target_rank=dst,
            edge_rank=None,
            bytes=1000,
            noc_path=deterministic_xy_route(hardware, src, dst),
            layer_start=0,
            layer_end=LAYERS)
        for src, dst in zip(PREFILL_RANKS, DECODE_RANKS))
    return KVTransfer(
        kind="noc_migrate",
        phase="decode",
        reason="decode_growth_capacity",
        session_id=session_id,
        trigger_request_id=trigger_request_id,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=1000 * len(shards),
        shards=shards,
        model_layers=LAYERS,
        layer_start=0,
        layer_end=LAYERS,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=0)


def _admission_runtime(history_evictions, prefill_evictions):
    runtime = _OnlineRequestRuntime({
        "request_id": RID,
        "session_id": "session_new",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": 50,
        "decode_length": 16,
        "history_tokens_before": 100,
        "prefill_context_tokens": 150,
        "final_context_tokens": 166,
    })
    runtime.prefill_instance_index = 0
    runtime.decode_instance_index = 0
    runtime.prefill_assignment_key = (0,)
    runtime.prefill_affinity_reason = "joint_stay"
    runtime.estimated_arrival_ns = 0
    runtime.admission_time_ns = TICK_ADMIT
    runtime.hbm_wait_ns = 0
    runtime.history_evictions = tuple(history_evictions)
    runtime.prefill_evictions = tuple(prefill_evictions)
    return runtime


def _session_view():
    return SessionKVView(
        session_id="session_new", home_instance=0, resident_instance=0,
        location="local_hbm", history_tokens=100,
        resident_prefix_layers=LAYERS,
        history_bytes_by_tp_rank=(400, 400),
        missing_bytes_by_tp_rank=(0, 0))


def _decode_joiner_plan(joiner_evictions, *, watch_id=None):
    joiner = {
        "request_id": RID,
        "session_id": "session_new",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "prefill_context_tokens": 300,
        "decode_evictions": list(joiner_evictions),
        "prefill_decode_transfer": KVTransfer(
            kind="local_hit",
            phase="prefill_to_decode",
            reason="train_test_same_tp_group",
            session_id="session_new",
            trigger_request_id=RID,
            source_instance_index=0,
            target_instance_index=1,
            total_bytes=0,
            shards=(),
            model_layers=LAYERS,
            layer_start=0,
            layer_end=LAYERS,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=LAYERS,
        ),
        "prefill_drain_block_ends": {rank: 0 for rank in PREFILL_RANKS},
    }
    if watch_id is not None:
        joiner["decode_eviction_watch_id"] = watch_id
    return joiner


def _decode_train_plan(joiner):
    return {
        "train_id": "batch_train_i1_1",
        "instance_index": 1,
        "stage": "decode",
        "joiners": [joiner],
        "pass_spans": [(1, 101), (1, 102)],
        "iterations": 1,
        "prefill_start_member": None,
        "first_chunk_member": None,
        "drain_members": [],
        "exit_members": [],
    }


def _drain_time_watch_entry(scheduler, runtime, evictions, tick):
    """_on_prefill_drain :1679-1697 drain 时登记内联块的同款镜像
    （该块非独立方法，无法单独调用；逐行使用同一组真实原语——
    _next_eviction_watch_id / _register_transfer_flows /
    _quota_enroll_eviction_branch / pending dict 形态逐字段一致，
    末行含 :1697 的 runtime 赋值）。

    2026-09-24 修复后全行可镜像：_OnlineRequestRuntime.__slots__ 已补
    ``decode_eviction_watch_id`` 并在 __init__ 置 None（修复前的
    AttributeError 潜伏缺陷由 DrainRegistrationSlotsFixPinTest 钉住
    修复后行为；可达性刻画——现行 joint 模式三来源结构性恒空——
    见该用例 docstring）。"""
    watch_id = scheduler._next_eviction_watch_id(runtime.request_id,
                                                 "decode_joiner")
    flow_owner = watch_id + "#flow"
    scheduler._register_transfer_flows(evictions, owner=flow_owner)
    quota_owner = scheduler._quota_enroll_eviction_branch(
        watch_id + "#quota", runtime.request_id, evictions, tick)
    scheduler._pending_eviction_watches[watch_id] = {
        "request_id": runtime.request_id,
        "flow_owners": (flow_owner,),
        "quota_owners": (
            (quota_owner,) if quota_owner is not None else ()),
        "scheduled": False,
    }
    # sh30_online_scheduler.py:1697 的逐字生产语句（slots 修复后生效）。
    runtime.decode_eviction_watch_id = watch_id
    return watch_id


def _assert_registry_owners(test, scheduler, expected):
    """三注册表 owner 在场断言（逐表显式——登记腿型决定归属：
    remote_store → 链路+池端口（_register_transfer_flows :2646-2649）；
    noc_migrate → 链路+HBM 端点（:2650-2655）；表内多余 owner 一并拒绝）。"""
    actual = {
        "link": set(scheduler._joint_flows.leaked_owners()),
        "pool_port": set(scheduler._pool_ports.leaked_owners()),
        "hbm_port": set(scheduler._hbm_ports.leaked_owners()),
    }
    test.assertEqual(
        actual, {name: set(owners) for name, owners in expected.items()},
        "flow registry owner mismatch")


def _assert_run_tail_clean(test, scheduler):
    """判据 (c)：run 尾守恒断言族（生产调用点 :5875/:5890/:5894/:5938）。"""
    test.assertEqual(scheduler._pending_eviction_watches, {},
                     "run tail :5938 condition violated")
    test.assertEqual(scheduler._quota_enrolled, {},
                     "run tail :5875 condition violated")
    scheduler._assert_quota_tracker_ledgers_clean()   # :3606（:5890 调用）
    scheduler._assert_no_flow_registry_leaks()        # :3637（:5894 调用）


# ======================================================= 路径 1：准入旁支 ==

class AdmissionEvictionWatchRealPathTest(unittest.TestCase):
    """准入 history/prefill 逐出旁支：真实 _emit_admission 全链。"""

    def setUp(self):
        self.builder = _make_builder()
        self.scheduler = _make_scheduler(self.builder)
        _open_batch(self.scheduler, 0, TICK_ADMIT)

    def _setup_production_admission_state(self, runtime):
        """_try_admit_request 的发射前生产序列（:4892-4897 流登记 +
        :4918 _quota_enroll_admission 入册——真实方法，非替身）。"""
        self.scheduler._register_transfer_flows(
            tuple(runtime.history_transfers), owner=runtime.request_id)
        self.scheduler._register_transfer_flows(
            runtime.history_evictions,
            owner=runtime.request_id + "#evict")
        self.assertTrue(self.scheduler._quota_enroll_admission(
            runtime, _session_view(), "stay", 0, TICK_ADMIT))

    def test_admission_history_and_prefill_release_on_tail_watch(self):
        """(a)(b)(c)：drain 边界释放后 #evict/#evict#prefill 仍在场；尾
        watch 逐个交付仅释放各自流与配额；run 尾守恒断言族全过。"""
        runtime = _admission_runtime(
            (_remote_store_eviction(
                SESSION_VICTIM, RID, source_instance_index=0,
                phase="admission",
                reason="admission_capacity_full_fallback"),),
            (_remote_store_eviction(
                SESSION_VICTIM_B, RID, source_instance_index=0,
                phase="admission",
                reason="admission_capacity_full_fallback"),))
        self._setup_production_admission_state(runtime)
        # 同场无关 owner（判定 (b) 的"仅释放本旁支"对照面）。
        edges, _ports = self.scheduler._quota_transfer_footprint(
            runtime.history_evictions)
        self.scheduler._register_transfer_flows(
            runtime.history_evictions, owner="r_other#evict")
        self.assertTrue(self.scheduler._quota_admit_flow(
            "r_other#evict", "r_other", flow_class=FLOW_ONESHOT,
            links=edges, now_ns=TICK_ADMIT))

        # ---- 真实路径 1：_emit_admission（:5317；watch 注册 :5322-5352）。
        self.scheduler._emit_admission(runtime, TICK_ADMIT)

        watches = self.scheduler._batch["watches"]
        self.assertEqual(len(watches), 2)
        self.assertEqual(
            {watch["request_id"] for watch in watches},
            {f"batch_train_evict_{RID}_admission_history",
             f"batch_train_evict_{RID}_admission_prefill"})
        for watch in watches:
            self.assertEqual(watch["stage"], STAGE_PREFILL)
            self.assertEqual(watch["generation"], 0)
            self.assertEqual(watch["statuses"], ["Success", "Skipped"])
            self.assertTrue(watch["members"])
            for rank, node_id in watch["members"].items():
                marker = next(
                    node for node in self.builder.batch["nodes"]
                    if node["rank"] == rank and node["id"] == node_id)
                self.assertIn("eviction_done_rank", marker["name"])
                # C++ 提交预检：成员节点上下文与注册一致（:2614-2616 注释
                # 声明的契约——标记节点 request_id 即 watch id）。
                self.assertEqual(marker["request_id"], watch["request_id"])
        pending = self.scheduler._pending_eviction_watches
        self.assertEqual(
            {watch_id: entry["flow_owners"]
             for watch_id, entry in pending.items()},
            {f"batch_train_evict_{RID}_admission_history":
             (RID + "#evict",),
             f"batch_train_evict_{RID}_admission_prefill":
             (RID + "#evict#prefill",)})
        self.assertTrue(all(entry["scheduled"] for entry in
                            pending.values()))
        self.assertEqual(
            pending[f"batch_train_evict_{RID}_admission_history"]
            ["quota_owners"], (RID + "#evict",))
        self.assertEqual(
            pending[f"batch_train_evict_{RID}_admission_prefill"]
            ["quota_owners"], (RID + "#evict#prefill#quota",))

        # ---- (a)：主链 drain 边界释放（_on_prefill_drain :1588/:1592 的
        # 真实释放方法）后，旁支流/配额/watch 全部仍在场。
        self.scheduler._release_transfer_flows(RID)
        self.scheduler._quota_release_admission_phase(RID, TICK_DELIVER)
        branch_owners = (RID + "#evict", RID + "#evict#prefill",
                         "r_other#evict")
        _assert_registry_owners(self, self.scheduler, expected={
            "link": branch_owners,          # remote_store：链路腿在册
            "pool_port": branch_owners,     # remote_store：池端口腿在册
            "hbm_port": (),                 # 非 noc_migrate：HBM 表不涉
        })
        self.assertIn(RID + "#evict", self.scheduler._quota_enrolled)
        self.assertIn(RID + "#evict#prefill#quota",
                      self.scheduler._quota_enrolled)
        self.assertIn("r_other#evict", self.scheduler._quota_enrolled)
        self.assertEqual(len(self.scheduler._pending_eviction_watches), 2)

        # ---- (b)：history 尾 watch 交付仅释放 #evict 半边。
        history_watch_id = f"batch_train_evict_{RID}_admission_history"
        self.scheduler._on_eviction_watch(
            history_watch_id, STAGE_PREFILL, TICK_DELIVER)
        self.assertNotIn(history_watch_id,
                         self.scheduler._pending_eviction_watches)
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (RID + "#evict#prefill", "r_other#evict"),
            "pool_port": (RID + "#evict#prefill", "r_other#evict"),
            "hbm_port": (),
        })
        self.assertNotIn(RID + "#evict", self.scheduler._quota_enrolled)
        self.assertIn(RID + "#evict#prefill#quota",
                      self.scheduler._quota_enrolled)
        self.assertIn("r_other#evict", self.scheduler._quota_enrolled)

        # ---- (b)：prefill 尾 watch 交付仅释放 #evict#prefill 半边。
        prefill_watch_id = f"batch_train_evict_{RID}_admission_prefill"
        self.scheduler._on_eviction_watch(
            prefill_watch_id, STAGE_PREFILL, TICK_DELIVER)
        _assert_registry_owners(self, self.scheduler, expected={
            "link": ("r_other#evict",),
            "pool_port": ("r_other#evict",),
            "hbm_port": (),
        })
        self.assertNotIn(RID + "#evict#prefill#quota",
                         self.scheduler._quota_enrolled)
        self.assertIn("r_other#evict", self.scheduler._quota_enrolled)

        # ---- (c)：清场后 run 尾守恒断言族（生产真方法）。
        self.scheduler._release_transfer_flows("r_other#evict")
        self.scheduler._quota_release_owner("r_other#evict",
                                            now_ns=TICK_DELIVER)
        self.assertEqual(self.scheduler._quota_admit_events,
                         self.scheduler._quota_release_events)
        _assert_run_tail_clean(self, self.scheduler)

    def test_admission_empty_shard_evictions_make_no_watch(self):
        """(d) 路径 1：空 shard 逐出无图节点 ⇒ 不造 watch/流/配额
        （graph_batch_builder 只对确实发出节点的旁支返回 watch）。"""
        victim = _remote_store_eviction(
            SESSION_VICTIM, RID, source_instance_index=0,
            phase="admission", reason="admission_capacity_full_fallback")
        runtime = _admission_runtime(
            (replace(victim, total_bytes=0, shards=()),), ())
        # 生产发射前序列对空 shard 转移为零登记（footprint 空 ⇒ 不入册）。
        self.scheduler._register_transfer_flows(
            runtime.history_evictions, owner=RID + "#evict")
        self.assertTrue(self.scheduler._quota_enroll_admission(
            runtime, _session_view(), "stay", 0, TICK_ADMIT))
        self.assertEqual(self.scheduler._quota_enrolled, {})

        self.scheduler._emit_admission(runtime, TICK_ADMIT)

        self.assertEqual(self.scheduler._batch["watches"], [])
        self.assertEqual(self.scheduler._pending_eviction_watches, {})
        self.assertFalse(any(
            "history_evictions" in node["name"]
            or "eviction_done_rank" in node["name"]
            for node in self.builder.batch["nodes"]))
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (), "pool_port": (), "hbm_port": ()})
        _assert_run_tail_clean(self, self.scheduler)


# ============================================ 路径 2+4：decode joiner/增长 ==

class DecodeJoinerAndGrowthWatchRealPathTest(unittest.TestCase):
    """decode joiner 旁支（drain 先登记 → 构图转正 → 尾 watch 释放）与
    decode 增长旁支（_emit_eviction_only_nodes）联合生命周期。"""

    def setUp(self):
        self.builder = _make_builder()
        self.scheduler = _make_scheduler(self.builder)
        _open_batch(self.scheduler, 0, TICK_ADMIT)
        # decode 实例 frontier（结构测试同款 fork 锚）。
        for rank in DECODE_RANKS:
            self.builder.builders[rank].comp(
                f"decode_frontier_rank{rank}", 1, 1)

    def test_joiner_watch_registered_at_drain_released_on_tail_watch(self):
        """(a)(b)(c)+路径 2/4：drain 内联块登记 → 真实构图返回同 id
        watch → _register_train_eviction_watches 转正 → 主链释放不提前
        注销 → 两个旁支 watch 逐个交付各自释放 → run 尾守恒。"""
        runtime = _admission_runtime((), ())
        joiner_evictions = (_remote_store_eviction(
            SESSION_VICTIM, RID, source_instance_index=1, phase="decode",
            reason="decode_capacity_full_fallback"),)
        # 路径 4：decode 增长经真实 _emit_eviction_only_nodes 发射
        # （noc_migrate 驱动 HBM 端口表登记臂，见模块头披露）。
        growth_evictions = (_noc_migrate_eviction(SESSION_VICTIM_B, RID),)

        # ---- 路径 2 前半：drain 时先登记（:1679-1697 同款真实原语）。
        joiner_watch_id = _drain_time_watch_entry(
            self.scheduler, runtime, joiner_evictions, TICK_ADMIT)
        self.assertEqual(joiner_watch_id,
                         f"batch_train_evict_{RID}_decode_joiner_0000")
        self.assertFalse(
            self.scheduler._pending_eviction_watches[joiner_watch_id]
            ["scheduled"])
        # 边缘同 rank 逐出（路径长 1）配额足迹为空 ⇒ 生产语义 = 不入册
        # （_quota_enroll_eviction_branch :2590-2592 空足迹早退）。
        self.assertEqual(
            self.scheduler._pending_eviction_watches[joiner_watch_id]
            ["quota_owners"], ())
        # decode 主链 owner（:1706-1708 同款登记；稍后作 (a) 对照面）。
        self.scheduler._register_transfer_flows(
            runtime.prefill_evictions, owner=RID + "#decode")

        # ---- 路径 4：decode 增长旁支经真实 _emit_eviction_only_nodes。
        growth_watch = self.scheduler._emit_eviction_only_nodes(
            growth_evictions, TICK_ADMIT, trigger_request_id=RID)
        growth_watch_id = growth_watch["request_id"]
        self.assertEqual(growth_watch_id,
                         f"batch_train_evict_{RID}_side_0000")
        # branch 字段由图侧发射器统一标注（emit_eviction_side_branch
        # :2054 的返回形态；调度器不改写——通道差异由 watch id 相位区分）。
        self.assertEqual(growth_watch["branch"], "failed_admission")
        # kv_eviction 咽喉点行（R17-1b，:5152-5158）已落。
        self.assertEqual(
            [row["kind"] for row in self.scheduler.online_log_rows
             if row["kind"] == "kv_eviction"],
            ["kv_eviction"])
        # HBM 端口表确有在册条目（(c) 非空转前提）。
        self.assertIn(growth_watch_id + "#flow",
                      self.scheduler._hbm_ports.leaked_owners())

        # ---- 路径 2 后半：真实构图（joiner 计划透传 drain 时 watch id，
        # graph_batch_builder :776-780）→ _register_train_eviction_watches
        # 转正（:2575-2583）。watch id 经 :1236-1238 的生产消费面
        # （getattr(runtime, "decode_eviction_watch_id", None)）取自
        # drain 登记写入的字段，非测试直填。
        self.assertEqual(runtime.decode_eviction_watch_id, joiner_watch_id)
        joiner = _decode_joiner_plan(
            joiner_evictions,
            watch_id=getattr(runtime, "decode_eviction_watch_id", None))
        result = self.builder.emit_iteration_train(_decode_train_plan(joiner))
        self.assertEqual(len(result["eviction_watches"]), 1)
        graph_watch = result["eviction_watches"][0]
        self.assertEqual(graph_watch["request_id"], joiner_watch_id)
        self.assertEqual(graph_watch["branch"], "decode_joiner")
        self.assertEqual(graph_watch["owner_request_id"], RID)
        self.assertEqual(
            {node["request_id"] for node in self.builder.batch["nodes"]
             if node["name"].startswith("eviction_done_rank")},
            # 批累加器含本测试已发射的两条旁支尾标记（joiner + growth）。
            {joiner_watch_id, growth_watch_id})
        self.scheduler._register_train_eviction_watches(result)
        self.assertTrue(
            self.scheduler._pending_eviction_watches[joiner_watch_id]
            ["scheduled"])
        self.assertEqual(len(self.scheduler._batch["watches"]), 2)

        # ---- (a)：decode 主链释放（service_done 边界 _release_transfer_
        # flows(rid#decode) :1839 + _quota_release_decode_phase :3122）后
        # 两个旁支 owner 与 watch 仍在场（joiner 逐出源=边缘同 rank，路径
        # 长 1 不进链路表、只进池端口表；growth noc_migrate 进链路+HBM 表）。
        self.scheduler._release_transfer_flows(RID + "#decode")
        self.scheduler._quota_release_decode_phase(RID, TICK_DELIVER)
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (growth_watch_id + "#flow",),
            "pool_port": (joiner_watch_id + "#flow",),
            "hbm_port": (growth_watch_id + "#flow",),
        })
        self.assertEqual(len(self.scheduler._pending_eviction_watches), 2)

        # ---- (b)：增长 watch 交付仅释放其流/配额；joiner 旁支不动。
        self.scheduler._on_eviction_watch(
            growth_watch_id, STAGE_PREFILL, TICK_DELIVER)
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (), "pool_port": (joiner_watch_id + "#flow",),
            "hbm_port": ()})
        self.assertNotIn(growth_watch_id + "#quota",
                         self.scheduler._quota_enrolled)
        # joiner 旁支无配额足迹（见上），其流半边仍在场由上方注册表断言
        # 覆盖——交付前不被增长 watch 的释放波及。

        # ---- (b)：joiner watch 交付仅释放 drain 登记的旁支 owner。
        self.scheduler._on_eviction_watch(
            joiner_watch_id, STAGE_PREFILL, TICK_DELIVER)
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (), "pool_port": (), "hbm_port": ()})
        self.assertEqual(self.scheduler._quota_enrolled, {})
        self.assertEqual(self.scheduler._quota_admit_events,
                         self.scheduler._quota_release_events)
        _assert_run_tail_clean(self, self.scheduler)

    def test_joiner_without_drain_registration_rejected(self):
        """(e)+路径 2：无 drain 时登记的同族 watch，构图转正 fail-closed
        （:2579-2582"has no drain-time owner registration"）；图侧缺省
        watch id 合成口径（:777-780）一并钉住。"""
        joiner = _decode_joiner_plan(
            (_remote_store_eviction(
                SESSION_VICTIM, RID, source_instance_index=1,
                phase="decode", reason="decode_capacity_full_fallback"),),
            watch_id=None)
        result = self.builder.emit_iteration_train(_decode_train_plan(joiner))
        self.assertEqual(len(result["eviction_watches"]), 1)
        graph_watch = result["eviction_watches"][0]
        # 图侧缺省合成 id（watch_id None 分支）。
        self.assertEqual(graph_watch["request_id"],
                         f"batch_train_evict_{RID}_decode_batch_train_i1_1")
        with self.assertRaisesRegex(
                RuntimeError, "has no drain-time owner registration"):
            self.scheduler._register_train_eviction_watches(result)


# ================================================== 路径 3：失败准入旁支 ==

class FailedAdmissionWatchRealPathTest(unittest.TestCase):
    """失败准入已提交逐出：真实 _emit_eviction_only_nodes 全链。"""

    def setUp(self):
        self.builder = _make_builder()
        self.scheduler = _make_scheduler(self.builder)
        _open_batch(self.scheduler, 0, TICK_ADMIT)

    def test_failed_admission_branch_releases_on_tail_watch(self):
        """(a)(b)(c)+路径 3：失败准入旁支发射/披露/注册 → 主链回滚边界
        （_rollback_admission_registrations :5110-5112 真实释放）不提前
        注销 → 尾 watch 交付释放 → run 尾守恒。"""
        evictions = (_remote_store_eviction(
            SESSION_VICTIM, RID, source_instance_index=0, phase="admission",
            reason="admission_failed_capacity_full"),)

        # ---- 真实路径 3：_emit_eviction_only_nodes（:5128-5170）。
        watch = self.scheduler._emit_eviction_only_nodes(
            evictions, TICK_ADMIT, trigger_request_id=RID)
        watch_id = watch["request_id"]
        self.assertEqual(watch_id, f"batch_train_evict_{RID}_side_0000")
        self.assertEqual(watch["owner_request_id"], RID)
        self.assertEqual(watch["branch"], "failed_admission")
        self.assertTrue(watch["members"])
        # 图侧节点与尾标记（evict_ 前缀 action 命名 + eviction_done 标记）。
        self.assertTrue(any(
            node["name"].startswith("evict_")
            for node in self.builder.batch["nodes"]))
        self.assertEqual(
            {node["request_id"] for node in self.builder.batch["nodes"]
             if node["name"].startswith("eviction_done_rank")},
            {watch_id})
        # 决策日志披露（R17-1b 咽喉点）。
        kv_rows = [row for row in self.scheduler.online_log_rows
                   if row["kind"] == "kv_eviction"]
        self.assertEqual(len(kv_rows), 1)
        self.assertEqual(kv_rows[0]["request_id"], RID)
        # watch 注册形态：独立 flow/quota owner（watch_id#flow/#quota）。
        pending = self.scheduler._pending_eviction_watches[watch_id]
        self.assertTrue(pending["scheduled"])
        self.assertEqual(pending["flow_owners"], (watch_id + "#flow",))
        self.assertEqual(pending["quota_owners"], (watch_id + "#quota",))
        self.assertIn(watch_id + "#quota", self.scheduler._quota_enrolled)

        # ---- 图侧缺省 watch id（emit_eviction_side_branch watch_id=None
        # 分支 :2021-2023）：failed_q 形态一并钉住。序号取该 trigger 的
        # per-transfer action state（:2019-2023）——上一分支已消费 seq 0，
        # 故缺省分支 = q001（同 trigger 发射序延续口径）。
        default_watch = self.builder.emit_eviction_side_branch(
            evictions, TICK_ADMIT)
        self.assertEqual(default_watch["request_id"],
                         f"batch_train_evict_{RID}_failed_q001")
        # 该缺省分支不经过调度器注册——不得凭空出现 pending 条目。
        self.assertNotIn(default_watch["request_id"],
                         self.scheduler._pending_eviction_watches)

        # ---- (a)：准入失败主链回滚边界（:5110-5112 真实释放）后旁支仍在。
        self.scheduler._release_transfer_flows(RID)
        self.scheduler._release_transfer_flows(RID + "#evict")
        self.scheduler._release_transfer_flows(RID + "#readplan")
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (watch_id + "#flow",),
            "pool_port": (watch_id + "#flow",),
            "hbm_port": (),
        })
        self.assertIn(watch_id, self.scheduler._pending_eviction_watches)

        # ---- (b)：尾 watch 交付仅释放本旁支。
        self.scheduler._on_eviction_watch(
            watch_id, STAGE_PREFILL, TICK_DELIVER)
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (), "pool_port": (), "hbm_port": ()})
        self.assertNotIn(watch_id + "#quota",
                         self.scheduler._quota_enrolled)
        self.assertEqual(self.scheduler._quota_admit_events,
                         self.scheduler._quota_release_events)
        _assert_run_tail_clean(self, self.scheduler)

    def test_emit_eviction_only_nodes_empty_cases_make_no_watch(self):
        """(d) 路径 3/4：空转移集与空 shard 转移都不造 watch/流/配额
        （:5146-5147 早退 + emit_eviction_side_branch members 空返 None，
        flow/quota/watch 注册段 :5161-5169 不可达）。"""
        self.assertIsNone(self.scheduler._emit_eviction_only_nodes(
            (), TICK_ADMIT, trigger_request_id=RID))
        victim = _remote_store_eviction(
            SESSION_VICTIM, RID, source_instance_index=0, phase="admission",
            reason="admission_failed_capacity_full")
        self.assertIsNone(self.scheduler._emit_eviction_only_nodes(
            (replace(victim, total_bytes=0, shards=()),),
            TICK_ADMIT, trigger_request_id=RID))
        self.assertEqual(self.scheduler._pending_eviction_watches, {})
        self.assertEqual(self.scheduler._batch["watches"], [])
        self.assertFalse(any(
            "eviction_done_rank" in node["name"]
            for node in self.builder.batch["nodes"]))
        _assert_registry_owners(self, self.scheduler, expected={
            "link": (), "pool_port": (), "hbm_port": ()})
        _assert_run_tail_clean(self, self.scheduler)


# ==================================================== 判据 (e)：守卫分支 ==

class EvictionWatchGuardBranchesTest(unittest.TestCase):
    """_register_eviction_watch / _on_eviction_watch 的重复/错阶段/未知
    回调拒绝分支（全部真实方法，fail-closed 语义钉死）。"""

    def setUp(self):
        self.builder = _make_builder()
        self.scheduler = _make_scheduler(self.builder)
        _open_batch(self.scheduler, 0, TICK_ADMIT)

    def _real_watch(self):
        """经真实 _emit_eviction_only_nodes 产出一枚已注册 watch。"""
        watch = self.scheduler._emit_eviction_only_nodes(
            (_remote_store_eviction(
                SESSION_VICTIM, RID, source_instance_index=0,
                phase="admission", reason="admission_failed"),),
            TICK_ADMIT, trigger_request_id=RID)
        return watch

    def test_malformed_or_empty_watch_rejected(self):
        bad_id = {"request_id": "not_batch_train", "owner_request_id": RID,
                  "branch": "failed_admission",
                  "members": {0: 1}}
        with self.assertRaisesRegex(
                RuntimeError, "malformed or empty eviction watch"):
            self.scheduler._register_eviction_watch(bad_id)
        empty_members = {"request_id": f"batch_train_evict_{RID}_side_0000",
                         "owner_request_id": RID,
                         "branch": "failed_admission", "members": {}}
        with self.assertRaisesRegex(
                RuntimeError, "malformed or empty eviction watch"):
            self.scheduler._register_eviction_watch(empty_members)

    def test_register_outside_batch_rejected(self):
        self.scheduler._batch = None
        watch = {"request_id": f"batch_train_evict_{RID}_side_0000",
                 "owner_request_id": RID, "branch": "failed_admission",
                 "members": {0: 1}}
        with self.assertRaisesRegex(
                RuntimeError, "registered outside a batch"):
            self.scheduler._register_eviction_watch(watch)

    def test_scheduled_twice_rejected(self):
        watch = self._real_watch()
        self.assertEqual(len(self.scheduler._batch["watches"]), 1)
        with self.assertRaisesRegex(RuntimeError, "scheduled twice"):
            self.scheduler._register_eviction_watch(watch)
        # 拒绝发生在追加之前：批内 watch 恰一枚。
        self.assertEqual(len(self.scheduler._batch["watches"]), 1)

    def test_duplicate_owners_rejected(self):
        """drain 先登记（scheduled=False、已带 owner）→ 构图阶段再带
        owner 注册 = duplicate owners（:2542-2545）。"""
        runtime = _admission_runtime((), ())
        watch_id = _drain_time_watch_entry(
            self.scheduler, runtime,
            (_remote_store_eviction(
                SESSION_VICTIM, RID, source_instance_index=1,
                phase="decode", reason="decode_capacity_full_fallback"),),
            TICK_ADMIT)
        watch = {"request_id": watch_id, "owner_request_id": RID,
                 "branch": "decode_joiner", "members": {2: 1, 3: 2}}
        with self.assertRaisesRegex(RuntimeError, "duplicate owners"):
            self.scheduler._register_eviction_watch(
                watch, flow_owners=("someone_else#flow",))

    def test_fired_before_scheduled_rejected_entry_survives(self):
        """drain 已登记但未转正（scheduled=False）时 watch 提前交付 =
        fired-before-scheduled，条目保留（:2561-2564）。"""
        runtime = _admission_runtime((), ())
        watch_id = _drain_time_watch_entry(
            self.scheduler, runtime,
            (_remote_store_eviction(
                SESSION_VICTIM, RID, source_instance_index=1,
                phase="decode", reason="decode_capacity_full_fallback"),),
            TICK_ADMIT)
        with self.assertRaisesRegex(
                RuntimeError, "fired before it was scheduled"):
            self.scheduler._on_eviction_watch(
                watch_id, STAGE_PREFILL, TICK_DELIVER)
        self.assertIn(watch_id, self.scheduler._pending_eviction_watches)
        # 在场不被消费：交付后流/配额仍登记（源=边缘同 rank 的 remote_store
        # 路径长 1 → 流登记落在池端口表）。
        self.assertIn(watch_id + "#flow",
                      self.scheduler._pool_ports.leaked_owners())

    def test_unknown_and_duplicate_delivery_rejected(self):
        with self.assertRaisesRegex(
                RuntimeError, "unknown or duplicate eviction watch"):
            self.scheduler._on_eviction_watch(
                f"batch_train_evict_{RID}_side_0042", STAGE_PREFILL,
                TICK_DELIVER)
        watch = self._real_watch()
        self.scheduler._on_eviction_watch(
            watch["request_id"], STAGE_PREFILL, TICK_DELIVER)
        # 交付即销账（:2569 del）：同 id 二次交付落入 unknown-or-duplicate。
        with self.assertRaisesRegex(
                RuntimeError, "unknown or duplicate eviction watch"):
            self.scheduler._on_eviction_watch(
                watch["request_id"], STAGE_PREFILL, TICK_DELIVER)

    def test_wrong_stage_rejected_entry_survives_then_delivers(self):
        """错阶段交付拒绝且不销账（stage 判据先于 del），随后正确阶段
        交付成功——旁支流与配额最终只经真实交付释放。"""
        watch = self._real_watch()
        watch_id = watch["request_id"]
        with self.assertRaisesRegex(
                RuntimeError, "fired on unexpected stage"):
            self.scheduler._on_eviction_watch(watch_id, "decode",
                                              TICK_DELIVER)
        self.assertIn(watch_id, self.scheduler._pending_eviction_watches)
        self.scheduler._on_eviction_watch(
            watch_id, STAGE_PREFILL, TICK_DELIVER)
        self.assertNotIn(watch_id, self.scheduler._pending_eviction_watches)
        _assert_run_tail_clean(self, self.scheduler)

    def test_owner_change_rejected(self):
        """同 watch id 换 owner = changed owner（:2535-2538）。"""
        runtime = _admission_runtime((), ())
        watch_id = _drain_time_watch_entry(
            self.scheduler, runtime,
            (_remote_store_eviction(
                SESSION_VICTIM, RID, source_instance_index=1,
                phase="decode", reason="decode_capacity_full_fallback"),),
            TICK_ADMIT)
        watch = {"request_id": watch_id, "owner_request_id": "r_impostor",
                 "branch": "decode_joiner", "members": {2: 1, 3: 2}}
        with self.assertRaisesRegex(RuntimeError, "changed owner from"):
            self.scheduler._register_eviction_watch(watch)


# ===================== 逐出尾 watch slots 修复钉测（2026-09-24）============

class DrainRegistrationSlotsFixPinTest(unittest.TestCase):
    """2026-09-24 修复行为的钉测：_OnlineRequestRuntime 此前 __slots__
    不含 ``decode_eviction_watch_id`` 且无 __dict__，_on_prefill_drain
    :1697 的 drain 登记赋值一旦执行必 AttributeError（潜伏缺陷——
    :1679 门在现行 joint 模式下结构性不可达，见下）。修复 = slots 补
    声明 + __init__ 置 None；本用例钉住修复后行为：

    - slot 在册、初值 None（:1236 消费面 getattr 容缺省的显式化）；
    - 生产语句逐字执行成功且值可读回。

    可达性刻画（维持登记，非缺陷移除理由）：:1679 门 = decode_
    evictions 非空且含 shard；三个来源在现行 joint 模式下结构性恒空
    ——move_request_capacity_reservation 同实例恒 ()（face_scheduler
    :2359-2377，红线 #4）、move_prefill_to_decode 同实例恒 ()（:4630-
    4645）、decode_growth_evictions 字面 ()（sh30:1675）——decode
    joiner 逐出 watch 链路现行 run 不 engag；一旦任何来源产生
    decode_evictions（P/D 拆分激活、红线松动、策略变更），链路即按
    本文件路径 2 的钉测形态运转。"""

    def test_production_drain_assignment_succeeds_after_slot_fix(self):
        runtime = _OnlineRequestRuntime({
            "request_id": RID,
            "session_id": "session_new",
            "turn_index": 0,
            "queue_index": 0,
            "prefill_length": 50,
            "decode_length": 16,
            "history_tokens_before": 100,
            "prefill_context_tokens": 150,
            "final_context_tokens": 166,
        })
        self.assertIn("decode_eviction_watch_id",
                      type(runtime).__slots__)
        self.assertIsNone(runtime.decode_eviction_watch_id)
        # sh30_online_scheduler.py:1697 的逐字生产语句。
        runtime.decode_eviction_watch_id = (
            f"batch_train_evict_{RID}_decode_joiner_0000")
        self.assertEqual(runtime.decode_eviction_watch_id,
                         f"batch_train_evict_{RID}_decode_joiner_0000")


if __name__ == "__main__":
    unittest.main()
