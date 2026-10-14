#!/usr/bin/env python3
"""test_c4_boundary_full_surface.py -- C4 钉测：D 边界完整动作面的执行
语义（v2.0 §19.1 P→D 专项 / D10 静默重建 / D19 多源与按缺失区间 copy）。

覆盖（执行侧 + 计价口径一致性；视图层基础钉测在 joint/test_d19；
原增量型布局族（P=remote-read@i_P 的 U/H 分置）与多源远读/credit
切片/远读折叠/远读臂用例已随 2026-10-14 远读退役删除——三动作下
边界布局恒完整型单族）：
  1. **完整型 recompute@j 单源形态**：base_* 转记录、重建副本持并集
     → 零载荷合并（重复副本释放）；
  2. **D recompute 静默重建且首 token 只消费一次**：重建 span 前置首
     个 decode 列车（(H_B, H_B) 首 span）、首 token 状态恰一次消费
     （二次消费 fail-closed 由 BoundaryFirstTokenState 构造层钉）；
  3. **JOINT_ALLOWED_ACTIONS=recompute + load-first 全链路**：P
     recompute → 边界 recompute@j → 静默重建 → decode → 完成结算
     （C1 遗留的全链路验收）；
  4. **完整型 copy 回 base 驻留实例就地折叠**（journal 闭合、轮末
     merge = stay）与完成路径 new_tokens 口径钉。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_c4_boundary_full_surface.py -q
"""
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_tokens,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.test_d18_boundary_decision import (  # noqa: E402
    _complete_boundary,
)
from joint.test_d17_schema_version_rounding import (  # noqa: E402
    _loads,
    _loads3,
)
from joint.test_joint_shard_pricing import (  # noqa: E402
    _load,
    _model,
    _request,
    _session,
)
from joint.joint_scheduler import select_instance_and_action  # noqa: E402
from joint.test_joint_mechanisms import _cost_model  # noqa: E402
from online.test_decision_route_serialization import (  # noqa: E402
    _GraphStub,
    _decision_shell,
    _make_runtime,
)
from online.test_joint_arrival_contract import (  # noqa: E402
    R0,
    SERVICE_DONE_NS,
)

HISTORY_TOKENS = 96        # 历史区间 H（token）
INPUT_TOKENS = 16          # 本轮输入 U（token）
HOME_INSTANCE = 2          # 历史权威驻留实例（三实例几何）
P_INSTANCE = 0             # i_P（P 阶段执行实例）
THIRD_INSTANCE = 1         # 第三目标


def _three_instance_topology(hardware):
    """三实例几何（home=2 / i_P=0 / 第三目标=1；2×3 mesh 6 rank，
    逐实例一列 × 两 rank = 轴对齐矩形）。"""
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 3)),
        FaceInstanceSpec("ins1", "2", (1, 4)),
        FaceInstanceSpec("ins2", "3", (2, 5)),
    ))


def _three_hardware():
    return FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=10 ** 9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)


def _instance_bytes(kv, instance_index):
    return tuple(
        kv._rank_states[rank].kv_cache_bytes
        for rank in kv.topology.instance(instance_index).ranks)


# ------------------------------------------------------ 边界事务（KV 账本） --


# （原增量型布局用例六类——C4HomeCopyFillsOnlyU/C4ThirdTargetCopyCoversHAndU/
# C4BaseMaterializationAtExec/C4MultiSourceRemote/C5D12ThreePositionMergeSurface/
# C4RecomputeRebuildElsewhere——已随 2026-10-14 远读退役删除：其会话几何
# （P=remote-read@i_P 的 U@i_P+H@home 分置）只能由 remote-read 准入产生，
# 三动作下不可构造；完整型单族的对应语义由下方
# C4CompleteFamilyTransitionsTest 与 G3CompleteCopyOntoBaseFoldsTest 承接。）


class C4CompleteFamilyTransitionsTest(unittest.TestCase):
    """完整型（P ∈ {stay,copy,recompute}）边界：recompute 的单源形态
    （base_* 转记录，无第二来源）。原 remote 单源形态用例已随
    2026-10-14 远读退役删除。"""

    def _complete_kv(self):
        hardware = _three_hardware()
        model = _tiny_model()
        topology = _three_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=0, trigger_request_id="r0")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        return kv, model

    def test_complete_recompute_elsewhere_single_source(self):
        kv, model = self._complete_kv()
        hb = HISTORY_TOKENS + INPUT_TOKENS
        transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="recompute",
            history_tokens=hb, trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "recompute_rebuild_elsewhere")
        self.assertEqual(outcome["sources"], (P_INSTANCE,))
        session = kv._sessions["s"]
        # 旧 H_B 副本转 base_*（无第二来源）；j 侧零化。
        self.assertEqual(session.base_instance_index, P_INSTANCE)
        self.assertIsNone(session.boundary_source_instance_index)
        self.assertEqual(session.shard_bytes, (0, 0))
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE),
            tuple(int(v) for v in kv_cache_shard_bytes_for_tokens(
                model, hb, kv.tp_degree)))
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=hb + 4, trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0", new_tokens=4)
        self.assertEqual(transfers, ())
        row = kv.last_merge_outcome
        self.assertTrue(row["zero_byte_flip"])
        self.assertEqual(row["winner_instance"], THIRD_INSTANCE)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))

# ------------------------------------------------------------ 计价口径 --

class C4RecomputePricingTest(unittest.TestCase):
    """recompute@j 计价：H_B 完整静默重建（历史基数 0）+ 正常剩余 D
    前向；工作副本不迁移（零准备腿）。"""

    def test_rebuild_charged_full_history_zero_base(self):
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1,
            action="recompute")
        self.assertTrue(candidate.applicable)
        self.assertFalse(candidate.is_continuation)
        # 零准备腿（工作副本不迁移）；计算段 = 重建（H_B × prefill）+
        # decode（8 步 × 2 ns）；merge 0（重建副本持并集）。
        self.assertEqual(candidate.breakdown.history_prep_ns, 0)
        self.assertEqual(
            candidate.breakdown.compute_ns, 100 * 1 + 16)
        self.assertEqual(
            candidate.cost_ns,
            candidate.breakdown.target_wait_ns
            + candidate.breakdown.compute_ns)
        notes = candidate.breakdown.notes
        self.assertIn("boundary_rebuild_tokens=100", notes)
        self.assertIn("boundary_rebuild_silent=1", notes)
        self.assertIn("boundary_rebuild_history_base=0", notes)
        self.assertIn("boundary_rebuild_ns=100", notes)

    def test_in_place_recompute_reuses_space(self):
        """recompute@i_P（完整型）：驻留旧分配空间复用（D10）——足迹
        新增 = 0（不触发驱逐等待），重建计算照常全额。"""
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action="recompute")
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 0)
        self.assertEqual(candidate.breakdown.history_prep_ns, 0)
        self.assertEqual(
            candidate.breakdown.compute_ns, 100 * 1 + 16)
        self.assertFalse(candidate.is_continuation)


# ---------------------------------------------------- 全链路（recompute 臂） --


class _TrainGraphStub(_GraphStub):
    """发射面 stub：列车 + 完成批（merge 零传输 → 无 merge watch）。"""

    last_train_plan = None
    pending_store_tails = {}

    def emit_iteration_train(self, train_plan):
        self.last_train_plan = train_plan
        return {
            "drain_members": {},
            "exit_members": {
                member["request_id"]: {
                    0: "exit_{}".format(member["request_id"]),
                    1: "exit_{}".format(member["request_id"])}
                for member in train_plan["exit_members"]
            },
            "sentinel_members": {},
            "block_ends": {0: "end_0", 1: "end_1"},
            "eviction_watches": [],
        }

    def emit_completion_batch(self, plan, *, block_end_stage="seg2"):
        return {"merge_done_members": []}


def _recompute_arm_shell(decode_length=8):
    """{recompute}+load-first 臂全链路基底：真 KVCacheManager + 真
    drain/train/finalize/完成链（d18 drain 壳 + 列车发射面）。"""
    hardware = _tiny_hardware()
    model = _tiny_model()
    topology = _two_instance_topology(hardware)
    kv = KVCacheManager(topology, model)
    scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                topology=topology)
    scheduler.graph = _TrainGraphStub()
    scheduler.next_request = {}
    scheduler._runtime_index = {}
    scheduler.runtimes = [None]
    scheduler.completed_requests = 0
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(next_trigger_type="human",
                        inter_request_interval_ns=1000)])
    scheduler.train_ledger_rows = []
    scheduler.train_ledger_sink = None
    scheduler._batch["watches"] = []
    scheduler._batch["tick"] = 2000
    scheduler._train_max_iter = 0        # 不截断（列车终点 = 自然边界）
    scheduler._train_instance_index = {}
    scheduler._pending_eviction_watches = {}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler._stalled_by_instance = {}
    scheduler._pending_merge_alarms = {}
    runtime = _make_runtime("r0", decode_length=decode_length)
    scheduler.runtime_by_request_id["r0"] = runtime
    scheduler.next_request["r0"] = None
    scheduler._runtime_index["r0"] = 0
    scheduler.runtimes.append(runtime)
    kv.reserve_request_capacity(
        request_id="r0", session_id=runtime.session_id, instance_index=0,
        final_context_tokens=runtime.prefill_context_tokens)
    kv.prepare_prefill(
        session_id=runtime.session_id, target_instance_index=0,
        history_tokens=0, trigger_request_id="r0")
    kv.expand_prefill(
        session_id=runtime.session_id, instance_index=0,
        context_tokens=runtime.prefill_context_tokens,
        trigger_request_id="r0")
    runtime.prefill_instance_index = 0
    runtime.joint_stage = "P_RUNNING"
    scheduler.instances[0].qp.append(runtime)
    runtime.queued_chunk_load_ns = 0
    config = parse_joint_config({
        "JOINT_ALLOWED_ACTIONS": "recompute",
        "JOINT_SCHEDULER_MODE": "load-first",
    })
    scheduler.joint_config = config
    scheduler._joint_mode = config.scheduler_mode
    # P 阶段动作按臂语义标注（P 工作副本已在实例 0 物化）。
    runtime.joint_action = "recompute"
    # 实例 0 压在服负载 → 边界 argmin 翻转到实例 1（recompute@j≠i_P）。
    for seq in range(3):
        hog = _make_runtime(f"active0_{seq}")
        scheduler.runtime_by_request_id[f"active0_{seq}"] = hog
        scheduler.instances[0].active_decode.append(hog)
        scheduler.instances[0].active_decode_lookup.add(hog)
    return scheduler, kv, runtime


class C4RecomputeArmFullChainTest(unittest.TestCase):
    """JOINT_ALLOWED_ACTIONS=recompute + load-first 全链路：P recompute
    → 边界 recompute@j → 静默重建 → decode → 完成结算。"""

    def test_full_chain_recompute_arm(self):
        scheduler, kv, runtime = _recompute_arm_shell()
        hb = runtime.history_tokens_before + runtime.joint_input_tokens
        # ---- P 腿：真实选择器（load-first + {recompute} 掩码）。----
        config = scheduler.joint_config
        p_model = _cost_model(
            {0: _load(0, 100), 1: _load(1, 5000)},
            allowed_actions=config.allowed_action_set)
        p = select_instance_and_action(
            mode=config.scheduler_mode, cost_model=p_model,
            session=_session(home=0, resident=1), request=_request())
        self.assertEqual(
            (p.chosen.instance_index, p.chosen.action), (0, "recompute"))
        # ---- 边界腿：真 drain → recompute@1（实例 0 在服负载重）。----
        scheduler._on_prefill_drain(runtime.request_id, 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index, 1)
        self.assertEqual(runtime.joint_decode_action, "recompute")
        self.assertFalse(runtime.boundary_decision["fallback"])
        self.assertEqual(
            runtime.boundary_decision["boundary_transition"]["kind"],
            "recompute_rebuild_elsewhere")
        self.assertEqual(runtime.joint_decode_rebuild_tokens, hb)
        self.assertTrue(runtime.joint_decode_rebuild_pending)
        self.assertTrue(runtime.joint_suppress_sampling)
        self.assertTrue(runtime.joint_suppress_output)
        first_token = runtime.joint_boundary_first_token
        self.assertTrue(first_token["issued"])
        self.assertFalse(first_token["consumed"])
        self.assertEqual(first_token["rng_advances"], 1)
        # 会话：旧 H_B 副本留实例 0（base 记录）、实例 1 工作副本零化。
        session = kv._sessions[runtime.session_id]
        self.assertEqual(session.instance_index, 1)
        self.assertEqual(session.base_instance_index, 0)
        self.assertEqual(session.shard_bytes, (0, 0))
        # ---- 列车腿：静默重建 span 前置 + 首 token 恰一次消费。----
        scheduler._plan_and_emit_trains(2000)
        self.assertEqual(runtime.joint_stage, "D_RUNNING")
        train = scheduler.instances[1].in_flight_train
        self.assertIsNotNone(train)
        # 重建 span = 成员首 span（(H_B, H_B)——工作量与上下文终点）。
        self.assertEqual(train["pass_spans"][0], (hb, hb))
        decode_spans = train["pass_spans"][1:]
        self.assertEqual(
            decode_spans,
            [(1, runtime.prefill_context_tokens + step)
             for step in range(1, runtime.decode_length + 1)])
        self.assertFalse(runtime.joint_decode_rebuild_pending)
        # 首 token 恰一次消费（首个 D 前向排定；重复消费被状态守卫拦）。
        self.assertTrue(runtime.joint_boundary_first_token["consumed"])
        self.assertEqual(runtime.joint_boundary_first_token["rng_advances"], 1)
        scheduler._consume_boundary_first_token(runtime)
        self.assertTrue(runtime.joint_boundary_first_token["consumed"])
        # ---- 完成腿：核销 → 重建物化 + 预约注销 → merge → DONE。----
        scheduler._finalize_completed_trains(
            drained=(), completed_now=("r0",), sentinel_trains=(),
            tick=2100)
        # 核销点：重建副本已物化在实例 1（H_B + decode 增长全量）、旧
        # 副本仍在实例 0（零载荷释放发生在 merge）；目标侧预约在重建
        # 物化点注销（"final KV allocation"守卫的因果点）。
        final = kv_cache_shard_bytes_for_tokens(
            scheduler.model,
            runtime.final_context_tokens, kv.tp_degree)
        old_copy = kv_cache_shard_bytes_for_tokens(
            scheduler.model,
            runtime.prefill_context_tokens, kv.tp_degree)
        self.assertEqual(_instance_bytes(kv, 1),
                         tuple(int(v) for v in final))
        self.assertEqual(_instance_bytes(kv, 0),
                         tuple(int(v) for v in old_copy))
        self.assertFalse(runtime.joint_rebuild_reservation_hold)
        self.assertNotIn("r0", kv._reservations)
        # 完成结算（生产由交付处理器在核销后调用——exit 标记核销 +
        # 完成归属实例一致）。
        scheduler._complete_requests(("r0",), 2100)
        self.assertTrue(runtime.completed)
        self.assertEqual(runtime.joint_stage, "SETTLE_DONE")
        self.assertEqual(runtime.decode_tokens_consumed,
                         runtime.decode_length)
        self.assertEqual(session.instance_index, 1)
        self.assertIsNone(session.working_kind)
        # merge 披露：零字节翻转（重建副本持并集；旧副本释放）；终轮
        # 无后续请求 → 会话 KV 随 retire 释放（两实例清零）。
        self.assertTrue(runtime.merge_outcome["zero_byte_flip"])
        self.assertEqual(runtime.merge_outcome["winner_instance"], 1)
        self.assertEqual(_instance_bytes(kv, 0), (0, 0))
        self.assertEqual(_instance_bytes(kv, 1), (0, 0))
        # 首 token 元数据核验：恰一次采样（rng=1）、已消费、重建期抑制。
        self.assertEqual(
            runtime.plan_dict()["joint_boundary_first_token"],
            {"issued": True, "consumed": True, "rng_advances": 1,
             "rebuild_sampling_suppressed": True})



# --------------------------------------------- G3 冒烟矩阵回归钉（2026-09-27）--

DECODE_TOKENS = 8             # D 增长（token）


def _turn0_local_prefill(kv, tokens, request_id):
    """turn-0 完整型 P（本地 stay 起步，无工作副本）。"""
    kv.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id=request_id)
    kv.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=tokens, trigger_request_id=request_id)


class G3CompleteCopyOntoBaseFoldsTest(unittest.TestCase):
    """G3 真实形态二：完整型 copy 的 D 边界迁移落到 base 驻留实例——
    就地并基折叠（R13 消除的组合不得到达轮末 merge）。

    P 阶段 copy 建 C13 逐块交接 journal（home 侧基础轮内释放），D 边
    界再选 copy 回 base 驻留实例：折叠清 working_kind/基础记录、闭合
    journal，轮末 merge = stay 本地提交（不再撞 N9 防御）。
    """

    def _arrange(self):
        hardware = _three_hardware()
        model = _tiny_model()
        kv = KVCacheManager(_three_instance_topology(hardware), model)
        _turn0_local_prefill(kv, HISTORY_TOKENS, "seed")
        kv.mark_complete("s", HISTORY_TOKENS)
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        kv._settle_copy_handoffs(
            "s", boundary="prefill_drain", trigger_request_id="r1")
        return kv, model

    def test_copy_onto_base_folds_and_settles_stay(self):
        kv, model = self._arrange()
        session = kv._sessions["s"]
        self.assertEqual(session.working_kind, "copy")
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        self.assertEqual(outcome["kind"], "complete_copy_base_fold")
        # 折叠后：本地全上下文，无工作副本、无基础记录、journal 闭合。
        self.assertIsNone(session.working_kind)
        self.assertEqual(session.instance_index, HOME_INSTANCE)
        self.assertEqual(
            session.context_tokens, HISTORY_TOKENS + INPUT_TOKENS)
        self.assertIsNone(session.base_instance_index)
        self.assertEqual(session.base_history_tokens, 0)
        self.assertIsNone(session.copy_handoff)
        full = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS, kv.tp_degree)
        self.assertEqual(session.shard_bytes, full)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(value) for value in full))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        # 轮末 merge = stay 本地提交（零流量），守恒不经过 N9 防御。
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=(HISTORY_TOKENS + INPUT_TOKENS
                                  + DECODE_TOKENS),
            trigger_request_id="r1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r1",
            new_tokens=INPUT_TOKENS + DECODE_TOKENS)
        self.assertEqual(transfers, ())
        self.assertEqual(kv.last_merge_outcome["direction"], "stay")
        kv.mark_complete(
            "s", HISTORY_TOKENS + INPUT_TOKENS + DECODE_TOKENS)


# （原远读落 base 折叠/预约 sizing/交接 journal 闭合与 growth-only
# 增量域四类用例已随 2026-10-14 远读退役删除：remote_read_base_fold
# 折叠支路、远读 stage 预约口径与 joint_remote_growth_only 增量域均
# 随远读机制删除；copy 折叠形态由 G3CompleteCopyOntoBaseFoldsTest
# 承接。）


class G3CompletionNewTokensDomainTest(unittest.TestCase):
    """完成路径 new_tokens 口径钉（调度器侧）：三动作完成路径统一按
    input+decode 披露（零字节翻转族口径）。"""

    @staticmethod
    def _completed_scheduler():
        from online.test_joint_arrival_contract import (
            _bare_scheduler,
        )
        scheduler = _bare_scheduler()
        runtime = scheduler.runtime_by_request_id[R0]
        runtime.joint_input_tokens = INPUT_TOKENS
        runtime.decode_length = DECODE_TOKENS
        runtime.decode_tokens_consumed = DECODE_TOKENS
        return scheduler, runtime

    def _merge_new_tokens(self, scheduler):
        for name, kwargs in scheduler.kv_manager.calls:
            if name == "merge_back":
                return kwargs["new_tokens"]
        raise AssertionError("merge_back was not called")

    # （原 growth-only/远读完成路径 new_tokens 两用例已随 2026-10-14
    # 远读退役删除：joint_remote_growth_only 与 ACTION_REMOTE 完成分支
    # 均退役，三动作完成路径统一按 input+decode 披露。）

    def test_flip_family_keeps_disclosure_formula(self):
        scheduler, runtime = self._completed_scheduler()
        runtime.joint_action = "copy"
        runtime.joint_decode_action = "copy"
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        self.assertEqual(
            self._merge_new_tokens(scheduler),
            INPUT_TOKENS + DECODE_TOKENS)


if __name__ == "__main__":
    unittest.main()
