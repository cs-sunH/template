#!/usr/bin/env python3
"""test_acc_acceptance_formula_conservation.py -- 验收纪律第 1 类逐项核对
后的**补齐**钉测（ACC；joint修改文档「验收纪律（v2.0 §19 三类，全部纳
入）」/ v2.0 §19.1 公式与守恒测试）。

本文件只补既有钉测未覆盖的清单项，**不重复既有覆盖**。逐项核对映射
（通用清单 → 既有看护 → 本文件补齐）：

  | 清单项                     | 既有看护（文件:类）                                  | 本文件 |
  |---------------------------|-----------------------------------------------------|--------|
  | 零历史/零传输              | test_joint_fix1_pricing ZeroHistory；quota 零传输支路 | — |
  | TP 不均匀                  | test_face_scheduler tp6 whole-head exact              | 补（§19.1 整头不整除 + 头复制 padding 记账 + 逐 rank 真值进边界空间等待）|
  | KV 头复制/padding          | （无直接钉）                                          | 补 |
  | 单跳/多跳                  | test_d04 far/near restore by hops                     | — |
  | 共享/独立池端口            | test_d04 SharedVsIndependentPoolPortTest              | — |
  | copy 层间迟到              | test_d02 FirstForwardRecurrenceTest 110ns late layer | — |
  | remote 双缓冲多来源/尾块   | （已随远读退役删除，2026-10-14）                      | — |
  | 长上下文 / 短/长 decode    | test_d08 long/short context decode differs            | — |
  | 干净/脏驱逐                | test_d11 clean/dirty split + atom counts              | — |
  | source 引用保护            | test_d20 StageReferenceProtectionTest                 | — |
  | 二/三处 merge 平局与容量回退| test_d12（tie/fallback/three-position）               | — |
  | 实际增长超预测             | channel2 decode-growth eviction row（发射面）          | 补（管理器级：承诺外因果增长真实 T+E；超预约天花板 fail-closed；Z→O 转账不双计）|
  | token 与前向步数关系       | test_d14（冷启动 32 步/边界 max(1,n̂)）+ test_d18 步精确增长 + test_d19 首 token 采样契约 | — |

P→D 专项（§19.1 第二段）逐条核对映射：

  | 专项条目                        | 既有看护                          | 本文件 |
  |--------------------------------|-----------------------------------|--------|
  | 1 原地 remote 不变成本地 hit     | （已随远读退役删除）              | — |
  | 2 回 home 仅补 U                | test_d19 HomeCopyFillsOnlyU       | — |
  | 3 第三目标来源齐全               | test_d19 ThirdTargetTwoSource     | — |
  | 4 D recompute 不重复输出/采样    | test_d19 DRecomputeSilentRebuild  | — |
  | 5 首 token 已终止时不选 D        | test_d18 terminal settles         | — |
  | 6 控制费用使小收益迁移不再合算    | （仅等价/严格改进钉，无控制费联合钉）| 补 |
  | 7 原地接续不重排队               | test_d19 WaitInterfaceSplit       | — |
  | 8 P 的未完 ack/引用不消失        | test_d20 引用/幂等 epoch          | — |
  | 9 新目标容量提交失败仍保持旧方案  | test_d20 abort + test_d18 fallback| — |
  | 10 第一笔新物理工作后不再次选点   | test_d20 重复通知幂等（同回调）     | 补（环境变化后的迟到重入不重选点 + argmin 确已翻转的正控制）|
  | 11 旧/新增长预留不重复计入        | test_d20 式20a rebase/gap         | — |
  | 12 P 末尾不预发 D                | test_d20 composer 拒绝未提交 joiner| 补（状态机面：真实边界前零 D 发射）|
  | 13 同一回调不提交两次             | test_d20 duplicate notification   | — |

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_acc_acceptance_formula_conservation.py   （或 pytest）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceModel,
    attention_heads_by_tp_rank,
    kv_cache_bytes_for_tokens,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
)
from joint.test_d18_boundary_decision import (  # noqa: E402
    KV_TOKEN_BYTES,
    _complete_boundary,
    _drain_ready_scheduler,
    _load,
    _loads,
    _model,
)
from joint.test_joint_mechanisms import _seed  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ACTION_STAY,
    InstanceLoadView,
)


# ------------------------------------------------------------ 通用清单 --


class TPUnevenAndHeadReplicationTest(unittest.TestCase):
    """式（2）逐 rank KV 字节 = 2·h^KV_{g,r}·d_g·s_KV：整头不整除与
    头复制 padding 按物理布局记录，逻辑总量守恒（§19.1 TP 不均匀与
    KV 头复制）。"""

    def _model(self, num_heads, hidden_size):
        return FaceModel(
            layers=4, hidden_size=hidden_size, ffn_size=64,
            num_heads=num_heads, vocab_size=128, bytes_per_elem=2,
            mlp_variant="swiglu")

    def test_uneven_tp_whole_head_partition_is_exact(self):
        """头数不整除 TP：逐 rank 字节 = 自身整头数 × 每头字节，不是
        总量机械除以 TP；合计守恒。"""
        model = self._model(num_heads=7, hidden_size=28)  # head_dim = 4
        shards = kv_cache_shard_bytes_for_tokens(model, 1_000, 3)
        bytes_per_head = (
            2 * model.layers * 1_000
            * (model.hidden_size // model.num_heads)
            * model.bytes_per_elem)
        self.assertEqual(
            attention_heads_by_tp_rank(model.num_heads, 3), (3, 2, 2))
        self.assertEqual(
            shards,
            (3 * bytes_per_head, 2 * bytes_per_head, 2 * bytes_per_head))
        # 无一 rank 拿到"总量/TP"的机械均分值。
        mechanical = sum(shards) // 3
        self.assertNotIn(mechanical, shards)
        self.assertEqual(sum(shards), kv_cache_bytes_for_tokens(model, 1_000))

    def test_tp_beyond_head_count_records_zero_kv_padding(self):
        """TP 头数超过 KV 头数（复制式部署面）：padding rank 的物理
        KV = 0（按物理布局如实记录），持头 rank 精确；合计 == 逻辑有
        效载荷（真实副本不双计入逻辑量）。层区间口径同守恒。"""
        model = self._model(num_heads=4, hidden_size=16)  # head_dim = 4
        shards = kv_cache_shard_bytes_for_tokens(model, 500, 8)
        bytes_per_head = (
            2 * model.layers * 500
            * (model.hidden_size // model.num_heads)
            * model.bytes_per_elem)
        self.assertEqual(
            attention_heads_by_tp_rank(model.num_heads, 8),
            (1, 1, 1, 1, 0, 0, 0, 0))
        self.assertEqual(
            shards,
            tuple(count * bytes_per_head
                  for count in (1, 1, 1, 1, 0, 0, 0, 0)))
        self.assertEqual(sum(shards), kv_cache_bytes_for_tokens(model, 500))
        # 层区间（后缀卸载口径）：padding rank 同为 0，合计 = 每头每层
        # 字节 × 区间层数（整数闭式）。
        ranged = kv_cache_shard_bytes_for_layer_range(
            model, 500, 8, layer_start=1, layer_end=3)
        per_head_per_layer = (
            2 * 500 * (model.hidden_size // model.num_heads)
            * model.bytes_per_elem)
        self.assertEqual(
            ranged,
            tuple(count * per_head_per_layer * 2
                  for count in (1, 1, 1, 1, 0, 0, 0, 0)))
        self.assertEqual(
            sum(ranged),
            kv_cache_bytes_for_tokens(model, 500) * 2 // model.layers)

    def test_per_rank_truth_drives_boundary_space_wait(self):
        """不均匀逐 rank 字节进边界空间等待：总余量充足但单 rank 缺口
        → 深缺口等待真实非零（总量/均值口径会说零）；该 rank 恰被覆
        盖 → 零等待（逐 rank 判据，无幻影缺口）。工作字节 = 整头不平
        均分割 (2,1) 形态。"""
        working = (KV_TOKEN_BYTES * 100, KV_TOKEN_BYTES * 56)
        boundary = _complete_boundary(working_bytes_by_tp_rank=working)

        def loads(remaining):
            return {
                0: InstanceLoadView(
                    instance_index=0, queued_task_load_ns=0,
                    running_task_load_ns=10 ** 7,
                    active_decode_task_load_ns=0,
                    hbm_remaining_bytes_by_tp_rank=remaining,
                    reclaimable_bytes_by_tp_rank=(0, 0)),
                1: _load(1),
            }

        # rank1 缺口 6,656 B：活跃工作排空前无合法释放 → 深缺口等待。
        short = _model(loads((10 ** 9, 512))).estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY)
        self.assertGreater(short.breakdown.eviction_wait_ns, 0)
        # rank1 恰被覆盖（7,168 = 128×56）：零等待——缺口逐 rank 判定。
        covered = _model(loads((10 ** 9, KV_TOKEN_BYTES * 56))
                         ).estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY)
        self.assertEqual(covered.breakdown.eviction_wait_ns, 0)
        # 镜像（缺口在 rank0）同样真实——真值在向量，不在单侧。
        mirrored = _model(loads((512, 10 ** 9))).estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY)
        self.assertGreater(mirrored.breakdown.eviction_wait_ns, 0)


class ActualGrowthBeyondPredictionTest(unittest.TestCase):
    """§19.1 实际增长超预测：边界承诺（式 20a 缺口）只覆盖 D 阶段足
    迹；decode 因果增长（§3.1）逐列车真实准备——承诺不被静默改写、
    驻留不超预约天花板（fail-closed）、增长 T+E 真实逐出、Z→O 转账
    不双计旧/新预留。"""

    def _prepared(self, kv, *, request_id, session_id, reserve_tokens,
                  prefill_tokens):
        kv.reserve_request_capacity(
            request_id=request_id, session_id=session_id,
            instance_index=0, final_context_tokens=reserve_tokens)
        kv.prepare_prefill(
            session_id=session_id, target_instance_index=0,
            history_tokens=0, trigger_request_id=request_id)
        kv.expand_prefill(
            session_id=session_id, instance_index=0,
            context_tokens=prefill_tokens, trigger_request_id=request_id)

    def test_causal_growth_transfers_reservation_not_double_count(self):
        """实际增长（35 > 边界承诺足迹 40 内的材料化 20）真实入账：
        逐 rank shard 精确、零逐出（宽裕），预约claim 恰减去增长量
        （Z→O 转账）——旧承诺与新驻留不双计。"""
        from joint.test_d20_stage_transaction import _manager, _token_shards
        kv = _manager()
        self._prepared(kv, request_id="r1", session_id="s1",
                       reserve_tokens=40, prefill_tokens=20)
        kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=0, stage_action="stay",
            stage_epoch=1, history_tokens=20, input_tokens=20)
        z_before = tuple(kv._reserved_bytes_by_tp_rank(0))
        self.assertEqual(z_before, _token_shards(kv, 20))  # 式20a：仅缺口
        evictions = kv.expand_decode(
            session_id="s1", instance_index=0, final_context_tokens=35,
            trigger_request_id="r1")
        self.assertEqual(evictions, ())
        session = kv._sessions["s1"]
        self.assertEqual(session.context_tokens, 35)
        self.assertEqual(tuple(session.shard_bytes), _token_shards(kv, 35))
        z_after = tuple(kv._reserved_bytes_by_tp_rank(0))
        growth = (_token_shards(kv, 35)[0] - _token_shards(kv, 20)[0])
        self.assertEqual(z_before[0] - z_after[0], growth)
        # 总占用声明（材料化 + 增量预约）恒 = 预约终态：40 tok 等值。
        self.assertEqual(
            z_after[0] + session.shard_bytes[0], _token_shards(kv, 40)[0])

    def test_growth_beyond_reserved_ceiling_fails_closed(self):
        """实际增长越过预约天花板（final_context_tokens）= fail-closed
        ——预约不被静默改写，不产生超驻留的沉默状态。"""
        from joint.test_d20_stage_transaction import _manager
        kv = _manager()
        self._prepared(kv, request_id="r1", session_id="s1",
                       reserve_tokens=40, prefill_tokens=20)
        kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=0, stage_action="stay",
            stage_epoch=1, history_tokens=20, input_tokens=20)
        with self.assertRaisesRegex(RuntimeError, "reservation is smaller"):
            kv.expand_decode(
                session_id="s1", instance_index=0,
                final_context_tokens=80, trigger_request_id="r1")

    def test_growth_beyond_materialized_pays_real_evictions(self):
        """紧容量下实际增长超材料化足迹（2 → 7 token，free 624 <
        640）：真实 T+E 逐出合法 victim（非占位、非虚构），会话逐
        rank shard 精确到实际终态。"""
        from joint.test_d20_stage_transaction import _manager, _token_shards
        kv = _manager(capacity_bytes=51_328)  # 恰满算术：增长必逐出
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True
        kv._refresh_capacity_summary(("hog",))
        _seed(kv, "victim", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s2", target_instance_index=0,
            history_tokens=0, trigger_request_id="r2")
        kv.expand_prefill(
            session_id="s2", instance_index=0, context_tokens=2,
            trigger_request_id="r2")
        self.assertEqual(kv._effective_remaining_by_tp_rank(0), (624, 624))
        evictions = kv.expand_decode(
            session_id="s2", instance_index=0, final_context_tokens=7,
            trigger_request_id="r2")
        self.assertEqual(
            [(t.session_id, t.reason) for t in evictions],
            [("victim", "decode_growth_capacity_full_fallback")])
        session = kv._sessions["s2"]
        self.assertEqual(session.context_tokens, 7)
        self.assertEqual(tuple(session.shard_bytes), _token_shards(kv, 7))


# ------------------------------------------------------- P→D 专项补齐 --


class ControlCostSmallGainMigrationTest(unittest.TestCase):
    """P→D 专项第 6 条：控制费用使小收益迁移不再合算（式 1c + §2.5
    "新增协议/元数据/预约工作只进入改变方案"）。切换-only 的迁移腿
    （copy 协议 + 元数据 + 预约 = 2,570 ns）是控制费用：目标负载差
    （毛收益）小于它 → 继续胜出；大于它 → 才切换。"""

    BOUNDARY_TOKENS = 100

    def test_switch_only_control_legs_are_the_migration_fee(self):
        """同负载下切换-only 腿 = 迁移费用（history_prep）恰为差额：
        继续候选不付这笔费用——控制费用只进入改变方案。"""
        model = _model(_loads())
        continuation = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY)
        switch = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1,
            action="copy")
        self.assertEqual(continuation.breakdown.history_prep_ns, 0)
        self.assertEqual(
            switch.cost_ns - continuation.cost_ns,
            switch.breakdown.history_prep_ns)
        self.assertGreater(switch.breakdown.history_prep_ns, 0)

    def test_small_gain_below_control_fee_keeps_continuation(self):
        """毛收益 2,000 ns < 控制费用 2,570 ns：目标更轻（收益为真）
        仍选继续——小收益迁移不再合算。

        C4 完整动作面：本例钉 copy 迁移腿的控制费用经济学，允许面掩
        成 {stay,copy}（recompute 的竞争属完整面的正常经济学，非控
        制费用机制的观测对象）。"""
        loads = _loads(active=(3_000, 1_000))
        model = _model(loads)
        model.allowed_actions = frozenset({"stay", "copy"})
        record = _selector(model)
        self.assertTrue(record.chosen.is_continuation)
        self.assertEqual(record.chosen.instance_index, 0)
        self.assertEqual(record.chosen.action, ACTION_STAY)
        # 逐项自证：毛收益为正且小于切换-only 费用。
        switch = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1,
            action="copy")
        continuation = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY)
        gross_gain = 3_000 - 1_000
        self.assertGreater(gross_gain, 0)
        self.assertGreater(
            switch.breakdown.history_prep_ns, gross_gain)
        self.assertGreater(switch.cost_ns, continuation.cost_ns)

    def test_gain_exceeding_control_fee_switches(self):
        """正控制：毛收益 2,700 ns > 控制费用 2,570 ns → 切换生效
        （费用不是禁迁移的伪惩罚，严格改进仍可通过；掩码 {stay,copy}
        ——控制费用机制观测面，同上）。"""
        model = _model(_loads(active=(3_000, 300)))
        model.allowed_actions = frozenset({"stay", "copy"})
        record = _selector(model)
        self.assertFalse(record.chosen.is_continuation)
        self.assertEqual(record.chosen.instance_index, 1)
        self.assertEqual(record.chosen.action, "copy")


def _selector(model):
    from joint.joint_scheduler import select_decode_target_and_action
    return select_decode_target_and_action(
        mode="joint", cost_model=model, boundary=_complete_boundary())


# ---------------------------------------------- P→D 专项第 10 / 12 条 --


class _SecondRequestMixin:
    """在 drain-ready 壳上登记第二个 P_RUNNING 请求（d18 夹具同款）。"""

    def _register_p_running(self, scheduler, kv, request_id,
                            decode_length=8):
        from online.test_decision_route_serialization import _make_runtime
        from types import SimpleNamespace
        runtime = _make_runtime(request_id, decode_length=decode_length)
        scheduler.runtime_by_request_id[request_id] = runtime
        scheduler.next_request[request_id] = None
        scheduler._runtime_index[request_id] = len(scheduler.runtimes)
        scheduler.runtimes.append(runtime)
        kv.reserve_request_capacity(
            request_id=request_id, session_id=runtime.session_id,
            instance_index=0,
            final_context_tokens=runtime.prefill_context_tokens)
        kv.prepare_prefill(
            session_id=runtime.session_id, target_instance_index=0,
            history_tokens=0, trigger_request_id=request_id)
        kv.expand_prefill(
            session_id=runtime.session_id, instance_index=0,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id=request_id)
        runtime.prefill_instance_index = 0
        runtime.joint_stage = "P_RUNNING"
        scheduler.instances[0].qp.append(runtime)
        runtime.queued_chunk_load_ns = 0
        return runtime

    def _active_decode_member(self, scheduler, request_id):
        from online.test_decision_route_serialization import _make_runtime
        runtime = _make_runtime(request_id, decode_length=4)
        scheduler.runtime_by_request_id[request_id] = runtime
        return runtime


class NoReselectAfterFirstNewPhysicalWorkTest(
        _SecondRequestMixin, unittest.TestCase):
    """P→D 专项第 10 条：第一笔新物理工作（边界预约换址提交）之后，
    本轮 D 不再选点（§16.4）——环境变化使 argmin 翻转也不重入边界；
    迟到的重复 P_COMPUTE_DONE 只落幂等披露行。"""

    def test_environment_change_after_commit_does_not_reselect(self):
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        first_decision = dict(runtime.boundary_decision)
        first_pending = list(scheduler.instances[0].pending_decode_ready)
        rows_after_first = len(scheduler.online_log_rows)

        # 环境变化：实例 0 压在服负载、实例 1 空闲——此刻若重新选点，
        # argmin 将翻转到实例 1（正控制见下）。但 r0 已越过第一笔新
        # 物理工作（预约事务已提交）。
        for seq in range(3):
            member = self._active_decode_member(
                scheduler, f"late_active_{seq}")
            scheduler.instances[0].active_decode.append(member)

        # 正控制：同一新环境下，新请求 r1 的真实边界决策确会切换到
        # 实例 1（证明环境变化真实翻转了 argmin，而非无效扰动）。
        r1 = self._register_p_running(scheduler, kv, "r1")
        scheduler._on_prefill_drain("r1", 2000)
        self.assertEqual(r1.joint_stage, "D_COMMITTED")
        self.assertFalse(r1.boundary_decision["fallback"])
        self.assertEqual(
            r1.boundary_decision["chosen_instance_index"], 1)
        self.assertFalse(r1.joint_decode_is_continuation)
        rows_after_r1 = len(scheduler.online_log_rows)

        # 迟到的重复 P_COMPUTE_DONE：幂等忽略，无第二次 D 决策——
        # 状态机、epoch、决策记录、就绪队列零副作用。
        scheduler._on_prefill_drain("r0", 3000)
        new_rows = scheduler.online_log_rows[rows_after_r1:]
        kinds = [row["kind"] for row in new_rows]
        self.assertNotIn("decode", kinds)
        self.assertIn("boundary_duplicate_ignored", kinds)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.joint_boundary_epoch,
                         first_decision["stage_epoch"])
        self.assertEqual(runtime.boundary_decision, first_decision)
        self.assertEqual(runtime.decode_instance_index,
                         first_decision["chosen_instance_index"])
        self.assertEqual(
            scheduler.instances[0].pending_decode_ready, first_pending)


class NoDPreissueAtPrefillEndTest(
        _SecondRequestMixin, unittest.TestCase):
    """P→D 专项第 12 条（§16.1）：不得因初始预测包含 D 就在 P 末尾
    同一不可中断列车中预提交本请求的 D——第二次决策只属于真实边界
    （P_COMPUTE_DONE 之后）。"""

    def test_no_d_emission_before_real_boundary(self):
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        # 初始预测包含 D（decode_length=8 > 0，准入预约即含 decode 终
        # 态足迹），但真实边界前：零 decode 决策行、零 D 就绪队列成员、
        # 阶段尚未越过 P_COMPUTE_DONE、边界 epoch 未前进。
        self.assertEqual(runtime.joint_stage, "P_RUNNING")
        self.assertEqual(runtime.joint_boundary_epoch, 0)
        self.assertEqual(
            [row for row in scheduler.online_log_rows
             if row["kind"] == "decode"], [])
        self.assertEqual(scheduler.instances[0].pending_decode_ready, [])
        self.assertIsNone(runtime.completion_instance_index)
        # 真实边界（P_COMPUTE_DONE）之后 D 才存在：决策行恰一、成员
        # 入就绪队列、epoch 前进——D 是边界二次决策的产物，不是 P 末
        # 尾的预发行。
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.joint_boundary_epoch, 1)
        decode_rows = [row for row in scheduler.online_log_rows
                       if row["kind"] == "decode"]
        self.assertEqual(len(decode_rows), 1)
        self.assertEqual(
            scheduler.instances[0].pending_decode_ready, [runtime])


if __name__ == "__main__":
    unittest.main()
