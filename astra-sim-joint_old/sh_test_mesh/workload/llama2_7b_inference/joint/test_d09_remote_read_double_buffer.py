#!/usr/bin/env python3
"""test_d09_remote_read_double_buffer.py -- D09 remote 有限字节双缓冲
（含 M1′ 执行侧过读修正）的验收单测（joint修改文档 第二批 D09 条目 +
v2.0 设计 §12/式 18a/附录 A.4；修复轮二 2026-09-27 固定 tile 口径）。

验收四条（条目原文；修复轮二口径）：
  1. 任意时刻远读暂存 ≤2 块——tile = 配置值**固定**（``remote_read_
     exec_tile_tokens`` 原样返回，不随基础区间放大——v2.0 §17.1 不
     做清单"为减少仿真事件把容量/缓冲暗改无限"），暂存字节恒 =
     2×配置 tile×kv_token_bytes；每 pass tile 数 = ⌈base/tile⌉ 可 >2，
     ≤2 块不变量由图侧逐 tile lag-2 缓冲复用释放链承载（生产锚定
     消费，见 online/test_remote_credit_stream 的
     test_buffer_reuse_release_chain）；
  2. 慢计算不导致源端无限泵出——图侧 tile t 的生产在消费 tile t-2 的
     体块发射点才释放（待发 tile 账本在体块点前排空 = 无未锚定生产）；
  3. 1 块/N 块/尾块/跨列车复用守恒——切片 Σ 字节 ≡ 计划总量（I1）
     在 1 tile/pass、N tile/pass（>2）、尾块、跨列车续切片四形态逐
     位成立；
  4. 不用宿主大数组冒充容量——暂存预约是 KVCacheManager 真实 HBM 容
     量账分量：准入 ``required`` 含暂存、占用视图/可行性可见、完成/
     异常注销后归零。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d09_remote_read_double_buffer.py（或 pytest）
"""
import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceModel,
    KVCacheManager,
)
from face_scheduler import KVCapacityError  # noqa: E402
from joint.joint_config import (  # noqa: E402
    parse_joint_config,
    REMOTE_READ_STAGING_BUFFERS,
)
from joint.joint_cost_model import LinkFlowRegistry  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _TASK_LOAD_CACHE_CAPACITY,
)

EXEC_INSTANCE = 0   # ranks (0, 1)
HOME_INSTANCE = 1   # ranks (2, 3)


def _scheduler(tile_bytes="auto", train_max_iter=8):
    """轻量调度器替身（test_remote_credit_stream 同族，F6 对齐初值）。"""
    s = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    s.p_chunk = 512
    s._pending_merge_alarms = {}
    s.instances = [_OnlineInstanceState(index=0),
                   _OnlineInstanceState(index=1)]
    s.runtime_by_request_id = {}
    s._train_max_iter = train_max_iter
    s._train_instance_index = {}
    s._ready_frontier = set()
    ranks_by_instance = {0: (0, 1), 1: (2, 3)}
    s.topology = SimpleNamespace(
        instance=lambda i: SimpleNamespace(
            size=2, ranks=ranks_by_instance[i]),
        instances=[SimpleNamespace(size=2), SimpleNamespace(size=2)])
    s.hardware = FaceHardware(
        mesh_rows=2, mesh_cols=2,
        local_hbm_capacity_bytes=10**9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    s.model = FaceModel(
        layers=2, hidden_size=64, ffn_size=128, num_heads=8,
        vocab_size=256, bytes_per_elem=2, mlp_variant="gelu")
    s._prefill_task_cache = {}
    s._decode_task_cache = {}
    s._decode_task_load_cache = {}
    s._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    s._quota_tracker = None
    s.joint_config = parse_joint_config(
        {"JOINT_REMOTE_READ_TILE_BYTES": tile_bytes})
    s._joint_flows = LinkFlowRegistry()
    s._pool_ports = SimpleNamespace(
        register=lambda *args, **kwargs: None,
        release_owner=lambda *args, **kwargs: None)
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    s._hbm_ports = HbmPortFlowRegistry()
    s._kv_ledger_epoch = 0
    return s


def _session(context_tokens=100, working_kind=None,
             base_history_tokens=0, base_prefix_layers=0):
    return SimpleNamespace(
        working_kind=working_kind, context_tokens=context_tokens,
        base_history_tokens=base_history_tokens,
        base_resident_prefix_layers=base_prefix_layers)


def _remote_runtime(request_id="rr", *, decode=8, input_tokens=16,
                    session_id="s_rr"):
    rt = _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0, "queue_index": 0,
        "prefill_length": input_tokens, "decode_length": decode,
        "history_tokens_before": 0,
        "prefill_context_tokens": input_tokens,
        "final_context_tokens": input_tokens + decode,
    }, 512)
    rt.joint_action = "remote-read"
    rt.origin_home_instance = HOME_INSTANCE
    rt.prefill_instance_index = HOME_INSTANCE
    rt.decode_instance_index = EXEC_INSTANCE
    rt.joint_input_tokens = input_tokens
    rt.decode_tokens_consumed = 0
    rt.remaining_chunks = 0
    rt.remote_read_credit_plan = None
    rt.remote_read_preplan = None
    rt.remote_read_credit_trains = 0
    rt.remote_read_slice_summaries = []
    return rt


def _plan_for(scheduler, runtime, session=None):
    scheduler.runtime_by_request_id[runtime.request_id] = runtime
    scheduler.kv_manager = SimpleNamespace(
        _sessions={runtime.session_id: session or _session()},
        tp_degree=2)
    runtime.remote_read_credit_plan = (
        scheduler._joint_remote_read_credit_plan(
            runtime, EXEC_INSTANCE))
    return runtime.remote_read_credit_plan


# ============================== 验收 1：任意时刻远读暂存 ≤2 块 ==

class StagingBoundTest(unittest.TestCase):
    """切片协议不变量（修复轮二，2026-09-27 固定 tile）：tile = 配置值
    原样（不随基础区间放大——放大使每块暂存随历史长度无界增长，正中
    v2.0 §17.1 不做清单"为减少仿真事件把容量/缓冲暗改无限"）；暂存字
    节恒 = 2×配置 tile×kv_token_bytes、每 pass tile 数 = ⌈base/tile⌉ 可
    >2——≤2 块暂存不变量由图侧逐 tile lag-2 释放链承载（见
    online/test_remote_credit_stream::test_buffer_reuse_release_chain）。"""

    def test_staging_bytes_fixed_at_configured_tile(self):
        # kv_token_bytes = 256 B/token（2 层模型夹具）；auto tile = 256
        # token 固定——任意基础区间下暂存恒 = 2×256 token×kv（不随
        # base 放大），tiles_per_pass = ⌈base/256⌉（base > 512 时可 >2，
        # 逐 tile lag-2 流水消费）。
        for base_tokens in (1, 100, 255, 256, 257, 512, 1000, 10000):
            scheduler = _scheduler("auto")
            runtime = _remote_runtime(session_id="s_%d" % base_tokens)
            session = _session(context_tokens=base_tokens)
            plan = _plan_for(scheduler, runtime, session)
            self.assertIsNotNone(plan, base_tokens)
            self.assertEqual(plan["tile_tokens"], 256, base_tokens)
            self.assertEqual(
                plan["staging_bytes_per_rank"],
                REMOTE_READ_STAGING_BUFFERS * 256 * plan["kv_token_bytes"],
                base_tokens)
            self.assertEqual(
                plan["tiles_per_pass"], -(-base_tokens // 256), base_tokens)
        # 显式锚：base 10000 → tiles_per_pass = ⌈10000/256⌉ = 40 块/pass
        # （>2；块数不设上限，暂存上限由固定 tile×双缓冲承载）。
        scheduler = _scheduler("auto")
        runtime = _remote_runtime(session_id="s_10000")
        plan = _plan_for(
            scheduler, runtime, _session(context_tokens=10000))
        self.assertEqual(plan["tiles_per_pass"], 40)

    def test_explicit_small_tile_stays_fixed(self):
        # 显式 1 KB tile（4 token @ 256 B/token）+ base 100 → tile 恒 =
        # 配置 4 token（固定，不放宽到 ⌈base/2⌉）→ 25 块/pass；暂存 =
        # 双缓冲 × 配置 tile（真实执行暂存量 = 配置名义值，不再放大）。
        scheduler = _scheduler("1024")
        runtime = _remote_runtime()
        plan = _plan_for(scheduler, runtime)
        self.assertEqual(plan["tile_tokens"], 4)
        self.assertEqual(plan["tiles_per_pass"], 25)
        self.assertEqual(
            plan["staging_bytes_per_rank"],
            REMOTE_READ_STAGING_BUFFERS * 4 * plan["kv_token_bytes"])

    def test_staging_buffers_protocol_constant(self):
        # 协议固定每 rank 两块（§12.1，非常量旋钮）。
        self.assertEqual(REMOTE_READ_STAGING_BUFFERS, 2)


# ============================== 验收 3：1/2/尾块/跨列车守恒 ==

class SliceConservationTest(unittest.TestCase):
    """I1 守恒：Σ 切片字节 ≡ 计划总量，在 1 tile/pass、2 tile/pass、
    尾块、跨列车续切片四形态逐位成立。"""

    def _plan_and_slice(self, tile_bytes, base_tokens, steps):
        scheduler = _scheduler(tile_bytes)
        runtime = _remote_runtime(
            decode=steps, session_id="s_%d" % base_tokens)
        plan = _plan_for(
            scheduler, runtime, _session(context_tokens=base_tokens))
        blocks = scheduler._joint_remote_read_slice(runtime, steps)
        return plan, blocks

    def test_single_tile_per_pass_conserves(self):
        # auto tile（256 token）≥ base 100 → 每 pass 1 tile、8 pass。
        plan, blocks = self._plan_and_slice("auto", 100, 8)
        self.assertEqual(len(blocks), 8)
        for block in blocks:
            self.assertEqual(
                block.total_bytes,
                plan["total_bytes"] // 8)
        self.assertEqual(
            sum(b.total_bytes for b in blocks), plan["total_bytes"])

    def test_n_tiles_per_pass_conserves(self):
        # N-tile/pass 守恒（修复轮二固定 tile）：显式 1 KB tile（4
        # token）+ base 100 + 4 pass → 每 pass 25 块、Σ块 ≡ 计划总量、
        # 块字节 = kv×整数 token；整除无尾 ⇒ 末 pass 与首 pass 逐位
        # 一致（同形 pass 复用同一对暂存缓冲）。
        plan, blocks = self._plan_and_slice("1024", 100, 4)
        self.assertEqual(plan["tiles_per_pass"], 25)
        self.assertEqual(len(blocks), 4 * 25)
        self.assertEqual(
            sum(b.total_bytes for b in blocks), plan["total_bytes"])
        # token 粒度：每块字节 = kv_token_bytes × 整数 token。
        kv_bytes = plan["kv_token_bytes"]
        for block in blocks:
            for shard in block.shards:
                self.assertEqual(shard.bytes % kv_bytes, 0)
        # 整除无尾：末 pass 与首 pass 逐位一致（100 = 25×4 整除）。
        first_pass = blocks[:plan["tiles_per_pass"]]
        last_pass = blocks[-plan["tiles_per_pass"]:]
        self.assertEqual(
            [tuple(shard.bytes for shard in block.shards)
             for block in first_pass],
            [tuple(shard.bytes for shard in block.shards)
             for block in last_pass])

    def test_tail_tile_smaller_than_full_tile_conserves(self):
        # 尾块形态（固定 tile 口径）：auto tile = 256 token 固定，base
        # 301 → ⌈301/256⌉ = 2 块/pass（256+45，末块 = 真尾块 45 token
        # < 完整 tile）。显式 tile 小于一个 token KV（如 128 B @ 256
        # B/token）由配置校验 fail-closed 拒绝（tile_protocol 面覆盖），
        # 不以放宽掩盖。
        plan, blocks = self._plan_and_slice("auto", 301, 2)
        self.assertEqual(plan["tiles_per_pass"], 2)
        kv_bytes = plan["kv_token_bytes"]
        # 每 pass 块 token 和 = 基础区间（301）：auto tile = 256 token
        # 固定 → 块界 = (256, 45)——末块 45 token 为真尾块（< 完整
        # tile；逐 shard 同分，取单 shard 字节折算 token）。
        first_tokens = blocks[0].shards[0].bytes // kv_bytes
        tail_tokens = blocks[1].shards[0].bytes // kv_bytes
        self.assertEqual(first_tokens, 256)
        self.assertEqual(tail_tokens, 45)
        self.assertEqual(first_tokens + tail_tokens, 301)
        self.assertEqual(
            sum(b.total_bytes for b in blocks), plan["total_bytes"])

    def test_cross_train_slice_reuse_conserves(self):
        # 跨列车复用：同一 8 步计划连续切片（T1 5 步 + T2 3 步）Σ ≡
        # 计划总量——缓冲跨列车复用下字节账不重不漏。
        scheduler = _scheduler("auto")
        runtime = _remote_runtime(decode=8)
        plan = _plan_for(scheduler, runtime)
        blocks_t1 = scheduler._joint_remote_read_slice(runtime, 5)
        blocks_t2 = scheduler._joint_remote_read_slice(runtime, 3)
        self.assertEqual(len(blocks_t1), 5)
        self.assertEqual(len(blocks_t2), 3)
        self.assertEqual(
            sum(b.total_bytes for b in blocks_t1)
            + sum(b.total_bytes for b in blocks_t2),
            plan["total_bytes"])

    def test_read_base_is_remote_base_interval_only(self):
        # M1′/式(18a)：每步读量 = 远端基础区间字节——计划不含 input/
        # decode 增量（旧终态上下文字节基数废除）。
        scheduler = _scheduler("auto")
        runtime = _remote_runtime(input_tokens=400, decode=8)
        plan = _plan_for(scheduler, runtime)
        base_per_step = sum(spec[2] for spec in plan["shard_specs"])
        self.assertEqual(plan["base_tokens"], 100)
        self.assertEqual(plan["total_bytes"], base_per_step * 8)
        # 若仍按旧终态口径（base+input+decode），每步字节严格更大
        # （构造性可分：424 token 的前缀字节 > 100 token）。
        from face_scheduler import kv_cache_shard_bytes_for_layer_range
        legacy_per_step = sum(
            kv_cache_shard_bytes_for_layer_range(
                scheduler.model, 100 + 400 + 8, 2,
                layer_start=0, layer_end=plan["read_prefix_layers"]))
        self.assertLess(base_per_step, legacy_per_step)


# ============================== 验收 4：暂存是真实容量预约 ==

class RealCapacityReservationTest(unittest.TestCase):
    """暂存预约 = KVCacheManager 真实 HBM 容量账分量：准入 required 含
    暂存、占用视图/可行性可见、注销后归零——不是宿主大数组口径。

    夹具：2 层模型、TP2——每 rank 每 token KV（K+V）= 256 B；HOME 实
    例 0 驻留 100 token 基础历史，预约在 exec 实例 1。"""

    def _manager(self, *, hbm_bytes, tile_tokens=256, kv_token_bytes=10):
        from face_scheduler import FaceInstanceSpec, build_instances
        hardware = FaceHardware(
            mesh_rows=2, mesh_cols=2,
            local_hbm_capacity_bytes=hbm_bytes,
            local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
            peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("ins0", "1", (0, 1)),
             FaceInstanceSpec("ins1", "2", (2, 3))))
        model = FaceModel(
            layers=2, hidden_size=64, ffn_size=128, num_heads=8,
            vocab_size=256, bytes_per_elem=2, mlp_variant="gelu")
        return KVCacheManager(
            topology, model, remote_read_tile_tokens=tile_tokens,
            remote_read_kv_token_bytes=kv_token_bytes)

    def _seed_local_base(self, manager, tokens=100):
        manager.prepare_prefill(
            session_id="s", target_instance_index=0, history_tokens=0,
            trigger_request_id="seed")
        manager.expand_prefill(
            session_id="s", instance_index=0, context_tokens=tokens,
            trigger_request_id="seed")

    def test_staging_enters_required_and_occupancy_view(self):
        manager = self._manager(hbm_bytes=10**9)
        self._seed_local_base(manager)
        staging_per_rank = 2 * 256 * 10  # 双缓冲 × 256 token × 10 B
        baseline = tuple(manager._effective_remaining_by_tp_rank(1))
        evictions = manager.reserve_request_capacity(
            request_id="r1", session_id="s", instance_index=1,
            final_context_tokens=150, action="remote-read")
        self.assertEqual(evictions, ())
        reservation = manager._reservations["r1"]
        self.assertEqual(
            reservation.staging_shard_bytes, (staging_per_rank,) * 2)
        self.assertEqual(reservation.staging_tile_tokens, 256)
        # 占用视图：预约后有效余量按 KV 基数（remote-read = input 基数）
        # + 暂存同步收缩（逐 rank；基线含模型权重，取差值断言）。
        expected_delta = tuple(
            int(basis) + staging_per_rank
            for basis in reservation.final_shard_bytes)
        self.assertEqual(
            tuple(manager._effective_remaining_by_tp_rank(1)),
            tuple(b - d for b, d in zip(baseline, expected_delta)))

    def test_staging_short_capacity_is_infeasible_not_faked(self):
        # 容量 < KV 基数 + 暂存 → 可行性判不可行 / 预约 fail-closed
        # （容量账不因"暂存只是计数"而放行——无宿主大数组冒充）。
        kv_need = 100 * 256  # remote-read 基数（input 100 token × 256 B/rank）
        staging = 2 * 256 * 10
        manager = self._manager(hbm_bytes=10**9)
        # 收紧 exec 实例容量到"权重 + KV 基数 + 暂存 − 1"（不可再松一
        # 位；seed 30 token 留在 home 实例 0，不占 exec 容量）。
        for rank in (2, 3):
            state = manager._rank_states[rank]
            state.capacity_bytes = (
                state.model_weight_bytes + kv_need + staging - 1)
        self._seed_local_base(manager, tokens=30)
        with self.assertRaises(KVCapacityError):
            manager.reserve_request_capacity(
                request_id="r1", session_id="s", instance_index=1,
                final_context_tokens=130, action="remote-read")
        self.assertNotIn("r1", manager._reservations)

    def test_release_unregisters_staging(self):
        manager = self._manager(hbm_bytes=10**9)
        self._seed_local_base(manager)
        before = tuple(manager._effective_remaining_by_tp_rank(1))
        manager.reserve_request_capacity(
            request_id="r1", session_id="s", instance_index=1,
            final_context_tokens=150, action="remote-read")
        staging_per_rank = 2 * 256 * 10
        held = tuple(manager._effective_remaining_by_tp_rank(1))
        # 每 rank 有效余量收缩 = KV 基数（input 50 token × 256 B）+ 暂存。
        self.assertEqual(
            held,
            tuple(b - 50 * 256 - staging_per_rank for b in before))
        # 注销预约（终态退休路径的账面动作 = 删除 + 不变量刷新）——暂存
        # 随预约一并注销，有效余量全额恢复（KV 物化占用由 session 账另
        # 计；物化守卫路径由既有 lifecycle 套件覆盖）。
        del manager._reservations["r1"]
        manager._check_invariants_after_mutation(reservation_ids=("r1",))
        after = tuple(manager._effective_remaining_by_tp_rank(1))
        self.assertEqual(after, before)

    def test_non_remote_read_has_no_staging(self):
        manager = self._manager(hbm_bytes=10**9)
        self._seed_local_base(manager)
        manager.reserve_request_capacity(
            request_id="r2", session_id="s", instance_index=1,
            final_context_tokens=150, action="stay")
        reservation = manager._reservations["r2"]
        self.assertEqual(reservation.staging_shard_bytes, ())
        self.assertEqual(reservation.staging_tile_tokens, 0)

    def test_tile_protocol_params_must_pair(self):
        with self.assertRaises(ValueError):
            self._manager(hbm_bytes=10**9, tile_tokens=0,
                          kv_token_bytes=10)
        with self.assertRaises(ValueError):
            self._manager(hbm_bytes=10**9, tile_tokens=256,
                          kv_token_bytes=0)


if __name__ == "__main__":
    unittest.main()
