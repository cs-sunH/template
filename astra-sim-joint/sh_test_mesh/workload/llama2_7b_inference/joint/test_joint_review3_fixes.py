#!/usr/bin/env python3
"""test_joint_review3_fixes.py -- R16 修复批定向测试
（2026-09-15，第二次错误 D2-1/D2-2/D2-3 + 四审 P1/P2-1/P2-2）。

覆盖（对应 joint第二次修改方案.md R16-4 测试矩阵）：
  1. 核心端到端：copy@PARTIAL×跨实例复合两笔物化（noc_migrate[0,prefix)
     + remote_load[prefix,L)）+ shard 校验 + 会话终态 + 双端账本守恒。
  2. 守卫语义：区间越界仍 raise；[0,L) 于全驻留恒过（move 路径回归）；
     move 字段按新区间惯例钉 (before, after) = (0, L)（A-F1/B-3）。
  3. 计价同源（R16-3）：copy 两腿闭式 + LOCAL 逐位一致 + REMOTE 不变；
     merge 拆分同型（前缀 NoC 腿 + 守卫后缀池腿 + home 等待按前缀增量）；
     L∤inc 合成视图公式钉死 prefix = inc×p//L; suffix = inc−prefix。
  4. 图构建：准入逐出 × 复合两笔同图交错（8a）；跨实例 noc_migrate
     无门控发射（无 trigger 中继）；后缀 remote_load 真实消费重建门
     （arm + exec≠edge 1B request 中继）；store→restore 前递边；
     readiness barrier 在两笔之后；链式顺序与链路账。
  5. 水印重放：复合 history_transfers 行目标端两腿入账、源端不动、
     prefill_grow 顶账差额正确。
  6. merge 回归：copy@PARTIAL 完成后 merge 两笔 + 工作副本释放 +
     last_merged_request_id 版本键单次。
  8b. copy@home×PARTIAL 负例：R16-6 退化守卫——不建工作副本、
      working_kind 保持 None、完成结算不撞 N9；深审二轮补图发射层
      用例（joint_action=copy 走 partial 快速路径分支，断言无 noc
      前缀腿 + 屏障结构 + _suffix_body_arms）。
  8c. "复合 copy → merge 落回 PARTIAL → 再跨实例 copy"序列回归。
  8d. 决策日志复合两笔序列化形状（_transfer_summary + 单/复数/字节
      聚合三字段合同）。
  R16-7：hopbytes collect_joint——复合两笔两腿聚合；单行产物回退
      逐位一致；local_hit 零贡献。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_review3_fixes.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_SLO_DIR = os.path.join(_WL, "..", "..", "slo_tools")
if _SLO_DIR not in sys.path:
    sys.path.insert(0, _SLO_DIR)

from face_scheduler import (  # noqa: E402
    KVTransfer,
    KVTransferShard,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
)
from generate_face_trace import _validate_transfer_shard  # noqa: E402
from generate_trace import COMM_RECV_NODE, COMM_SEND_NODE  # noqa: E402
from joint.test_joint_fixes import (  # noqa: E402
    _manager,
    _make_partial,
    _model,
    _load,
)
from joint.test_joint_mechanisms import _seed  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    SessionKVView,
    _pool_transfer_ns,
    _transfer_ns,
)
import hbm_watermark  # noqa: E402
import hopbytes  # noqa: E402


def _session_view(home=0, resident=0, history_bytes=(1000, 1000),
                  missing=(0, 0), location="local_hbm", prefix=4):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=50, resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request_view(input_tokens=5, decode=10, input_bytes=(500, 500)):
    from joint.joint_cost_model import RequestView
    return RequestView(
        request_id="r", session_id="s",
        input_tokens=input_tokens, history_tokens_before=50,
        estimated_decode_tokens=decode, horizon_source="run_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


def _scan(tokens_requests):
    mapping = hbm_watermark.REPO_VARIANTS["astra-sim-joint"]
    tokens = {"requests": tokens_requests, "path": "test"}
    return hbm_watermark.WatermarkScan(
        Path("/tmp/nonexistent-run"), "astra-sim-joint", mapping, tokens,
        coef=100, capacity=None)


def _token_row(history, prefill_context, final_context, session_id="sess"):
    return {
        "session_id": session_id,
        "history_tokens_before": history,
        "prefill_context_tokens": prefill_context,
        "final_context_tokens": final_context,
    }


# ===================================================== R16-4-1 核心端到端 ==


class CompositeCopyPartialE2ETest(unittest.TestCase):
    """copy@PARTIAL×跨实例：复合两笔物化 + 终态 + 账本守恒（D2-1/D2-2）。"""

    def test_composite_two_transfer_materialization(self):
        kv = _manager()
        _make_partial(kv)                       # PARTIAL@0，prefix=2
        session = kv._sessions["s"]
        prefix = session.resident_prefix_layers
        layers = kv.model.layers
        self.assertTrue(0 < prefix < layers)
        full = kv_cache_shard_bytes_for_tokens(kv.model, 10, kv.tp_degree)
        prefix_shards = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree,
            layer_start=0, layer_end=prefix)
        suffix_shards = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree,
            layer_start=prefix, layer_end=layers)
        remaining0_before = kv._effective_remaining_by_tp_rank(0)
        remaining1_before = kv._effective_remaining_by_tp_rank(1)

        _, transfers, evictions = kv.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="t1", reservation_request_id="t1",
            action="copy")

        # (a) 两笔：noc_migrate[0,prefix) + remote_load[prefix,L)。
        self.assertEqual(len(transfers), 2)
        noc, restore = transfers
        self.assertEqual(noc.kind, "noc_migrate")
        self.assertEqual((noc.layer_start, noc.layer_end), (0, prefix))
        self.assertEqual(
            (noc.resident_prefix_layers_before,
             noc.resident_prefix_layers_after), (0, prefix))
        self.assertEqual(restore.kind, "remote_load")
        self.assertEqual(
            (restore.layer_start, restore.layer_end), (prefix, layers))
        self.assertEqual(evictions, ())

        # (b) 字节：NoC = 前缀层、restore = 后缀层、合计 = 整份。
        self.assertEqual(noc.total_bytes, sum(prefix_shards))
        self.assertEqual(restore.total_bytes, sum(suffix_shards))
        self.assertEqual(
            noc.total_bytes + restore.total_bytes, sum(full))

        # (c) 会话终态：LOCAL@exec、全驻留、shard_bytes 整份。
        self.assertEqual(session.location, kv.LOCAL_HBM)
        self.assertEqual(session.instance_index, 1)
        self.assertEqual(session.resident_prefix_layers, layers)
        self.assertEqual(session.shard_bytes, full)
        self.assertEqual(session.total_bytes, sum(full))
        self.assertEqual(session.working_kind, "copy")

        # (d) 双端 rank 账本守恒：home 前缀不动、exec +整份工作副本。
        self.assertEqual(
            kv._effective_remaining_by_tp_rank(0), remaining0_before)
        self.assertEqual(
            tuple(before - after for before, after in zip(
                remaining1_before,
                kv._effective_remaining_by_tp_rank(1))),
            full)

        # (e) 发射器 shard 校验通过（层区间一致性/边界端点）。
        config = SimpleNamespace(
            remote_memory=SimpleNamespace(edge_npus=set(range(4))))
        for transfer in transfers:
            for shard in transfer.shards:
                _validate_transfer_shard(config, transfer, shard)


# ===================================================== R16-4-2 守卫语义 ==


class NocTransferLayerRangeGuardTest(unittest.TestCase):
    """R16-1：区间包含判定——越界 raise、[0,L) 恒过、move 路径回归。"""

    def test_out_of_range_raises(self):
        kv = _manager()
        _seed(kv, "s", 0, 10, 10)               # LOCAL@0 全驻留（L=4）
        session = kv._sessions["s"]
        for layer_start, layer_end in ((0, 5), (2, 2), (-1, 2), (3, 1)):
            with self.assertRaises(
                    RuntimeError,
                    msg=f"[{layer_start},{layer_end}) 应 fail-closed"):
                kv._noc_transfer(
                    phase="history", reason="x", session=session,
                    trigger_request_id="t", source_instance_index=0,
                    target_instance_index=1,
                    layer_start=layer_start, layer_end=layer_end)
        # PARTIAL 会话：区间 ⊆ [0, resident_prefix) 合法、越过后缀 raise。
        kv2 = _manager()
        _make_partial(kv2)
        partial = kv2._sessions["s"]
        prefix = partial.resident_prefix_layers
        ok = kv2._noc_transfer(
            phase="history", reason="x", session=partial,
            trigger_request_id="t", source_instance_index=0,
            target_instance_index=1,
            layer_start=0, layer_end=prefix)
        self.assertEqual(ok.layer_end, prefix)
        with self.assertRaises(RuntimeError):
            kv2._noc_transfer(
                phase="history", reason="x", session=partial,
                trigger_request_id="t", source_instance_index=0,
                target_instance_index=1,
                layer_start=0, layer_end=prefix + 1)

    def test_move_path_full_range_bitwise_shape(self):
        """move_prefill_to_decode 显式传 [0,L)：字段按区间惯例 (0, L)。"""
        kv = _manager()
        _seed(kv, "s", 0, 10, 10)
        full = kv_cache_shard_bytes_for_tokens(kv.model, 10, kv.tp_degree)
        transfer, evictions = kv.move_prefill_to_decode(
            session_id="s", target_instance_index=1,
            trigger_request_id="t")
        self.assertEqual(transfer.kind, "noc_migrate")
        self.assertEqual((transfer.layer_start, transfer.layer_end),
                         (0, kv.model.layers))
        # A-F1/B-3：resident_prefix_layers_before 由 L 变 0（区间传输
        # 惯例；唯一行为性消费者 _mark_pending_history_store 仅门控
        # remote_store，P3-a 已核无影响）。
        self.assertEqual(
            (transfer.resident_prefix_layers_before,
             transfer.resident_prefix_layers_after),
            (0, kv.model.layers))
        self.assertEqual(transfer.total_bytes, sum(full))
        self.assertEqual(
            tuple(shard.bytes for shard in transfer.shards), full)


# ===================================================== R16-4-3 计价同源 ==


class CopyCompositePricingTest(unittest.TestCase):
    """R16-3 copy 段：两腿闭式 + LOCAL 逐位一致 + REMOTE 不变。

    C4 金值重推导（2026-09-22）：C1 三腿 min + A1' 并集除数后，闭式
    期望值不变——字节均衡（(3000,3000)）+ 空链路 + u_port=0（离线
    口径）下逐 shard noc 腿 3000/(10/2)=600 与旧聚合 6000/10=600 同
    wall（A1' 桥接锚）；copy 端点腿 home=3000/100=30、exec 写腿=30
    （A4' 前即有）< noc 腿不触 max。下述三测试的期望因此逐位保留，
    手算过程按现行公式写进各用例注释。"""

    def test_partial_base_two_legs_closed_form(self):
        model = _model({0: _load(), 1: _load()})
        session = _session_view(
            home=0, resident=0, history_bytes=(3000, 3000),
            missing=(1000, 1000), location="partial_hbm_remote", prefix=3)
        candidate = model.estimate_action(
            session=session, request=_request_view(), instance_index=1,
            action="copy", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        # 手算（现行公式）：NoC 前缀腿逐 shard 字节 3000、并集除数 2
        # （双 shard 共链 (0,1)）→ noc 腿 = 3000/5 = 600、home 读腿
        # 3000/100 = 30、exec 写腿 30 → max = 600；wall = 10（单跳
        # 时延）+ 600 = 610；后缀池腿 = 100 + 2000/5 = 500 →
        # history_prep = 1110。闭式 _transfer_ns(6000, 除数 1) =
        # 10 + 600 = 610 逐位同值（A1' 桥接锚）。
        expected = _transfer_ns(
            total_bytes=6000, path_hops=1, divisor=1, rates=model.rates,
            per_hop_latency_ns=None, startup_ns=0)
        expected += _pool_transfer_ns(
            total_bytes=2000, divisor=1, rates=model.rates)
        self.assertEqual(candidate.breakdown.history_prep_ns, expected)
        self.assertIn(
            "noc_prefix+pool_suffix_restore", candidate.breakdown.notes)

    def test_local_base_bit_identical(self):
        """LOCAL 基（missing=0）：后缀腿守卫为零，与旧整份 NoC 口径
        逐位一致（旧公式 = _transfer_ns(resident + 0)；现行公式下
        同值 = A1' 桥接锚：逐 shard 3000/(10/2) ≡ 聚合 6000/10，
        端点腿 30 各不触 max，wall = 10 + 600 = 610）。"""
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3000, 3000),
                missing=(0, 0), location="local_hbm", prefix=4),
            request=_request_view(), instance_index=1,
            action="copy", remote_enabled=True)
        expected = _transfer_ns(
            total_bytes=6000, path_hops=1, divisor=1, rates=model.rates,
            per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(candidate.breakdown.history_prep_ns, expected)

    def test_remote_base_unchanged(self):
        # REMOTE 基池腿独走（无 NoC 腿——C1/A1'/A4' 均不触及）：
        # pool(8000) = 100 + 8000/5 = 1700。
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=None, history_bytes=(0, 0),
                missing=(4000, 4000), location="remote_memory", prefix=0),
            request=_request_view(), instance_index=1,
            action="copy", remote_enabled=True)
        expected = _pool_transfer_ns(
            total_bytes=8000, divisor=1, rates=model.rates)
        self.assertEqual(candidate.breakdown.history_prep_ns, expected)
        self.assertIn("pool_restore_to_target", candidate.breakdown.notes)


class MergeV2PricingTest(unittest.TestCase):
    """merge v2 少并多计价（需求②，2026-09-17《部分层逐出kv管理改造
    方案》§4.3.4）：merge_ns = min(前向, 反向)；前向 = noc(exec 侧保留
    量) + home 空间准备等待；反向 = noc(home 侧保留量) + exec 空间准备
    等待；copy/recompute 反向零字节翻转（=0）；REMOTE 基就地保留（=0）。
    守恒拆分纪律（prefix = value × p // L）迁移至 remote-read 读流基数
    （需求①前缀份额）。
    金样留档（改造前口径，git 9a95e06 可复算）——旧"增量按 base_prefix
    /L 逐 rank 分裂"五用例期望值：partial 拆分 noc(2250)+pool(750)；
    LOCAL 逐位一致 noc(3000)；REMOTE 池写 pool(3000)；不可除钉
    noc(6)+pool(4)；home 等待 noc(2250)+pool(750)+pool(1025)。"""

    def _noc(self, rates, total_bytes):
        return _transfer_ns(
            total_bytes=total_bytes, path_hops=1, divisor=1, rates=rates,
            per_hop_latency_ns=None, startup_ns=0)

    def test_copy_partial_base_zero_byte_flip(self):
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3000, 3000),
                missing=(1000, 1000), location="partial_hbm_remote",
                prefix=3),
            request=_request_view(), instance_index=1,
            action="copy", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.merge_ns, 0)
        self.assertIn("merge_v2_zero_byte_flip",
                      candidate.breakdown.notes)

    def test_remote_read_local_forward_bit_identical(self):
        """LOCAL 基 remote-read 前向腿数值锚（p=L 退化）：home 保留
        (4000) ≥ exec 增量 (3000) → 前向胜出，merge_ns = noc(3000)——与
        改造前 LOCAL 公式逐位一致（旧 prefix = inc × L // L = inc，
        home 等待充足为 0）。"""
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(history_bytes=(2000, 2000)),
            request=_request_view(), instance_index=1,
            action="remote-read", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(
            candidate.breakdown.merge_ns, self._noc(model.rates, 3000))
        self.assertIn("merge_to_home=0", candidate.breakdown.notes)

    def test_remote_read_local_reverse_when_home_smaller(self):
        """home 保留 (2000) < exec 增量 (3000) → 反向胜出（少并多翻转，
        home 迁移 exec 的计价预见）：merge_ns = noc(2000)。"""
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(history_bytes=(1000, 1000)),
            request=_request_view(), instance_index=1,
            action="remote-read", remote_enabled=True)
        self.assertEqual(
            candidate.breakdown.merge_ns, self._noc(model.rates, 2000))

    def test_remote_read_partial_hybrid_three_segments(self):
        """PARTIAL 混合形态三段式（需求①）：历史准备 = 后缀池恢复
        pool(2000)；读流基数层区间化（remote_read_prefix_layers=3）；
        merge 双向 = noc(S+I=5000) vs noc(H=6000) → 前向胜出 noc(5000)。"""
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3000, 3000),
                missing=(1000, 1000), location="partial_hbm_remote",
                prefix=3),
            request=_request_view(), instance_index=1,
            action="remote-read", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(
            candidate.breakdown.history_prep_ns,
            _pool_transfer_ns(
                total_bytes=2000, divisor=1, rates=model.rates))
        self.assertIn("pool_suffix_restore_hybrid",
                      candidate.breakdown.notes)
        self.assertEqual(
            candidate.breakdown.merge_ns, self._noc(model.rates, 5000))
        self.assertIn("remote_read_prefix_layers=3",
                      candidate.breakdown.notes)

    def test_remote_read_partial_reverse_when_home_smaller(self):
        """PARTIAL 混合形态反向：home (2000) < exec (S+I=5000) →
        反向 noc(2000) 胜出。"""
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(1000, 1000),
                missing=(1000, 1000), location="partial_hbm_remote",
                prefix=3),
            request=_request_view(), instance_index=1,
            action="remote-read", remote_enabled=True)
        self.assertEqual(
            candidate.breakdown.merge_ns, self._noc(model.rates, 2000))

    def test_read_prefix_share_conservation_pinned(self):
        """读流基数钉（K8 重钉，2026-09-23 外部审计）：D1 口径 read_base
        = history_bytes_by_tp_rank 账本真值（(3009,3009) → 6018），input/
        decode 增量本地读取**不计**远读；read_passes = max(1, 0) = 1 →
        remote_read_ns = noc(6018)。字节取 3009/rank 使截断桶敏感：旧
        （D1 前）口径 base = 6018 + input 前缀份额 6 = 6024 → noc 桶 602
        ≠ 601（原 6000/6006 形态两桶同为 600，钉不住任何口径）。"""
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3009, 3009),
                missing=(1000, 1000), location="partial_hbm_remote",
                prefix=3),
            request=_request_view(
                input_tokens=5, decode=0, input_bytes=(5, 5)),
            instance_index=1, action="remote-read", remote_enabled=True)
        self.assertEqual(
            candidate.breakdown.remote_read_ns,
            self._noc(model.rates, 6018))
        self.assertNotEqual(
            candidate.breakdown.remote_read_ns,
            self._noc(model.rates, 6024))

    def test_home_wait_in_forward_formula(self):
        """前向公式含 home 空间准备等待（基数 = exec 侧保留量）：home 仅
        剩 100 B/rank → max 缺口 2500−100=2400 按池写回口径估。K5 重钉
        （2026-09-23 外部审计）：merge 方向 = 字节判据镜像物化侧
        （Σ exec_retained 5000 ≤ Σ home 6000 → forward，F15 平局含等号
        取前向）——merge_ns = 前向值 1090（反向 noc(6000)=610 虽更廉，
        非物化方向不取；容量兜底翻转不可计价）。"""
        tight_home = {0: _load(remaining=(100, 100),
                               reclaimable=(10**9, 10**9)),
                      1: _load()}
        model = _model(tight_home)
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3000, 3000),
                missing=(1000, 1000), location="partial_hbm_remote",
                prefix=3),
            request=_request_view(), instance_index=1,
            action="remote-read", remote_enabled=True)
        expected_forward = self._noc(model.rates, 5000) + _pool_transfer_ns(
            total_bytes=2400, divisor=1, rates=model.rates)
        expected_reverse = self._noc(model.rates, 6000)
        self.assertIn(
            "home_merge_eviction_writeback_est",
            candidate.breakdown.notes)
        self.assertIn(
            "merge_v2_direction=forward", candidate.breakdown.notes)
        self.assertEqual(
            candidate.breakdown.merge_ns, expected_forward)
        self.assertNotEqual(
            candidate.breakdown.merge_ns, expected_reverse)


# ===================================================== R16-4-4 图构建 ==


def _graph_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="i0"),
            SimpleNamespace(ranks=(2, 3), pg_name="i1"),
        ],
        layers=4,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        remote_memory=SimpleNamespace(edge_npus=(0, 1)),
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0, inter_request_interval_ns=None),
            SimpleNamespace(
                session_arrival_time_ns=0, inter_request_interval_ns=2500),
        ],
    )


def _noc_migrate_composite(session_id):
    shards = tuple(
        KVTransferShard(
            source_rank=source, target_rank=target, edge_rank=None,
            bytes=640, noc_path=(source, target),
            layer_start=0, layer_end=2)
        for source, target in ((0, 2), (1, 3)))
    return KVTransfer(
        kind="noc_migrate", phase="history",
        reason="history_prefix_working_copy",
        session_id=session_id, trigger_request_id="s_request_1",
        source_instance_index=0, target_instance_index=1,
        total_bytes=1280, shards=shards, model_layers=4,
        layer_start=0, layer_end=2,
        resident_prefix_layers_before=0, resident_prefix_layers_after=2)


def _remote_load_composite(session_id):
    shards = tuple(
        KVTransferShard(
            source_rank=edge, target_rank=target, edge_rank=edge,
            bytes=640, noc_path=(edge, target),
            layer_start=2, layer_end=4)
        for target, edge in ((2, 0), (3, 1)))
    return KVTransfer(
        kind="remote_load", phase="history",
        reason="history_suffix_pool_restore_working_copy",
        session_id=session_id, trigger_request_id="s_request_1",
        source_instance_index=None, target_instance_index=1,
        total_bytes=1280, shards=shards, model_layers=4,
        layer_start=2, layer_end=4,
        resident_prefix_layers_before=2, resident_prefix_layers_after=4)


def _suffix_store(session_id):
    shards = tuple(
        KVTransferShard(
            source_rank=source, target_rank=edge, edge_rank=edge,
            bytes=640, noc_path=(source, edge),
            layer_start=2, layer_end=4)
        for source, edge in ((0, 0), (1, 1)))
    return KVTransfer(
        kind="remote_store", phase="admission",
        reason="admission_capacity_suffix_half",
        session_id=session_id, trigger_request_id="s_request_0",
        source_instance_index=0, target_instance_index=None,
        total_bytes=1280, shards=shards, model_layers=4,
        layer_start=2, layer_end=4,
        resident_prefix_layers_before=4, resident_prefix_layers_after=2)


class CompositeCopyGraphEmissionTest(unittest.TestCase):
    """R16-4-4/8a：准入逐出 + 复合两笔同图；noc 无门控、后缀腿真门控。"""

    def _harness(self):
        from online.graph_batch_builder import GraphBatchBuilder
        builder = GraphBatchBuilder(_graph_config())
        builder.begin_batch()
        return builder

    def _rank_node_names(self, builder, rank):
        return [node["name"] for node in builder.batch["nodes"]
                if node.get("rank") == rank]

    def test_composite_emission_shape(self):
        from generate_face_trace import PendingHistoryGate
        builder = self._harness()
        # (8a) 前一轮准入逐出：后缀 store 入池（同缘），尾部登记。
        builder.emit_admission_batch({
            "request_id": "r0", "session_id": "other", "turn_index": 0,
            "queue_index": 0, "prefill_instance_index": 0,
            "decode_instance_index": 0, "admission_time_ns": 1000,
            "history_location_before": None, "history_transfer": None,
            "history_evictions": [_suffix_store("session_x")],
            "prefill_evictions": [],
            "history_tokens_before": 0,
            "prefill_context_tokens": 300,
        })
        self.assertIn("session_x", builder.pending_store_tails)
        tails = builder.pending_store_tails["session_x"]
        # 前递补边硬化（需求①配套，2026-09-17）：条目 5 元组
        # （edge, store 节点, ack 节点, layer_start, layer_end）。
        self.assertEqual({entry[0] for entry in tails}, {0, 1})
        self.assertTrue(all(
            len(entry) == 5 and entry[3] == 2 and entry[4] == 4
            for entry in tails))

        # 本轮：跨实例 copy 复合两笔（gate 在源实例 0，prefill 在 1）。
        gate_ids = {}
        for rank in (0, 1):
            builder.builders[rank].comp(f"gate_seed_rank{rank}", 1, 1)
            gate_ids[rank] = builder.builders[rank].previous_id
        builder.pending_history["s_request_1"] = PendingHistoryGate(
            source_instance_index=0,
            timer_gates=(gate_ids[0], gate_ids[1]),
            location="partial_hbm_remote")
        builder.emit_admission_batch({
            "request_id": "s_request_1", "session_id": "session_x",
            "turn_index": 1, "queue_index": 1,
            "prefill_instance_index": 1, "decode_instance_index": 1,
            "admission_time_ns": None, "hbm_wait_ns": 0,
            "joint_action": "copy",
            "history_location_before": SimpleNamespace(
                location="partial_hbm_remote", instance_index=0,
                resident_prefix_layers=2),
            "history_transfers": [
                _noc_migrate_composite("session_x"),
                _remote_load_composite("session_x")],
            "history_evictions": [], "prefill_evictions": [],
            "history_tokens_before": 100,
            "prefill_context_tokens": 300,
        })

        nodes = builder.batch["nodes"]
        names = [node["name"] for node in nodes]

        # noc 腿无门控发射：无 trigger 中继（R16-5 删死参数后的行为钉死）。
        # 守卫注记（R16-4-4）：若未来给 noc_migrate 接通 trigger 消费
        # （离线/回放模式的前置条件，见 PROVENANCE §11-3），本断言须
        # 同步改写为"trigger 中继存在且依赖到达/间隔 gate"。
        self.assertFalse([n for n in names if "trigger" in n])
        # 后缀 remote_load 腿真实消费重建门：exec≠edge → 1B request 中继。
        request_relays = [n for n in names if "_request_to_edge" in n]
        self.assertEqual(len(request_relays), 2)

        # 链路账：noc 数据腿（*_send 结尾）与 restore 数据腿
        # （*_send_to_rank 结尾）各 640×2——两腿链路账精确分立。
        def data_bytes(name_suffix):
            return sum(
                node["comm"]["bytes"] for node in nodes
                if node["name"].endswith(name_suffix)
                and "shard" in node["name"])
        self.assertEqual(data_bytes("_send"), 1280)
        self.assertEqual(data_bytes("_send_to_rank2") +
                         data_bytes("_send_to_rank3"), 1280)
        restore_writes = [n for n in names if n.endswith("_target_hbm_write")]
        self.assertEqual(len(restore_writes), 2)
        mem_loads = [n for n in names if n.endswith("_remote_load")]
        self.assertEqual(len(mem_loads), 2)

        # store→restore 前递边：restore mem_load 的祖先闭包含同缘 store
        # 尾（同缘 arm 由该 rank 回迁链首节点消费，经链传递到 mem_load）。
        parents = {}
        for edge in builder.batch["parent_edges"]:
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                (edge["rank"], edge["from"]))
        # comm 配对：recv 依赖 send（同 src/dst/tag，跨 rank 依赖载体）。
        sends = {}
        for node in nodes:
            if node["type"] == COMM_SEND_NODE:
                comm = node["comm"]
                sends[(comm["src"], comm["dst"], comm["tag"])] = (
                    node["rank"], node["id"])
        for node in nodes:
            if node["type"] == COMM_RECV_NODE:
                comm = node["comm"]
                send = sends.get((comm["src"], comm["dst"], comm["tag"]))
                if send is not None:
                    parents.setdefault(
                        (node["rank"], node["id"]), []).append(send)

        def ancestors(rank, node_id):
            seen = {(rank, node_id)}
            frontier = [(rank, node_id)]
            while frontier:
                current = frontier.pop()
                for parent in parents.get(current, ()):
                    if parent not in seen:
                        seen.add(parent)
                        frontier.append(parent)
            return seen

        store_ids = {(entry[0], entry[1]) for entry in tails}
        for edge, store in store_ids:
            loads_on_edge = [
                node for node in nodes
                if node.get("rank") == edge
                and node["name"].endswith("_remote_load")]
            self.assertTrue(loads_on_edge)
            for load in loads_on_edge:
                self.assertIn(
                    (edge, store),
                    ancestors(edge, load["id"]),
                    "restore mem_load 必须传递依赖同缘 store 尾（前递补边）")

        # 链式顺序（rank 2）：重建 gate → noc recv → ack → request 中继 →
        # target restore；readiness barrier 在两笔之后。
        rank2 = self._rank_node_names(builder, 2)
        index = {name: position for position, name in enumerate(rank2)}
        self.assertLess(
            index[min(n for n in rank2 if "interval_gate_rebuilt" in n)],
            index[min(n for n in rank2 if n.endswith("_recv"))])
        self.assertLess(
            index[min(n for n in rank2 if n.endswith("_recv"))],
            index[min(n for n in rank2 if "_request_to_edge" in n)])
        self.assertLess(
            index[min(n for n in rank2 if "_request_to_edge" in n)],
            index[min(n for n in rank2 if n.endswith("_target_hbm_write"))])
        barrier_positions = [
            position for position, name in enumerate(rank2)
            if "prefill_kv_ready_barrier" in name]
        self.assertTrue(barrier_positions)
        self.assertGreater(
            barrier_positions[0],
            index[min(n for n in rank2 if n.endswith("_target_hbm_write"))])
        # rank 3 同型（TP 并行）。
        rank3_names = self._rank_node_names(builder, 3)
        self.assertTrue(
            [n for n in rank3_names if "prefill_kv_ready_barrier" in n])


class CopyAtHomePartialGraphFastPathTest(unittest.TestCase):
    """R16-6 深审补项：copy@home×PARTIAL 退化（单笔后缀 remote_load、
    同实例）在图发射层走 partial 快速路径分支（此前仅 stay 可达）——
    断言分支结构完整、无 noc 前缀腿、suffix 完成门入 _suffix_body_arms。
    """

    def test_degenerate_copy_emits_partial_fast_path(self):
        from generate_face_trace import PendingHistoryGate
        from online.graph_batch_builder import GraphBatchBuilder
        builder = GraphBatchBuilder(_graph_config())
        builder.begin_batch()
        # 前置：前一轮准入逐出的后缀 store（[2,4) 同缘）——前递补边
        # fail-closed 硬化后，无登记的 restore 会 raise（需求①配套，
        # 2026-09-17；物理序列 = 会话先 PARTIAL 化才有后缀可恢复）。
        builder.emit_admission_batch({
            "request_id": "r0", "session_id": "other", "turn_index": 0,
            "queue_index": 0, "prefill_instance_index": 0,
            "decode_instance_index": 0, "admission_time_ns": 1000,
            "history_location_before": None, "history_transfer": None,
            "history_evictions": [_suffix_store("session_x")],
            "prefill_evictions": [],
            "history_tokens_before": 0,
            "prefill_context_tokens": 300,
        })
        request_id = "s_request_1"
        gate_ids = {}
        for rank in (0, 1):
            builder.builders[rank].comp(f"gate_seed_rank{rank}", 1, 1)
            gate_ids[rank] = builder.builders[rank].previous_id
        builder.pending_history[request_id] = PendingHistoryGate(
            source_instance_index=0,
            timer_gates=(gate_ids[0], gate_ids[1]),
            location="partial_hbm_remote")
        # 同实例退化：后缀恢复目标 = prefill 实例自身 ranks（0,1）。
        degenerate_restore = KVTransfer(
            kind="remote_load", phase="history",
            reason="history_remote_suffix_restore",
            session_id="session_x", trigger_request_id=request_id,
            source_instance_index=None, target_instance_index=0,
            total_bytes=1280,
            shards=tuple(
                KVTransferShard(
                    source_rank=edge, target_rank=target, edge_rank=edge,
                    bytes=640, noc_path=(edge, target),
                    layer_start=2, layer_end=4)
                for target, edge in ((0, 0), (1, 1))),
            model_layers=4, layer_start=2, layer_end=4,
            resident_prefix_layers_before=2,
            resident_prefix_layers_after=4)
        builder.emit_admission_batch({
            "request_id": request_id, "session_id": "session_x",
            "turn_index": 1, "queue_index": 1,
            "prefill_instance_index": 0, "decode_instance_index": 0,
            "admission_time_ns": None, "hbm_wait_ns": 0,
            "joint_action": "copy",
            "history_location_before": SimpleNamespace(
                location="partial_hbm_remote", instance_index=0,
                resident_prefix_layers=2),
            "history_transfers": [degenerate_restore],
            "history_evictions": [], "prefill_evictions": [],
            "history_tokens_before": 100,
            "prefill_context_tokens": 300,
        })
        nodes = builder.batch["nodes"]
        names = [node["name"] for node in nodes]
        # 无 noc 前缀腿（退化 = 仅后缀恢复，不建工作副本）。
        self.assertFalse(
            [n for n in names if "noc_migrate" in n],
            "退化 copy 不得发射前缀 NoC 腿")
        # partial 快速路径分支结构：驻留前缀屏障 + 逐 rank 后缀完成门 +
        # p2p suffix readiness + 恢复链。
        for fragment in ("prefill_resident_prefix_ready_barrier",
                         "_target_hbm_write", "_remote_load"):
            self.assertTrue(
                [n for n in names if fragment in n],
                f"缺 {fragment}")
        self.assertIn(request_id, builder._suffix_body_arms)
        arms = builder._suffix_body_arms[request_id]
        self.assertEqual(set(arms), {0, 1})
        self.assertTrue(all(isinstance(v, int) for v in arms.values()))


# ===================================================== R16-4-5 水印重放 ==


class CompositeWatermarkReplayTest(unittest.TestCase):
    """R16-4-5：复合两笔行 → 目标端两腿入账、源端不动、grow 顶账差额。"""

    def test_composite_history_transfers_replay(self):
        scan = _scan({
            "q1": _token_row(0, 10, 10),
            "q2": _token_row(10, 14, 16),
        })
        # turn-1（stay@0）：建立 home 驻留 10 token（1000 B）。
        scan.consume({"request_id": "q1", "kind": "prefill", "tick": 1,
                      "decision": {"joint_action": "stay",
                                   "prefill_instance_index": 0}})
        scan.consume({"request_id": "q1", "kind": "decode", "tick": 2,
                      "decision": {"joint_action": "stay",
                                   "decode_instance_index": 0}})
        scan.consume({"request_id": "q1", "kind": "completion", "tick": 3,
                      "decision": {"joint_action": "stay"}})
        self.assertEqual(scan.replay.occupancy, {0: 1000})
        # turn-2（复合 copy@1）：前缀 noc 600 + 后缀 remote_load 400。
        scan.consume({"request_id": "q2", "kind": "prefill", "tick": 4,
                      "decision": {
                          "joint_action": "copy",
                          "prefill_instance_index": 1,
                          "origin_home_instance": 0,
                          "history_transfers": [
                              {"kind": "noc_migrate", "total_bytes": 600},
                              {"kind": "remote_load", "total_bytes": 400}]}})
        # 目标端 = 600 + 400；源端（home 0）前缀不动；grow 到 14 token。
        self.assertEqual(scan.replay.occupancy, {0: 1000, 1: 1400})
        scan.consume({"request_id": "q2", "kind": "decode", "tick": 5,
                      "decision": {"joint_action": "copy",
                                   "decode_instance_index": 1}})
        self.assertEqual(scan.replay.occupancy, {0: 1000, 1: 1600})
        scan.consume({"request_id": "q2", "kind": "completion", "tick": 6,
                      "decision": {
                          "joint_action": "copy",
                          "origin_home_instance": 0,
                          "joint_working_copy": True,
                          "merge_transfers": [
                              {"kind": "noc_migrate", "total_bytes": 200},
                              {"kind": "remote_store",
                               "reason": "merge_increment_suffix_pool_store",
                               "session_id": "sess",
                               "total_bytes": 200}]}})
        # PARTIAL 基结算：home 只并入前缀增量 200（后缀池写不占 home）；
        # 执行端工作副本释放归零——两腿合计不得漏账或双计。
        self.assertEqual(scan.replay.occupancy, {0: 1200, 1: 0})
        session = scan.replay.sessions["sess"]
        self.assertEqual((session.instance, session.bytes), (0, 1200))


# ===================================================== R16-4-6 merge 回归 ==


class CompositeCopyMergeRegressionTest(unittest.TestCase):
    """copy@PARTIAL 完成 → merge v2 零字节翻转 + 版本键单次。
    金样留档（改造前口径，git 9a95e06）：两笔拆分——增量前缀 noc 回
    home [0,prefix) + 后缀增量 remote_store [prefix,L)，终态 PARTIAL@home。
    C4 triage（2026-09-22）：C13 copy 块级交接/源端立即释放后，home 基础
    前缀的释放点自 merge 迁至交接块到达（expand_prefill 的 prefill_drain
    结算边界 `_settle_copy_handoffs`）——merge 时刻 home 剩余量回升量由
    base_prefix 改为 0（释放前移、总量守恒不变，断言对象随之刷新）。"""

    def test_merge_after_composite_copy(self):
        kv = _manager()
        _make_partial(kv)
        prefix = kv._sessions["s"].resident_prefix_layers
        kv.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="t1", reservation_request_id="t1",
            action="copy")
        remaining_home_before_expand = kv._effective_remaining_by_tp_rank(0)
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=14,
            trigger_request_id="t1")
        # C13 源端立即释放：交接块在 prefill_drain 边界结算，home（实例
        # 0）基础前缀 H = kv(10)@[0,prefix) 于 expand 内释放——不再等
        # merge（旧口径断言点）。
        base_prefix = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree, layer_start=0, layer_end=prefix)
        self.assertEqual(
            tuple(after - before for before, after in zip(
                remaining_home_before_expand,
                kv._effective_remaining_by_tp_rank(0))),
            base_prefix)
        remaining_home_before = kv._effective_remaining_by_tp_rank(0)

        transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=4)

        # 零字节翻转：零传输（无 noc/remote_store——I6 合并零池写）。
        self.assertEqual(transfers, ())
        session = kv._sessions["s"]
        # 工作副本转正：LOCAL@exec（胜者），home 迁移 exec。
        self.assertEqual(session.working_kind, None)
        self.assertEqual(session.location, kv.LOCAL_HBM)
        self.assertEqual(session.instance_index, 1)
        self.assertEqual(session.home_instance, 1)
        self.assertEqual(session.resident_prefix_layers, kv.model.layers)
        self.assertEqual(session.context_tokens, 14)
        self.assertEqual(session.last_merged_request_id, "t1")
        # home（实例 0）已在交接结算释放全部基础前缀（上断言）——merge
        # 时刻零增量（不重复释放已交接源块，C12 §2 规则 4 / C13 守恒式
        # D_handoff 轮末归零）。
        self.assertEqual(
            tuple(after - before for before, after in zip(
                remaining_home_before,
                kv._effective_remaining_by_tp_rank(0))),
            (0, 0))
        # 版本键：同请求重复 merge = 合同类违规。
        with self.assertRaises(RuntimeError):
            kv.merge_back(
                session_id="s", trigger_request_id="t1", new_tokens=4)


# ===================================================== R16-4-8b 退化负例 ==


class CopyAtHomePartialDegradeTest(unittest.TestCase):
    """R16-6：copy@home×PARTIAL 镜像 stay-partial 退化（不撞 N9）。"""

    def test_degenerates_to_stay_partial(self):
        kv = _manager()
        _make_partial(kv)
        prefix = kv._sessions["s"].resident_prefix_layers
        suffix_shards = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree,
            layer_start=prefix, layer_end=kv.model.layers)
        remaining_before = kv._effective_remaining_by_tp_rank(0)

        _, transfers, _ = kv.prepare_prefill(
            session_id="s", target_instance_index=0, history_tokens=10,
            trigger_request_id="t1", reservation_request_id="t1",
            action="copy")

        # 只恢复缺失后缀：单笔 remote_load、无 NoC 前缀笔。
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].kind, "remote_load")
        self.assertEqual(
            (transfers[0].layer_start, transfers[0].layer_end),
            (prefix, kv.model.layers))
        session = kv._sessions["s"]
        # 不建工作副本：working_kind 保持 None、不双计整份。
        self.assertIsNone(session.working_kind)
        self.assertEqual(session.location, kv.LOCAL_HBM)
        self.assertEqual(session.instance_index, 0)
        self.assertEqual(session.resident_prefix_layers, kv.model.layers)
        self.assertEqual(
            tuple(before - after for before, after in zip(
                remaining_before, kv._effective_remaining_by_tp_rank(0))),
            suffix_shards)
        # 完成结算：stay 等价本地提交（不撞 N9 防御）。
        settled = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=4)
        self.assertEqual(settled, ())
        self.assertEqual(session.last_merged_request_id, "t1")


# ===================================================== R16-4-8c 序列回归 ==


class CompositeCycleSequenceTest(unittest.TestCase):
    """复合 copy → merge v2 零字节翻转（LOCAL@胜者、home 漂移）→ 再跨
    实例 copy 序列。金样留档（改造前口径，git 9a95e06）：merge 落回
    PARTIAL@home、home 恒 0；第 2 轮复合两腿 [0,prefix)+[prefix,L)。"""

    def test_cycle_sequence(self):
        kv = _manager()
        _make_partial(kv)
        prefix = kv._sessions["s"].resident_prefix_layers

        # 第 1 轮：复合 copy@1（两腿）+ 增长 + merge 零字节翻转 LOCAL@1。
        kv.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="t1", reservation_request_id="t1",
            action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=14,
            trigger_request_id="t1")
        kv.merge_back(session_id="s", trigger_request_id="t1", new_tokens=4)
        self.assertEqual(kv._sessions["s"].location, kv.LOCAL_HBM)
        self.assertEqual(kv._sessions["s"].instance_index, 1)
        self.assertEqual(kv._sessions["s"].home_instance, 1)
        self.assertEqual(kv._sessions["s"].context_tokens, 14)

        # 第 2 轮：再跨实例 copy（LOCAL 基@1 → 目标 0）——单腿全层 noc。
        _, transfers, _ = kv.prepare_prefill(
            session_id="s", target_instance_index=0, history_tokens=14,
            trigger_request_id="t2", reservation_request_id="t2",
            action="copy")
        self.assertEqual(
            [(t.kind, t.layer_start, t.layer_end) for t in transfers],
            [("noc_migrate", 0, kv.model.layers)])
        full14 = kv_cache_shard_bytes_for_tokens(kv.model, 14, kv.tp_degree)
        self.assertEqual(
            sum(t.total_bytes for t in transfers), sum(full14))
        kv.expand_prefill(
            session_id="s", instance_index=0, context_tokens=18,
            trigger_request_id="t2")
        settled = kv.merge_back(
            session_id="s", trigger_request_id="t2", new_tokens=4)
        # 零字节翻转（home 漂移 1 → 0）。
        self.assertEqual(settled, ())
        self.assertEqual(kv._sessions["s"].context_tokens, 18)
        self.assertEqual(kv._sessions["s"].location, kv.LOCAL_HBM)
        self.assertEqual(kv._sessions["s"].home_instance, 0)


# ===================================================== R16-4-8d 日志形状 ==


class CompositeDecisionLogShapeTest(unittest.TestCase):
    """决策日志复合两笔序列化形状 + 单/复数/字节聚合三字段合同。"""

    def test_transfer_summary_and_aggregation_contract(self):
        from online.sh30_online_scheduler import _transfer_summary
        kv = _manager()
        _make_partial(kv)
        prefix = kv._sessions["s"].resident_prefix_layers
        _, transfers, _ = kv.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="t1", reservation_request_id="t1",
            action="copy")
        summaries = [_transfer_summary(t) for t in transfers]
        # 复数列表逐笔带层区间 + shard 级路由（hopbytes 采集面）。
        self.assertEqual(
            [(row["kind"], row["layer_start"], row["layer_end"])
             for row in summaries],
            [("noc_migrate", 0, prefix),
             ("remote_load", prefix, kv.model.layers)])
        for row in summaries:
            self.assertGreater(len(row["shards"]), 0)
            for shard in row["shards"]:
                self.assertIn("noc_hops", shard)
                self.assertIn("noc_path", shard)
                self.assertIsInstance(shard["bytes"], int)
        # sh30 :2404-2414 三字段合同：单数 = 复数首笔；字节 = 两腿求和。
        singular = (
            transfers[0] if transfers else None)
        self.assertEqual(_transfer_summary(singular), summaries[0])
        history_transfer_bytes = sum(
            t.total_bytes for t in transfers if t.kind != "local_hit")
        self.assertEqual(
            history_transfer_bytes,
            sum(row["total_bytes"] for row in summaries))


# ===================================================== R16-7 hopbytes ==


class HopbytesCompositeCollectionTest(unittest.TestCase):
    """R16-7：collect_joint 复数列表两腿采集 + 单行回退逐位一致。"""

    @staticmethod
    def _acc():
        return {"actions_with_hops": 0, "hop_bytes_total": 0,
                "bytes_with_hops": 0, "bytes_without_hops": 0}

    def test_composite_two_legs_collected(self):
        record = {"kind": "prefill", "request_id": "q1", "decision": {
            "joint_action": "copy",
            "history_transfer": {          # 单数恒为第一笔（前缀腿）
                "kind": "noc_migrate", "total_bytes": 640,
                "shards": [{"bytes": 640, "noc_hops": 1}]},
            "history_transfers": [
                {"kind": "noc_migrate", "total_bytes": 640,
                 "shards": [{"bytes": 640, "noc_hops": 1}]},
                {"kind": "remote_load", "total_bytes": 640,
                 "shards": [{"bytes": 640, "noc_hops": 2}]}]}}
        acc, per = self._acc(), {}
        hopbytes.collect_joint(record, acc, per)
        # 两腿都入账：hop_bytes = 640×1 + 640×2（旧 collect_sh30 只见
        # 单数前缀腿 → 640×1，后缀 NoC 段系统性漏计——本修复点）。
        self.assertEqual(acc["hop_bytes_total"], 640 * 1 + 640 * 2)
        self.assertEqual(acc["bytes_with_hops"], 1280)
        self.assertEqual(acc["actions_with_hops"], 2)
        # 旧收集器对照：同记录单数口径（回归保护——差值即修复面）。
        old_acc, old_per = self._acc(), {}
        hopbytes.collect_sh30(record, old_acc, old_per)
        self.assertEqual(old_acc["hop_bytes_total"], 640)
        # 注册表路径钉死（五轮深审补）：生产 CLI 经 REPO_HOP_SOURCES 分发，
        # 上述用例直接调用 collect_joint、不覆盖该映射——错配时测试全绿
        # 而实跑静默漏计后缀腿。
        self.assertIs(
            hopbytes.REPO_HOP_SOURCES["astra-sim-joint"]["collector"],
            hopbytes.collect_joint)

    def test_legacy_single_field_fallback_bit_identical(self):
        record = {"kind": "prefill", "request_id": "q1", "decision": {
            "history_transfer": {
                "kind": "noc_migrate", "total_bytes": 640,
                "shards": [{"bytes": 640, "noc_hops": 1}]}}}
        acc, per = self._acc(), {}
        hopbytes.collect_joint(record, acc, per)
        old_acc, old_per = self._acc(), {}
        hopbytes.collect_sh30(record, old_acc, old_per)
        self.assertEqual(acc, old_acc)
        self.assertEqual(per, old_per)

    def test_local_hit_and_no_route_semantics(self):
        record = {"kind": "prefill", "request_id": "q1", "decision": {
            "history_transfers": [
                {"kind": "local_hit", "total_bytes": 0, "shards": []},
                {"kind": "remote_load", "total_bytes": 300,
                 "shards": [{"bytes": 300}]}]}}     # 无路由 → fallback 桶
        acc, per = self._acc(), {}
        hopbytes.collect_joint(record, acc, per)
        self.assertEqual(acc["hop_bytes_total"], 0)
        self.assertEqual(acc["bytes_without_hops"], 300)
        self.assertEqual(acc["bytes_with_hops"], 0)

    def test_decode_and_completion_kinds_passthrough(self):
        record = {"kind": "decode", "request_id": "q1", "decision": {
            "prefill_decode_transfer": {
                "kind": "noc_migrate", "total_bytes": 100,
                "shards": [{"bytes": 100, "noc_hops": 2}]}}}
        acc, per = self._acc(), {}
        hopbytes.collect_joint(record, acc, per)
        self.assertEqual(acc["hop_bytes_total"], 200)
        record = {"kind": "completion", "request_id": "q1", "decision": {
            "completion_evictions": [{
                "kind": "remote_store", "total_bytes": 50,
                "shards": [{"bytes": 50, "noc_hops": 1}]}]}}
        acc, per = self._acc(), {}
        hopbytes.collect_joint(record, acc, per)
        self.assertEqual(acc["hop_bytes_total"], 50)

    def test_kv_eviction_collected_and_dead_prefill_evictions_ignored(self):
        """R17-2c（kimi 终审 P3 迁入基线车道，2026-09-17）：
        kind=kv_eviction 决策行 evictions[].shards 入账（decode 增长/
        准入失败逐出池写流，语义同 decode_evictions；跳数按实际路由——
        本批 96gib 实测 noc_hops=2）；collect_joint 的 prefill_evictions
        为结构性恒空死通道（准入 R1' 预约覆盖 prefill 全动作足迹、drain
        expand gap≡0，三方裁决）已移除读取，history_evictions 照常消费。
        原置于 slo_tools/tests/test_slo_contract.py（--ignore 面，仅目录
        内直跑生效）——迁入后进 320 门禁。"""
        acc, per = self._acc(), {}
        record = {"kind": "kv_eviction", "request_id": "r0", "tick": 5,
                  "decision": {"evictions": [{
                      "shards": [
                          {"bytes": 1024, "noc_hops": 1,
                           "noc_path": [8, 9]},
                          {"bytes": 512, "noc_hops": 0,
                           "noc_path": [10]},
                          {"bytes": 256, "noc_path": [4, 5, 6]}]}]}}
        hopbytes.collect_joint(record, acc, per)
        # 1024*1 + 512*0 + 256*2 = 1536；0 跳贡献零 hop_bytes、缺显式
        # noc_hops 由 noc_path 推导（len-1，同 _shard_hops 语义）。
        self.assertEqual(acc["hop_bytes_total"], 1536)
        self.assertEqual(acc["bytes_with_hops"], 1792)
        self.assertEqual(acc["actions_with_hops"], 3)
        self.assertEqual(per["r0"]["hop_bytes"], 1536)
        # prefill_evictions 即使非空也不入账（死通道移除钉子）；
        # history_evictions 消费如常（100*2=200）。
        record2 = {"kind": "prefill", "request_id": "r1", "tick": 6,
                   "decision": {
                       "prefill_evictions": [{
                           "shards": [{"bytes": 900, "noc_hops": 1}]}],
                       "history_evictions": [{
                           "shards": [{"bytes": 100, "noc_hops": 2}]}]}}
        hopbytes.collect_joint(record2, acc, per)
        self.assertEqual(acc["hop_bytes_total"], 1736)
        self.assertEqual(acc["bytes_with_hops"], 1892)
        self.assertEqual(per["r1"]["hop_bytes"], 200)


# ================================================= R17-1b 咽喉点披露 ==


class KvEvictionChokePointLogTest(unittest.TestCase):
    """R17-5a：容量逐出咽喉点决策日志（kind=kv_eviction）四用例。

    R17-7 探针（2026-09-17）管理器级三通道裁决的单测显式化：
      通道 1（expand_prefill）结构性恒空——R1' 预约不变量；
      通道 2（expand_decode）真凶主通道——kv_eviction 行落盘且
      victim/区间/字节/前缀字段与 KVCacheManager 逐出一致；
      通道 3（准入失败 exc.evictions）当批零触发（feasible 预检
      拦截 + 预约不变量双重覆盖；在盘 4i 33+6i 1 条失败全为预检
      拦截形态）——潜伏位点，本类以直调咽喉点验证披露接线。
    tick 守护（kimi N4）：_batch 不在场 = 生命周期破损，断言式熔断。
    """

    def _pressure_manager(self):
        """3 个 complete victim + active grower 的容量压力构造。

        容量（每 rank）= 4 × seed(100 tok) 整份字节——与 R17-7 探针
        同口径；typed/minimal 与 _manager() 缺省一致。
        """
        from face_scheduler import FaceModel, KVCacheManager
        from joint.test_joint_fixes import _tiny_hardware
        from joint.test_joint_mechanisms import _two_instance_topology
        model = FaceModel(
            layers=4, hidden_size=16, ffn_size=32, num_heads=4,
            vocab_size=32, bytes_per_elem=2, mlp_variant="swiglu")
        per_seed = kv_cache_shard_bytes_for_tokens(model, 100, 2)[0]
        return KVCacheManager(
            _two_instance_topology(_tiny_hardware(4 * per_seed)), model,
            category_mode="typed", layer_policy="minimal_layer_groups")

    def _seed_victims(self, kv):
        for i, name in enumerate(("v1", "v2", "v3")):
            _seed(kv, name, 0, 100, (i + 1) * 10, "human")

    def _scheduler(self, kv, tick=777):
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.kv_manager = kv
        scheduler._kv_ledger_epoch = 0
        scheduler._stalled_by_instance = {}
        scheduler._batch = {"tick": tick, "assignments": [], "watches": []}
        scheduler._pending_eviction_watches = {}
        scheduler._eviction_watch_seq = {}
        scheduler.online_log_count = 0
        scheduler.decision_log_sink = None
        scheduler.online_log_rows = []
        graph_emitted = []
        def emit_eviction_side_branch(transfers, event_tick, *, watch_id):
            graph_emitted.append((tuple(transfers), event_tick))
            return {"request_id": watch_id,
                    "owner_request_id": transfers[0].trigger_request_id,
                    "members": {0: 1}}

        scheduler.graph = SimpleNamespace(
            emit_eviction_side_branch=emit_eviction_side_branch,
            sync_pending_history_after_evictions=lambda evictions: None,
        )
        # 计价流登记面属 R18 计量批（方案 §7 红线：流登记不在本批）。
        scheduler._register_transfer_flows = lambda transfers, owner: None
        # 对齐 __init__ 初值（F6 销账：类级软缺省已删；off 档 None，
        # 替身漏设 = AttributeError）。
        scheduler._quota_tracker = None
        return scheduler, graph_emitted

    def _grower_runtime(self, consumed=250):
        """stay 动作 grower：working = history(0)+input(50)+consumed。"""
        return SimpleNamespace(
            session_id="sG", request_id="rG", joint_action="stay",
            history_tokens_before=0, joint_input_tokens=50,
            decode_tokens_consumed=consumed,
            joint_span_base_context=None, joint_prefill_work=0,
            decode_stalled=False)

    def test_channel2_decode_growth_emits_kv_eviction_row(self):
        """① 通道 2：逐列车增长逐出 → kv_eviction 行 + 图发射同点同刻，
        字段与 KVCacheManager 一致（victim/区间/前缀/R17-1d 新键）。"""
        kv = self._pressure_manager()
        self._seed_victims(kv)
        kv.prepare_prefill(
            session_id="sG", target_instance_index=0, history_tokens=0,
            trigger_request_id="rG")
        kv.expand_prefill(
            session_id="sG", instance_index=0, context_tokens=50,
            trigger_request_id="rG")
        scheduler, emitted = self._scheduler(kv)
        scheduler._joint_grow_decode(self._grower_runtime(), 0)
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "kv_eviction"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["request_id"], "rG")
        self.assertEqual(row["tick"], 777)     # 逐出精确 tick = 批次 tick
        entries = row["decision"]["evictions"]
        self.assertTrue(entries)
        victims = {entry["session_id"] for entry in entries}
        self.assertTrue(victims <= {"v1", "v2", "v3"})
        for entry in entries:
            self.assertTrue(entry["reason"].startswith(
                "decode_growth_capacity"))
            self.assertGreater(entry["total_bytes"], 0)
            # R17-1d 新键：前缀迁移自证 + victim 归位。
            self.assertIn("resident_prefix_layers_before", entry)
            self.assertIn("resident_prefix_layers_after", entry)
            self.assertIn("source_instance_index", entry)
            self.assertEqual(entry["source_instance_index"], 0)
            snapshot = kv.session_snapshot(entry["session_id"])
            self.assertEqual(
                entry["resident_prefix_layers_after"],
                snapshot.resident_prefix_layers)
        # 披露与物理同点同刻：图侧旁路支链同 tick、同一批 transfers。
        self.assertEqual(len(emitted), 1)
        emitted_transfers, emitted_tick = emitted[0]
        self.assertEqual(emitted_tick, 777)
        self.assertEqual(
            [t.session_id for t in emitted_transfers],
            [e["session_id"] for e in entries])

    def test_channel3_latent_site_wiring_logs_row(self):
        """② 通道 3 潜伏位点：exc.evictions 非空的假想形态走咽喉点，
        行落盘接线成立（当批结构不可达，R17-7 探针 C 实证）。"""
        kv = _manager()
        _seed(kv, "s1", 0, 10, 10, "human")
        transfer = kv._evict_suffix(
            kv._sessions["s1"], phase="history", reason="probe_admit_fail",
            trigger_request_id="rX")
        scheduler, emitted = self._scheduler(kv, tick=999)
        scheduler._emit_eviction_only_nodes(
            (transfer,), 999, trigger_request_id="rX")
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "kv_eviction"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_id"], "rX")
        self.assertEqual(rows[0]["tick"], 999)
        entry = rows[0]["decision"]["evictions"][0]
        self.assertEqual(entry["session_id"], "s1")
        # reason 携带逐层策略后缀（如 _suffix_half），前缀匹配。
        self.assertTrue(entry["reason"].startswith("probe_admit_fail"))
        self.assertEqual(len(emitted), 1)

    def test_channel1_prefill_growth_structurally_empty(self):
        """③ 通道 1 谱系钉死显式化：压力下准入逐出非空（前提成立），
        drain expand_prefill 恒空（R1' 预约不变量，gap≡0）。"""
        kv = self._pressure_manager()
        self._seed_victims(kv)
        final_tokens = 200
        admission_evictions = kv.reserve_request_capacity(
            request_id="rA", session_id="sA", instance_index=0,
            final_context_tokens=final_tokens, action="recompute")
        kv.prepare_prefill(
            session_id="sA", target_instance_index=0, history_tokens=0,
            trigger_request_id="rA", reservation_request_id="rA",
            action="recompute")
        drain_evictions = kv.expand_prefill(
            session_id="sA", instance_index=0, context_tokens=final_tokens,
            trigger_request_id="rA", reservation_request_id="rA")
        self.assertTrue(admission_evictions)   # 压力前提：预约路径有逐出
        self.assertEqual(drain_evictions, ())  # 死通道：结构性恒空

    def test_require_batch_tick_asserts_when_batch_missing(self):
        """④ tick 守护（kimi N4）：_batch 缺失 = 生命周期破损，
        断言式熔断（原 `else 0` 虚构回退会破坏重放全序单调）。"""
        kv = _manager()
        scheduler, _ = self._scheduler(kv)
        scheduler._batch = None
        with self.assertRaises(AssertionError):
            scheduler._require_batch_tick()
        # 在场时返回批次 tick。
        scheduler._batch = {"tick": 42}
        self.assertEqual(scheduler._require_batch_tick(), 42)


if __name__ == "__main__":
    unittest.main()
