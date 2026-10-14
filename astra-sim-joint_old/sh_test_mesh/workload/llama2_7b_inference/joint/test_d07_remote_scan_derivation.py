#!/usr/bin/env python3
"""test_d07_remote_scan_derivation.py -- D07（M1′ 计价侧，v2.0 式 11/18、
§10.3/§13.3）remote 扫描次数从 chunk/tile 推导的零后端单测。

覆盖（条目原文三点 + 验收三条）：
  1. 模板身份推导：RequestView 携带 chunk/query tile 合法模板身份
     （q_c/q_tile）时 `_remote_read_passes` 从已知 U 闭式推导
     Σν = a⌈q_c/q_tile⌉ + 1_{b>0}⌈b/q_tile⌉（单一裁决点
     `prefill_remote_read_scan_passes`，与执行侧同源）——prefill_scan_
     passes 供给值缺省（None）也推导同值，不依赖外部传入的未说明数字；
  2. 一致性校验：执行侧 prefill_scan_passes 与推导值不一致即
     fail-closed（拒价，不静默采信）；
  3. 合法模板身份：chunk/query tile 须成对供给且为正整数（半份身份/
     零/负/bool 构造即拒）；
  4. 验收①：U 从一个 chunk 增为多个 chunk（含尾 chunk）时基础历史
     远读量按扫描数增长（read_ns 随遍数线性、字节基数 = 基础历史）；
  5. 验收②：改 query tile 时扫描数与执行侧闭式逐值一致；
  6. 验收③：输入/生成增量不重复跨 Mesh——decode 步数与 input 增量
     字节不进入远读基数（基数恒 = 基础历史 × 遍数）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d07_remote_scan_derivation.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import (  # noqa: E402
    prefill_remote_read_scan_passes,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    InstanceLoadView,
    JointCostError,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)


# ============================================================ 夹具构造 ==

def _rates(noc=1000.0, hbm=10.0, lat=0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=5.0, local_hbm_gbps=hbm,
        d2d_latency_ns=lat, pool_latency_ns=100)


def _load(index=0, total=0):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=0, history_bytes=(1_000_000, 1_000_000),
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location="local_hbm", history_tokens=history_tokens,
        resident_prefix_layers=4,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=(0, 0))


def _request(input_tokens=512, decode=3, input_bytes=(50_000, 50_000),
             prefill_scan_passes=None, chunk_tokens=None, tile_tokens=None):
    """identity = (chunk_tokens, tile_tokens)；None = 夹具旧路径（无模板
    身份）。缺省 (512, 128) 与 REMOTE_READ_QUERY_TILE_TOKENS 无关——
    本文件锚的是推导语义本身，U=512 → Σν = ⌈512/128⌉ = 4。"""
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes,
        prefill_scan_passes=prefill_scan_passes,
        prefill_chunk_tokens=chunk_tokens,
        query_tile_tokens=tile_tokens)


def _disjoint_paths(source, target):
    """tp=2 逐 rank 不相交路径（rank 对 (2s,2t)/(2s+1,2t+1)，各 1 跳）。"""
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


def _route(source, target):
    return ((source, target), 1)


def _model():
    return JointCostModel(
        rates=_rates(),
        loads={0: _load(0), 1: _load(1)},
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        route_paths_fn=_disjoint_paths,
        remote_read_tile_bytes="auto",
        # tile 基准注入（与 history 视图 1_000_000/100 同值）——input 增
        # 量字节膨胀不改变 tile 基准（验收③的隔离前提）。
        remote_read_kv_token_bytes=10_000)


def _remote(session, request):
    candidate = _model().estimate_P_with_continuation(
        session=session, request=request, instance_index=1,
        action=ACTION_REMOTE, remote_enabled=True)
    if not candidate.applicable:
        raise AssertionError(
            "remote candidate unexpectedly inapplicable: "
            f"{candidate.inapplicable_reason}")
    return candidate


def _notes(candidate):
    return "\n".join(candidate.breakdown.notes)


# ======================================================== 1. 模板身份推导 ==

class TemplateIdentityDerivationTest(unittest.TestCase):
    """身份在场即闭式推导：供给值缺省也推导同值（不依赖外部传入数字）。"""

    def test_derives_scan_passes_without_supplied_value(self):
        # U=512、q_c=512、q_tile=128 → a=1、b=0 → Σν = ⌈512/128⌉ = 4。
        candidate = _remote(
            _session(), _request(chunk_tokens=512, tile_tokens=128))
        self.assertIn("prefill_scan_passes=4", _notes(candidate))
        self.assertIn("remote_read_passes=7", _notes(candidate))  # 4 + decode 3

    def test_derived_value_ignores_unsupplied_passes_field(self):
        # 同一身份下 prefill_scan_passes=None 与 =4 同值——扫描数来自
        # 推导而非供给（供给仅作一致性校验对象）。
        with_supply = _remote(
            _session(),
            _request(prefill_scan_passes=4, chunk_tokens=512,
                     tile_tokens=128))
        without_supply = _remote(
            _session(), _request(chunk_tokens=512, tile_tokens=128))
        self.assertEqual(
            with_supply.breakdown.remote_read_ns,
            without_supply.breakdown.remote_read_ns)
        self.assertIn("prefill_scan_passes=4", _notes(without_supply))

    def test_tail_chunk_closed_form_via_jcm(self):
        # 尾 chunk 闭式：U=600、q_c=512、q_tile=128 → a=1、b=88 →
        # Σν = 1×4 + ⌈88/128⌉ = 5（不构造长列表）。
        candidate = _remote(
            _session(),
            _request(input_tokens=600, chunk_tokens=512, tile_tokens=128))
        self.assertIn("prefill_scan_passes=5", _notes(candidate))

    def test_derivation_matches_execution_adjudication_point(self):
        # 与执行侧单一裁决点逐值一致（式 11/18 同源——改 q_c/q_tile/U
        # 任何一者，计价侧推导恒等于执行侧闭式）。
        for input_tokens, chunk, tile in (
                (512, 512, 128), (1024, 512, 128), (600, 512, 128),
                (512, 512, 256), (100, 512, 128), (0, 512, 128),
                (129, 64, 32)):
            self.assertEqual(
                prefill_remote_read_scan_passes(input_tokens, chunk, tile),
                JointCostModel._remote_read_passes(
                    _model(), _request(
                        input_tokens=input_tokens, chunk_tokens=chunk,
                        tile_tokens=tile))[0],
                f"U={input_tokens} q_c={chunk} q_tile={tile}")


# ====================================================== 2. 一致性校验 ==

class SuppliedPassesConsistencyTest(unittest.TestCase):
    """执行侧供给值 = 一致性校验对象：与推导不一致即拒价。"""

    def test_matching_supply_passes(self):
        candidate = _remote(
            _session(),
            _request(prefill_scan_passes=4, chunk_tokens=512,
                     tile_tokens=128))
        self.assertIn("prefill_scan_passes=4", _notes(candidate))

    def test_stale_single_pass_supply_fails_closed(self):
        # 旧"单遍扫描"供给值与多 chunk 推导矛盾 → fail-closed（不静默
        # 采信外部传入的未说明数字）。
        with self.assertRaisesRegex(JointCostError, "contradicts"):
            _remote(
                _session(),
                _request(prefill_scan_passes=1, chunk_tokens=512,
                         tile_tokens=128))

    def test_mismatch_message_carries_both_sides(self):
        try:
            _remote(
                _session(),
                _request(input_tokens=1024, prefill_scan_passes=4,
                         chunk_tokens=512, tile_tokens=128))
        except JointCostError as exc:
            message = str(exc)
            self.assertIn("prefill_scan_passes=4", message)
            self.assertIn("-> 8", message)  # 推导值 a⌈q_c/q_tile⌉ = 2×4
        else:
            self.fail("expected JointCostError for contradicting supply")


# ==================================================== 3. 合法模板身份 ==

class TemplateIdentityLegalityTest(unittest.TestCase):
    """RequestView 模板身份合法性（构造即 fail-closed）。"""

    def _build(self, **kwargs):
        return _request(**kwargs)

    def test_half_identity_rejected(self):
        with self.assertRaisesRegex(JointCostError, "pair"):
            self._build(chunk_tokens=512)
        with self.assertRaisesRegex(JointCostError, "pair"):
            self._build(tile_tokens=128)

    def test_non_positive_values_rejected(self):
        with self.assertRaisesRegex(JointCostError, "positive int"):
            self._build(chunk_tokens=0, tile_tokens=128)
        with self.assertRaisesRegex(JointCostError, "positive int"):
            self._build(chunk_tokens=512, tile_tokens=-128)

    def test_bool_identity_rejected(self):
        with self.assertRaisesRegex(JointCostError, "positive int"):
            self._build(chunk_tokens=True, tile_tokens=128)

    def test_absent_identity_still_legal(self):
        # 直接构造的测试夹具路径：无身份 = 沿用供给值/因果缺省（零漂移）。
        request = self._build()
        self.assertIsNone(request.prefill_chunk_tokens)
        self.assertIsNone(request.query_tile_tokens)


# ======================================================== 4. 验收（三条）==

class AcceptanceTest(unittest.TestCase):
    """条目验收：多 chunk 增长 / tile 一致 / 增量不重复跨 Mesh。"""

    def test_acceptance_multi_chunk_grows_base_history_reads(self):
        # U 512→1024（1→2 个完整 chunk）：遍数 7→11（+4 扫描），基础
        # 历史远读量按扫描数增长；且 +4 扫描的增量与 +4 decode 步的
        # 增量同值（字节基数恒为基础历史 × 遍数——遍数来源不改变基数）。
        one_chunk = _remote(
            _session(), _request(decode=3, chunk_tokens=512,
                                 tile_tokens=128))
        two_chunks = _remote(
            _session(), _request(input_tokens=1024, decode=3,
                                 chunk_tokens=512, tile_tokens=128))
        four_more_decode = _remote(
            _session(), _request(decode=7, chunk_tokens=512,
                                 tile_tokens=128))
        self.assertIn("prefill_scan_passes=4", _notes(one_chunk))
        self.assertIn("prefill_scan_passes=8", _notes(two_chunks))
        self.assertGreater(
            two_chunks.breakdown.remote_read_ns,
            one_chunk.breakdown.remote_read_ns)
        # 每遍读量守恒：Δ(2 chunks) = Δ(+4 decode 步) = 4 × 单遍。
        self.assertEqual(
            two_chunks.breakdown.remote_read_ns
            - one_chunk.breakdown.remote_read_ns,
            four_more_decode.breakdown.remote_read_ns
            - one_chunk.breakdown.remote_read_ns)

    def test_acceptance_query_tile_change_matches_execution(self):
        # 改 query tile：q_tile 128→256 → 每 chunk 扫描 4→2，计价侧
        # notes 与执行侧闭式同值（式 11 同源）。
        fine = _remote(
            _session(), _request(chunk_tokens=512, tile_tokens=128))
        coarse = _remote(
            _session(), _request(chunk_tokens=512, tile_tokens=256))
        self.assertIn("prefill_scan_passes=4", _notes(fine))
        self.assertIn("prefill_scan_passes=2", _notes(coarse))
        self.assertEqual(
            prefill_remote_read_scan_passes(512, 512, 256), 2)
        # 7→5 遍：同一字节基数（基础历史 × 遍数）下读量按遍数精确线性
        # 缩减（每遍 = 200_000 ns，两配置同源同基数）。
        self.assertLess(
            coarse.breakdown.remote_read_ns,
            fine.breakdown.remote_read_ns)
        self.assertEqual(
            coarse.breakdown.remote_read_ns * 7,
            fine.breakdown.remote_read_ns * 5)

    def test_acceptance_increments_not_remeshed(self):
        # 输入/生成增量不重复跨 Mesh：基数 = 基础历史。
        # (a) input 增量字节膨胀（U 不变）→ 远读量不变（输入本地读）；
        base = _remote(
            _session(),
            _request(input_bytes=(50_000, 50_000), chunk_tokens=512,
                     tile_tokens=128))
        inflated = _remote(
            _session(),
            _request(input_bytes=(5_000_000, 5_000_000), chunk_tokens=512,
                     tile_tokens=128))
        self.assertEqual(
            base.breakdown.remote_read_ns,
            inflated.breakdown.remote_read_ns)
        # (b) decode 步只乘基础历史（非 base+growth 终态口径）：
        # +1 decode 步的远读增量 = 单遍基础历史读量（与 (a) 同构锚：
        # decode 增量 KV 本地读、不过 NoC）。
        n = _remote(
            _session(), _request(decode=3, chunk_tokens=512,
                                 tile_tokens=128))
        n_plus1 = _remote(
            _session(), _request(decode=4, chunk_tokens=512,
                                 tile_tokens=128))
        per_pass = (n_plus1.breakdown.remote_read_ns
                    - n.breakdown.remote_read_ns)
        passes_total = 7  # prefill 4 + decode 3
        # 单遍远读 = 全轮远读 ÷ 遍数（同基数同份额下线性成立）。
        self.assertEqual(
            per_pass * passes_total,
            n.breakdown.remote_read_ns)
        self.assertGreater(per_pass, 0)


if __name__ == "__main__":
    unittest.main()
