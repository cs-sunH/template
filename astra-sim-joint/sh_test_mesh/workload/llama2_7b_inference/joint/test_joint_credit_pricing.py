#!/usr/bin/env python3
"""test_joint_credit_pricing.py -- remote-read 决策计价流水式的手算
锚点单测（§4.3.1，路线 B credit 交错流，P0）。

计价合同（唯一执行口径，2026-09-17 用户裁定：v1 串行加法口径删除，
不作为开关可选项保留）：

* effective = first_credit_ns + max(remaining_stream_ns, compute_ns)
  ——joint_config 同源公式，与 Workload.cc:558-566 闭式
  first_tile + max(remaining, compute) 同构；K 与执行侧切片同源
  （remote_credit_block_size 单一裁决点）；
* 不变量：K >= read_passes（单 credit）时 first_credit_ns ==
  remote_read_ns、remaining_stream_ns == 0 ⇒ 合成退化为
  remote_read_ns + compute_ns（旧加法形态的数值——单块切片等价锚）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_credit_pricing.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import remote_credit_block_size  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)


# ============================================================ 夹具构造 ==

def _route(source, target):
    return ((source, target), 1)


def _load(index, total):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=0, history_bytes=(1000, 1000)):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location="local_hbm", history_tokens=100,
        resident_prefix_layers=4,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=(0, 0))


def _request(decode=10):
    return RequestView(
        request_id="r", session_id="s", input_tokens=50,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=(25, 25))


def _model(loads, **overrides):
    kwargs = dict(
        rates=JointHardwareRates.from_gbps(
            noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
            d2d_latency_ns=10, pool_latency_ns=100),
        loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route)
    kwargs.update(overrides)
    return JointCostModel(**kwargs)


def _anchor_loads():
    return {0: _load(0, 5000), 1: _load(1, 1200)}


def _remote(model, session=None, request=None):
    session = session if session is not None else _session()
    request = request if request is not None else _request()
    return model.estimate_action(
        session=session, request=request, instance_index=1,
        action=ACTION_REMOTE, remote_enabled=True)


# ---- 手算锚（C4 金值重推导，2026-09-22；rates：B_link = 10 B/ns、
# B_HBM = 100 B/ns、单跳时延 10 ns、流注册表空、u_port = 0（离线
# 口径——port_registry 未注入））。对旧金值（3286/3356/2070/422/
# 1648）的公式变更来源逐项列明：
#   1. C1 三腿 min（F1 冻结）：读流 wall = startup + 跳数×时延
#      + max(noc_stream, home_read, exec_write)，逐 shard 取 max——
#      TP 并行不再按"聚合字节 ÷ 单链路率"串行放大；
#   2. A1' 并集瓶颈除数（§4.3 补遗）：2 个 TP shard 共链 (0→1) →
#      divisor_multi 并集除数 = 2、noc 有效率 = 10/2 = 5 B/ns；字节
#      均衡 + 空链路下逐 shard 3000/(10/2)=600 与旧聚合 6000/10=600
#      同 wall（桥接锚——CopyCompositePricingTest 的闭式仍逐位可用）；
#   3. D1 读放大因果（C1 步骤 2b）：远读基数 = 仅远端基础历史
#      (1000,1000)/rank，input/decode 增量本地读、不入远读基数
#      （旧口径基数 = 终态上下文 2060 B）；read_passes = prefill 遍数
#      + decode 步数 = 1 + 10 = 11（A2' 缺省单遍计入基数，§4.3 补遗
#      ——本卡核验执行器无 prefill 分块重扫、缺省单遍即真值）；
#   4. C3 上下文依赖 + A4' 补价（C2 rider）：本夹具 compute =
#      50×1 + 10×2 = 70 不变（C3 改 decode 上下文依赖计价，本夹具
#      参数下同值）；kind="read" 增执行端 HBM 写腿 11000/100 = 110
#      < noc 腿 2200（u_exec = 0 时写腿与源端读腿同值、不触 max）。
# decode 增长逐 rank = 25*10//50 = 5（入空间足迹与 merge 增量）；
# remote_read_ns：逐 shard 字节 = 1000×11 = 11000，noc 腿 = 11000/5
# = 2200、端点腿各 110 → max = 2200，wall = 10 + 2200 = 2210；
# compute_ns = 70；merge：increments = (30,30)、并集除数 2 → noc
# = 30/5 = 6、wall = 16（与旧值 16 逐位一致——桥接锚），反向腿
# (1000,1000) → 10 + 200 = 210 → merge_ns = min(16, 210) = 16。
ANCHOR_TARGET_WAIT = 1200
ANCHOR_COMPUTE_NS = 70
ANCHOR_REMOTE_READ_NS = 2210
ANCHOR_MERGE_NS = 16
# credit(auto) K = max(1, ceil(11/8)) = 2：credit1 逐 shard 2000 B →
# noc = 2000/5 = 400 → first = 10 + 400 = 410；其余流送逐 shard
# 9000 B → noc = 1800 → stream = 1800（纯流送段，无 startup/时延）。
ANCHOR_K = 2
ANCHOR_FIRST_CREDIT_NS = 410
ANCHOR_STREAM_NS = 1800
# 单块退化值（K = read_passes = 11）：first == 全流时延 2210、
# stream == 0 ⇒ effective = 2210 + 70 = 2280（旧加法形态数值）
# ⇒ cost = 1200 + 0 + 2280 + 16 = 3496。
ANCHOR_SINGLE_CREDIT_COST = 3496


# ======================================================= 流水式手算锚 ==

class PipelinePricingAnchorTest(unittest.TestCase):
    """唯一形态手算锚：auto K（流送段主导）与 compute-bound 两分支。"""

    def test_auto_k_stream_bound_cost_by_hand(self):
        candidate = _remote(_model(_anchor_loads()))
        self.assertTrue(candidate.applicable)
        # 全链手算（C4 重推导）：1200 + 0 + (410 + max(1800, 70)) + 16
        # = 3426。流送段主导（1800 > 70）⇒ 计算段完全被其余读流隐藏。
        # effective 只剩首 credit + 其余流送 = 410 + 1800 = 2210（与
        # 全流 wall 2210 同值——时延只计一次、其余流送段无时延；旧
        # 加法形态为 compute 70 + 全流 2210 = 2280）。
        self.assertEqual(candidate.cost_ns, 3426)
        breakdown = candidate.breakdown
        # 拆分披露 = 手算值；remote_read_ns 仍为全流总时延口径。
        self.assertEqual(
            breakdown.remote_read_first_credit_ns, ANCHOR_FIRST_CREDIT_NS)
        self.assertEqual(breakdown.remote_read_stream_ns, ANCHOR_STREAM_NS)
        self.assertEqual(breakdown.remote_read_ns, ANCHOR_REMOTE_READ_NS)
        self.assertEqual(breakdown.compute_ns, ANCHOR_COMPUTE_NS)
        self.assertEqual(breakdown.merge_ns, ANCHOR_MERGE_NS)
        self.assertEqual(breakdown.target_wait_ns, ANCHOR_TARGET_WAIT)
        self.assertIn("remote_read_passes=11", breakdown.notes)
        self.assertIn(f"remote_credit_k={ANCHOR_K}", breakdown.notes)
        # 合成式逐位：effective = first + max(stream, compute)。
        self.assertEqual(
            candidate.cost_ns - breakdown.target_wait_ns
            - max(breakdown.history_prep_ns, breakdown.eviction_wait_ns)
            - breakdown.merge_ns,
            ANCHOR_FIRST_CREDIT_NS
            + max(ANCHOR_STREAM_NS, ANCHOR_COMPUTE_NS))
        # 流水重叠恒 ≤ 旧加法形态数值（3356）。
        self.assertLess(candidate.cost_ns, ANCHOR_SINGLE_CREDIT_COST)

    def test_compute_bound_takes_max_compute(self):
        # 流送段短于计算段时 max 取 compute：decode 速率放大（2 → 50
        # ns/token）+ 小历史（200 B）。手算（C4 重推导）：远读基数
        # = (100,100)/rank × read_passes 11 → 逐 shard 1100 B，noc 腿
        # = 1100/5 = 220（并集除数 2）、端点腿各 11 → remote_read =
        # 10 + 220 = 230；compute = 50 + 500 = 550；K = 2 → credit1
        # 逐 shard 200 B → first = 10 + 40 = 50、其余流送 900 B →
        # stream = 900/5 = 180；effective = 50 + max(180, 550) = 600
        # （旧加法形态为 550 + 230 = 780，被 max 吸收的流送段即重叠
        # 收益）；merge：forward = 10 + 30/5 = 16、reverse = 10 +
        # 100/5 = 30 → merge_ns = 16 → cost = 0 + 0 + 600 + 16。
        model = _model(
            {0: _load(0, 0), 1: _load(1, 0)}, decode_ns_per_token=50.0)
        candidate = _remote(model, session=_session(history_bytes=(100, 100)))
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.compute_ns, 550)
        self.assertEqual(breakdown.remote_read_first_credit_ns, 50)
        self.assertEqual(breakdown.remote_read_stream_ns, 180)
        self.assertEqual(
            candidate.cost_ns - breakdown.target_wait_ns
            - max(breakdown.history_prep_ns, breakdown.eviction_wait_ns)
            - breakdown.merge_ns,
            50 + max(180, 550))
        self.assertEqual(candidate.cost_ns, 616)

    def test_default_k_equals_explicit_auto(self):
        # 缺省 remote_credit_iters 与显式 "auto" 同夹具逐位相等。
        default = _remote(_model(_anchor_loads()))
        explicit = _remote(_model(
            _anchor_loads(), remote_credit_iters="auto"))
        self.assertEqual(default.cost_ns, explicit.cost_ns)


# ================================================== 单块退化等价锚 ==

class SingleCreditDegenerateTest(unittest.TestCase):
    """K >= read_passes（单 credit）⇒ 合成退化为旧加法形态数值。"""

    def test_explicit_k_equals_steps_degenerates_to_additive(self):
        candidate = _remote(_model(
            _anchor_loads(), remote_credit_iters="11"))  # K = read_passes
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.cost_ns, ANCHOR_SINGLE_CREDIT_COST)
        breakdown = candidate.breakdown
        # 不变量落地：first_credit == 全流时延、stream == 0。
        self.assertEqual(
            breakdown.remote_read_first_credit_ns, ANCHOR_REMOTE_READ_NS)
        self.assertEqual(breakdown.remote_read_stream_ns, 0)
        self.assertIn("remote_credit_k=11", breakdown.notes)

    def test_explicit_k_clamped_to_steps(self):
        # 显式 K = 100 > 步数 11 → 钳到 11（单块退化锚的另一面）；
        # remote_credit_block_size 对显式值与步数独立钳位（100, 10）
        # → 10 仍成立（配置函数单查锚，步数参数与本夹具无关）。
        self.assertEqual(remote_credit_block_size("100", 10), 10)
        candidate = _remote(_model(
            _anchor_loads(), remote_credit_iters="100"))
        self.assertIn("remote_credit_k=11", candidate.breakdown.notes)
        self.assertEqual(candidate.cost_ns, ANCHOR_SINGLE_CREDIT_COST)


# ==================================================== 自适应 K 边界 ==

class AdaptiveBlockSizeTest(unittest.TestCase):
    """auto K = max(1, ceil(S/8)) 边界 + 计价 notes 披露。"""

    def test_auto_k_formula_boundaries(self):
        for steps, expected_k in ((1, 1), (8, 1), (9, 2), (100, 13)):
            self.assertEqual(
                remote_credit_block_size("auto", steps), expected_k)

    def test_pricing_notes_disclose_adaptive_k(self):
        # D1 后 read_passes = prefill 缺省单遍 1 + decode 步数 →
        # auto K = max(1, ceil(read_passes/8))：decode=1 → passes 2 →
        # K=1；decode=8 → 9 → 2；decode=9 → 10 → 2；decode=100 →
        # 101 → 13。
        for decode, expected_k in ((1, 1), (8, 2), (9, 2), (100, 13)):
            model = _model(_anchor_loads())
            candidate = _remote(model, request=_request(decode=decode))
            self.assertTrue(candidate.applicable, decode)
            self.assertIn(
                f"remote_credit_k={expected_k}",
                candidate.breakdown.notes, decode)


# ================================================ 非 remote 动作隔离 ==

class NonRemoteActionsUnaffectedTest(unittest.TestCase):
    """credit 拆分仅存在于 remote-read 候选（其余动作审计字段恒 0）。"""

    def test_stay_recompute_copy_have_zero_credit_split(self):
        model = _model(_anchor_loads())
        resident_session = _session(home=1, resident=1)
        away_session = _session(home=0, resident=0)
        cases = (
            ("stay", 1, resident_session),       # 驻留实例本地续用
            ("recompute", 1, resident_session),  # @驻留仅缺失后缀（此处 0）
            ("copy", 1, away_session),           # 历史在实例 0、目标 1
        )
        for action, instance, session in cases:
            candidate = model.estimate_action(
                session=session, request=_request(),
                instance_index=instance, action=action,
                remote_enabled=True)
            self.assertTrue(candidate.applicable, action)
            self.assertEqual(
                candidate.breakdown.remote_read_first_credit_ns, 0, action)
            self.assertEqual(
                candidate.breakdown.remote_read_stream_ns, 0, action)
            self.assertEqual(candidate.breakdown.remote_read_ns, 0, action)


if __name__ == "__main__":
    unittest.main()
