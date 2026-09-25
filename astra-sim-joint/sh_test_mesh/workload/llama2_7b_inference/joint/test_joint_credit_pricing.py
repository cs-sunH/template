#!/usr/bin/env python3
"""test_joint_credit_pricing.py -- remote-read 决策计价分阶段关键路径的
手算锚点单测（规格书 §5，2026-09-25；前身 §4.3.1 credit 交错流锚）。

计价合同（分阶段关键路径，旧"后缀恢复全量串在 prefill 计算之前"的
max(history_prep, eviction) + first_credit + max(stream, compute) 合成
废除、无开关可选项）：

* prefill_stage = prefix_first_credit
                  + max(prefix_remaining_stream, suffix_restore,
                        prefill_compute)
  decode_stage  = decode_first_credit
                  + max(decode_remaining_stream, decode_compute)
  cost = target_wait + eviction_wait + prefill_stage + decode_stage
         + merge
  ——prefill 前缀读流与后缀池恢复同准入 frontier 并行分叉、prefill 计
  算按层段随数据到达推进（无全局 barrier）；decode 只对 home 前缀新
  发 credit 读。阶段量经 notes 四键披露（remote_read_prefill_ns /
  remote_read_decode_ns / suffix_restore_ns /
  prefill_pipeline_overlap_ns）；K 与执行侧切片同源
  （remote_credit_block_size 单一裁决点，两腿共用同一 K）；
* 旧 breakdown 字段保持日志兼容口径：remote_read_ns 恒为合并流
  （read_passes 全程、时延单计）全流总时延，remote_read_first_credit_ns
  / remote_read_stream_ns 仍为该合并流的全局 credit 拆分——不再进关
  键路径；不变量：K >= read_passes（单 credit）时 first_credit_ns ==
  remote_read_ns、remaining_stream_ns == 0。

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
# 分阶段关键路径（LOCAL 基 suffix_restore = 0）：prefill 腿 = 1 遍
# （1000 B/rank）→ leg = 10 + 1000/5 = 210、first = 210（K=2 ≥ 1 遍
# 全覆盖）、remaining = 0；decode 腿 = 10 遍（10000 B/rank）→ leg =
# 10 + 10000/5 = 2010、first = 10 + 2000/5 = 410、remaining = 8000/5
# = 1600。计算分量 prefill = 50×1.0 = 50、decode = 10×2.0 = 20。
# prefill_stage = 210 + max(0, 0, 50) = 260；decode_stage = 410 +
# max(1600, 20) = 2010 ⇒ cost = 1200 + 0 + 260 + 2010 + 16 = 3486。
ANCHOR_PREFILL_LEG_NS = 210
ANCHOR_DECODE_LEG_NS = 2010
# 逐腿 credit 拆分（同一 K = 2）：prefill 腿 1 遍全进首块 → first 210、
# remaining 0；decode 腿首块 2 遍 → first 410、其余 8 遍纯流送 1600。
ANCHOR_PREFILL_FIRST_NS = 210
ANCHOR_PREFILL_REMAINING_NS = 0
ANCHOR_DECODE_FIRST_NS = 410
ANCHOR_DECODE_REMAINING_NS = 1600
ANCHOR_PREFILL_COMPUTE_NS = 50
ANCHOR_DECODE_COMPUTE_NS = 20
ANCHOR_STAGED_COST = 3486
# 单块退化值（K = read_passes = 11）：全局拆分 first == 全流时延
# 2210、stream == 0；分阶段逐腿同退化——prefill_first = 210、
# decode_first = 10 + 10000/5 = 2010、两腿 remaining = 0 ⇒
# prefill_stage = 210 + 50 = 260、decode_stage = 2010 + 20 = 2030
# ⇒ cost = 1200 + 260 + 2030 + 16 = 3506。
ANCHOR_SINGLE_CREDIT_COST = 3506


# ======================================================= 流水式手算锚 ==

class PipelinePricingAnchorTest(unittest.TestCase):
    """分阶段关键路径手算锚：auto K（流送段主导）与 compute-bound 两
    分支。"""

    def test_auto_k_stream_bound_cost_by_hand(self):
        candidate = _remote(_model(_anchor_loads()))
        self.assertTrue(candidate.applicable)
        # 全链手算（分阶段）：1200 + 0 + (210 + max(0, 0, 50)) + (410 +
        # max(1600, 20)) + 16 = 3486。LOCAL 基无后缀恢复项（suffix_
        # restore = 0）；decode 腿流送段主导（1600 > 20）⇒ decode 计算
        # 段被其余读流隐藏；prefill 腿 K=2 ≥ 1 遍全覆盖 ⇒ remaining = 0、
        # prefill 计算 50 暴露在首 credit 之后（旧合并流口径下 compute
        # 70 全被 stream 1800 吸收——分阶段后 prefill/decode 计算各归
        # 所属阶段）。
        self.assertEqual(candidate.cost_ns, ANCHOR_STAGED_COST)
        breakdown = candidate.breakdown
        # 全局拆分披露 = 手算值（日志兼容口径）；remote_read_ns 仍为合
        # 并流全流总时延。
        self.assertEqual(
            breakdown.remote_read_first_credit_ns, ANCHOR_FIRST_CREDIT_NS)
        self.assertEqual(breakdown.remote_read_stream_ns, ANCHOR_STREAM_NS)
        self.assertEqual(breakdown.remote_read_ns, ANCHOR_REMOTE_READ_NS)
        self.assertEqual(breakdown.compute_ns, ANCHOR_COMPUTE_NS)
        self.assertEqual(breakdown.merge_ns, ANCHOR_MERGE_NS)
        self.assertEqual(breakdown.target_wait_ns, ANCHOR_TARGET_WAIT)
        self.assertIn("remote_read_passes=11", breakdown.notes)
        self.assertIn(f"remote_credit_k={ANCHOR_K}", breakdown.notes)
        # 阶段量 notes 披露（规格书 §5 四键）与手算值逐位一致。
        self.assertIn(
            f"remote_read_prefill_ns={ANCHOR_PREFILL_LEG_NS}",
            breakdown.notes)
        self.assertIn(
            f"remote_read_decode_ns={ANCHOR_DECODE_LEG_NS}",
            breakdown.notes)
        self.assertIn("suffix_restore_ns=0", breakdown.notes)
        self.assertIn("prefill_pipeline_overlap_ns=0", breakdown.notes)
        # 合成式逐位：cost = target + eviction + prefill_stage +
        # decode_stage + merge（LOCAL 基 eviction = 0、suffix_restore =
        # 0）；prefill_stage = first + max(remaining, suffix, compute)、
        # decode_stage = first + max(remaining, compute)。
        prefill_stage = ANCHOR_PREFILL_FIRST_NS + max(
            ANCHOR_PREFILL_REMAINING_NS, 0, ANCHOR_PREFILL_COMPUTE_NS)
        decode_stage = ANCHOR_DECODE_FIRST_NS + max(
            ANCHOR_DECODE_REMAINING_NS, ANCHOR_DECODE_COMPUTE_NS)
        self.assertEqual(
            candidate.cost_ns - breakdown.target_wait_ns
            - breakdown.eviction_wait_ns - breakdown.merge_ns,
            prefill_stage + decode_stage)
        # 分阶段流水恒 ≤ 单块退化数值（3506）。
        self.assertLess(candidate.cost_ns, ANCHOR_SINGLE_CREDIT_COST)

    def test_compute_bound_takes_max_compute(self):
        # prefill/decode 计算分量各自在所属阶段取 max：decode 速率放大
        # （2 → 50 ns/token）+ 小历史（200 B）。手算（C4 重推导）：远读
        # 基数 = (100,100)/rank × read_passes 11 → 逐 shard 1100 B，noc
        # 腿 = 1100/5 = 220（并集除数 2）、端点腿各 11 → remote_read =
        # 10 + 220 = 230；compute = 50 + 500 = 550（prefill 50、decode
        # 500）；K = 2 → prefill 腿 1 遍全覆盖：leg = first = 10 +
        # 100/5 = 30、remaining = 0 → prefill_stage = 30 + 50 = 80；
        # decode 腿 first = 10 + 200/5 = 50、remaining = 800/5 = 160 →
        # decode_stage = 50 + max(160, 500) = 550；merge：forward =
        # 10 + 30/5 = 16、reverse = 10 + 100/5 = 30 → merge_ns = 16 ⇒
        # cost = 0 + 0 + 80 + 550 + 16 = 646。
        model = _model(
            {0: _load(0, 0), 1: _load(1, 0)}, decode_ns_per_token=50.0)
        candidate = _remote(model, session=_session(history_bytes=(100, 100)))
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.compute_ns, 550)
        self.assertEqual(breakdown.remote_read_first_credit_ns, 50)
        self.assertEqual(breakdown.remote_read_stream_ns, 180)
        self.assertIn("remote_read_prefill_ns=30", breakdown.notes)
        self.assertIn("remote_read_decode_ns=210", breakdown.notes)
        # 合成式逐位：prefill_stage = 30 + max(0, 0, 50)、decode_stage
        # = 50 + max(160, 500)。
        self.assertEqual(
            candidate.cost_ns - breakdown.target_wait_ns
            - breakdown.eviction_wait_ns - breakdown.merge_ns,
            30 + max(0, 0, 50) + 50 + max(160, 500))
        self.assertEqual(candidate.cost_ns, 646)

    def test_default_k_equals_explicit_auto(self):
        # 缺省 remote_credit_iters 与显式 "auto" 同夹具逐位相等。
        default = _remote(_model(_anchor_loads()))
        explicit = _remote(_model(
            _anchor_loads(), remote_credit_iters="auto"))
        self.assertEqual(default.cost_ns, explicit.cost_ns)


# ================================================== 单块退化等价锚 ==

class SingleCreditDegenerateTest(unittest.TestCase):
    """K >= read_passes（单 credit）⇒ 合并流全局拆分退化为 first ==
    remote_read_ns、stream == 0；分阶段逐腿同退化（首块 = 整腿、
    remaining = 0）。"""

    def test_explicit_k_equals_steps_degenerates(self):
        candidate = _remote(_model(
            _anchor_loads(), remote_credit_iters="11"))  # K = read_passes
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.cost_ns, ANCHOR_SINGLE_CREDIT_COST)
        breakdown = candidate.breakdown
        # 合并流不变量落地：first_credit == 全流时延、stream == 0。
        self.assertEqual(
            breakdown.remote_read_first_credit_ns, ANCHOR_REMOTE_READ_NS)
        self.assertEqual(breakdown.remote_read_stream_ns, 0)
        self.assertIn("remote_credit_k=11", breakdown.notes)
        # 分阶段逐腿退化：prefill 腿（1 遍）全进首块 = 整腿 210、
        # decode 腿（10 遍）全进首块 = 整腿 2010、两腿 remaining = 0
        # ⇒ prefill_stage = 210 + 50 = 260、decode_stage = 2010 + 20
        # = 2030（decode 计算不再被流送吸收——单块下无其余流送段）。
        self.assertIn(
            f"remote_read_prefill_ns={ANCHOR_PREFILL_LEG_NS}",
            breakdown.notes)
        self.assertIn(
            f"remote_read_decode_ns={ANCHOR_DECODE_LEG_NS}",
            breakdown.notes)
        self.assertEqual(
            candidate.cost_ns - breakdown.target_wait_ns
            - breakdown.eviction_wait_ns - breakdown.merge_ns,
            (ANCHOR_PREFILL_FIRST_NS + ANCHOR_PREFILL_COMPUTE_NS)
            + (ANCHOR_DECODE_LEG_NS + ANCHOR_DECODE_COMPUTE_NS))

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
