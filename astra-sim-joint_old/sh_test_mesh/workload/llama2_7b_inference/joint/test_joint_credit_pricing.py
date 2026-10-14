#!/usr/bin/env python3
"""test_joint_credit_pricing.py -- remote-read 决策计价 D09 有限字节
双缓冲闭式的手算锚点单测（v2.0 §12/式 14/15/附录 A.1，2026-09-26；
前身 §4.3.1 credit 交错流锚与规格书 §5 分阶段合成）。

计价合同（双缓冲闭式，旧"首 credit + max(余量流, 计算)"合成废除）：

* 读流按配置字节 tile 连续切块（首块发射约束、两块暂存、尾块 A.1 显
  式处理）；prefill 腿消费池 = prefill 计算、decode 腿消费池 = decode
  计算（按字节份额均匀分摊，§12.3 等大等服时间近似）；
* D02 递推 + D03 统一时间轴（2026-09-26）：prefill 首前向 =
  estimate_first_forward(start_ns=Q, ready_space_ns=A, ready_data=决策
  相对绝对量——读流腿决策时刻发射、与排队重叠)；decode_stage =
  decode_leg_闭式（自首前向后发射）；cost = prefill 首前向 + decode 腿
  + merge（target_wait 为观测字段、不前置相加——D03）；
* 旧 breakdown 字段保持日志兼容口径：remote_read_ns 恒为合并流全流
  总时延，remote_read_first_credit_ns / remote_read_stream_ns 为首
  tile / 其余流送的审计拆分——不再进关键路径；不变量：流字节 ≤ 一个
  tile 时 first_credit_ns == remote_read_ns、stream == 0。

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

from joint.joint_config import (  # noqa: E402
    prefill_remote_read_scan_passes,
    remote_read_exec_tile_tokens,
    resolve_remote_read_tile_tokens,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    _double_buffer_recurrence_ns,
    remote_read_double_buffer_ns,
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
    return model.estimate_P_with_continuation(
        session=session, request=request, instance_index=1,
        action=ACTION_REMOTE, remote_enabled=True)


# ---- 手算锚（D09 重推导，2026-09-26；rates：B_link = 10 B/ns、
# B_HBM = 100 B/ns、单跳时延 10 ns、流注册表空、u_port = 0（离线
# 口径——port_registry 未注入））。保留基座：C1 三腿 min（F1）、A1'
# 并集瓶颈除数 = 2（noc 有效率 5 B/ns）、D1 读放大因果（基数 = 仅远
# 端基础历史 (1000,1000)/rank，input/decode 增量本地读）、C3/A4'
# compute 与执行端写腿；merge 手算 16 不变。
# D09 tile 口径（逐 rank 求值取 max，§12.3）：kv_token_bytes 惰性派生
# = history 1000//100 = 10 B/token；tile auto = 256 token = 2560
# B/shard；rank 流 = 基础 1000 B/rank × 遍数。
# remote_read_ns：逐 shard 11000 B → noc 腿 11000/5 = 2200 → wall =
# 10 + 2200 = 2210（全流审计，不变）；compute_ns = 50×1 + 10×2 = 70。
# 审计拆分：首 tile = min(2560, 11000) = 2560 → 10 + 2560/5 = 522；
# 其余流送 = 8440/5 = 1688。
ANCHOR_TARGET_WAIT = 1200
ANCHOR_COMPUTE_NS = 70
ANCHOR_REMOTE_READ_NS = 2210
ANCHOR_MERGE_NS = 16
ANCHOR_TILE_TOKENS = 256
ANCHOR_FIRST_TILE_NS = 522
ANCHOR_STREAM_NS = 1688
# prefill 审计腿（1 遍、rank 流 1000 < tile 2560 → m=0 纯尾块）：
# tail_x = 10 + 1000/5 = 210、tail_c = int(50 × 1000/2000) = 25 →
# F = 235（notes 审计披露，不进关键路径）。
# decode 腿（修复轮三 H1 逐 pass 重切，2026-09-27：full_tiles =
# passes×⌈tokens/tile⌉ = 10×⌈100/256⌉ = 10 整块/rank——每 pass 尾块
# 并入整块服务 tail=0，不再跨 pass 合并）：x = latency+tile_bytes/bw
# = 10+2560/5 = 522、c = int(20×2560/(2×10×2560)) = 1 < x；闭式
# T_j = T_{j-1}+x → F = x+c+9x = 522+1+4698 = 5221（旧连续链 m=3+尾
# 2320 口径 = 2042——逐 pass 重切消除尾块合并低估，只增不减）。
# D02 逐层就绪递推（式 12/13，2026-09-26）：LOCAL 基 = 全 4 层由读流
# 供给；末遍 tile 覆盖偏移 = 1000×(g+1)//4 = 250/500/750/1000，j=1
# （单尾块事务）→ 各层就绪 = A + 210 = 210（A = 0：容量无缺口）。
# D03 统一时间轴（2026-09-26）：读流腿决策时刻发射、与排队重叠——
# 到达 210 ⊆ Q = 1200 排空视界被真实吸收；F_0 = Q ⇒ F_L =
# 1200+13+13+12+12 = 1250（递推仍保留零 Q 时的到达后计算——D02 零载
# 锚 260 不变）；repeat 段自 F_L 后发射 ⇒ cost = 1250 + 5221 + 16 =
# 6487（旧串行前置 3518 双计排队×读流重叠 210——v2.0 §8.3"Q 只门控
# 计算"）。
ANCHOR_PREFILL_LEG_NS = 235
ANCHOR_DECODE_LEG_NS = 5221
ANCHOR_PREFILL_FIRST_FORWARD_NS = 1250
ANCHOR_STAGED_COST = 6487
ANCHOR_PREFILL_COMPUTE_NS = 50
ANCHOR_DECODE_COMPUTE_NS = 20
# 显式 tile_bytes = 500（50 token → 500 B/shard）：decode rank 流
# 10000 → m=20 无尾块：tile_c = int(20×500/20000) = 0（份额 < 1 截断）
# → F = 110 + 0 + 19×110 = 2200；prefill 审计腿 m=2：tile_c =
# int(50×500/2000) = 12 → F = 110 + 12 + 110 = 232（审计）。
# D02 递推：层偏移 250/500/750/1000 → tile 索引 j = 1,1,2,2；到达
# T_1 = A+110 = 110、T_2 = 220（通道串行、决策相对绝对量）。D03：
# F_0 = Q = 1200（到达 ⊆ 排空视界被吸收）⇒ F_L = 1200+13+13+12+12
# = 1250 ⇒ dynamic event cost = 1250 + 2201 + 16 = 3467（旧串行前置
# 排队×读流重叠 194）。
EXPLICIT_TILE_COST = 3467
EXPLICIT_TILE_FIRST_FORWARD_NS = 1250


# ======================================================= 双缓冲闭式锚 ==

class DoubleBufferClosedFormAnchorTest(unittest.TestCase):
    """D09 双缓冲闭式手算锚：尾块腿（prefill 单 tile）与多 tile 腿
    （decode m=7 + 尾块 A.1）。"""

    def test_auto_tile_stream_bound_cost_by_hand(self):
        candidate = _remote(_model(_anchor_loads()))
        self.assertTrue(candidate.applicable)
        # 全链手算（D02 递推 + D03 时间轴）：1250 + 2042 + 16 = 3308
        # （读流到达 210 与排队 1200 重叠——旧串行前置 3518 双计）。
        self.assertEqual(candidate.cost_ns, ANCHOR_STAGED_COST)
        breakdown = candidate.breakdown
        # 全局审计拆分 = 手算值；remote_read_ns 仍为全流总时延。
        self.assertEqual(
            breakdown.remote_read_first_credit_ns, ANCHOR_FIRST_TILE_NS)
        self.assertEqual(breakdown.remote_read_stream_ns, ANCHOR_STREAM_NS)
        self.assertEqual(breakdown.remote_read_ns, ANCHOR_REMOTE_READ_NS)
        self.assertEqual(breakdown.compute_ns, ANCHOR_COMPUTE_NS)
        self.assertEqual(breakdown.merge_ns, ANCHOR_MERGE_NS)
        self.assertEqual(breakdown.target_wait_ns, ANCHOR_TARGET_WAIT)
        self.assertIn("remote_read_passes=11", breakdown.notes)
        self.assertIn(
            f"remote_read_tile_tokens={ANCHOR_TILE_TOKENS}",
            breakdown.notes)
        # 阶段量 notes 披露与手算值逐位一致（审计腿 + 递推首前向）。
        self.assertIn(
            f"remote_read_prefill_ns={ANCHOR_PREFILL_LEG_NS}",
            breakdown.notes)
        self.assertIn(
            f"remote_read_decode_ns={ANCHOR_DECODE_LEG_NS}",
            breakdown.notes)
        self.assertIn(
            "prefill_first_forward_ns="
            f"{ANCHOR_PREFILL_FIRST_FORWARD_NS}",
            breakdown.notes)
        self.assertIn("suffix_restore_ns=0", breakdown.notes)
        self.assertIn("prefill_pipeline_overlap_ns=0", breakdown.notes)
        # 合成式逐位（D03）：cost = prefill 首前向（含 Q 门）+ decode
        # 腿相对量 + merge；target_wait 为观测字段（值 = Q，不进合成
        # ——其与读流到达的重叠已被递推吸收）；递推保留单尾块到达
        # （210）之后的全部 prefill 计算——旧 max 合成只付字节份额
        # tail_c = 25、低估 25 ns（D02 零载锚保留该性质）。
        self.assertEqual(
            candidate.cost_ns - breakdown.merge_ns,
            ANCHOR_PREFILL_FIRST_FORWARD_NS + ANCHOR_DECODE_LEG_NS)
        self.assertEqual(breakdown.target_wait_ns, ANCHOR_TARGET_WAIT)
        self.assertLess(
            candidate.cost_ns,
            ANCHOR_TARGET_WAIT + 260 + ANCHOR_DECODE_LEG_NS
            + ANCHOR_MERGE_NS)
        self.assertGreater(
            ANCHOR_PREFILL_FIRST_FORWARD_NS, ANCHOR_PREFILL_LEG_NS)

    def test_explicit_tile_bytes_multi_tile_hand(self):
        # 显式 tile = 500 B（50 token × 10 B/token）：decode 腿 20 个
        # 完整 tile/rank 无尾块（x=110；20 ns compute floor 以最大余数
        # 分配给 20 个 tile，各 1 ns，故 F = 110+19×110+1 = 2201）；
        # prefill 审计腿 2 个完整 tile/rank
        # （c=12；跨 rank 余数守恒后 F = 233）。
        # 修复轮三 H1 例外锚（不变）：tokens=100 整除 tile=50 token
        # （每 pass 恰 2 整块、无每 pass 尾块）⇒ 逐 pass 重切
        # passes×⌈tokens/tile⌉ = 10×2 = 20 与旧连续链 ⌈10000/500⌉ = 20
        # 逐 pass 重切仍保持字节守恒；compute floor 的余数不再被丢弃。
        # D02 递推：tile 索引 j = 1,1,2,2、到达 110/220（通道串行、
        # 决策相对）；D03：到达 ⊆ Q 排空视界被吸收 → F_L = 1250
        # （第二对层等到 220 后仍各付整段份额的性质由 D02 零载锚保留
        # ——此处递推同样不吸收到达后计算，只是门值 ≤ Q）。
        model = _model(_anchor_loads(), remote_read_tile_bytes="500")
        candidate = _remote(model)
        self.assertIn("remote_read_tile_tokens=50",
                      candidate.breakdown.notes)
        self.assertIn("remote_read_decode_ns=2201",
                      candidate.breakdown.notes)
        self.assertIn("remote_read_prefill_ns=233",
                      candidate.breakdown.notes)
        self.assertIn(
            f"prefill_first_forward_ns={EXPLICIT_TILE_FIRST_FORWARD_NS}",
            candidate.breakdown.notes)
        self.assertEqual(candidate.cost_ns, EXPLICIT_TILE_COST)

    def test_compute_bound_first_tile_transfer_exposed(self):
        # decode 速率放大（2 → 50 ns/token）+ 小历史（(100,100)、
        # kv_token_bytes = 1 → tile = 256 B）：修复轮三 H1 逐 pass 重
        # 切（full_tiles = passes×⌈tokens/tile⌉ = 10×⌈100/256⌉ = 10 整
        # 块/rank、每 pass 尾块并入整块服务 tail=0）：重切后流总量 =
        # 10×256 = 2560 B/rank → tile_c = int(500×256/5120) = 25、
        # tile_x = 10+⌈256/5⌉ = 62 ≥ c；闭式 T_j = T_{j-1}+x →
        # F = x+c+9x = 62+25+558 = 645（旧连续链 m=3+尾 232、x<c 口径
        # = 312——重切后 c 份额随块数摊薄、块数×10，只增不减）；
        # prefill 审计尾块腿 = 30+25 = 55（1 遍连续链 = 逐 pass，不动）。
        # D02 递推：单尾块事务到达 = 0+30，全 4 层就绪 = 30 →
        # F_L = 30 + 50 = 80（整段 prefill 计算在到达后支付，不再按
        # 字节份额缩水）⇒ cost = 80 + 645 + 16 = 741。
        model = _model(
            {0: _load(0, 0), 1: _load(1, 0)}, decode_ns_per_token=50.0)
        candidate = _remote(model, session=_session(history_bytes=(100, 100)))
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.compute_ns, 550)
        self.assertIn("remote_read_prefill_ns=55", breakdown.notes)
        self.assertIn("remote_read_decode_ns=645", breakdown.notes)
        self.assertIn("prefill_first_forward_ns=80", breakdown.notes)
        # D03 合成恒等式（Q = A = 0 退化锚）：cost = 首前向 + decode
        # 腿相对量 + merge；target_wait/eviction_wait 观测位恒 0。
        # W3（v4-projection）：decode 腿相对量来自事件世界——逐 tile
        # 完成取整（闭式的每 tile int(x) 向上取整在事件侧只对总完成
        # 时刻取整：x = 61.2 × 10 块 ⇒ 闭式 10×62 + 25 = 645、事件
        # ⌈612⌉ + 25 = 637，差 = 逐 tile 取整累积，物理关系不变）。
        self.assertEqual(
            candidate.cost_ns - breakdown.merge_ns,
            80 + 637)
        self.assertEqual(breakdown.target_wait_ns, 0)
        self.assertEqual(breakdown.eviction_wait_ns, 0)
        # W3（事件侧完成取整）：741 − 8（逐 tile 取整累积）= 733。
        self.assertEqual(candidate.cost_ns, 733)

    def test_default_tile_config_equals_explicit_auto(self):
        # 缺省 remote_read_tile_bytes 与显式 "auto" 同夹具逐位相等。
        default = _remote(_model(_anchor_loads()))
        explicit = _remote(_model(
            _anchor_loads(), remote_read_tile_bytes="auto"))
        self.assertEqual(default.cost_ns, explicit.cost_ns)


# ================================================== 配置裁决点锚 ==

class TileConfigAdjudicationTest(unittest.TestCase):
    """tile 解析/固定执行 tile/扫描次数闭式的单一裁决点锚。"""

    def test_resolve_aligns_down_to_token_granularity(self):
        self.assertEqual(resolve_remote_read_tile_tokens("auto", 10), 256)
        self.assertEqual(resolve_remote_read_tile_tokens("1000", 10), 100)
        # 向下对齐：1001 B @ 10 B/token → 100 token（非 101）。
        self.assertEqual(resolve_remote_read_tile_tokens("1001", 10), 100)

    def test_resolve_rejects_sub_token_tile(self):
        from joint.joint_config import JointConfigError
        with self.assertRaises(JointConfigError):
            resolve_remote_read_tile_tokens("9", 10)

    def test_exec_tile_stays_configured(self):
        # 执行 tile 恒 = 配置值（修复轮二 2026-09-27：不随基础区间放
        # 大——放宽使每块暂存随历史长度无界增长，正中 v2.0 §17.1 不做
        # 清单"为减少仿真事件把容量/缓冲暗改无限"；base_tokens 仅校
        # 验非负，每 pass tile 数 = ⌈base/tile⌉ 可 >2，由逐 tile lag-2
        # 双缓冲释放链承载 ≤2 块暂存不变量）。
        self.assertEqual(remote_read_exec_tile_tokens(256, 100), 256)
        self.assertEqual(remote_read_exec_tile_tokens(256, 1000), 256)
        self.assertEqual(remote_read_exec_tile_tokens(64, 1000), 64)
        # 非法输入仍 fail-closed（0/负 tile 或负 base）。
        from joint.joint_config import JointConfigError
        with self.assertRaises(JointConfigError):
            remote_read_exec_tile_tokens(0, 1000)
        with self.assertRaises(JointConfigError):
            remote_read_exec_tile_tokens(-1, 1000)
        with self.assertRaises(JointConfigError):
            remote_read_exec_tile_tokens(256, -1)

    def test_scan_passes_closed_form(self):
        # 式(11) + §13.3：Σν = a⌈q_c/q_t⌉ + 1_{b>0}⌈b/q_t⌉。
        self.assertEqual(prefill_remote_read_scan_passes(0, 512, 128), 0)
        self.assertEqual(prefill_remote_read_scan_passes(128, 512, 128), 1)
        self.assertEqual(prefill_remote_read_scan_passes(512, 512, 128), 4)
        # 1000 = 1×512 + 488：Σν = 4 + ⌈488/128⌉ = 4 + 4 = 8。
        self.assertEqual(prefill_remote_read_scan_passes(1000, 512, 128), 8)
        # 1024 = 2×512：Σν = 2×4 = 8（无尾 chunk）。
        self.assertEqual(prefill_remote_read_scan_passes(1024, 512, 128), 8)


# ================================================== 闭式 vs 式(14) 对照 ==

class ClosedFormVsRecurrenceTest(unittest.TestCase):
    """闭式（式 15/A.1）与逐块递推（式 14 直译）的一致性对照（A.3 检查
    的确定性采样版——双缓冲完整块/含尾块两分支）。"""

    def _expected_recurrence(self, tile_x, tile_c, full, tail_frac):
        tile_bytes = 1000
        stream = full * tile_bytes + int(tile_bytes * tail_frac)
        transfers = [float(tile_x)] * full
        consumptions = [float(tile_c)] * full
        if tail_frac:
            transfers.append(tile_x * tail_frac)
            consumptions.append(tile_c * tail_frac)
        t_prev2 = None
        t_prev = transfers[0]
        a_1 = max(0, t_prev)
        f_prev = a_1 + consumptions[0]
        for j in range(1, len(transfers)):
            if j == 1:
                t_j = a_1 + transfers[1]
            else:
                t_j = max(t_prev, t_prev2) + transfers[j]
            f_j = max(t_j, f_prev) + consumptions[j]
            t_prev2, t_prev, f_prev = t_prev, t_j, f_j
        return int(f_prev)

    def test_full_tiles_only_matches_recurrence(self):
        for m in (1, 2, 3, 7, 16):
            for x, c in ((522, 2), (110, 120), (61, 64)):
                self.assertEqual(
                    remote_read_double_buffer_ns(
                        tile_x_ns=x, tile_c_ns=c, full_tiles=m),
                    self._expected_recurrence(x, c, m, 0),
                    (m, x, c))

    def test_tail_block_matches_recurrence(self):
        for m in (0, 1, 2, 5):
            for tail_frac in (0.25, 0.5, 0.9):
                self.assertEqual(
                    remote_read_double_buffer_ns(
                        tile_x_ns=100, tile_c_ns=40, full_tiles=m,
                        tail_x_ns=int(100 * tail_frac),
                        tail_c_ns=int(40 * tail_frac)),
                    self._expected_recurrence(
                        100, 40, m, tail_frac),
                    (m, tail_frac))

    def test_recurrence_reference_helper_agrees(self):
        # 模块内参照递推（式 14 直译）与闭式在整除/含尾块两形态一致。
        for m in (1, 3, 9):
            closed = remote_read_double_buffer_ns(
                tile_x_ns=70, tile_c_ns=30, full_tiles=m)
            rec = _double_buffer_recurrence_ns(
                tile_x_ns=70, tile_c_ns=30, tile_bytes=1000,
                stream_bytes=m * 1000)
            self.assertEqual(closed, rec, m)
        closed = remote_read_double_buffer_ns(
            tile_x_ns=70, tile_c_ns=30, full_tiles=3,
            tail_x_ns=35, tail_c_ns=15)
        rec = _double_buffer_recurrence_ns(
            tile_x_ns=70, tile_c_ns=30, tile_bytes=1000,
            stream_bytes=3 * 1000 + 500)
        self.assertEqual(closed, rec)


# ================================================ 非 remote 动作隔离 ==

class NonRemoteActionsUnaffectedTest(unittest.TestCase):
    """tile 拆分仅存在于 remote-read 候选（其余动作审计字段恒 0）。"""

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
            candidate = model.estimate_P_with_continuation(
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
