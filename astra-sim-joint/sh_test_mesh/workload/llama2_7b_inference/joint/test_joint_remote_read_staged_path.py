#!/usr/bin/env python3
"""test_joint_remote_read_staged_path.py -- remote-read 分阶段关键路径
（规格书 §5，2026-09-25）的零后端手算锚点单测。

语义合同（PARTIAL 基）：

* home HBM 存历史前缀 [0,p)、remote KV pool 存历史后缀 [p,L)；选择
  remote-read 且 exec != home 时，prefill 前缀 NoC 读流与后缀池恢复
  自同一准入 frontier 并行分叉，prefill 计算按层段等待对应数据——
  [0,p) 段等前缀首 credit 到达、[p,L) 段等恢复到达，无"全量后缀恢
  复完成才开始 prefill"的全局 barrier；
* decode 只对 home 前缀 [0,p) 新发 credit 读，后缀直接复用 exec HBM
  中已恢复的 KV（decode 腿字节基恒 = 前缀，与 missing 无关）；
* merge 仍为 merge v2（少并多、字节平局取 forward）；prefill 前缀
  读流是瞬时 staging，不入 session 容量账本（JCM 侧对应 F14：读流
  字节不进 _space_footprint，足迹 = 池恢复后缀 + input 增量）。

计价合成（D02 逐层就绪递推，2026-09-26，v2.0 式(12)/(13)；"首 credit
+ max(余量流, 计算)"与 D09 后的"max(prefill 腿, suffix_restore)"阶段
级 max 合成都已废除）：

    ready_data[g] = 末遍覆盖第 g 层 KV 的 tile 到达（g < p，通道串行
                    T 递推闭式）；g ≥ p = 池后缀恢复完成（一次性传输
                    只门控真实消费它的层）
    prefill 首前向 = estimate_first_forward（式 13 逐层递推；prefill
                    计算按层均分 (13,13,12,12)——数据晚到层之后的
                    计算由递推保留）
    repeat 段     = decode 双缓冲闭式（消费池 = decode 计算；自首前
                    向完成后发射）
    cost = 首前向 + repeat 段 + merge（D03 统一时间轴：Q =
            target_wait 只门控计算（F_0 = Q）、A = eviction_wait 为空
            间可用时刻（式 12 A_dest / 式 13 A^work）；可提前的准备自
            决策时刻起估并与排队重叠——本文件锚 target = eviction =
            0，与旧串行口径逐位同值）

阶段量经 notes 披露：remote_read_prefill_ns / remote_read_decode_ns
（腿闭式相对量，审计）/ suffix_restore_ns / prefill_pipeline_overlap_ns
（两供给并行项成对吸收量，审计）/ prefill_first_forward_ns（递推结
果，关键路径）；breakdown 冻结字段保持日志兼容口径（remote_read_ns =
合并流全流总时延、first/stream = 首 tile / 其余流送的审计拆分）。
tile 口径（kv_token_bytes = 10 B/token、auto = 256 token = 2560
B/shard、逐 rank 求值）：prefill 审计腿（1 遍、rank 流 1000 < tile）
= 纯尾块 210 + int(50×1000/2000) = 235；decode 腿（10 遍、rank 流
10000 → m=3、尾 2320）= max(1566+474, 1568)+2 = 2042；PARTIAL 基递推
（p=2）：前缀两层到达 210、后缀两层等池恢复（D04 逐 shard）300 →
首前向 = 210+13+13+max(236,300)+12+12 = 324；LOCAL 基（p=L=4）全层
到达 210 → 260。

手算锚硬件（与 test_joint_credit_pricing 同族）：B_link = 10 B/ns、
B_pool = 5 B/ns、B_HBM = 100 B/ns、单跳时延 10 ns、池端口事务时延
100 ns、流注册表空、u_port = 0（离线口径）；TP2 共链 (0,1) → 并集除
数 2、noc 有效率 5 B/ns。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_remote_read_staged_path.py（或 pytest）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
    _transfer_ns,
)


# ============================================================ 夹具构造 ==

def _rates(pool_latency_ns=100):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=pool_latency_ns)


def _load(index, total=0, remaining=(10**9, 10**9)):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=0, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25),
             prefill_scan_passes=None):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes,
        prefill_scan_passes=prefill_scan_passes)


def _route(source, target):
    return ((source, target), 1)


def _model(loads, *, rates=None):
    return JointCostModel(
        rates=rates if rates is not None else _rates(),
        loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route)


def _partial_session(missing=(1000, 1000)):
    """PARTIAL 基：L=4、p=2——home 存前缀 [0,2)（1000 B/rank）、池存
    后缀 [2,4)（missing B/rank）。"""
    return _session(
        home=0, resident=0, history_bytes=(1000, 1000), missing=missing,
        location="partial_hbm_remote", prefix=2)


def _remote(model, session, request=None):
    return model.estimate_P_with_continuation(
        session=session, request=request if request is not None else _request(),
        instance_index=1, action=ACTION_REMOTE, remote_enabled=True)


def _note_value(breakdown, key):
    """notes 四键披露位的整数值提取（缺键即失败——披露契约钉死）。"""
    prefix = key + "="
    for note in breakdown.notes:
        if note.startswith(prefix):
            return int(note[len(prefix):])
    raise AssertionError(f"missing disclosure note {key!r}")


# ================================================== PARTIAL 分阶段锚 ==

class PartialStagedCriticalPathTest(unittest.TestCase):
    """PARTIAL 基分阶段关键路径手算锚（prefill 读流 ∥ 后缀池恢复）。"""

    def test_partial_staged_cost_by_hand(self):
        # 手算（D02 逐层就绪递推全链推导）：target = 0、eviction = 0
        # （HBM 充裕）；suffix_restore（D04 逐 shard）= 100 +
        # max(1000/5, 1000/100) = 300；读流基数 = 前缀 (1000,1000)/
        # rank，read_passes = 1 + 10 = 11；prefill 审计腿（rank 流
        # 1000 < tile 2560 → 纯尾块）= 210 + int(50×1000/2000) = 235；
        # decode 腿（rank 流 10000 → m=3、尾 2320）= max(1566+474,
        # 1568)+2 = 2042；递推首前向：前缀两层 [0,2) 就绪 = 单尾块到
        # 达 210（末遍覆盖偏移 500/1000 同一 j=1 事务），后缀两层
        # [2,4) 就绪 = 300 → F = 210+13 + 13 + max(236,300)+12 + 12 =
        # 324；merge：exec 保留 (1030,1030) > home (1000,1000) → 反向
        # 10 + 1000/5 = 210。cost = 324 + 2042 + 210 = 2576（后缀恢复
        # 只门控 [2,4) 层——无全局 barrier，也无阶段级 max 吸收）。
        model = _model({0: _load(0), 1: _load(1)})
        candidate = _remote(model, _partial_session())
        self.assertTrue(candidate.applicable)
        breakdown = candidate.breakdown
        self.assertEqual(_note_value(breakdown, "remote_read_prefill_ns"),
                         235)
        self.assertEqual(_note_value(breakdown, "remote_read_decode_ns"),
                         5221)
        self.assertEqual(_note_value(breakdown, "suffix_restore_ns"), 300)
        self.assertEqual(
            _note_value(breakdown, "prefill_pipeline_overlap_ns"), 235)
        self.assertEqual(
            _note_value(breakdown, "prefill_first_forward_ns"), 324)
        self.assertEqual(breakdown.history_prep_ns, 300)
        self.assertEqual(breakdown.merge_ns, 210)
        # 合成式逐位：cost = target + eviction + 首前向递推 + repeat
        # 段 + merge；首前向严格小于两供给串行和（读流 235 与恢复 300
        # 并行分叉、只在其消费层等待）。
        prefill_stage = candidate.cost_ns - breakdown.merge_ns - 5221
        self.assertEqual(prefill_stage, 324)
        self.assertLess(prefill_stage, 235 + 300)
        self.assertEqual(candidate.cost_ns, 324 + 5221 + 210)
        self.assertIn("pool_suffix_restore_hybrid", breakdown.notes)
        self.assertIn("remote_read_prefix_layers=2", breakdown.notes)

    def test_suffix_hidden_under_prefill_compute(self):
        # 重叠方向正向钉：prefill 计算远大于后缀恢复时，恢复被计算吸
        # 收（旧口径恒串行前置、无此吸收）。input 2000（decode 0）：
        # passes = 1 → prefill 审计腿（尾块消费池 = prefill 计算
        # 2000）= 210 + int(2000×1000/2000) = 1210、decode 腿 0；
        # 递推：前缀两层到达 210，层份额 500×4 → F1 = 710、F2 = 1210
        # （前缀读流尾段在计算内消化）、F3 = max(1210,300)+500 = 1710、
        # F4 = 2210 —— 后缀恢复 300 完全隐藏（< F2 = 1210）。正式事件
        # 图还会在共享 HBM 上重算 prefill compute 的物理 byte stream，
        # 因此 service_done=2255；merge 反向 210 ⇒ cost = 2465。
        # 300 + 210 + 2000 + 210 = 2720。
        model = _model({0: _load(0), 1: _load(1)})
        request = _request(
            input_tokens=2000, decode=0, input_bytes=(1000, 1000))
        candidate = _remote(model, _partial_session(), request)
        self.assertTrue(candidate.applicable)
        breakdown = candidate.breakdown
        self.assertEqual(_note_value(breakdown, "remote_read_prefill_ns"),
                         1210)
        self.assertEqual(_note_value(breakdown, "remote_read_decode_ns"), 0)
        self.assertEqual(_note_value(breakdown, "suffix_restore_ns"), 300)
        self.assertEqual(
            _note_value(breakdown, "prefill_pipeline_overlap_ns"), 300)
        self.assertEqual(
            _note_value(breakdown, "prefill_first_forward_ns"), 2210)
        self.assertEqual(candidate.cost_ns, 2465)
        # 旧"后缀恢复全量串在 prefill 计算之前"形态（全局 barrier）。
        barrier_form = 300 + 210 + 2000 + breakdown.merge_ns
        self.assertLess(candidate.cost_ns, barrier_form)

    def test_pool_rate_does_not_touch_decode_stage(self):
        # decode 阶段与池速率解耦钉：decode 腿只读 home 前缀 credit，
        # 池端口时延放大只经 suffix_restore（prefill 阶段 max 内）与
        # merge 空间准备进入——decode 腿披露值逐位不变。
        model = _model({0: _load(0), 1: _load(1)},
                       rates=_rates(pool_latency_ns=100))
        slow = _model({0: _load(0), 1: _load(1)},
                      rates=_rates(pool_latency_ns=10**6))
        base = _remote(model, _partial_session())
        lifted = _remote(slow, _partial_session())
        self.assertEqual(
            _note_value(base.breakdown, "remote_read_decode_ns"),
            _note_value(lifted.breakdown, "remote_read_decode_ns"))
        self.assertGreater(
            _note_value(lifted.breakdown, "suffix_restore_ns"),
            _note_value(base.breakdown, "suffix_restore_ns"))


# ================================================= decode 复用后缀钉 ==

class DecodeStageReusesExecSuffixTest(unittest.TestCase):
    """decode 只对 home 前缀发 credit：decode 腿与后缀体量无关。"""

    def test_decode_leg_reads_prefix_only(self):
        # 同前缀、后缀 500 vs 5000：decode 腿（10 遍 × 1000 B/rank，
        # 修复轮三 H1 逐 pass 重切：10×⌈100/256⌉ = 10 整块/rank、
        # x=522、c=1 → 522+1+9×522 = 5221）逐位相等；suffix_restore
        # 随后缀体量变化（D04 逐 shard：200 vs 1100）。
        model = _model({0: _load(0), 1: _load(1)})
        small = _remote(model, _partial_session(missing=(500, 500)))
        large = _remote(model, _partial_session(missing=(5000, 5000)))
        self.assertEqual(
            _note_value(small.breakdown, "remote_read_decode_ns"), 5221)
        self.assertEqual(
            _note_value(large.breakdown, "remote_read_decode_ns"), 5221)
        self.assertEqual(
            _note_value(small.breakdown, "suffix_restore_ns"), 200)
        self.assertEqual(
            _note_value(large.breakdown, "suffix_restore_ns"), 1100)
        # 读流基数恒 = 前缀（D1 账本真值），不随后缀膨胀：全流
        # remote_read_ns（合并流口径）两夹具同值。
        self.assertEqual(
            small.breakdown.remote_read_ns,
            large.breakdown.remote_read_ns)


# ===================================================== LOCAL 基退化 ==

class LocalBaseNoSuffixTermTest(unittest.TestCase):
    """LOCAL 基（p=L）：无后缀恢复项——池速率整链不可见、suffix 披露 0。"""

    def test_local_base_pool_rate_invariant(self):
        session = _session()  # local_hbm、prefix=4=L、missing=0
        model = _model({0: _load(0), 1: _load(1)})
        slow = _model({0: _load(0), 1: _load(1)},
                      rates=_rates(pool_latency_ns=10**6))
        base = _remote(model, session)
        lifted = _remote(slow, session)
        self.assertEqual(base.cost_ns, lifted.cost_ns)
        self.assertEqual(_note_value(base.breakdown, "suffix_restore_ns"), 0)
        self.assertEqual(base.breakdown.history_prep_ns, 0)
        # 四键披露在 LOCAL 基同样在场（suffix 恒 0 的显式锚）。
        for key in ("remote_read_prefill_ns", "remote_read_decode_ns",
                    "suffix_restore_ns", "prefill_pipeline_overlap_ns"):
            self.assertTrue(
                any(note.startswith(key + "=")
                    for note in base.breakdown.notes), key)

    def test_local_base_staged_composition(self):
        # LOCAL 基合成（无 suffix 项）：递推首前向 = 单尾块到达 210 +
        # 全部 prefill 计算 50 = 260（旧 max 合成只付字节份额 25、
        # 235——D02 保留到达后计算）；cost = target + eviction + 260 +
        # decode 腿 5221（H1 逐 pass 重切新锚，手算同上）+ merge 16
        # （target/eviction = 0）。
        model = _model({0: _load(0), 1: _load(1)})
        candidate = _remote(model, _session())
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.target_wait_ns, 0)
        self.assertEqual(breakdown.eviction_wait_ns, 0)
        self.assertEqual(breakdown.merge_ns, 16)
        self.assertEqual(
            _note_value(breakdown, "prefill_first_forward_ns"), 260)
        self.assertEqual(
            candidate.cost_ns - breakdown.merge_ns, 260 + 5221)


# ================================================== 旧字段日志兼容 ==

class LegacyFieldCompatTest(unittest.TestCase):
    """breakdown 冻结字段保持合并流日志口径（不进关键路径）。"""

    def test_combined_stream_fields_unchanged(self):
        # remote_read_ns = 合并流（11 遍 × 2000 B）全流时延 = 10 + 2200
        # = 2210，与旧聚合口径 _transfer_ns 逐位同值（桥接锚）；审计
        # 拆分（D09 tile）：首 tile = min(2560, 11000)/rank →
        # 10 + 2560/5 = 522、其余流送 = 8440/5 = 1688、和 = 全流。
        model = _model({0: _load(0), 1: _load(1)})
        candidate = _remote(model, _partial_session())
        breakdown = candidate.breakdown
        self.assertEqual(
            breakdown.remote_read_ns,
            _transfer_ns(
                total_bytes=22000, path_hops=1, divisor=1,
                rates=model.rates, per_hop_latency_ns=None, startup_ns=0))
        self.assertEqual(breakdown.remote_read_ns, 2210)
        self.assertEqual(breakdown.remote_read_first_credit_ns, 522)
        self.assertEqual(breakdown.remote_read_stream_ns, 1688)
        self.assertEqual(
            breakdown.remote_read_first_credit_ns
            + breakdown.remote_read_stream_ns,
            breakdown.remote_read_ns)
        # history_prep_ns 仍为后缀池恢复全量审计值（与 suffix 披露同源
        # ；D04 逐 shard：每 rank 1000 B → 100 + 1000/5 = 300）。
        self.assertEqual(
            breakdown.history_prep_ns,
            _note_value(breakdown, "suffix_restore_ns"))
        self.assertEqual(breakdown.history_prep_ns, 100 + 1000 // 5)

    def test_single_tile_stream_degeneration_kept(self):
        # tile >= 全流（20000 B 显式 tile @ 10 B/token → 2000 token）：
        # 合并流 first == remote_read_ns、stream == 0；decode 腿 = 修复
        # 轮三 H1 逐 pass 重切：tokens=100 < tile=2000 → 每 pass 恰 1
        # 块（尾块形态）×10 遍 = 10 整块/rank、尾块并入整块服务
        # （tail=0 保守）：x = 10+20000/5 = 4010、c = int(20×20000/
        # 400000) = 1 → T_j = T_{j-1}+x ⇒ 4010+1+9×4010 = 40101（旧纯
        # 尾块连续链 = 2020——重切消除低估，只增不减）；递推与 auto
        # tile 同形（前缀单尾块事务到达 210、后缀等恢复 300）→ 首前
        # 向 324 ⇒ cost = 324 + 40101 + 210 = 40635。
        model = _model({0: _load(0), 1: _load(1)})
        model_tile = JointCostModel(
            rates=model.rates, loads=model.loads,
            flow_registry=model.flow_registry,
            service_factors=model.service_factors,
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2, route_fn=_route,
            remote_read_tile_bytes="20000")
        candidate = _remote(model_tile, _partial_session())
        breakdown = candidate.breakdown
        self.assertEqual(
            breakdown.remote_read_first_credit_ns,
            breakdown.remote_read_ns)
        self.assertEqual(breakdown.remote_read_stream_ns, 0)
        self.assertEqual(
            _note_value(breakdown, "prefill_first_forward_ns"), 324)
        self.assertEqual(candidate.cost_ns, 324 + 40101 + 210)


# ================================================= eviction 串行披露 ==

class EvictionWaitSerialTest(unittest.TestCase):
    """D03（v2.0 §22-D03）：A = eviction_wait 成为相对本次决策的空
    间可用时刻——式(12) A_dest 门控全部目标写腿（tile 到达/池恢复自
    A 起发射）、式(13) A^work 门控计算投影；旧 max(history_prep,
    eviction) 形态废除——空间未就绪前传输不发射、A 在 cost 中逐位可
    见（不被供给/计算吸收）。本锚 Q = 0：A 门控与旧 frontier 平移逐
    位同值（退化锚）。"""

    def test_eviction_serial_before_stages(self):
        # exec HBM 清空：空间缺口 = 池恢复后缀 + input + 增长 + D09 双
        # 缓冲暂存（2 × 256 token × 10 B = 5120）= 6150 B/rank → A =
        # 写回 = 100 + 6150/5 = 1330（可逐出、无深缺口）；merge 反向
        # 腿加 exec 侧 home 保留量空间准备 300 → merge = 210 + 300 =
        # 510。A 门控递推与供给（tile 到达 1330+210、恢复 1330+300、
        # 投影 ≥ 1330）：首前向 = 1330 + 324 = 1654（324 与零驱逐锚同
        # 值——A 平移不变）⇒ cost = 1654 + 5221 + 510 = 7385（decode
        # 腿 = H1 逐 pass 重切新锚 5221，旧 2042——只增不减）——
        # eviction 与 suffix_restore（300）不取 max（写腿等空间可用），
        # A 完整可见。
        tight = {0: _load(0), 1: _load(1, remaining=(0, 0))}
        model = _model(tight)
        candidate = _remote(model, _partial_session())
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.eviction_wait_ns, 1330)
        self.assertIn(
            "execution_growth_eviction_writeback_est", breakdown.notes)
        self.assertEqual(_note_value(breakdown, "suffix_restore_ns"), 300)
        self.assertEqual(breakdown.merge_ns, 510)
        self.assertEqual(
            candidate.cost_ns,
            breakdown.eviction_wait_ns + 324 + 5221 + 510)
        # 串行方向对照：旧形态 prep = max(300, eviction) 会把 eviction
        # 完全吸收——新口径下 eviction 在 cost 中逐位可见。
        self.assertEqual(
            candidate.cost_ns - breakdown.merge_ns - 5221 - 324, 1330)


if __name__ == "__main__":
    unittest.main()
