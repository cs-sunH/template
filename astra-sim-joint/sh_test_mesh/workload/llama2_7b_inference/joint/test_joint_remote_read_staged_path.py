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

计价合成（旧 max(history_prep, eviction) + remote_read_and_compute
的"后缀恢复全量前置"形态废除）：

    prefill_stage = prefix_first_credit
                    + max(prefix_remaining_stream, suffix_restore,
                          prefill_compute)
    decode_stage  = decode_first_credit
                    + max(decode_remaining_stream, decode_compute)
    cost = target_wait + eviction_wait + prefill_stage + decode_stage
           + merge

阶段量经 notes 四键披露：remote_read_prefill_ns / remote_read_decode_ns
/ suffix_restore_ns / prefill_pipeline_overlap_ns（最后者 = prefill
阶段三并行项之和被 max 吸收的隐藏量）；breakdown 冻结字段保持日志兼
容口径（remote_read_ns = 合并流全流总时延、first/stream = 该合并流的
全局 credit 拆分）。

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
    _pool_transfer_ns,
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
    return model.estimate_action(
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
        # 手算（C4 式全链推导）：target = 0、eviction = 0（HBM 充裕）；
        # suffix_restore = pool(2000) = 100 + 2000/5 = 500；读流基数 =
        # 前缀 (1000,1000)/rank，read_passes = 1 + 10 = 11、K = 2；
        # prefill 腿 = 10 + 1000/5 = 210（1 遍全进首块、remaining 0）；
        # decode 腿 = 10 + 10000/5 = 2010（first 410 + 纯流送 1600）；
        # prefill 计算 50×1.0 = 50、decode 计算 10×2.0 = 20；merge：exec
        # 保留 (1030,1030) > home (1000,1000) → 反向 10 + 1000/5 = 210。
        # prefill_stage = 210 + max(0, 500, 50) = 710（后缀恢复主导、
        # 与前缀读流尾段及 prefill 计算重叠——无全局 barrier：串行和
        # 210 + 0 + 500 + 50 = 760 被吸收 50）；decode_stage = 410 +
        # max(1600, 20) = 2010 ⇒ cost = 710 + 2010 + 210 = 2930。
        model = _model({0: _load(0), 1: _load(1)})
        candidate = _remote(model, _partial_session())
        self.assertTrue(candidate.applicable)
        breakdown = candidate.breakdown
        self.assertEqual(_note_value(breakdown, "remote_read_prefill_ns"),
                         210)
        self.assertEqual(_note_value(breakdown, "remote_read_decode_ns"),
                         2010)
        self.assertEqual(_note_value(breakdown, "suffix_restore_ns"), 500)
        self.assertEqual(
            _note_value(breakdown, "prefill_pipeline_overlap_ns"), 50)
        self.assertEqual(breakdown.history_prep_ns, 500)
        self.assertEqual(breakdown.merge_ns, 210)
        # 合成式逐位：cost = target + eviction + prefill_stage +
        # decode_stage + merge；prefill_stage 严格小于三并行项串行和。
        prefill_stage = candidate.cost_ns - breakdown.merge_ns - 2010
        self.assertEqual(prefill_stage, 710)
        self.assertLess(prefill_stage, 210 + 0 + 500 + 50)
        self.assertEqual(candidate.cost_ns, 710 + 2010 + 210)
        self.assertIn("pool_suffix_restore_hybrid", breakdown.notes)
        self.assertIn("remote_read_prefix_layers=2", breakdown.notes)

    def test_suffix_hidden_under_prefill_compute(self):
        # 重叠方向正向钉：prefill 计算远大于后缀恢复时，恢复被计算吸
        # 收（旧口径恒串行前置、无此吸收）。input 2000（decode 0）：
        # passes = 1、K = 1 → prefill 腿 210 = first、decode 腿 0；
        # prefill 计算 2000 → prefill_stage = 210 + 2000 = 2210（后缀
        # 500 全隐藏，overlap = 500）；merge 反向 210 ⇒ cost = 2420
        # << 旧 barrier 形态 500 + 210 + 2000 + 210 = 2920。
        model = _model({0: _load(0), 1: _load(1)})
        request = _request(
            input_tokens=2000, decode=0, input_bytes=(1000, 1000))
        candidate = _remote(model, _partial_session(), request)
        self.assertTrue(candidate.applicable)
        breakdown = candidate.breakdown
        self.assertEqual(_note_value(breakdown, "remote_read_prefill_ns"),
                         210)
        self.assertEqual(_note_value(breakdown, "remote_read_decode_ns"), 0)
        self.assertEqual(_note_value(breakdown, "suffix_restore_ns"), 500)
        self.assertEqual(
            _note_value(breakdown, "prefill_pipeline_overlap_ns"), 500)
        self.assertEqual(candidate.cost_ns, 2210 + 210)
        # 旧"后缀恢复全量串在 prefill 计算之前"形态（全局 barrier）。
        barrier_form = 500 + 210 + 2000 + breakdown.merge_ns
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
        # 同前缀、后缀 500 vs 5000：decode 腿（10 遍 × 1000 B/rank）=
        # 2010 逐位相等；suffix_restore 随后缀体量变化（300 vs 2100）。
        model = _model({0: _load(0), 1: _load(1)})
        small = _remote(model, _partial_session(missing=(500, 500)))
        large = _remote(model, _partial_session(missing=(5000, 5000)))
        self.assertEqual(
            _note_value(small.breakdown, "remote_read_decode_ns"), 2010)
        self.assertEqual(
            _note_value(large.breakdown, "remote_read_decode_ns"), 2010)
        self.assertEqual(
            _note_value(small.breakdown, "suffix_restore_ns"), 300)
        self.assertEqual(
            _note_value(large.breakdown, "suffix_restore_ns"), 2100)
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
        # LOCAL 基合成（无 suffix 项）：cost = target + eviction +
        # (210 + 50) + (410 + 1600) + merge 16 = 2286（target/eviction
        # = 0；prefill 腿 1 遍、decode 腿 first 410 + 流送 1600）。
        model = _model({0: _load(0), 1: _load(1)})
        candidate = _remote(model, _session())
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.target_wait_ns, 0)
        self.assertEqual(breakdown.eviction_wait_ns, 0)
        self.assertEqual(breakdown.merge_ns, 16)
        self.assertEqual(
            candidate.cost_ns - breakdown.merge_ns,
            (210 + 50) + (410 + 1600))


# ================================================== 旧字段日志兼容 ==

class LegacyFieldCompatTest(unittest.TestCase):
    """breakdown 冻结字段保持合并流日志口径（不进关键路径）。"""

    def test_combined_stream_fields_unchanged(self):
        # remote_read_ns = 合并流（11 遍 × 2000 B）全流时延 = 10 + 2200
        # = 2210，与旧聚合口径 _transfer_ns 逐位同值（桥接锚）；first =
        # 10 + 2000/5 = 410、stream = 9000/5×... = 1800、和 = 全流
        # （流送段主导下时延单计）。
        model = _model({0: _load(0), 1: _load(1)})
        candidate = _remote(model, _partial_session())
        breakdown = candidate.breakdown
        self.assertEqual(
            breakdown.remote_read_ns,
            _transfer_ns(
                total_bytes=22000, path_hops=1, divisor=1,
                rates=model.rates, per_hop_latency_ns=None, startup_ns=0))
        self.assertEqual(breakdown.remote_read_ns, 2210)
        self.assertEqual(breakdown.remote_read_first_credit_ns, 410)
        self.assertEqual(breakdown.remote_read_stream_ns, 1800)
        self.assertEqual(
            breakdown.remote_read_first_credit_ns
            + breakdown.remote_read_stream_ns,
            breakdown.remote_read_ns)
        # history_prep_ns 仍为后缀池恢复全量审计值（与 suffix 披露同源）。
        self.assertEqual(
            breakdown.history_prep_ns,
            _note_value(breakdown, "suffix_restore_ns"))
        self.assertEqual(
            breakdown.history_prep_ns,
            _pool_transfer_ns(
                total_bytes=2000, divisor=1, rates=model.rates))

    def test_single_credit_global_degeneration_kept(self):
        # K >= read_passes：合并流 first == remote_read_ns、stream == 0；
        # 分阶段逐腿同退化（整腿进首块）——prefill_stage 不变（后缀主
        # 导）、decode_stage = 整腿 2010 + decode 计算 20 ⇒ cost = 710
        # + 2030 + 210 = 2950。
        model = _model({0: _load(0), 1: _load(1)})
        model_credits = JointCostModel(
            rates=model.rates, loads=model.loads,
            flow_registry=model.flow_registry,
            service_factors=model.service_factors,
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2, route_fn=_route,
            remote_credit_iters="11")
        candidate = _remote(model_credits, _partial_session())
        breakdown = candidate.breakdown
        self.assertEqual(
            breakdown.remote_read_first_credit_ns,
            breakdown.remote_read_ns)
        self.assertEqual(breakdown.remote_read_stream_ns, 0)
        self.assertEqual(candidate.cost_ns, 710 + (2010 + 20) + 210)


# ================================================= eviction 串行披露 ==

class EvictionWaitSerialTest(unittest.TestCase):
    """规格书合成式：eviction_wait 串行于阶段之前，suffix 并行项不入
    该 max（旧 max(history_prep, eviction) 形态废除）。"""

    def test_eviction_serial_before_stages(self):
        # exec HBM 清空：空间缺口 = 池恢复后缀 + input + 增长 =
        # 1030 B/rank → 驱逐写回等待 = 100 + 1030/5 = 306（可逐出、
        # 无深缺口）；merge 反向腿加 exec 侧 home 保留量空间准备 300
        # → merge = 210 + 300 = 510。cost = 306 + 710 + 2010 + 510
        # = 3536——eviction 与 suffix_restore（500）不取 max、串行级联。
        tight = {0: _load(0), 1: _load(1, remaining=(0, 0))}
        model = _model(tight)
        candidate = _remote(model, _partial_session())
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.eviction_wait_ns, 306)
        self.assertIn(
            "execution_growth_eviction_writeback_est", breakdown.notes)
        self.assertEqual(_note_value(breakdown, "suffix_restore_ns"), 500)
        self.assertEqual(breakdown.merge_ns, 510)
        self.assertEqual(
            candidate.cost_ns,
            breakdown.eviction_wait_ns + 710 + 2010 + 510)
        # 串行方向对照：旧形态 prep = max(500, 306) = 500 会把 eviction
        # 完全吸收——新口径下 eviction 在 cost 中逐位可见。
        self.assertEqual(
            candidate.cost_ns - breakdown.merge_ns - 2010 - 710, 306)


if __name__ == "__main__":
    unittest.main()
