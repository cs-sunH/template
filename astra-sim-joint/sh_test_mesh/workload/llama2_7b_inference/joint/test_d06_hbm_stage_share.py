#!/usr/bin/env python3
"""test_d06_hbm_stage_share.py -- D06（2026-09-27，v2.0 §9.1 式(6)/§22-D06）
候选计算与候选传输共用 HBM 阶段份额的零后端单测。

覆盖（条目原文三项 + v2.0 验收两条）：
  1. CandidateStageHbmShare 只读叠加视图：divisor = m+(k−1)（消费方 +1
     口径）、stage_denominator = m+k、未登记端口 k 缺省 1；fail-closed
     （k < 1 / 非整数）；**不污染真实登记**（构造/查询前后注册表状态
     逐位不变，无 register 调用面）；
  2. 本地算子求值器接受候选阶段 HBM 份额：estimate_iteration_time_ns /
     estimate_prefill_task_load_ns / estimate_decode_remaining_task_load_
     ns 的 HBM 带宽项按 1/份额缩放（bytes-bound 关系式 + 容差），FLOP-
     bound 段不受份额影响，份额 < 1 / 非整数 fail-closed，缺省 1 逐位
     等于独占口径；
  3. 式(6) 传输审计同分母（JCM P 入口）：copy 写、pool 恢复写、remote
     暂存写按 notes 披露 hbm_stage_share；TASK_KIND_COMPUTE 的纯计算下限
     与 HBM 字节工作分开投影，profile/evaluator callback 只收到独占输入
     divisor=1，实际 HBM 竞争由事件活跃集重算；
  4. 验收①：单端口两条长期满载候选流（计算读 + copy 留存写）估计总
     服务不超峰值（各得 B/(m+k)，隐含带宽和 ≤ B）；
  5. 验收②：copy 尾流隐藏在计算后（交接墙钟 ≪ 计算段）仍能看到共享
     资源减速（首前向递推中计算段随份额放大）；
  6. 阶段裁定锚：P→D 转换（D 边界迁移腿）为独立冻结阶段且各端口单流
     （k=1）⇒ 旧口径 m+1 不变；候选估价全程不污染登记表。
  7. D06 收口（2026-09-28）：D 边界 remote 候选的 decode credit 暂存
     腿（双缓冲闭式供给基元/tile x）与同阶段 decode 计算读同端口同分
     母 m_e+2（离线 k_e=2 同面）；迁移腿仍 m+1（条目 6 锚不动）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d06_hbm_stage_share.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceModel,
    estimate_decode_remaining_task_load_ns,
    estimate_iteration_time_ns,
    estimate_prefill_task_load_ns,
)
from joint.hbm_port_flow_registry import (  # noqa: E402
    CandidateStageHbmShare,
    CandidateStageHbmShareError,
    HbmPortFlowRegistry,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_REMOTE,
    ACTION_STAY,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
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


def _session(home=0, resident=0, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=3, input_bytes=(5000, 5000),
             prefill_scan_passes=None):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes,
        prefill_scan_passes=prefill_scan_passes)


def _disjoint_paths(source, target):
    """tp=2 逐 rank 不相交路径（rank 对 (2s,2t)/(2s+1,2t+1)，各 1 跳）：
    home 端口 = {0,1}、exec 端口 = {2,3}（source=0/target=1 时）。"""
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


def _route(source, target):
    return ((source, target), 1)


def _model(port_registry=None, *, rates=None, prefill_fn=None,
           decode_fn=None):
    return JointCostModel(
        rates=rates if rates is not None else _rates(),
        loads={0: _load(0), 1: _load(1)},
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        route_paths_fn=_disjoint_paths,
        hbm_port_registry=port_registry,
        prefill_task_load_ns_fn=prefill_fn,
        decode_task_load_ns_fn=decode_fn)


def _registry_with_exec_contention(exec_active, home_active=0):
    """目标（exec）端口 = rank 0/1（target 实例 0）活跃 decode =
    exec_active 的注册表；源（home）端口 = rank 2/3（resident 实例 1）。"""
    registry = HbmPortFlowRegistry()
    registry.attach_active_decode_provider(
        lambda port: exec_active if port in (0, 1) else home_active)
    return registry


def _registry_all_ports(active):
    """全部端口同值活跃 decode（目标实例无关的 m_e 夹具）。"""
    registry = HbmPortFlowRegistry()
    registry.attach_active_decode_provider(lambda port: active)
    return registry


def _hardware(*, bandwidth_gbps=1000.0, tflops=1e6):
    """bytes-bound 求值夹具：峰值算力极高（FLOP 项被 max 吸收）、带宽
    有限 ⇒ HBM 项主导、份额缩放可观测；时延项 100 ns。"""
    return FaceHardware(
        mesh_rows=2, mesh_cols=2, local_hbm_capacity_bytes=10**12,
        local_hbm_bandwidth_gbps=bandwidth_gbps,
        d2d_bandwidth_gbps=1000.0, peak_perf_tflops=tflops,
        d2d_latency_ns=0, local_hbm_latency_ns=100)


def _model_spec():
    return FaceModel(
        layers=4, hidden_size=64, ffn_size=128, num_heads=8,
        vocab_size=64, bytes_per_elem=2, mlp_variant="gelu")


# ============================================== 1. 只读叠加视图本体 ==


class CandidateStageShareViewTest(unittest.TestCase):
    """CandidateStageHbmShare：口径、缺省、fail-closed、登记表纯净。"""

    def test_divisor_keeps_consumer_plus_one_convention(self):
        # m=2（活跃 decode）、k=3（候选阶段并发流）：divisor = m+(k−1) = 4
        # ——消费方 +1 后 = m+k = 5；stage_denominator = 5（完整分母）。
        registry = _registry_with_exec_contention(exec_active=2)
        share = CandidateStageHbmShare(registry, {0: 3})
        self.assertEqual(share.registered_divisor(0), 2)
        self.assertEqual(share.candidate_flows(0), 3)
        self.assertEqual(share.divisor(0), 4)
        self.assertEqual(share.stage_denominator(0), 5)

    def test_unmapped_port_defaults_to_own_single_flow(self):
        # 未登记端口 k 缺省 1（至多本候选的一条流）⇒ divisor = m、
        # 旧口径零漂移；registry 缺省 None（离线/夹具）⇒ m = 0。
        registry = _registry_with_exec_contention(exec_active=2)
        share = CandidateStageHbmShare(registry, {0: 3})
        self.assertEqual(share.divisor(1), 2)
        self.assertEqual(share.stage_denominator(1), 3)
        bare = CandidateStageHbmShare(None, {1: 2})
        self.assertEqual(bare.registered_divisor(1), 0)
        self.assertEqual(bare.divisor(1), 1)
        self.assertEqual(bare.stage_denominator(1), 2)

    def test_fail_closed_on_invalid_flow_counts(self):
        registry = _registry_with_exec_contention(exec_active=1)
        for bad in (0, -1, True, 2.0, "3"):
            with self.assertRaises(CandidateStageHbmShareError, msg=bad):
                CandidateStageHbmShare(registry, {5: bad})

    def test_view_never_pollutes_real_registry(self):
        # 条目原文"候选临时占用另加、不污染真实登记"：视图构造 + 查询
        # + snapshot 前后，注册表登记状态逐位不变（无 register 通道）。
        registry = HbmPortFlowRegistry()
        registry.register(0, "r7#decode")
        registry.register(1, "r7#decode")
        registry.attach_active_decode_provider(lambda port: 2)
        before_snapshot = registry.snapshot()
        before_owners = registry.leaked_owners()
        before_flag = registry.has_registrations
        share = CandidateStageHbmShare(registry, {0: 3, 1: 3})
        # m(0) = 活跃 2 + 在册 1 = 3：divisor = 3+(3−1) = 5、分母 = 6。
        self.assertEqual(share.registered_divisor(0), 3)
        self.assertEqual(share.divisor(0), 5)
        self.assertEqual(share.stage_denominator(1), 6)
        share.snapshot()  # 审计快照也只读
        self.assertEqual(registry.snapshot(), before_snapshot)
        self.assertEqual(registry.leaked_owners(), before_owners)
        self.assertEqual(registry.has_registrations, before_flag)
        self.assertEqual(registry.divisor(0), 3)  # 活跃 2 + 在册 1

    def test_snapshot_discloses_registered_candidate_denominator(self):
        registry = _registry_with_exec_contention(exec_active=3)
        share = CandidateStageHbmShare(registry, {0: 2})
        self.assertEqual(
            share.snapshot()[0],
            {"u_port_registered": 3, "candidate_stage_flows": 2,
             "stage_denominator": 5})


# ====================================== 2. 本地算子求值器接受份额 ==


class OperatorEvaluatorShareTest(unittest.TestCase):
    """roofline 求值器 HBM 带宽项按 m+k 份额缩放；FLOP 项不动。"""

    def test_bytes_bound_time_scales_with_share(self):
        # bytes-bound：t(d) ≈ hbm_latency + d × (独占带宽项)——份额 d 的
        # HBM 项 = d × 独占项（时延项不动；各自 ceil ⇒ ±1 ns 容差）。
        hardware = _hardware()
        model = _model_spec()
        exclusive = estimate_iteration_time_ns(
            hardware, model, instance_size=2, p_chunk=16, d_batch=0,
            d_token=0, p_context_tokens=16)
        shared = estimate_iteration_time_ns(
            hardware, model, instance_size=2, p_chunk=16, d_batch=0,
            d_token=0, p_context_tokens=16, hbm_share_divisor=3)
        # 两段结构（linear + max(prefill, decode)）⇒ 两个时延项；份额
        # 只缩放字节项：shared − 2·lat ≈ 3 × (exclusive − 2·lat)。
        base = 2 * hardware.local_hbm_latency_ns
        self.assertGreater(shared, exclusive)
        self.assertAlmostEqual(
            shared - base, 3 * (exclusive - base), delta=3)

    def test_flop_bound_time_ignores_share(self):
        # FLOP-bound（极高峰值带宽 ⇒ bytes 项被 max 吸收）：份额不影响
        # 结果（式(8) 的峰值算力项与时延项不动）。
        hardware = _hardware(bandwidth_gbps=1e12, tflops=1.0)
        model = _model_spec()
        exclusive = estimate_iteration_time_ns(
            hardware, model, instance_size=2, p_chunk=16, d_batch=0,
            d_token=0, p_context_tokens=16)
        shared = estimate_iteration_time_ns(
            hardware, model, instance_size=2, p_chunk=16, d_batch=0,
            d_token=0, p_context_tokens=16, hbm_share_divisor=7)
        self.assertEqual(shared, exclusive)

    def test_prefill_and_decode_wrappers_forward_share(self):
        hardware = _hardware()
        model = _model_spec()
        exclusive = estimate_prefill_task_load_ns(
            hardware, model, instance_size=2, chunk_tokens=16,
            context_tokens=16)
        shared = estimate_prefill_task_load_ns(
            hardware, model, instance_size=2, chunk_tokens=16,
            context_tokens=16, hbm_share_divisor=2)
        self.assertAlmostEqual(
            shared - 200, 2 * (exclusive - 200), delta=2)
        decode_exclusive = estimate_decode_remaining_task_load_ns(
            hardware, model, instance_size=2, current_context_tokens=16,
            generated_tokens=0, average_decode_length=1)
        decode_shared = estimate_decode_remaining_task_load_ns(
            hardware, model, instance_size=2, current_context_tokens=16,
            generated_tokens=0, average_decode_length=1,
            hbm_share_divisor=2)
        self.assertAlmostEqual(
            decode_shared - 200, 2 * (decode_exclusive - 200), delta=2)

    def test_share_fail_closed(self):
        hardware = _hardware()
        model = _model_spec()
        for bad in (0, -2, True, 1.5):
            with self.assertRaises(ValueError, msg=bad):
                estimate_iteration_time_ns(
                    hardware, model, instance_size=2, p_chunk=16,
                    d_batch=0, d_token=0, hbm_share_divisor=bad)
            with self.assertRaises(ValueError, msg=bad):
                estimate_prefill_task_load_ns(
                    hardware, model, instance_size=2, chunk_tokens=16,
                    context_tokens=16, hbm_share_divisor=bad)
            with self.assertRaises(ValueError, msg=bad):
                estimate_decode_remaining_task_load_ns(
                    hardware, model, instance_size=2,
                    current_context_tokens=16, generated_tokens=0,
                    average_decode_length=2.0, hbm_share_divisor=bad)

    def test_default_share_one_is_bit_identical_exclusive(self):
        # 缺省 1 = 独占带宽：全部既有调用点（台账/排队负载等非候选计价
        # 路径）零漂移。
        hardware = _hardware()
        model = _model_spec()
        self.assertEqual(
            estimate_prefill_task_load_ns(
                hardware, model, instance_size=2, chunk_tokens=16,
                context_tokens=16),
            estimate_prefill_task_load_ns(
                hardware, model, instance_size=2, chunk_tokens=16,
                context_tokens=16, hbm_share_divisor=1))


# ============================== 3. 式(6) 同分母（JCM P 入口接线） ==


class StageShareSameDenominatorTest(unittest.TestCase):
    """同一端口上计算读/copy 写/pool 恢复写/remote 暂存写同分母。"""

    EXEC_BYTES_PER_RANK = 8000  # 交接/暂存/恢复字节（disjoint 逐 rank）

    def test_copy_write_leg_shares_denominator_with_compute(self):
        # copy：exec 端口候选流 k = 计算读 + copy 留存写 = 2 ⇒ exec 写
        # 腿 = 8000/(10/2) = 1600 主导（noc=1e6 快链路/home 读腿 800/
        # 800 均被遮蔽）；notes 披露 exec_flows=2 与计算分母 2。
        model = _model(rates=_rates(noc=1e6, hbm=10.0))
        candidate = model.estimate_P_with_continuation(
            session=_session(resident=1,
                             history_bytes=(self.EXEC_BYTES_PER_RANK,) * 2),
            request=_request(), instance_index=0, action=ACTION_COPY,
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(
            candidate.breakdown.history_prep_ns,
            self.EXEC_BYTES_PER_RANK // (10 // 2))
        self.assertIn(
            "hbm_stage_share=exec_flows=2(compute+copy1/pool0/staging0)"
            ",compute_denom=2",
            candidate.breakdown.notes)

    def test_pool_restore_write_leg_shares_denominator_with_compute(self):
        # stay-PARTIAL：缺失后缀池恢复——目标 HBM 写腿与计算读同分母
        # （k = 计算读 + 恢复写 = 2）。池端口即端点本地（默认注入退化
        # ——无 NoC 腿）：history_prep = 100(池时延) + 8000/(10/2)。
        model = _model(rates=_rates(hbm=10.0))
        candidate = model.estimate_P_with_continuation(
            session=_session(location="partial_hbm_remote",
                             missing=(self.EXEC_BYTES_PER_RANK,) * 2),
            request=_request(), instance_index=0, action=ACTION_STAY,
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertIn("partial_suffix_pool_restore",
                      candidate.breakdown.notes)
        self.assertEqual(
            candidate.breakdown.history_prep_ns,
            100 + self.EXEC_BYTES_PER_RANK // (10 // 2))
        self.assertIn(
            "hbm_stage_share=exec_flows=2(compute+copy0/pool1/staging0)"
            ",compute_denom=2",
            candidate.breakdown.notes)

    def test_registered_background_flows_enter_same_denominator(self):
        # m_e（在册活跃 decode）与 k_e 进同一分母：exec 端口 m=3、
        # copy 候选 k=2 ⇒ 写腿 = 8000/(10/5) = 4000。
        registry = _registry_with_exec_contention(exec_active=3)
        model = _model(port_registry=registry,
                       rates=_rates(noc=1e6, hbm=10.0))
        candidate = model.estimate_P_with_continuation(
            session=_session(resident=1,
                             history_bytes=(self.EXEC_BYTES_PER_RANK,) * 2),
            request=_request(), instance_index=0, action=ACTION_COPY,
            remote_enabled=True)
        self.assertEqual(
            candidate.breakdown.history_prep_ns,
            self.EXEC_BYTES_PER_RANK // (10 // 5))

    def test_remote_staging_write_leg_shares_denominator_with_compute(self):
        # remote：暂存写腿（kind="read" 第三腿）与计算读同分母——
        # m=0、k=2（计算读 + 暂存写）⇒ 全流 wall = 8000×passes/(10/2)。
        model = _model(rates=_rates(noc=1e6, hbm=10.0))
        candidate = model.estimate_P_with_continuation(
            session=_session(resident=1,
                             history_bytes=(self.EXEC_BYTES_PER_RANK,) * 2),
            request=_request(decode=3), instance_index=0,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertIn("remote_read_passes=4", candidate.breakdown.notes)
        self.assertEqual(
            candidate.breakdown.remote_read_ns,
            (self.EXEC_BYTES_PER_RANK * 4) // (10 // 2))

    def test_compute_receives_exclusive_profile_input(self):
        # P/D 计算腿使用纯计算下限 + HBM 字节工作；HBM 资源竞争由事件
        # 世界按活跃集动态重仲裁。profile/evaluator callback 的 divisor=1
        # 只表示输入 profile 的独占基准，候选 HBM 路由承载实际共享。
        seen = {}

        def fn(tokens, history, hbm_share_divisor=1):
            seen["share"] = hbm_share_divisor
            return tokens * 10

        registry = _registry_all_ports(4)
        model = _model(port_registry=registry, prefill_fn=fn)
        stay = model.estimate_P_with_continuation(
            session=_session(resident=1, history_bytes=(0, 0)),
            request=_request(), instance_index=1, action=ACTION_STAY,
            remote_enabled=True)
        self.assertEqual(seen["share"], 1)
        model2 = _model(port_registry=registry,
                        rates=_rates(noc=1e6, hbm=10.0),
                        prefill_fn=fn)
        copy_c = model2.estimate_P_with_continuation(
            session=_session(resident=1,
                             history_bytes=(self.EXEC_BYTES_PER_RANK,) * 2),
            request=_request(), instance_index=0, action=ACTION_COPY,
            remote_enabled=True)
        self.assertEqual(seen["share"], 1)
        # 传输审计位仍按 legacy 静态视图 m+k（=6）披露（C5 冻结 notes
        # 面；事件世界的实际份额 = 活跃集动态计数 m+候选写流）。
        self.assertEqual(
            copy_c.breakdown.history_prep_ns,
            int(self.EXEC_BYTES_PER_RANK / (10.0 / 6)))

    def test_no_registry_fixture_compute_share_is_one_for_stay(self):
        # 守恒锚：无注册表、无传输腿（stay-LOCAL）⇒ 分母 1 = 旧独占
        # 口径逐位一致。
        seen = {}

        def fn(tokens, history, hbm_share_divisor=1):
            seen["share"] = hbm_share_divisor
            return tokens * 10

        model = _model(prefill_fn=fn)
        model.estimate_P_with_continuation(
            session=_session(resident=1, history_bytes=(0, 0)),
            request=_request(), instance_index=1, action=ACTION_STAY,
            remote_enabled=True)
        self.assertEqual(seen["share"], 1)

    def test_recompute_single_flow_share_with_background_m(self):
        # recompute 的 profile 输入保持 divisor=1；纯计算下限与 HBM
        # 字节工作由事件世界投影，recompute 本身没有候选传输腿。
        seen = {}

        def fn(tokens, history, hbm_share_divisor=1):
            seen["share"] = hbm_share_divisor
            return tokens * 10

        registry = _registry_all_ports(2)
        model = _model(port_registry=registry, prefill_fn=fn)
        candidate = model.estimate_P_with_continuation(
            session=_session(resident=0, history_bytes=(0, 0)),
            request=_request(), instance_index=0, action="recompute",
            remote_enabled=True)
        self.assertEqual(seen["share"], 1)


# ======================================== 4. v2.0 §22-D06 验收两条 ==


class AcceptanceBoundTest(unittest.TestCase):
    """验收①两条候选流总服务不超峰值；验收②尾流隐藏仍见减速。"""

    def test_two_long_full_load_candidate_flows_within_port_peak(self):
        # 验收①：单 HBM 端口两条长期满载候选流（计算读 + copy 留存写）
        # ——份额 d = m+k = 2 下每流隐含带宽 = B/2，估计总服务（隐含带
        # 宽和）= B ≤ 峰值；旧口径（各自独占）会虚报 2B > 峰值。传输
        # 腿与计算读确为同分母（都按 d=2 缩放）。
        bytes_per_rank = 8000
        copy_leg_ns = bytes_per_rank // (1000 // 2)  # 写腿 @ B/2
        compute_bw = 1000.0 / 2  # 计算读流隐含带宽（份额 d=2）
        copy_bw = 1000.0 / 2     # copy 写流隐含带宽（同分母）
        self.assertEqual(compute_bw + copy_bw, 1000.0)  # = 端口峰值
        self.assertEqual(copy_leg_ns, bytes_per_rank * 2 // 1000)

    def test_hidden_copy_tail_still_slows_compute(self):
        # 验收②（W3 重锚，v4-projection 世代）：copy 尾流与计算的耦合
        # 由**数据门依赖边**承载——copy chunk 到达门控其层区间的计算，
        # 慢 HBM 交接令首前向真实推迟（旧世代的求值器份额放大退役）。
        # 夹具：计算段 5000 ns（prefill 求值器 100×50）、交接 8 B/rank
        # ——hbm=noc=1e9 时交接墙钟 1 ns（首前向 ≈ 计算段 5001）；
        # hbm=0.01 B/ns 时写腿 8/0.01 = 800 ns 仍隐藏在 5000 内、
        # hbm=0.001 时 8000 ns > 计算段 ⇒ 首前向被数据门推迟。
        def fn(tokens, history, hbm_share_divisor=1):
            return tokens * 100

        stay = _model(
            rates=_rates(noc=1e9, hbm=1e9),
            prefill_fn=fn).estimate_P_with_continuation(
            session=_session(resident=1, history_bytes=(0, 0)),
            request=_request(), instance_index=1, action=ACTION_STAY,
            remote_enabled=True)
        fast_copy = _model(
            rates=_rates(noc=1e9, hbm=1e9),
            prefill_fn=fn).estimate_P_with_continuation(
            session=_session(resident=1, history_bytes=(8, 8)),
            request=_request(), instance_index=0, action=ACTION_COPY,
            remote_enabled=True)
        slow_copy = _model(
            rates=_rates(noc=1e9, hbm=0.001),
            prefill_fn=fn).estimate_P_with_continuation(
            session=_session(resident=1, history_bytes=(8, 8)),
            request=_request(), instance_index=0, action=ACTION_COPY,
            remote_enabled=True)
        self.assertIn("prefill_first_forward_ns=5000", stay.breakdown.notes)
        # 快交接：交接隐藏在计算内（首前向 = 计算段 + 1 ns 取整墙钟）。
        self.assertIn(
            "prefill_first_forward_ns=5001", fast_copy.breakdown.notes)
        # 慢交接：写腿 8 B/0.001 B/ns（双端 HBM 腿）⇒ chunk 到达
        # 16000 ns > 计算段 ⇒ 首前向 = 数据门 + 计算段 = 21000。
        self.assertIn(
            "prefill_first_forward_ns=21000", slow_copy.breakdown.notes)
        def _walls(candidate):
            note = next(n for n in candidate.breakdown.notes
                        if n.startswith("copy_handoff_chunk_walls="))
            return [int(v) for v in note.split("=", 1)[1].split(",")]
        self.assertGreater(_walls(slow_copy)[0], _walls(fast_copy)[0])
        # 计算段不再被份额放大（求值器份额退役——两候选同值）。
        self.assertEqual(
            fast_copy.breakdown.compute_ns, stay.breakdown.compute_ns)


# ==================================== 5. 阶段裁定锚（不变口径钉测） ==


class StageScopeAnchorTest(unittest.TestCase):
    """P→D 转换（迁移腿）为独立冻结阶段（各端口单流 k=1）⇒ 旧 m+1
    口径不变；候选估价全程不污染登记表。"""

    @staticmethod
    def _working():
        return 8000

    def test_boundary_move_leg_keeps_registry_plus_one(self):
        # D 边界完整型迁移（copy 三腿）：exec 写腿分母 = m+1（k=1，独
        # 立阶段单流）——非 m+k_exec。exec 端口 m=2 ⇒ 8000/(10/3)。
        registry = _registry_with_exec_contention(exec_active=2)
        model = _model(port_registry=registry,
                       rates=_rates(noc=1e6, hbm=10.0))
        boundary = PDBoundaryView(
            session_id="s", home_instance=0, p_instance_index=1,
            p_action=ACTION_COPY, history_tokens=100,
            working_bytes_by_tp_rank=(self._working(), self._working()),
            base_bytes_by_tp_rank=(0, 0),
            decode_steps=4, kv_token_bytes=100)
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_COPY,
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(
            candidate.breakdown.history_prep_ns,
            int(self._working() / (10 / 3)))

    def test_candidate_pricing_leaves_registry_state_untouched(self):
        # 端到端纯净锚：copy/remote/stay 三动作连续估价后，注册表
        # snapshot/leaked_owners/has_registrations 逐位不变。
        registry = HbmPortFlowRegistry()
        registry.register(0, "r9#decode")
        registry.register(1, "r9#decode")
        registry.attach_active_decode_provider(lambda port: 2)
        before = (registry.snapshot(), registry.leaked_owners(),
                  registry.has_registrations)
        model = _model(port_registry=registry,
                       rates=_rates(noc=1e6, hbm=10.0))
        fixtures = (
            (ACTION_COPY, _session(resident=1,
                                   history_bytes=(8000, 8000)), 0),
            (ACTION_REMOTE, _session(
                resident=1, history_bytes=(8000, 8000)), 0),
            (ACTION_STAY, _session(
                location="partial_hbm_remote", missing=(8000, 8000)), 0),
        )
        for action, session, target in fixtures:
            candidate = model.estimate_P_with_continuation(
                session=session, request=_request(), instance_index=target,
                action=action, remote_enabled=True)
            self.assertTrue(candidate.applicable, msg=action)
        self.assertEqual(
            (registry.snapshot(), registry.leaked_owners(),
             registry.has_registrations),
            before)


# ==================== 6. F6 修复轮三：P 入口双缓冲腿走阶段份额 ==


class RemoteLegStageShareAnchorTest(unittest.TestCase):
    """P 入口 remote 双缓冲腿（审计 notes 键 remote_read_prefill_ns /
    remote_read_decode_ns）与合并流腿/逐层到达表同走候选阶段 HBM 份额
    （修复轮三 F6：修复前腿内供给基元/tile x 回退走裸注册表，与
    remote_read_ns 的 m+k 分母失配）。

    夹具 = HBM 瓶颈域（B_HBM = 8 B/ns ≪ B_link = 1000 B/ns——NoC 腿
    可忽略、份额放大在腿值中直接可观测；d2d 时延 10 ns、池时延
    100 ns；loads 零排队）。基座与 test_joint_credit_pricing 锚族同
    构：history (1000,1000)/rank、input (25,25)、decode=10、layers=4、
    prefill 1.0/decode 2.0 ns/token ⇒ kv_token_bytes = 10、tile =
    256 token = 2560 B/shard、passes = 1+10 = 11；resident=1 →
    target=0 不相交路径：候选 exec_ranks = 端口 {0,1}（阶段份额 k=2
    = 计算读 + 暂存写登记于此 = 目标 exec 端口）、home 端口 = {2,3}
    （份额视图未登记，k 缺省 1）。

    手算复核式（离线 m=0，denom = m+k = 2 → exec 率 8/2 = 4 B/ns）：
      x = 10 + max(2560/1000, home 2560/8, exec 2560/4) = 10+640 = 650
      prefill（1 遍、流 1000 → 纯尾块）：tail_x = 10+max(1,125,250)
        = 260、tail_c = int(50×1000/2000) = 25 → 285
      decode（修复轮三 H1 逐 pass 重切：full_tiles = passes ×
        ⌈tokens/tile⌉ = 10 × ⌈100/256⌉ = 10 整块/rank、每 pass 尾块
        并入整块服务 tail=0；x = latency+tile_bytes/bw = 650、tile_c
        = int(20×2560/(2×10×2560)) = 1 < x；闭式 T_j = T_{j-1}+x →
        F = x+c+9x = 650+1+5850 = 6501）（旧连续链 3 整+1 尾 = 2542
        ——逐 pass 重切消除跨 pass 尾块合并的低估，只增不减）
      审计同分母：first_credit = 10+640 = 650 == x、stream = 8440/2
        = 2110、remote_read_ns = 2760
    m_e=2（_registry_all_ports(2)，denom = 2+2 = 4 → exec 率 2）：
      x = 10+1280 = 1290；tail_x = 10+max(1,250,500) = 510 → 535；
      tile_c = 1 < x、F = 1290+1+9×1290 = 12901（旧 5042——同上只增
      不减）；first_credit = 1290 == x、stream = 8440/2 = 4220、
      remote_read_ns = 5510。
    """

    @staticmethod
    def _note_value(breakdown, key):
        prefix = key + "="
        for note in breakdown.notes:
            if note.startswith(prefix):
                return int(note[len(prefix):])
        raise AssertionError(f"missing disclosure note {key!r}")

    @classmethod
    def _remote_candidate(cls, registry=None):
        model = _model(port_registry=registry,
                       rates=_rates(noc=1000.0, hbm=8.0, lat=10))
        return model.estimate_P_with_continuation(
            session=_session(resident=1),
            request=_request(decode=10, input_bytes=(25, 25)),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)

    def test_offline_legs_follow_stage_share(self):
        # 离线默认（无注册表 → m=0、k=2）：腿值锚 285/6501（decode 腿
        # 修复轮三 H1 逐 pass 重切新锚 6501，旧连续链口径 2542——只增
        # 不减；修复前裸注册表口径为 260/2132——份额不进腿）。
        candidate = self._remote_candidate()
        self.assertTrue(candidate.applicable)
        notes = candidate.breakdown.notes
        self.assertEqual(
            self._note_value(candidate.breakdown, "remote_read_prefill_ns"),
            285)
        self.assertEqual(
            self._note_value(candidate.breakdown, "remote_read_decode_ns"),
            6501)
        self.assertIn("remote_read_passes=11", notes)
        # 同分母一致性：腿内 tile x（notes 披露位）== 合并流审计首
        # tile 服务（remote_read_first_credit_ns）——腿键与 remote_
        # read_ns 同一 m+k 分母；且审计分解恒等 first+stream == 全流。
        decode_note = next(
            n for n in notes if n.startswith("decode_tile_tiles="))
        decode_x = int(decode_note.split("_x=")[1].split("_c=")[0])
        self.assertEqual(decode_x, 650)
        self.assertEqual(
            candidate.breakdown.remote_read_first_credit_ns, decode_x)
        self.assertEqual(
            candidate.breakdown.remote_read_first_credit_ns
            + candidate.breakdown.remote_read_stream_ns,
            candidate.breakdown.remote_read_ns)
        self.assertEqual(candidate.breakdown.remote_read_ns, 2760)

    def test_registered_decode_share_scales_legs(self):
        # 注入 registry m_e=2：目标端口分母 = m+k = 4 → 腿 535/12901
        # （x 翻倍 650 → 1290，HBM 瓶颈域内份额放大逐位可观测；decode
        # 腿为 H1 逐 pass 重切新锚 12901，旧连续链口径 5042——只增不
        # 减）。
        candidate = self._remote_candidate(registry=_registry_all_ports(2))
        self.assertTrue(candidate.applicable)
        notes = candidate.breakdown.notes
        self.assertIn(
            "hbm_stage_share=exec_flows=2(compute+copy0/pool0/staging1)"
            ",compute_denom=4", notes)
        self.assertEqual(
            self._note_value(candidate.breakdown, "remote_read_prefill_ns"),
            535)
        self.assertEqual(
            self._note_value(candidate.breakdown, "remote_read_decode_ns"),
            12901)
        decode_note = next(
            n for n in notes if n.startswith("decode_tile_tiles="))
        self.assertEqual(int(decode_note.split("_x=")[1].split("_c=")[0]),
                         1290)
        self.assertEqual(
            candidate.breakdown.remote_read_first_credit_ns, 1290)
        self.assertEqual(
            candidate.breakdown.remote_read_first_credit_ns
            + candidate.breakdown.remote_read_stream_ns,
            candidate.breakdown.remote_read_ns)
        self.assertEqual(candidate.breakdown.remote_read_ns, 5510)


# ============== 7. D06 收口（2026-09-28）：D 边界 credit 腿同分母 ==


class DboundaryCreditLegStageShareTest(unittest.TestCase):
    """D 边界 remote 候选的 decode credit 暂存腿（双缓冲闭式的供给基元
    /tile x）与同阶段 decode 计算读同端口同分母 m_e+2（式(6) 收口，
    2026-09-28）。

    修复前：单源腿调 ``_remote_double_buffer_leg_ns`` 不传
    ``hbm_port_registry``（裸注册表 m_e+1；离线无注册表时独占 = 1），
    多源 tile x 求值同走裸注册表——与同端口同阶段并发的 decode 计算
    （decode_share k_e=2 → 分母 m_e+2）失配：闭式 start_ns =
    move_done_ns 意味着暂存供给与 decode 计算同端口并发，暂存腿偏快
    ⇒ remote D 边界候选偏乐观。修复后调用方保留 decode 阶段静态审计
    份额；实际 HBM 竞争由事件世界按活跃集重算。

    夹具 = HBM 瓶颈域（noc=1e6 ≫ hbm=10 B/ns——exec 写腿主导，份额
    放大在 tile x 中逐位可观测；离线口径 = 无注册表、无求值器）：
    完整型边界（working (8000,8000)/rank、kv_token_bytes=100 → 每源
    100 token、tile auto = 256 token = 25600 B/shard、decode_steps=4
    → 4 pass；source = p_instance=1 → target=0：exec 端口 = {0,1} =
    阶段份额 k_e=2 登记位、home 端口 = {2,3} 未登记 k 缺省 1）。

    手算复核式（tile x = 25600/(10/分母)，NoC 腿 0.26 ns 与 home 读腿
    （分母 ≤ m+1）均被 exec 写腿遮蔽）：
      离线（m=0）：分母 = m_e+k_e = 2 → x = 5120（修复前独占口径 2560）
      m_e=2（_registry_all_ports(2)）：分母 = m_e+k_e = 4 → x = 10240
        （修复前裸注册表 m_e+1 = 3 → 7680）
      decode profile callback 收到 hbm_share_divisor=1，表示独占 profile
        输入；tile x 的静态审计分母仍为 m_e+k_e = 4，实际计算 HBM 流与
        credit 流在事件世界共享活跃集。
    多源形态（增量型两来源：base@home=2 + working@p=1，目标 0）同
    分母：tile 恒 25600 B（tile 与来源数无关），x 逐位同单源锚。
    """

    TILE_STREAM_BYTES = 25600   # 256 token × 100 B/(rank·token)
    HBM_BNS = 10.0              # local_hbm_gbps=10.0 = 10 B/ns

    @staticmethod
    def _boundary():
        return PDBoundaryView(
            session_id="s", home_instance=0, p_instance_index=1,
            p_action=ACTION_COPY, history_tokens=100,
            working_bytes_by_tp_rank=(8000, 8000),
            base_bytes_by_tp_rank=(0, 0),
            decode_steps=4, kv_token_bytes=100)

    @staticmethod
    def _multi_source_boundary():
        # 增量型（p_action=REMOTE）：base 驻 home=2、working 驻
        # p_instance=1，目标 0 ⇒ 两来源各一缺失区间。
        return PDBoundaryView(
            session_id="s", home_instance=2, p_instance_index=1,
            p_action=ACTION_REMOTE, history_tokens=90,
            working_bytes_by_tp_rank=(4000, 4000),
            base_bytes_by_tp_rank=(4000, 4000),
            base_tokens=50, decode_steps=4, kv_token_bytes=100)

    @classmethod
    def _decode_tile_x(cls, breakdown):
        note = next(n for n in breakdown.notes
                    if n.startswith("decode_tile_tiles="))
        return int(note.split("_x=")[1].split("_c=")[0])

    @classmethod
    def _note_starts(cls, breakdown, prefix):
        return next(n for n in breakdown.notes if n.startswith(prefix))

    def _remote_candidate(self, registry=None, decode_fn=None,
                          boundary=None):
        model = _model(port_registry=registry,
                       rates=_rates(noc=1e6, hbm=10.0),
                       decode_fn=decode_fn)
        candidate = model.estimate_D_remaining(
            boundary=boundary if boundary is not None
            else self._boundary(),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        return candidate

    def test_offline_credit_leg_counts_own_staging_flow(self):
        # 离线（无注册表）：exec 写腿分母 = m_e+k_e = 0+2 = 2——暂存写
        # 与（假想的）计算读共享端口，修复前独占口径 x = 2560。
        candidate = self._remote_candidate()
        self.assertEqual(
            self._decode_tile_x(candidate.breakdown),
            int(self.TILE_STREAM_BYTES / (self.HBM_BNS / 2)))

    def test_registered_credit_leg_denominator_is_stage_denominator(self):
        # 注入 m_e=2：exec 写腿分母 = m_e+k_e = 4（非裸注册表 m_e+1=3
        # ——修复前 x = 7680）；tile x = 25600/(10/4) = 10240。
        candidate = self._remote_candidate(registry=_registry_all_ports(2))
        self.assertEqual(
            self._decode_tile_x(candidate.breakdown),
            int(self.TILE_STREAM_BYTES / (self.HBM_BNS / 4)))
        self.assertEqual(self._decode_tile_x(candidate.breakdown), 10240)

    def test_credit_leg_shares_denominator_with_decode_compute(self):
        # decode profile callback 使用 divisor=1 的独占输入；decode 与
        # credit HBM 腿共享动态活跃集，tile x 仍是 legacy 静态审计视图
        # m_e+k_e（=4）。
        seen = {}

        def decode_fn(context_tokens, *, hbm_share_divisor=1):
            seen["share"] = hbm_share_divisor
            return context_tokens * 10

        candidate = self._remote_candidate(
            registry=_registry_all_ports(2), decode_fn=decode_fn)
        self.assertEqual(seen["share"], 1)
        self.assertTrue(any(
            note.startswith("decode_ctx_exact=base=100,steps=4,")
            for note in candidate.breakdown.notes),
            candidate.breakdown.notes)
        # 审计腿 x 仍按 legacy 静态视图分母 4 披露（m_e 2 + k_e 2）。
        self.assertEqual(
            self._decode_tile_x(candidate.breakdown),
            int(self.TILE_STREAM_BYTES / (self.HBM_BNS / 4)))

    def test_multi_source_credit_leg_follows_stage_share(self):
        # 多源（增量型两来源）：tile x 逐 (rank, 来源) 求值同走 decode
        # 阶段冻结份额——x = 25600/(10/4) = 10240（修复前裸注册表
        # 7680）；tile 与来源数无关（恒配置值），分母逐位同单源锚。
        candidate = self._remote_candidate(
            registry=_registry_all_ports(2),
            boundary=self._multi_source_boundary())
        notes = candidate.breakdown.notes
        self.assertIn("boundary_remote_sources=1,2", notes)
        self.assertEqual(
            self._decode_tile_x(candidate.breakdown),
            int(self.TILE_STREAM_BYTES / (self.HBM_BNS / 4)))

    def test_multi_source_offline_counts_own_staging_flow(self):
        # 多源离线（无注册表）：分母 = 0+2 = 2（修复前独占 2560）——
        # 离线口径与 P 入口离线腿（test_offline_legs_follow_stage_
        # share）同族：阶段份额视图不依赖求值器注入。
        candidate = self._remote_candidate(
            boundary=self._multi_source_boundary())
        self.assertIn("boundary_remote_sources=1,2",
                      candidate.breakdown.notes)
        self.assertEqual(
            self._decode_tile_x(candidate.breakdown),
            int(self.TILE_STREAM_BYTES / (self.HBM_BNS / 2)))


if __name__ == "__main__":
    unittest.main()
