#!/usr/bin/env python3
"""test_joint_c6_sample_integrity.py -- C6 三钉（P2-1/P2-2/P2-3）。

  1. **P2-1 EWMA 对象**（v2.0 §9.2）：平滑对象 = 效率 η = base/actual
     （汇总样本 Σbase/Σactual），因子 = 1/η_new；取倒数与线性平均不可
     交换，对 actual/base 比值直接平滑是公式错对象（actual=2×base 纯
     样本应得 32/31，不是 17/16）。η_sample>1 维持 anomalous 拒收。
  2. **P2-2 采样纯度门**：观测窗口内本实例 rank 端口存在**他请求**在册
     HBM 传输流（HbmPortFlowRegistry 在册登记）即拒收并计 contended
     （与 anomalous/rejected 分列），落 joint_sample_contended 决策行
     披露端口占用摘要；本列车成员自身归属键与他实例 rank 流不误伤。
  3. **P2-3 决策日志可重建**（D17）：决策行 capacity_summary_view 冻结
     逐桶逐 rank 完整计价输入（R/clean/dirty/原子数/最大原子/
     shard_ratio/写回端口）+ 稳定 summary_id（内容哈希）——仅凭日志
     重建桶并经 consume_gap_theta 重算桶选择与逐 rank θ 份额，与在线
     决策一致。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_c6_sample_integrity.py   （或 pytest）
"""
import json
import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_HERE)
for _path in (_HERE, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    build_instances,
)
from joint.capacity_summary import (  # noqa: E402
    PHASE_DECODE,
    CapacityBucket,
    CapacityBucketKey,
    InstanceCapacitySummary,
    PhaseFootprint,
    SUFFIX_EASY,
    SUFFIX_REST,
    consume_gap_theta,
)
from joint.event_recursion_predictor import ServiceFactorGroup  # noqa: E402
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _capacity_summary_audit_entry,
    _TASK_LOAD_CACHE_CAPACITY,
)


# ===================================================== P2-1 EWMA 对象 ==


class EtaEwmaObjectPinnedTest(unittest.TestCase):
    """平滑对象 = η（v2.0 §9.2 原式），因子 = 1/η_new。"""

    def test_double_time_single_sample_factor_is_32_over_31(self):
        # 冷启动 η=1；纯样本 actual=2×base ⇒ η_sample=1/2，
        # η_new = 15/16 + 1/32 = 31/32，因子 = 32/31 ≈ 1.032258。
        # 对 ratio 直接平滑得 17/16 = 1.0625（+3.02% 系统偏差）——
        # 取倒数与线性平均不可交换。
        factors = ServiceFactors()
        factors.observe_prefill(actual_ns=200, base_ns=100, tick_ns=1)
        factors.flush()
        self.assertAlmostEqual(factors.prefill_factor, 32.0 / 31.0,
                               places=12)
        self.assertNotAlmostEqual(factors.prefill_factor, 17.0 / 16.0,
                                  places=12)
        self.assertEqual(factors.updates["prefill_factor"], 1)

    def test_aggregate_sample_eta_is_base_sum_over_actual_sum(self):
        # 同时刻样本先汇总再取一次效率：η_sample = Σbase/Σactual =
        # 250/400（不是逐条 η 的算术平均 (1/2+2/3)/2）。
        factors = ServiceFactors()
        factors.observe_decode(actual_ns=100, base_ns=50, tick_ns=5)
        factors.observe_decode(actual_ns=300, base_ns=200, tick_ns=5)
        factors.flush()
        eta_new = (15.0 / 16.0) * 1.0 + (1.0 / 16.0) * (250.0 / 400.0)
        self.assertAlmostEqual(factors.decode_factor, 1.0 / eta_new,
                               places=12)

    def test_sub_roofline_sample_rejected_as_anomalous(self):
        # η_sample > 1（⟺ 旧 ratio<1 判据：实测快于 roofline）= 记量
        # 异常：计数进 anomalous、不更新因子、不钳位。
        factors = ServiceFactors()
        factors.observe_decode(actual_ns=80, base_ns=100, tick_ns=1)
        factors.flush()
        self.assertEqual(factors.decode_factor, 1.0)
        self.assertEqual(factors.anomalous["decode_factor"], 1)
        self.assertNotIn("decode_factor", factors.updates)

    def test_contended_channel_disclosed_separately(self):
        # contended 与 rejected/anomalous 分列披露（as_dict 键恒在）。
        factors = ServiceFactors()
        factors.reject_contended("prefill_factor")
        payload = factors.as_dict()
        self.assertEqual(payload["contended"], {"prefill_factor": 1})
        self.assertEqual(payload["rejected"], {})
        self.assertEqual(payload["anomalous"], {})
        self.assertEqual(payload["updates"], {})


# ================================================== P2-2 纯度门争用 ==


def _seam_scheduler():
    """_observe_service_factors 生产函数的最小 seam（__new__ 直装）。

    拓扑：实例 0 = ranks (0,1)，实例 1 = ranks (2,3)——端口归属判定
    （他实例 rank 流不得误伤）的判别基础。
    """
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.hardware = FaceHardware(
        mesh_rows=2, mesh_cols=2,
        local_hbm_capacity_bytes=10**9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    scheduler.model = FaceModel(
        layers=2, hidden_size=16, ffn_size=32, num_heads=4, vocab_size=32,
        bytes_per_elem=2, mlp_variant="gelu")
    scheduler.topology = build_instances(
        scheduler.hardware,
        (FaceInstanceSpec("ins0", "1", (0, 1)),
         FaceInstanceSpec("ins1", "2", (2, 3))))
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._joint_factors = ServiceFactors()
    # §8.1 单账本（2026-09-30 修复）：样本唯一更新点 = 统一 recorder
    # ——替身显式补设（不入环境配置懒加载路径，保持 seam 封闭）。
    from joint.history_estimator import (
        PredictionParams, ServiceSampleRecorder)
    scheduler._joint_service_samples = ServiceSampleRecorder(
        PredictionParams())
    scheduler.kv_manager = SimpleNamespace(
        service_factors=ServiceFactorGroup())
    scheduler.runtime_by_request_id = {
        "r1": SimpleNamespace(
            joint_span_base_context=0, prefill_tokens_completed=0),
        "rd": SimpleNamespace(
            joint_decode_action=None, joint_action="stay",
            prefill_context_tokens=16, decode_tokens_consumed=0),
    }
    scheduler._hbm_ports = HbmPortFlowRegistry()
    # log_decision 缓冲路径（基类无 sink 时驻留 online_log_rows）。
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    return scheduler


def _prefill_train(emit_tick=100):
    return {
        "emit_tick": emit_tick,
        "members": [],
        "prefill_chunk_tokens": [("r1", 8)],
        "had_joiners": False,
        "suffix_gated": False,
        "history_transfer_gated": False,
        "merge_tail_gated": False,
    }


def _decode_train(emit_tick=100):
    return {
        "emit_tick": emit_tick,
        "members": [("rd", 2)],
        "prefill_chunk_tokens": (),
        "had_joiners": False,
        "merge_tail_gated": False,
    }


class ContendedSamplePurityGateTest(unittest.TestCase):
    """P2-2：跨请求在册 HBM 流期间的样本拒收（审查复现场景仓内化）。"""

    def test_audit_repro_foreign_copy_rejects_prefill_sample(self):
        # 审查复现：端口 0 有 other-request:copy 在册、四纯度标记全
        # False、span=10,000ns——修前样本照收（prefill factor 1→
        # 1.2696…）；修后拒收：因子不动、计 contended、face 侧 γ 桥接
        # 同门（同一 span 的争用不得写进两套因子）。
        scheduler = _seam_scheduler()
        scheduler._hbm_ports.register(0, "other-request:copy")
        state = SimpleNamespace(index=0, last_train_finalize_tick=100)
        scheduler._observe_service_factors(
            state, _prefill_train(), 100 + 10_000)
        scheduler._joint_factors.flush()
        self.assertEqual(scheduler._joint_factors.prefill_factor, 1.0)
        self.assertEqual(
            scheduler._joint_factors.contended, {"prefill_factor": 1})
        self.assertNotIn("prefill_factor", scheduler._joint_factors._pending)
        self.assertEqual(scheduler._joint_factors.updates, {})
        self.assertTrue(scheduler.kv_manager.service_factors.cold_start)

    def test_contention_rejection_discloses_port_occupancy(self):
        # 拒收行 kind=joint_sample_contended：窗口端点、他请求在册流与
        # 端口占用摘要（活跃 decode/在册传输/合计分列）齐备，整行 JSON
        # 可序列化。
        scheduler = _seam_scheduler()
        scheduler._hbm_ports.register(0, "other-request:copy")
        state = SimpleNamespace(index=0, last_train_finalize_tick=100)
        scheduler._observe_service_factors(
            state, _prefill_train(), 100 + 10_000)
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "joint_sample_contended"]
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertEqual(decision["instance_index"], 0)
        self.assertEqual(decision["factor"], "prefill_factor")
        self.assertEqual(decision["window_start_ns"], 100)
        self.assertEqual(decision["window_end_ns"], 100 + 10_000)
        self.assertEqual(decision["own_request_ids"], ["r1"])
        self.assertEqual(decision["foreign_hbm_flows"],
                         {"other-request:copy": [0]})
        self.assertEqual(
            decision["port_occupancy"]["0"],
            {"u_port_active_decode_streams": 0,
             "u_port_registered_transfer_flows": 1,
             "u_port_total": 1})
        self.assertEqual(json.loads(json.dumps(rows[0])), rows[0])

    def test_own_request_owner_keys_do_not_reject(self):
        # 本列车成员自身的全部归属键形态（bare rid / #decode / #readplan
        # / 逐出尾 watch 前缀）不判为争用——样本正常入统一 recorder
        #（§8.1 单账本：同刻汇总缓冲键 = (类别, tick)）。
        scheduler = _seam_scheduler()
        for owner in ("r1", "r1#decode", "r1#readplan",
                      "batch_train_evict_r1_side_0001#flow"):
            scheduler._hbm_ports.register(0, owner)
        state = SimpleNamespace(index=0, last_train_finalize_tick=100)
        scheduler._observe_service_factors(state, _prefill_train(), 200)
        self.assertIn(
            ("prefill", 200),
            scheduler._joint_service_samples._pending)
        self.assertEqual(scheduler._joint_factors.contended, {})

    def test_foreign_flow_on_other_instance_ranks_not_rejected(self):
        # 他请求流落在他实例 rank（2 ∈ 实例 1）——与本列车实例端口无
        # 关，不拒收（不误伤）。
        scheduler = _seam_scheduler()
        scheduler._hbm_ports.register(2, "other-request:copy")
        state = SimpleNamespace(index=0, last_train_finalize_tick=100)
        scheduler._observe_service_factors(state, _prefill_train(), 200)
        self.assertIn(
            ("prefill", 200),
            scheduler._joint_service_samples._pending)
        self.assertEqual(scheduler._joint_factors.contended, {})

    def test_decode_train_foreign_flow_rejected(self):
        # decode 分支同门：他请求在册流期间样本被拒并计 contended。
        scheduler = _seam_scheduler()
        scheduler._hbm_ports.register(1, "r_other#decode")
        state = SimpleNamespace(index=0, last_train_finalize_tick=100)
        scheduler._observe_service_factors(state, _decode_train(), 300)
        scheduler._joint_factors.flush()
        self.assertEqual(scheduler._joint_factors.decode_factor, 1.0)
        self.assertEqual(
            scheduler._joint_factors.contended, {"decode_factor": 1})
        self.assertNotIn("decode_factor", scheduler._joint_factors._pending)
        # 争用样本同样只入统一 recorder 的残差通道（contended，不进 η）。
        self.assertEqual(
            scheduler._joint_service_samples.contended, {"decode": 1})
        self.assertNotIn("decode", scheduler._joint_service_samples.updates)


# ============================================ P2-3 决策行可重建冻结 ==


def _bucket(key, reclaimable, atoms=None, max_atom=None):
    atoms = tuple(1 if value > 0 else 0 for value in reclaimable
                  ) if atoms is None else atoms
    max_atom = tuple(reclaimable) if max_atom is None else max_atom
    return CapacityBucket(
        key=key, reclaimable_by_rank=tuple(reclaimable),
        clean_by_rank=tuple(0 for _ in reclaimable),
        dirty_by_rank=tuple(reclaimable),
        atom_count_by_rank=tuple(atoms), max_atom_by_rank=tuple(max_atom))


def _load_with_two_buckets():
    """两桶（human/easy_suffix 100/100、tool/rest_layers 400/200）+
    remaining=(100,50)——缺口 d 由足迹派生，θ 消费跨两桶。"""
    summary = InstanceCapacitySummary(instance_index=0, buckets=(
        _bucket(CapacityBucketKey(
            category="human", suffix_class=SUFFIX_EASY,
            shard_ratio=(1, 1), writeback_ports=(0, 0)), (100, 100)),
        _bucket(CapacityBucketKey(
            category="tool", suffix_class=SUFFIX_REST,
            shard_ratio=(2, 1), writeback_ports=(0, 1)), (400, 200)),
    ))
    return InstanceLoadView(
        instance_index=0, queued_task_load_ns=0, running_task_load_ns=0,
        active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(100, 50),
        reclaimable_bytes_by_tp_rank=(500, 300),
        capacity_summary=summary)


def _model(load):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads={0: load}, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2,
        route_fn=lambda source, target: ((source, target), 1))


def _parse_theta_note(notes):
    """在线 notes 的 θ 披露行 → [(桶描述, 'θ 三位小数字符串'), ...]。"""
    for note in notes:
        if note.startswith("execution_growth_capacity_summary_theta("):
            body = note[len("execution_growth_capacity_summary_theta("):-1]
            return [tuple(item.split(":theta=")) for item in body.split(",")]
    raise AssertionError("theta disclosure note missing")


class CapacitySummaryReconstructionTest(unittest.TestCase):
    """P2-3（D17）：仅凭决策行冻结块重算桶选择与 θ 份额，与在线一致。"""

    def test_frozen_block_recomputes_bucket_selection_and_theta(self):
        load = _load_with_two_buckets()
        model = _model(load)
        footprint = PhaseFootprint.single(PHASE_DECODE, (300, 300))
        notes = []
        model._space_ready_ns_estimate(
            load, footprint, notes, instance_index=0)
        online = _parse_theta_note(notes)
        self.assertEqual(
            [key for key, _ in online],
            ["human/easy_suffix", "tool/rest_layers"])

        # 外部审计者路径：决策行冻结块（JSON round-trip 后）重建桶，
        # 缺口向量按与在线同一式（足迹峰值 − 日志 hbm_remaining）派生。
        frozen = json.loads(json.dumps(_capacity_summary_audit_entry(load)))
        rebuilt = [
            CapacityBucket(
                key=CapacityBucketKey(
                    category=bucket["category"],
                    suffix_class=bucket["suffix_class"],
                    shard_ratio=tuple(bucket["shard_ratio"]),
                    writeback_ports=tuple(bucket["writeback_ports"])),
                reclaimable_by_rank=tuple(bucket["reclaimable_by_rank"]),
                clean_by_rank=tuple(bucket["clean_by_rank"]),
                dirty_by_rank=tuple(bucket["dirty_by_rank"]),
                atom_count_by_rank=tuple(bucket["atom_count_by_rank"]),
                max_atom_by_rank=tuple(bucket["max_atom_by_rank"]))
            for bucket in frozen["buckets"]]
        logged_remaining = load.hbm_remaining_bytes_by_tp_rank
        gaps = footprint.binding_gap_by_rank(logged_remaining)
        recomputed = [
            (consumption.key.describe(),
             "{:.3f}".format(consumption.theta))
            for consumption in consume_gap_theta(rebuilt, gaps)]
        # 桶选择（固定桶序的消费序列）与逐桶 θ 份额逐位一致。
        self.assertEqual(recomputed, online)

    def test_summary_id_is_content_addressed_and_stable(self):
        # 同内容同 ID（跨决策行去重键），内容变即变；非行序号。
        load_a = _load_with_two_buckets()
        load_b = _load_with_two_buckets()  # 同内容、不同实例索引
        object.__setattr__(  # dataclass frozen；实例索引不入 ID 内容
            load_b, "instance_index", 1)
        id_a = _capacity_summary_audit_entry(load_a)["summary_id"]
        id_b = _capacity_summary_audit_entry(load_b)["summary_id"]
        self.assertEqual(id_a, id_b)
        self.assertTrue(id_a.startswith("capsum-"))
        load_c = _load_with_two_buckets()
        object.__setattr__(
            load_c.capacity_summary.buckets[0],
            "reclaimable_by_rank", (101, 100))
        id_c = _capacity_summary_audit_entry(load_c)["summary_id"]
        self.assertNotEqual(id_a, id_c)

    def test_uninjected_view_freezes_synthesised_anchor(self):
        # 未注入增量摘要的视图按与计价侧同一退路（reclaimable 单桶合成
        # 锚）冻结——日志侧不出现第二套桶。
        view = InstanceLoadView(
            instance_index=3, queued_task_load_ns=0, running_task_load_ns=0,
            active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(10, 10),
            reclaimable_bytes_by_tp_rank=(40, 60))
        frozen = _capacity_summary_audit_entry(view)
        self.assertEqual(len(frozen["buckets"]), 1)
        bucket = frozen["buckets"][0]
        self.assertEqual(bucket["reclaimable_by_rank"], [40, 60])
        self.assertEqual(bucket["dirty_by_rank"], [40, 60])
        self.assertEqual(bucket["clean_by_rank"], [0, 0])


class DecisionRowCapacitySummaryViewTest(unittest.TestCase):
    """P2-3 全路径：joint_admission 决策行携带 capacity_summary_view。"""

    def test_admission_row_carries_per_instance_frozen_blocks(self):
        from online.test_joint_decision_schema import _make_scheduler, _runtime
        scheduler = _make_scheduler()
        self.assertTrue(scheduler._try_admit_request(_runtime(), 0))
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "joint_admission"]
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertIn("capacity_summary_view", decision)
        summary_view = decision["capacity_summary_view"]
        self.assertEqual(
            [entry["instance_index"] for entry in summary_view], [0, 1])
        for entry in summary_view:
            self.assertTrue(entry["summary_id"].startswith("capsum-"))
            self.assertIsInstance(entry["buckets"], list)
            for bucket in entry["buckets"]:
                self.assertEqual(
                    sorted(bucket),
                    sorted(("category", "suffix_class", "shard_ratio",
                            "writeback_ports", "reclaimable_by_rank",
                            "clean_by_rank", "dirty_by_rank",
                            "atom_count_by_rank", "max_atom_by_rank")))
        # 冻结块与 load_view 聚合量对账（D11 钉死的恒等式：
        # reclaimable == hbm_remaining + Σ桶 R——聚合标量减 remaining 即
        # 逐桶 R 之和，冻结块与聚合口径互洽）。
        by_index = {entry["instance_index"]: entry
                    for entry in decision["load_view"]}
        for entry in summary_view:
            view = by_index[entry["instance_index"]]
            ranks = len(view["reclaimable_bytes_by_tp_rank"])
            bucket_total = [
                sum(bucket["reclaimable_by_rank"][rank]
                    for bucket in entry["buckets"])
                for rank in range(ranks)]
            self.assertEqual(
                [bucket_total[rank] + view["hbm_remaining_bytes_by_tp_rank"][rank]
                 for rank in range(ranks)],
                view["reclaimable_bytes_by_tp_rank"])
        self.assertEqual(json.loads(json.dumps(decision)), decision)


if __name__ == "__main__":
    unittest.main(verbosity=2)
