#!/usr/bin/env python3
"""test_history_estimator.py -- W6 验收（方案 W6 段 的 V1/V8 面）。

被测模块：``joint/history_estimator.py``（shadow 统计与未知工作量接口）。
覆盖矩阵（方案 §13 的 W6 行）：

  * **条件剩余长度**（§8.2/主文 §3.2）：未结束且 d 已超均值仍有正剩余
    （非"均值−d 截零"）；删失请求不进完成样本；样本不足/d 超观测最大
    回退正尾先验；回退链 session→类别→run→先验逐级；直方图溢出归并
    尾桶不截尾；同输入确定性 + 估计只读；隐藏真值形参结构杜绝；参数
    从 PredictionParams 注入生效。
  * **到达统计**（§8.3/§8.4/主文 §5.2/§5.3）：α 派生与手算一致
    （1−2^(−Δ/L)）；空桶必须衰减（连续 10 空桶 == 闭式）；批量推进与
    逐桶推进等价（随机对拍）；快慢双尺度（Lf<Ls ⇒ α_f>α_s）；互斥分类
    恰更新一类（队列/在途/已提交通信不计）；回归公式 h=0 退化 λ_fast、
    h→∞ 收敛 λ_slow；observed_until 因果快照（双实例重放等价）；未闭合
    桶 full-buckets-only；advance 幂等；参数注入。
  * **服务样本**（§8.1/主文 §5.1）：纯样本进效率（EWMA 手算锚）；争用
    样本拒收计数；等待不可分离 → 残差记录不吸收；EWMA 权重经 params；
    与现有 ServiceFactors 默认参数逐点全等（语义同源锚）；效率更新与到
    达衰减互不影响（双向）。
  * **成熟/回退/epoch**（§8.5）：未成熟不产误差；成熟恰一次（幂等）；
    缺真值挂起；过期 EXPIRED 可复现；七码 FallbackRecord 可复现；epoch
    隔离窗 + 冻结"不重置"规则位。
  * **shadow 纪律**：模块不 import 调度器（无影响正式选择的调用面）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_history_estimator.py   （或 pytest 同路径）
"""
import inspect
import math
import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_HERE)
for _p in (_HERE, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joint.history_estimator as history_estimator  # noqa: E402
from joint.history_estimator import (  # noqa: E402
    ARRIVAL_CLASS_NEW_SESSION_FIRST,
    ARRIVAL_CLASS_SESSION_RETURN,
    ConditionalRemainingLengthEstimator,
    ArrivalStatistics,
    EpochLog,
    FallbackCode,
    FallbackRecord,
    ForecastMaturityLedger,
    ServiceSampleRecorder,
    ServiceWorkSample,
)
from joint.joint_cost_model import ServiceFactors  # noqa: E402
from joint.prediction_config import PredictionParams  # noqa: E402

DELTA = 100_000_000            # PredictionParams.arrival_bucket_delta_ns
LF = 1_000_000_000             # arrival_half_life_fast_ns
LS = 30_000_000_000            # arrival_half_life_slow_ns
TAU = 5_000_000_000            # revert_tau_ns
MIN_SAMPLES = 16               # history_min_mature_samples

#: 结构上不得出现的隐藏真值形参（防"顺引用读未来"白名单的负向钉）。
FORBIDDEN_PARAM_NAMES = (
    "decode_length", "final_context_tokens", "prefill_length",
    "inter_request_interval_ns", "history_tokens_before",
    "prefill_context_tokens",
)


def _feed(est, session, values, category=None):
    for v in values:
        est.observe_completed(session_id=session,
                              completed_forward_steps=v, category=category)


# ==================================================== 条件剩余长度（§8.2） ==


class ConditionalRemainderTest(unittest.TestCase):
    """E[N_total−d | N_total>d]：正剩余/删失/回退链/溢出/确定性/白名单。"""

    def test_unfinished_beyond_mean_still_positive(self):
        # d 已超历史均值（均值 4.5）且未结束 ⇒ 剩余仍正（非截零），
        # 且不小于 1（条件于 N>d ⇒ 至少一步，同源式(4a) max(1,n̂)）。
        est = ConditionalRemainingLengthEstimator()
        _feed(est, "s", [1, 2, 3, 4, 5, 6, 7, 8] * 2)
        res = est.estimate_conditional(
            session_id="s", exposed_progress_steps=5)
        self.assertEqual(res.source, "session_conditional")
        self.assertGreaterEqual(res.remaining_forward_steps, 1)
        # 对照：均值−d 截零会给 4.5−5 < 0；条件量保持正。
        mean_minus_d = 4.5 - 5
        self.assertLess(mean_minus_d, 0)
        self.assertGreater(res.remaining_forward_steps, mean_minus_d)

    def test_close_to_max_still_positive(self):
        # 16 个 40 步样本，d=39（贴近最大观测）仍有正剩余。
        est = ConditionalRemainingLengthEstimator()
        _feed(est, "s", [40] * MIN_SAMPLES)
        res = est.estimate_conditional(
            session_id="s", exposed_progress_steps=39)
        self.assertEqual(res.source, "session_conditional")
        self.assertGreaterEqual(res.remaining_forward_steps, 1)

    def test_censored_not_in_completed_samples(self):
        # 删失（活跃未完成）只进删失计数：不入完成样本/直方图/max_observed。
        est = ConditionalRemainingLengthEstimator()
        _feed(est, "s", [10] * MIN_SAMPLES)
        before = est.as_dict()
        version_before = est.version
        est.observe_censored(session_id="s", exposed_progress_steps=1000,
                             category="tool_a")
        after = est.as_dict()
        self.assertEqual(after["run_completed_samples"],
                         before["run_completed_samples"])
        self.assertEqual(after["run_max_observed"],
                         before["run_max_observed"])
        self.assertEqual(after["histogram_overflow_tail_count"], 0)
        self.assertEqual(after["run_censored_count"], 1)
        self.assertEqual(est.version, version_before + 1)
        # 完成样本不受删失影响 ⇒ 估计不走"把活跃请求当短完成样本"的路径。
        res = est.estimate_conditional(
            session_id="s", exposed_progress_steps=5)
        self.assertEqual(res.source, "session_conditional")
        self.assertEqual(res.censored_count, 1)

    def test_integer_bucket_does_not_invent_survivors_at_right_edge(self):
        est = ConditionalRemainingLengthEstimator()
        _feed(est, "s", [4] * 48 + [12] * 16)
        # d=7 时 [0,8) 已无 N>d 的整数。剩余只来自 [8,16)，
        # 桶均值 11.5 按已登记的 floor 规则给出 4；旧实现错误计入短桶质量而得到 3。
        result = est.estimate_conditional(
            session_id="s", exposed_progress_steps=7)
        self.assertEqual(result.remaining_forward_steps, 4)

    def test_bad_length_sample_does_not_partially_change_history(self):
        estimator = ConditionalRemainingLengthEstimator()
        before = estimator.as_dict()
        for value in (True, 1.5, float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimator.observe_completed(session_id="s", completed_forward_steps=value)
            self.assertEqual(estimator.as_dict(), before)
            self.assertEqual(estimator.version, 0)

    def test_no_samples_fallback_positive_tail_prior(self):
        est = ConditionalRemainingLengthEstimator()
        res = est.estimate_conditional(
            session_id="fresh", exposed_progress_steps=0)
        self.assertEqual(res.source, "cold_start_prior")
        self.assertEqual(res.remaining_forward_steps,
                         PredictionParams().cold_start_forward_steps)
        self.assertGreater(res.remaining_forward_steps, 0)
        self.assertEqual(res.fallback_reason, "no_samples")
        self.assertEqual(res.disclosure["cold_start_tail_distribution"],
                         "geometric-remainder-mean32-v1")

    def test_insufficient_samples_fallback(self):
        # 15 < 16 个样本：逐级门槛不满足 ⇒ 正尾先验。
        est = ConditionalRemainingLengthEstimator()
        _feed(est, "s", [10] * (MIN_SAMPLES - 1))
        res = est.estimate_conditional(
            session_id="s", exposed_progress_steps=0)
        self.assertEqual(res.source, "cold_start_prior")
        self.assertGreater(res.remaining_forward_steps, 0)
        self.assertEqual(res.fallback_reason, "insufficient_mature_samples")

    def test_d_beyond_observed_max_fallback(self):
        # d 超过完成样本最大值 ⇒ 回正尾先验（不能把仍活跃请求估为零步）。
        est = ConditionalRemainingLengthEstimator()
        _feed(est, "s", [40] * MIN_SAMPLES)
        res = est.estimate_conditional(
            session_id="s", exposed_progress_steps=50)
        self.assertEqual(res.source, "cold_start_prior")
        self.assertGreaterEqual(res.remaining_forward_steps, 1)
        self.assertEqual(res.fallback_reason, "d_beyond_observed_max")

    def test_fallback_chain_levels(self):
        # session → 类别 → run → 先验：逐级降级，命中即停。
        est = ConditionalRemainingLengthEstimator()
        _feed(est, "hot", [12] * MIN_SAMPLES)                    # session 级
        _feed(est, "other", [16] * MIN_SAMPLES, category="catA")  # 类别级
        # (1) session 级命中。
        res = est.estimate_conditional(
            session_id="hot", exposed_progress_steps=0)
        self.assertEqual(res.source, "session_conditional")
        # (2) 新 session 无样本、类别 catA 有 16 ⇒ 类别级。
        res = est.estimate_conditional(
            session_id="cold_s", exposed_progress_steps=0, category="catA")
        self.assertEqual(res.source, "category_conditional")
        # (3) 无类别键且 session 无样本 ⇒ run 级（32 个完成样本）。
        res = est.estimate_conditional(
            session_id="cold_s2", exposed_progress_steps=0)
        self.assertEqual(res.source, "run_conditional")
        # (4) 全新估计器 ⇒ 先验。
        fresh = ConditionalRemainingLengthEstimator()
        res = fresh.estimate_conditional(
            session_id="x", exposed_progress_steps=0, category="catB")
        self.assertEqual(res.source, "cold_start_prior")
        self.assertEqual(res.fallback_reason, "no_samples")

    def test_histogram_overflow_merges_not_truncates(self):
        # 溢出样本归并尾桶（保 count+sum）：估计不被桶上界截尾。
        est = ConditionalRemainingLengthEstimator()
        bound = (PredictionParams().length_histogram_max_buckets - 1) * (
            PredictionParams().length_histogram_bucket_steps)  # 2040
        _feed(est, "s", [8] * (MIN_SAMPLES - 1) + [1_000_000])
        res = est.estimate_conditional(
            session_id="s", exposed_progress_steps=100)
        self.assertEqual(res.source, "session_conditional")
        # 尾桶均值 1e6 主导：估计远超桶上界（截尾实现会给 ~2040）。
        self.assertGreater(res.remaining_forward_steps, bound)
        self.assertEqual(res.disclosure["histogram_overflow_tail_count"], 1)
        # 生存尾计数（保守整桶口径）：尾桶起点 2040 > 2039 ⇒ 计入；
        # 超过尾桶均值上界（200 万）⇒ 不计。
        self.assertEqual(
            est.survival_count_above(bound - 1, session_id="s"), 1)
        self.assertEqual(est.survival_count_above(2_000_000,
                                                  session_id="s"), 0)

    def test_deterministic_and_readonly(self):
        values = [1, 8, 15, 40, 63, 100, 255, 256, 999, 2040, 5000,
                  7, 33, 64, 128, 300] * 3
        est_a = ConditionalRemainingLengthEstimator()
        est_b = ConditionalRemainingLengthEstimator()
        for v in values:
            for est in (est_a, est_b):
                est.observe_completed(session_id="s", completed_forward_steps=v)
                est.observe_censored(session_id="s", exposed_progress_steps=v)
        self.assertEqual(est_a.version, est_b.version)
        for d in (0, 3, 63, 300, 2039, 4999):
            ra = est_a.estimate_conditional(
                session_id="s", exposed_progress_steps=d)
            rb = est_b.estimate_conditional(
                session_id="s", exposed_progress_steps=d)
            self.assertEqual(ra.remaining_forward_steps,
                             rb.remaining_forward_steps, msg=f"d={d}")
            self.assertEqual(ra.source, rb.source)
            self.assertEqual(ra.level_samples, rb.level_samples)
            self.assertEqual(ra.censored_count, rb.censored_count)
            self.assertEqual(dict(ra.disclosure), dict(rb.disclosure))
        # 估计只读：不 bump version、不改样本。
        version_before = est_a.version
        est_a.estimate_conditional(session_id="s", exposed_progress_steps=10)
        self.assertEqual(est_a.version, version_before)

    def test_hidden_truth_parameters_structurally_rejected(self):
        # 白名单的结构性钉：隐藏真值字段在构造/观测/估计接口无处可传。
        est = ConditionalRemainingLengthEstimator()
        with self.assertRaises(TypeError):
            est.observe_completed(session_id="s",
                                  completed_forward_steps=5,
                                  decode_length=5)
        with self.assertRaises(TypeError):
            est.observe_censored(session_id="s", exposed_progress_steps=3,
                                 final_context_tokens=128)
        with self.assertRaises(TypeError):
            est.estimate_conditional(session_id="s",
                                     exposed_progress_steps=0,
                                     prefill_length=64)
        with self.assertRaises(TypeError):
            ConditionalRemainingLengthEstimator(
                decode_length=8)  # type: ignore[arg-type]
        # 签名层面：公开方法形参不含任何隐藏真值名。
        for name in ("__init__", "observe_completed", "observe_censored",
                     "estimate_conditional"):
            signature = inspect.signature(
                getattr(ConditionalRemainingLengthEstimator, name))
            got = set(signature.parameters)
            self.assertFalse(got & set(FORBIDDEN_PARAM_NAMES),
                             msg=f"{name}: {sorted(got)}")

    def test_params_injection(self):
        # 缺省构造 = PredictionParams()；改 PredictionParams 生效。
        default = ConditionalRemainingLengthEstimator()
        self.assertEqual(default.prior_remainder(),
                         PredictionParams().cold_start_forward_steps)
        self.assertEqual(default.prior_remainder(), 32)
        params = PredictionParams(
            cold_start_forward_steps=7,
            length_histogram_max_buckets=4,
            length_histogram_bucket_steps=1,
            history_min_mature_samples=1)
        est = ConditionalRemainingLengthEstimator(params)
        self.assertEqual(est.prior_remainder(), 7)
        # min_samples=1：单样本即可命中 session 级。
        est.observe_completed(session_id="s", completed_forward_steps=2)
        res = est.estimate_conditional(
            session_id="s", exposed_progress_steps=0)
        self.assertEqual(res.source, "session_conditional")
        # 未知尾部分布 ID：fail-closed，不用静默别名。
        with self.assertRaises(ValueError):
            ConditionalRemainingLengthEstimator(
                PredictionParams(cold_start_tail_distribution="weird-v0"))


# ======================================================== 到达统计（§8.3/8.4） ==


class ArrivalStatisticsTest(unittest.TestCase):
    """互斥类别 + 时间桶快慢强度 + 空桶衰减 + 回归预测。"""

    def test_alpha_derived_matches_hand_computation(self):
        params = PredictionParams()
        stats = ArrivalStatistics(params)
        # α = 1 − 2^(−Δ/L)，只读派生属性（不另行配置 alpha）。
        self.assertAlmostEqual(params.arrival_alpha_fast,
                               1.0 - 2.0 ** (-DELTA / LF), places=15)
        self.assertAlmostEqual(params.arrival_alpha_slow,
                               1.0 - 2.0 ** (-DELTA / LS), places=15)
        # 行为对拍：闭合一个 n=5 的桶后 λ == α·n/Δ（手算锚）。
        for i in range(5):
            stats.observe_arrival(now_ns=i * 1000, turn_index=0)
        stats.advance_to(DELTA)
        lam_f, lam_s = stats.lambdas(("new_session_first", None))
        self.assertAlmostEqual(
            lam_f, params.arrival_alpha_fast * 5 / DELTA, places=15)
        self.assertAlmostEqual(
            lam_s, params.arrival_alpha_slow * 5 / DELTA, places=15)

    def test_empty_buckets_must_decay(self):
        # 连续 10 个空桶：λ == λ0·(1−α)^10（闭式手算；Lf=1e9, Δ=1e8
        # ⇒ 10 桶恰一个半衰期，λ_f 减半）。
        params = PredictionParams()
        stats = ArrivalStatistics(params)
        cat = ("new_session_first", None)
        for i in range(5):
            stats.observe_arrival(now_ns=i * 1000, turn_index=0)
        stats.advance_to(DELTA)
        lam_f0, lam_s0 = stats.lambdas(cat)
        self.assertGreater(lam_f0, 0.0)
        stats.advance_to(11 * DELTA)   # 桶 1..10 全空，闭式批量衰减
        lam_f1, lam_s1 = stats.lambdas(cat)
        self.assertAlmostEqual(
            lam_f1, lam_f0 * (1.0 - params.arrival_alpha_fast) ** 10,
            places=12)
        self.assertAlmostEqual(
            lam_s1, lam_s0 * (1.0 - params.arrival_alpha_slow) ** 10,
            places=12)
        self.assertAlmostEqual(lam_f1, lam_f0 * 0.5, places=12)

    def test_batch_advance_equals_per_bucket_advance(self):
        # 随机到达序列：一次批量推进 vs 逐桶推进，终态等价（对拍）。
        rng = random.Random(20260929)
        counts = [rng.randrange(0, 4) for _ in range(30)]
        times = []
        for k, n in enumerate(counts):
            for j in range(n):
                times.append(k * DELTA + 1000 + j * 7)

        def build(advance_each_bucket):
            stats = ArrivalStatistics()
            idx = 0
            for k, n in enumerate(counts):
                for _ in range(n):
                    stats.observe_arrival(now_ns=times[idx], turn_index=k % 3)
                    idx += 1
                if advance_each_bucket:
                    stats.advance_to((k + 1) * DELTA)
            stats.advance_to(31 * DELTA)
            return stats

        batch = build(False)
        stepwise = build(True)
        snap_a = batch.snapshot()
        snap_b = stepwise.snapshot()
        self.assertEqual(set(snap_a["categories"]),
                         set(snap_b["categories"]))
        for key, st_a in snap_a["categories"].items():
            st_b = snap_b["categories"][key]
            self.assertAlmostEqual(st_a["lambda_fast"], st_b["lambda_fast"],
                                   places=12, msg=key)
            self.assertAlmostEqual(st_a["lambda_slow"], st_b["lambda_slow"],
                                   places=12, msg=key)
            self.assertEqual(st_a["total_arrivals"], st_b["total_arrivals"],
                             msg=key)

    def test_fast_slow_two_scales(self):
        params = PredictionParams()
        # Lf < Ls ⇒ α_f > α_s（快尺度跟得更紧）。
        self.assertLess(params.arrival_half_life_fast_ns,
                        params.arrival_half_life_slow_ns)
        self.assertGreater(params.arrival_alpha_fast,
                           params.arrival_alpha_slow)
        stats = ArrivalStatistics(params)
        cat = ("session_return", "tool_a")
        for i in range(8):
            stats.observe_arrival(now_ns=i * 500, turn_index=2,
                                  revealed_type="tool_a")
        stats.advance_to(DELTA)
        lam_f, lam_s = stats.lambdas(cat)
        # 同一突发桶：λ_f = α_f·r > λ_s = α_s·r。
        self.assertGreater(lam_f, lam_s)
        self.assertGreater(lam_s, 0.0)

    def test_mutually_exclusive_classification(self):
        # 三类事件：新 session 到达、session 返回、队列占位（非到达）——
        # 每个真实到达恰更新一个互斥类别；非到达不进任何类别。
        stats = ArrivalStatistics()
        self.assertEqual(
            ArrivalStatistics.classify_arrival(turn_index=0),
            (ARRIVAL_CLASS_NEW_SESSION_FIRST, None))
        self.assertEqual(
            ArrivalStatistics.classify_arrival(
                turn_index=3, revealed_type="tool_a"),
            (ARRIVAL_CLASS_SESSION_RETURN, "tool_a"))
        with self.assertRaises(ValueError):
            ArrivalStatistics.classify_arrival(turn_index=-1)

        key_new = stats.observe_arrival(now_ns=10, turn_index=0)
        snap1 = stats.snapshot()
        self.assertEqual(snap1["categories"]["new_session_first|"]
                         ["total_arrivals"], 1)
        key_ret = stats.observe_arrival(now_ns=20, turn_index=1,
                                        revealed_type="tool_a")
        stats.note_excluded_event(kind="current_queue", now_ns=30)
        stats.note_excluded_event(kind="current_request_remaining_decode",
                                  now_ns=31)
        stats.note_excluded_event(kind="submitted_communications", now_ns=32)
        snap2 = stats.snapshot()
        # 恰更新一类：new 类别计数不变，return 类别 +1，无第三类别生长。
        self.assertEqual(snap2["categories"]["new_session_first|"]
                         ["total_arrivals"], 1)
        self.assertEqual(snap2["categories"]["session_return|tool_a"]
                         ["total_arrivals"], 1)
        self.assertEqual(len(snap2["categories"]), 2)
        # 排除事件只落披露计数。
        self.assertEqual(snap2["excluded_event_counts"]["current_queue"], 1)
        self.assertEqual(
            snap2["excluded_event_counts"]
            ["current_request_remaining_decode"], 1)
        self.assertEqual(snap2["excluded_event_counts"]
                         ["submitted_communications"], 1)
        # 未闭合桶规则 = full_buckets_only：桶内计数不进 λ。
        self.assertEqual(snap2["categories"]["new_session_first|"]
                         ["open_bucket_count"], 1)
        lam_new = stats.lambdas(key_new)
        lam_ret = stats.lambdas(key_ret)
        self.assertEqual(lam_new, (0.0, 0.0))
        self.assertEqual(lam_ret, (0.0, 0.0))

    def test_excluded_events_do_not_update_intensity(self):
        # 同桶内的非到达事件不改变任何 λ / 到达计数。
        stats = ArrivalStatistics()
        cat = ("new_session_first", None)
        stats.observe_arrival(now_ns=10, turn_index=0)
        stats.advance_to(2 * DELTA)
        before = stats.snapshot()
        stats.note_excluded_event(kind="current_queue",
                                  now_ns=2 * DELTA + 5)
        after = stats.snapshot()
        self.assertEqual(
            after["categories"]["new_session_first|"]["lambda_fast"],
            before["categories"]["new_session_first|"]["lambda_fast"])
        self.assertEqual(
            after["categories"]["new_session_first|"]["lambda_slow"],
            before["categories"]["new_session_first|"]["lambda_slow"])
        self.assertEqual(
            after["categories"]["new_session_first|"]["total_arrivals"],
            before["categories"]["new_session_first|"]["total_arrivals"])
        records = stats.exclusion_records()
        self.assertEqual(records["counts"], {"current_queue": 1})
        self.assertEqual(records["not_new_arrival_reasons"],
                         ["current_queue",
                          "current_request_remaining_decode",
                          "submitted_communications"])

    def test_reverting_formula_endpoints(self):
        # h=0 退化 λ_fast；h→∞ 收敛 λ_slow；中点与闭式手算一致。
        stats = ArrivalStatistics()
        cat = ("new_session_first", None)
        for i in range(6):
            stats.observe_arrival(now_ns=i * 1000, turn_index=0)
        stats.advance_to(2 * DELTA)
        lam_f, lam_s = stats.lambdas(cat)
        self.assertGreater(lam_f, lam_s)
        self.assertEqual(stats.intensity_recent(cat), lam_f)
        self.assertEqual(stats.intensity_reverting(cat, 0), lam_f)
        self.assertAlmostEqual(
            stats.intensity_reverting(cat, 50 * TAU), lam_s, places=18)
        h = TAU
        self.assertAlmostEqual(
            stats.intensity_reverting(cat, h),
            lam_s + math.exp(-h / TAU) * (lam_f - lam_s), places=15)
        h2 = 2.5 * TAU
        self.assertAlmostEqual(
            stats.intensity_reverting(cat, h2),
            lam_s + math.exp(-h2 / TAU) * (lam_f - lam_s), places=15)

    def test_open_bucket_full_buckets_only(self):
        # 登记规则：开放桶部分计数不进 λ，直至闭合（缺省完整桶规则）。
        stats = ArrivalStatistics()
        self.assertEqual(stats.OPEN_BUCKET_RULE, "full_buckets_only")
        cat = ("new_session_first", None)
        for i in range(3):
            stats.observe_arrival(now_ns=i * 100, turn_index=0)
        self.assertEqual(stats.lambdas(cat), (0.0, 0.0))
        snap = stats.snapshot()
        self.assertEqual(snap["open_bucket_rule"], "full_buckets_only")
        self.assertEqual(snap["categories"]["new_session_first|"]
                         ["open_bucket_count"], 3)
        stats.advance_to(DELTA)
        lam_f, lam_s = stats.lambdas(cat)
        alpha_f = PredictionParams().arrival_alpha_fast
        self.assertAlmostEqual(lam_f, alpha_f * 3 / DELTA, places=15)
        self.assertAlmostEqual(lam_s,
                               PredictionParams().arrival_alpha_slow * 3
                               / DELTA, places=15)

    def test_causal_snapshot_and_observed_until(self):
        # 统计只依赖 event_time ≤ observed_until 的事件（双实例重放等价）；
        # 已取出的快照不被后续"未来"输入改写。
        stats = ArrivalStatistics()
        stats.observe_arrival(now_ns=1_000, turn_index=0)
        stats.observe_arrival(now_ns=DELTA + 5_000, turn_index=1,
                              revealed_type="tool_a")
        stats.advance_to(2 * DELTA)
        self.assertEqual(stats.observed_until_ns, 2 * DELTA)
        snapshot_at_t = stats.snapshot()
        # 未来输入（T 之后的到达）不改写 T 时刻已取出的快照。
        stats.observe_arrival(now_ns=2 * DELTA + 1_000, turn_index=0)
        stats.advance_to(4 * DELTA)
        self.assertEqual(snapshot_at_t, snapshot_at_t)  # 冻结副本
        replay = ArrivalStatistics()
        replay.observe_arrival(now_ns=1_000, turn_index=0)
        replay.observe_arrival(now_ns=DELTA + 5_000, turn_index=1,
                               revealed_type="tool_a")
        replay.advance_to(2 * DELTA)
        self.assertEqual(replay.snapshot(), snapshot_at_t)
        # 时间倒退 fail-closed。
        with self.assertRaises(ValueError):
            stats.observe_arrival(now_ns=DELTA, turn_index=0)

    def test_advance_idempotent(self):
        stats = ArrivalStatistics()
        stats.observe_arrival(now_ns=5_000, turn_index=0)
        stats.advance_to(3 * DELTA)
        snap1, version1 = stats.snapshot(), stats.version
        stats.advance_to(3 * DELTA)          # 重复推进同刻：无新闭合
        snap_again = stats.snapshot()
        self.assertEqual(snap1["categories"], snap_again["categories"])
        self.assertEqual(stats.version, version1)
        # 同桶内推进：不闭合任何桶，类别统计不变（observed_until 前移）。
        stats.advance_to(3 * DELTA + 50)
        snap2 = stats.snapshot()
        self.assertEqual(snap1["categories"], snap2["categories"])
        self.assertEqual(snap2["observed_until_ns"], 3 * DELTA + 50)

    def test_params_injection_arrival(self):
        # Δ/半衰期改 PredictionParams 生效（桶边界随 Δ 变化）。
        params = PredictionParams(
            arrival_bucket_delta_ns=50,
            arrival_half_life_fast_ns=100,
            arrival_half_life_slow_ns=1000)
        stats = ArrivalStatistics(params)
        for i in range(5):
            stats.observe_arrival(now_ns=1 + i * 10, turn_index=0)
        stats.advance_to(50)
        lam_f, _ = stats.lambdas(("new_session_first", None))
        alpha_f = 1.0 - 2.0 ** (-50 / 100)
        self.assertAlmostEqual(lam_f, alpha_f * 5 / 50, places=15)

    def test_no_implicit_total_category(self):
        # 无隐式总到达率类别：聚合 = 互斥类别求和（避免总率+类别率双计）。
        stats = ArrivalStatistics()
        stats.observe_arrival(now_ns=1_000, turn_index=0)
        stats.observe_arrival(now_ns=2_000, turn_index=4)
        stats.advance_to(DELTA)
        snap = stats.snapshot()
        self.assertEqual(set(snap["categories"]),
                         {"new_session_first|", "session_return|"})
        lam_new = stats.lambdas(("new_session_first", None))
        lam_ret = stats.lambdas(("session_return", None))
        self.assertGreater(lam_new[0], 0.0)
        self.assertGreater(lam_ret[0], 0.0)


# ======================================================== 服务样本账本（§8.1） ==


def _sample(**overrides):
    base = dict(work_id="w", category="decode", observation_tick_ns=10,
                baseline_work_ns=100, busy_service_ns=200)
    base.update(overrides)
    return ServiceWorkSample(**base)


class ServiceSampleRecorderTest(unittest.TestCase):
    """纯度门 + 1/16 EWMA（params 注入）+ 与到达衰减分离。"""

    def test_non_unit_efficiency_prior_is_a_time_slowdown(self):
        params = PredictionParams(service_efficiency_init=0.5)
        recorder = ServiceSampleRecorder(params)
        self.assertEqual(recorder.efficiency_factor("decode"), 2.0)
        recorder.observe(_sample(baseline_work_ns=100, busy_service_ns=200))
        self.assertEqual(recorder.efficiency_factor("decode"), 2.0)

    def test_pure_sample_updates_efficiency_hand_anchor(self):
        rec = ServiceSampleRecorder()
        rec.observe(_sample(work_id="w1"))
        rec.flush()
        # η1 = 15/16·1 + 1/16·(100/200)；因子 = 1/η1（手算锚）。
        eta1 = (15.0 / 16.0) * 1.0 + (1.0 / 16.0) * 0.5
        self.assertAlmostEqual(rec.efficiency_factor("decode"),
                               1.0 / eta1, places=12)
        self.assertEqual(rec.updates["decode"], 1)
        rec.observe(_sample(work_id="w2", observation_tick_ns=20,
                            busy_service_ns=400))
        rec.flush()
        eta2 = (15.0 / 16.0) * eta1 + (1.0 / 16.0) * 0.25
        self.assertAlmostEqual(rec.efficiency_factor("decode"),
                               1.0 / eta2, places=12)

    def test_contended_sample_rejected_and_counted(self):
        rec = ServiceSampleRecorder()
        rec.observe(_sample(work_id="c1", foreign_participants=2))
        rec.flush()
        self.assertEqual(rec.contended["decode"], 1)
        self.assertEqual(rec.efficiency_factor("decode"), 1.0)
        self.assertNotIn("decode", rec.updates)
        residuals = rec.as_dict()["residuals"]
        self.assertEqual(residuals["decode|contended"]["count"], 1)
        self.assertEqual(residuals["decode|contended"]["sum_actual_ns"], 200)
        self.assertEqual(residuals["decode|contended"]["sum_baseline_ns"],
                         100)

    def test_unseparable_waits_recorded_as_residual(self):
        rec = ServiceSampleRecorder()
        rec.observe(_sample(work_id="u1", queue_wait_ns=50))
        rec.observe(_sample(work_id="u2", data_wait_ns=10))
        rec.flush()
        # 不吸收成"有效带宽"：η 不更新，残差按原因聚合。
        self.assertEqual(rec.efficiency_factor("decode"), 1.0)
        residuals = rec.as_dict()["residuals"]
        self.assertEqual(residuals["decode|unseparable_waits"]["count"], 2)
        self.assertEqual(residuals["decode|unseparable_waits"]
                         ["sum_actual_ns"], 400)

    def test_anomalous_and_rejected_discipline(self):
        rec = ServiceSampleRecorder()
        # η_sample > 1（实测快于 roofline）= 记量异常：计数跳过，不钳位。
        rec.observe(_sample(work_id="a1", busy_service_ns=50))
        rec.flush()
        self.assertEqual(rec.anomalous["decode"], 1)
        self.assertEqual(rec.efficiency_factor("decode"), 1.0)
        # 零时长样本：拒收计数。
        rec.observe(_sample(work_id="a2", busy_service_ns=0))
        rec.flush()
        self.assertEqual(rec.rejected["decode"], 1)
        self.assertEqual(rec.efficiency_factor("decode"), 1.0)

    def test_same_tick_aggregation(self):
        rec = ServiceSampleRecorder()
        rec.observe(_sample(work_id="s1", observation_tick_ns=10,
                            busy_service_ns=300))
        rec.observe(_sample(work_id="s2", observation_tick_ns=10,
                            baseline_work_ns=100, busy_service_ns=100))
        rec.flush()
        # 同 tick 汇总：η_sample = Σbase/Σactual = 200/400 = 0.5（一条更新）。
        eta1 = (15.0 / 16.0) * 1.0 + (1.0 / 16.0) * 0.5
        self.assertAlmostEqual(rec.efficiency_factor("decode"),
                               1.0 / eta1, places=12)
        self.assertEqual(rec.updates["decode"], 1)

    def test_service_sample_identity_is_idempotent_after_flush(self):
        rec = ServiceSampleRecorder()
        sample = _sample(work_id="train:1")
        self.assertTrue(rec.observe(sample))
        before = rec.efficiency_factor("decode")
        version = rec.version
        self.assertFalse(rec.observe(sample))
        self.assertEqual(rec.efficiency_factor("decode"), before)
        self.assertEqual(rec.updates["decode"], 1)
        self.assertEqual(rec.version, version)
        self.assertTrue(rec.observe(_sample(work_id="train:2")))
        rec.flush()
        self.assertEqual(rec.updates["decode"], 2)
        with self.assertRaises(ValueError):
            rec.observe(_sample(work_id="train:2", busy_service_ns=300))
        rec.observe(_sample(work_id="train:3", observation_tick_ns=20))
        self.assertFalse(rec.observe(sample))
        self.assertEqual(rec.as_dict()["stale_samples"], 1)

    def test_service_samples_reject_invalid_waits_and_time(self):
        for field in ("queue_wait_ns", "data_wait_ns", "capacity_wait_ns",
                      "foreign_participants", "observation_tick_ns"):
            for value in (-1, 1.5, True, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _sample(**{field: value})

    def test_hardware_version_fork_marks_epoch_once(self):
        # §8.5 挂点（2026-09-30 修复）：样本硬件版本混分叉 = 分布变化
        # ——恰一次 mark_epoch_change(kind="hardware")，不重置统计
        # （no_reset_quarantine 规则在 EpochLog 侧）；同版本延续不重复
        # 登记；未注入 epoch_log 时只登记版本集合、零行为。
        params = PredictionParams()
        log = EpochLog(params)
        rec = ServiceSampleRecorder(params, epoch_log=log)
        rec.observe(_sample(work_id="h1", observation_tick_ns=100,
                            hardware_version="sem-v1"))
        rec.observe(_sample(work_id="h2", observation_tick_ns=200,
                            hardware_version="sem-v1"))
        self.assertEqual(log.epochs, [])
        rec.observe(_sample(work_id="h3", observation_tick_ns=300,
                            hardware_version="sem-v2"))
        self.assertEqual(len(log.epochs), 1)
        epoch = log.epochs[0]
        self.assertEqual(epoch["kind"], "hardware")
        self.assertEqual(epoch["at_ns"], 300)
        self.assertEqual(
            epoch["detail"]["known_versions"], ["sem-v1", "sem-v2"])
        self.assertEqual(epoch["reset_rule"], EpochLog.EPOCH_RESET_RULE)
        rec.observe(_sample(work_id="h4", observation_tick_ns=400,
                            hardware_version="sem-v2"))
        self.assertEqual(len(log.epochs), 1)
        # 争用/不可分离样本同样携带版本证据（纯度裁决前登记）。
        rec.observe(_sample(work_id="h5", observation_tick_ns=500,
                            hardware_version="sem-v3",
                            foreign_participants=1))
        self.assertEqual(len(log.epochs), 2)
        self.assertEqual(log.epochs[1]["at_ns"], 500)
        # 争用样本不进 η（纯度门不变；h1–h4 均为有效样本）——版本登记
        # 不改变估计语义。
        rec.flush()
        self.assertEqual(rec.updates.get("decode"), 4)
        self.assertEqual(rec.contended, {"decode": 1})
        # 无 epoch_log 注入：版本集合照常登记、零 epoch 行为。
        rec_solo = ServiceSampleRecorder()
        rec_solo.observe(_sample(work_id="h1", hardware_version="x"))
        rec_solo.observe(_sample(work_id="h2", hardware_version="y"))
        self.assertEqual(rec_solo.as_dict()["hardware_versions"],
                         ["x", "y"])

    def test_ewma_weight_from_params(self):
        params = PredictionParams(service_efficiency_ewma_weight=0.25)
        rec = ServiceSampleRecorder(params)
        rec.observe(_sample(work_id="p1"))
        rec.observe(_sample(work_id="p2", observation_tick_ns=20,
                            busy_service_ns=400))
        rec.flush()
        eta1 = 0.75 * 1.0 + 0.25 * 0.5
        eta2 = 0.75 * eta1 + 0.25 * 0.25
        self.assertAlmostEqual(rec.efficiency_factor("decode"),
                               1.0 / eta2, places=12)

    def test_same_source_as_existing_service_factors(self):
        # 语义同源锚：默认参数下与现有 ServiceFactors 同样本流逐点全等。
        rec = ServiceSampleRecorder()
        factors = ServiceFactors()
        stream = [
            (300, 100, 10), (200, 100, 10),   # 同 tick 汇总
            (500, 100, 20),
            (150, 100, 30), (250, 200, 30),
            (1000, 400, 40),
        ]
        for i, (actual, base, tick) in enumerate(stream):
            rec.observe(_sample(work_id=f"w{i}",
                                observation_tick_ns=tick,
                                baseline_work_ns=base,
                                busy_service_ns=actual))
            factors.observe_decode(actual_ns=actual, base_ns=base,
                                   tick_ns=tick)
        rec.flush()
        factors.flush()
        self.assertAlmostEqual(rec.efficiency_factor("decode"),
                               factors.decode_factor, places=12)
        self.assertEqual(rec.updates["decode"],
                         factors.updates["decode_factor"])
        # 争用拒收通道同源：两边同样不更新 η。
        rec.observe(_sample(work_id="cx", observation_tick_ns=50,
                            foreign_participants=1))
        factors.reject_contended("decode_factor")
        self.assertEqual(rec.contended["decode"],
                         factors.contended["decode_factor"])
        rec.flush()
        self.assertAlmostEqual(rec.efficiency_factor("decode"),
                               factors.decode_factor, places=12)

    def test_efficiency_independent_of_arrival_decay(self):
        # 双向独立性：改 service 权重不动到达 α/λ；改到达半衰期不动效率轨迹。
        base_params = PredictionParams()
        service_changed = PredictionParams(
            service_efficiency_ewma_weight=0.5)
        arrival_changed = PredictionParams(
            arrival_half_life_fast_ns=2_000_000_000)
        # (1) 到达侧：半衰期改变 ⇒ α_f 改变；service 权重改变 ⇒ α_f 不变。
        self.assertNotEqual(arrival_changed.arrival_alpha_fast,
                            base_params.arrival_alpha_fast)
        self.assertEqual(service_changed.arrival_alpha_fast,
                         base_params.arrival_alpha_fast)
        stats_a = ArrivalStatistics(base_params)
        stats_b = ArrivalStatistics(service_changed)
        for stats in (stats_a, stats_b):
            for i in range(4):
                stats.observe_arrival(now_ns=i * 100, turn_index=0)
            stats.advance_to(DELTA)
        self.assertEqual(stats_a.snapshot()["categories"],
                         stats_b.snapshot()["categories"])
        # (2) 效率侧：到达半衰期改变 ⇒ 同样本流效率轨迹不变。
        rec_a = ServiceSampleRecorder(base_params)
        rec_b = ServiceSampleRecorder(arrival_changed)
        for i, (actual, base, tick) in enumerate(
                [(200, 100, 10), (400, 100, 20), (800, 200, 30)]):
            for rec in (rec_a, rec_b):
                rec.observe(_sample(work_id=f"w{i}",
                                    observation_tick_ns=tick,
                                    baseline_work_ns=base,
                                    busy_service_ns=actual))
        rec_a.flush()
        rec_b.flush()
        self.assertEqual(rec_a.efficiency_factor("decode"),
                         rec_b.efficiency_factor("decode"))
        # 权重来源披露钉死：params.service_efficiency_ewma_weight。
        self.assertEqual(rec_a.as_dict()["ewma_weight_source"],
                         "params.service_efficiency_ewma_weight")


# ==================================================== 成熟/回退/epoch（§8.5） ==


class MaturityAndFallbackTest(unittest.TestCase):
    """issued/matures 恰一次 + 过期 + 可复现回退 + epoch 隔离。"""

    def test_immature_forecast_never_updates_error(self):
        ledger = ForecastMaturityLedger()
        ledger.issue(forecast_id="f1", issued_at_ns=0, matures_at_ns=100,
                     origin="arrival:new_session_first", predicted=5.0)
        newly = ledger.mature_entries(now_ns=50, actuals={"f1": 4.0})
        self.assertEqual(newly, [])
        self.assertFalse(ledger.entry("f1").matured)
        self.assertEqual(ledger.matured_count(), 0)
        self.assertIsNone(ledger.mean_error())
        self.assertEqual(len(ledger.pending_entries()), 1)

    def test_matures_exactly_once_idempotent(self):
        ledger = ForecastMaturityLedger()
        ledger.issue(forecast_id="f1", issued_at_ns=0, matures_at_ns=100,
                     origin="arrival:session_return", predicted=5.0)
        newly = ledger.mature_entries(now_ns=100, actuals={"f1": 4.0})
        self.assertEqual(len(newly), 1)
        self.assertTrue(ledger.entry("f1").matured)
        self.assertEqual(ledger.entry("f1").error, 1.0)
        self.assertEqual(ledger.matured_count(), 1)
        self.assertEqual(ledger.mean_error(), 1.0)
        # 重复成熟（更晚时刻、同真值）：幂等，不二次更新。
        again = ledger.mature_entries(now_ns=200, actuals={"f1": 4.0})
        self.assertEqual(again, [])
        self.assertEqual(ledger.matured_count(), 1)
        self.assertEqual(ledger.entry("f1").error, 1.0)
        self.assertEqual(ledger.mean_error(), 1.0)

    def test_interval_passed_but_actual_missing_stays_pending(self):
        ledger = ForecastMaturityLedger()
        ledger.issue(forecast_id="f1", issued_at_ns=0, matures_at_ns=100,
                     origin="arrival:new_session_first", predicted=5.0)
        newly = ledger.mature_entries(now_ns=150, actuals={})
        self.assertEqual(newly, [])
        self.assertFalse(ledger.entry("f1").matured)
        self.assertEqual(len(ledger.pending_entries()), 1)
        # 事件完成后（真值揭示）才结算。
        newly = ledger.mature_entries(now_ns=160, actuals={"f1": 6.0})
        self.assertEqual(len(newly), 1)
        self.assertEqual(ledger.entry("f1").error, -1.0)

    def test_expired_records_get_reproducible_fallback(self):
        params = PredictionParams(history_sample_validity_ns=1000)
        ledger = ForecastMaturityLedger(params)
        ledger.issue(forecast_id="old", issued_at_ns=0, matures_at_ns=10,
                     origin="arrival:new_session_first", predicted=5.0)
        newly = ledger.mature_entries(now_ns=5000, actuals={"old": 4.0})
        self.assertEqual(newly, [])
        entry = ledger.entry("old")
        self.assertTrue(entry.expired)
        self.assertFalse(entry.matured)
        self.assertEqual(ledger.matured_count(), 0)
        record = entry.fallback
        self.assertIsNotNone(record)
        self.assertEqual(record.code, FallbackCode.EXPIRED)
        self.assertEqual(record.detail["forecast_id"], "old")
        self.assertEqual(record.detail["issued_at_ns"], 0)
        self.assertEqual(record.detail["validity_ns"], 1000)
        # 可复现：同字段重建 ⇒ 同 reproduce_key。
        rebuilt = FallbackRecord(
            code=FallbackCode.EXPIRED, at_ns=5000, origin=record.origin,
            detail=dict(record.detail))
        self.assertEqual(record.reproduce_key(), rebuilt.reproduce_key())
        self.assertIn(
            {"code": "expired", "at_ns": 5000, "origin": record.origin,
             "detail": dict(record.detail)},
            ledger.as_dict()["fallbacks"])

    def test_fallback_codes_reproducible(self):
        details = {
            FallbackCode.NO_SAMPLES: {"level": "run", "samples": 0},
            FallbackCode.ANOMALY: {"metric": "lambda_fast",
                                   "value": float("inf")},
            FallbackCode.EXPIRED: {"forecast_id": "f"},
            FallbackCode.TEMPLATE_UNCOVERED: {"category": "tool_x"},
            FallbackCode.BUDGET_EXCEEDED: {"budget": 262144},
            FallbackCode.DISTRIBUTION_SHIFT: {"kind": "policy"},
            FallbackCode.UNVALIDATED_DISTANCE: {"horizon_ns": 10 ** 12},
        }
        self.assertEqual(len(details), 7)
        records = [FallbackRecord(code=code, at_ns=7, origin="test",
                                  detail=detail)
                   for code, detail in details.items()]
        for record in records:
            twin = FallbackRecord(code=record.code, at_ns=7,
                                  origin="test", detail=dict(record.detail))
            self.assertEqual(record.reproduce_key(), twin.reproduce_key())
        # 不同 detail / code ⇒ 不同键（可区分、可复现）。
        keys = {record.reproduce_key() for record in records}
        self.assertEqual(len(keys), 7)
        mutated = FallbackRecord(
            code=FallbackCode.NO_SAMPLES, at_ns=7, origin="test",
            detail={"level": "session", "samples": 0})
        self.assertNotEqual(mutated.reproduce_key(),
                            records[0].reproduce_key())

    def test_ledger_discipline_fail_closed(self):
        ledger = ForecastMaturityLedger()
        ledger.issue(forecast_id="f1", issued_at_ns=10, matures_at_ns=20,
                     origin="arrival:new_session_first", predicted=1.0)
        with self.assertRaises(ValueError):
            ledger.issue(forecast_id="f1", issued_at_ns=30, matures_at_ns=40,
                         origin="x", predicted=1.0)  # 重复 id
        with self.assertRaises(ValueError):
            ledger.issue(forecast_id="f2", issued_at_ns=40, matures_at_ns=30,
                         origin="x", predicted=1.0)  # matures < issued
        with self.assertRaises(ValueError):
            ledger.issue(forecast_id="f3", issued_at_ns=0, matures_at_ns=1,
                         origin="x", predicted=1.0,
                         exclusion_set=("not_a_registered_reason",))
        ledger.mature_entries(now_ns=100, actuals={})
        with self.assertRaises(ValueError):
            ledger.mature_entries(now_ns=50, actuals={})  # 时间倒退

    def test_epoch_quarantine_and_frozen_reset_rule(self):
        params = PredictionParams(history_drift_quarantine_ns=1000)
        log = EpochLog(params)
        self.assertFalse(log.in_quarantine(0))
        self.assertIsNone(log.fallback_if_blocked(0))
        log.mark_epoch_change(kind="policy", at_ns=500,
                              detail={"axis": "layer_policy"})
        self.assertEqual(log.EPOCH_RESET_RULE, "no_reset_quarantine")
        self.assertTrue(log.in_quarantine(1499))
        self.assertFalse(log.in_quarantine(1500))
        record = log.fallback_if_blocked(600)
        self.assertIsNotNone(record)
        self.assertEqual(record.code, FallbackCode.DISTRIBUTION_SHIFT)
        self.assertEqual(record.detail["kind"], "policy")
        self.assertEqual(record.detail["reset_rule"],
                         "no_reset_quarantine")
        self.assertIsNone(log.fallback_if_blocked(2000))
        # epoch 时刻单调 fail-closed。
        with self.assertRaises(ValueError):
            log.mark_epoch_change(kind="config", at_ns=400)
        self.assertEqual(log.as_dict()["epochs"][0]["kind"], "policy")

    def test_min_mature_samples_gate_material(self):
        # W7 准入素材：成熟误差样本数可与 history_min_mature_samples 比较；
        # 未成熟/过期不进 matured 集合。
        ledger = ForecastMaturityLedger()
        for i in range(MIN_SAMPLES):
            ledger.issue(forecast_id=f"f{i}", issued_at_ns=0,
                         matures_at_ns=10 + i, origin="arrival:total",
                         predicted=float(i))
        ledger.issue(forecast_id="late", issued_at_ns=0, matures_at_ns=10 ** 9,
                     origin="arrival:total", predicted=0.0)
        actuals = {f"f{i}": float(i) + 1.0 for i in range(MIN_SAMPLES)}
        ledger.mature_entries(now_ns=100, actuals=actuals)
        self.assertEqual(ledger.matured_count(), MIN_SAMPLES)
        self.assertGreaterEqual(ledger.matured_count(),
                                PredictionParams().history_min_mature_samples)
        self.assertEqual(len(ledger.matured_entries()), MIN_SAMPLES)
        self.assertEqual(len(ledger.pending_entries()), 1)
        self.assertEqual(ledger.mean_error(), -1.0)


# ============================================================ shadow 纪律 ==


class ShadowDisciplineTest(unittest.TestCase):
    """shadow：模块无调度调用面（不 import 调度器/图/账本）。"""

    def test_module_imports_no_scheduler_surface(self):
        source = inspect.getsource(history_estimator)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                self.assertNotIn("online", stripped, msg=stripped)
                self.assertNotIn("face_scheduler", stripped, msg=stripped)
                self.assertNotIn("sh30", stripped, msg=stripped)

    def test_module_docstring_registers_shadow_contract(self):
        doc = history_estimator.__doc__
        self.assertIn("shadow", doc)
        self.assertIn("前向步", doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
