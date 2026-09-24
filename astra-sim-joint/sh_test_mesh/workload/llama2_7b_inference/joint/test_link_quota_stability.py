#!/usr/bin/env python3
"""test_link_quota_stability.py -- WP3b AIMD 控制律 + 五剧本零发射
稳定性套件（C10 卡；零后端、合成遥测字典）。

范式：零发射合成剧本（轻量构造范式，参考 online/
test_remote_credit_stream.py:56-93 的 ``__new__`` 构造——重型调度器
被绕过、只构造被测路径所需的最小状态；本套件被测对象 = 纯 Python
的 ``LinkQuotaTracker``，直接构造 + 合成事件钟 + 合成遥测字典
``{link_id: 实测有效速率}``（§2.1 冻结契约，SH 解析归 C8、接线归
C11），不触碰 SH/JCM/后端。合成负载模型只作单测稳定性证据（设计
文档 §4.4"不能以'有迟滞'代替稳定性证据"；实验思路 §7.3 P2"只改
queue/占用计数的合成输入可作单测，不能作为真实性能证据"）。

五剧本（卡内清单）与断言 (a)–(e) 的映射：

1. ``ArrivalBurstScenarioTest`` 到达突发 —— (c) 突发点不过饱和
  （链路占用+预留 ≤ Q、bulk ≤ N_bulk、平价门逐笔、原子零部分登记）
  + 同 tick 零 dt 遥测采样；
2. ``LongTrainScenarioTest`` 长列车 —— (a) 收缩不发散（Q ≥ 1、
   在册 grandfathered 不逐出、负余量封新准入、排空恢复）+ 流寿命
   EWMA 配对/边界规则 + 时钟纪律 fail-closed；
3. ``MergeStampedeScenarioTest`` merge 挤兑 —— 并发预留挤兑下
   N_bulk/链路信用封顶 + 借还配对零泄漏 + 交错裁决（(c) 端口侧）；
4. ``EwmaQuotaCouplingScenarioTest`` EWMA 服务因子与配额耦合 ——
   (a) 不发散（Q ∈ [1, floor(B_link/r_KV)]、收敛到不动点）/
   (b) 阈值带不重合下无极限环（触发集不相交 + 死区平衡 + 上界
   封顶 + 外生扰动后单调恢复）/ (e) EWMA→决策→负载反馈回路不振荡
   （保载-卸载-回载-挤压-恢复全程零自持方向翻转）；
5. ``DeferredRetryChainScenarioTest`` quota_deferred 重试链 ——
   (d) 信用释放必唤醒 deferred（释放侧事件逐个恰 bump +1 代数）、
   无漏唤醒/重复唤醒（非释放侧事件零 bump）、唤醒后再准入闭合。

AIMD 冻结常数钉：k=10、带 [r_KV, 1.2×r_KV] 不重合、MD 减半（裁定
披露）、EWMA α=1/8、扩张上界 = max(1, floor(B_link/r_KV))。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_link_quota_stability.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT, os.path.join(_PARENT, "online")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.link_quota import (  # noqa: E402
    AIMD_ACTION_COMFORT,
    AIMD_ACTION_COMFORT_COLD_START,
    AIMD_ACTION_EXPAND,
    AIMD_ACTION_EXPAND_CAPPED,
    AIMD_ACTION_HOLD,
    AIMD_ACTION_SHRINK,
    AIMD_ACTION_SHRINK_FLOOR,
    AIMD_BAND_UPPER_FACTOR,
    AIMD_EXPAND_K,
    AIMD_SHRINK_DIVISOR,
    AIMD_SIGNAL_COMFORT,
    AIMD_SIGNAL_HOLD,
    AIMD_SIGNAL_SHRINK,
    FLOW_ELASTIC,
    FLOW_LIFETIME_EWMA_ALPHA,
    FLOW_ONESHOT,
    FLOW_REALTIME,
    MERGE_DIRECTION_FORWARD,
    MERGE_DIRECTION_REVERSE,
    QUOTA_AIMD,
    QUOTA_OFF,
    QUOTA_STATIC,
    WAIT_CAPACITY,
    WAIT_QUOTA_LINK,
    WAIT_QUOTA_PORT,
    FlowLifetimeEwma,
    LinkQuotaError,
    LinkQuotaTracker,
    aimd_band_signal,
    aimd_expand_ceiling,
    aimd_shrink_quota,
    quota_deferred_requeue,
)

#: 合成锚点（bytes/ns 量纲，与 JCM 单位合同一致）：B_link=240、
#: B_HBM=100 → ρ=2.4 → Q_init=2；r_KV=10 → 阈值带 [10, 12)、
#: 收缩等值线/扩张上界 = floor(240/10) = 24。
_B_NOC = 240.0
_B_HBM = 100.0
_R_KV = 10.0
_CEILING = 24

#: 闭环剧本专用 B_HBM：ρ = 240/1000 < 1 → Q_init = 1（max(1,·) 下限
#: 派生），端口平价门 u ≤ 99 全程不绑定（隔离链路控制律）。
_B_HBM_CLOSED = 1000.0


def _tracker(mode=QUOTA_AIMD, noc=_B_NOC, hbm=_B_HBM, **kwargs):
    return LinkQuotaTracker(
        mode=mode, noc_link_bytes_per_ns=noc, local_hbm_bytes_per_ns=hbm,
        **kwargs)


class _SyntheticLinkLoad:
    """合成闭环链路负载（零发射范式）：深需求实时流 + 寿命结算 +
    服务遥测。

    每 tick 事件序：补足准入（深需求 = 永有等待者）→ 结算到期流
    （流 settle 喂 EWMA）→ 遥测采样 ``rate = B_link / (occupancy +
    bg_divisor)``（在册流 + 外部背景除数合并分母；occupancy +
    bg == 0 时无在册流 → 链路不出现在遥测字典 = 无测量无信号，
    §4.1 因果口径）。时钟与在册表跨 run 延续；bg_divisor 只影响
    遥测分母（外部竞争的合成注入，不改配额簿记）。
    """

    def __init__(self, tracker, link=(0, 1), port=5,
                 r_hat_kv=_R_KV, tick_ns=400, lifetimes=(800, 1200),
                 noc=_B_NOC):
        self.t = tracker
        self.link = link
        self.port = port
        self.r_hat = r_hat_kv
        self.tick = tick_ns
        self.lifetimes = lifetimes
        self.noc = noc
        self.now = 0
        self.enrolled = {}            # owner -> (release_at_ns, seq)
        self.seq = 0
        self.actions = []             # (now, action) 逐采样动作
        self.quota_history = []       # (now, Q, occupancy)

    def _top_up(self):
        while True:
            owner = f"syn{self.seq}#decode#0"
            verdict = self.t.admit_flow(
                owner=owner, flow_class=FLOW_REALTIME, links=[self.link],
                port_id=self.port, r_hat_kv_bytes_per_ns=self.r_hat,
                now_ns=self.now)
            if not verdict.admitted:
                return
            lifetime = self.lifetimes[self.seq % len(self.lifetimes)]
            self.enrolled[owner] = (self.now + lifetime, self.seq)
            self.seq += 1

    def _settle_due(self):
        # 结算序 = 准入因果序（同刻按 release_at 再按准入 seq），不做
        # 名字字典序（否则寿命奇偶被打乱、EWMA 样本流失去确定性）。
        due = sorted((at, seq, owner) for owner, (at, seq)
                     in self.enrolled.items() if at <= self.now)
        for _, _, owner in due:
            self.t.release_flow(owner, now_ns=self.now)
            del self.enrolled[owner]

    def run(self, ticks, demand=True, bg_divisor=0.0):
        """推进 ticks 个 tick；返回本段 (now, Q, occupancy) 历史。"""
        segment = []
        for _ in range(ticks):
            self.now += self.tick
            if demand:
                self._top_up()
            self._settle_due()
            occupancy = self.t.link_occupancy(self.link)
            denom = occupancy + bg_divisor
            if denom > 0:
                obs = self.t.observe_telemetry(
                    self.now, {self.link: self.noc / denom}, self.r_hat)
                record = obs["links"][repr(self.link)]
                self.actions.append((self.now, record["action"]))
            entry = (self.now, self.t.link_quota(self.link), occupancy)
            self.quota_history.append(entry)
            segment.append(entry)
        return segment


# ==================================================== 剧本 1：到达突发 ==


class ArrivalBurstScenarioTest(unittest.TestCase):
    """剧本 1：到达突发——(c) 突发点不过饱和 + 同 tick 零 dt 采样。"""

    def test_link_burst_never_oversaturates(self):
        # 8 笔实时流同 tick 挤同一链路：准入封顶在突发点逐笔成立。
        t = _tracker(mode=QUOTA_STATIC)          # Q_init = 2
        verdicts = [t.admit_flow(
            owner=f"burst{i}#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=5,
            r_hat_kv_bytes_per_ns=1.0) for i in range(8)]
        for i in range(len(verdicts)):
            # (c) 逐笔不变式：占用+预留 ≤ Q（准入原子，无超饱和点）。
            self.assertLessEqual(
                t.link_occupancy((0, 1)) + t.link_reserved((0, 1)),
                t.q_init, msg=f"after attempt {i}")
        admitted = [v for v in verdicts if v.admitted]
        deferred = [v for v in verdicts if not v.admitted]
        self.assertEqual(len(admitted), 2)
        self.assertEqual(len(deferred), 6)
        for v in deferred:
            self.assertEqual(v.wait_reason, WAIT_QUOTA_LINK)
            self.assertEqual(v.resource_kind, "link")
            self.assertEqual(v.remaining, 0)
        self.assertEqual(t.link_occupancy((0, 1)), 2)

    def test_port_burst_respects_parity_headroom(self):
        # 平价门突发：每笔已准入满足 B/(u+1) >= r̂（B=100、r̂=40 →
        # headroom=2）；链路各用独立链路避免先触链路门。
        t = _tracker(mode=QUOTA_STATIC)
        admitted = 0
        for i in range(6):
            v = t.admit_flow(
                owner=f"pb{i}#decode#0", flow_class=FLOW_REALTIME,
                links=[(i, i + 1)], port_id=9,
                r_hat_kv_bytes_per_ns=40.0)
            if v.admitted:
                admitted += 1
                self.assertLessEqual(t.port_enrolled(9), 2)
            else:
                self.assertEqual(v.wait_reason, WAIT_QUOTA_PORT)
        self.assertEqual(admitted, 2)
        # 负载视图突发注入（u_port_base）同样被 headroom 封顶。
        v = t.admit_flow(
            owner="pbx#decode#0", flow_class=FLOW_REALTIME,
            links=[(9, 10)], port_id=9, r_hat_kv_bytes_per_ns=40.0,
            u_port_base=3)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_PORT)

    def test_burst_mixed_with_merge_reserves_stays_capped(self):
        # 突发混合：1 实时流 + 4 merge 预留挤同一链路——链路与 bulk
        # 侧封顶逐笔成立，失败预留零登记（原子性）。
        t = _tracker(mode=QUOTA_STATIC)
        self.assertTrue(t.admit_flow(
            owner="f0#decode#0", flow_class=FLOW_REALTIME, links=[(0, 1)],
            port_id=5, r_hat_kv_bytes_per_ns=1.0).admitted)
        verdicts = [t.reserve_merge(
            f"rid-{i}", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8) for i in range(4)]
        for i, v in enumerate(verdicts):
            self.assertLessEqual(
                t.link_occupancy((0, 1)) + t.link_reserved((0, 1)),
                t.q_init, msg=f"after reserve {i}")
            self.assertLessEqual(t.bulk_used(7), t.n_bulk)
            self.assertLessEqual(t.bulk_used(8), t.n_bulk)
        self.assertTrue(verdicts[0].admitted)     # 余量 1 恰容首笔预留
        self.assertFalse(any(v.admitted for v in verdicts[1:]))
        self.assertEqual(t.link_reserved((2, 3)), 0)   # 失败零登记
        self.assertEqual(t.bulk_used(9), 0)

    def test_same_tick_telemetry_samples_have_zero_dt(self):
        # 同 tick 两次遥测采样：dt=0，comfort streak 不重复累计。
        t = _tracker()
        t.admit_flow(owner="z#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=_R_KV, now_ns=0)
        t.release_flow("z#decode#0", now_ns=1000)   # EWMA=1000 → T=10000
        obs1 = t.observe_telemetry(2000, {(0, 1): 240.0}, _R_KV)
        obs2 = t.observe_telemetry(2000, {(0, 1): 240.0}, _R_KV)
        rec1 = obs1["links"]["(0, 1)"]
        rec2 = obs2["links"]["(0, 1)"]
        self.assertEqual(rec1["dt_ns"], 0)          # 首采样 = 基线
        self.assertEqual(rec2["dt_ns"], 0)          # 同 tick → dt 0
        self.assertEqual(rec1["action"], AIMD_ACTION_COMFORT)
        self.assertEqual(rec2["quiet_ns"], rec1["quiet_ns"])
        self.assertEqual(rec2["quota_after"], rec2["quota_before"])  # 无扩张


# ====================================================== 剧本 2：长列车 ==


class LongTrainScenarioTest(unittest.TestCase):
    """剧本 2：长列车——(a) AIMD 收缩不发散；grandfathered 在册不逐出。"""

    def test_shrink_grandfathers_long_trains(self):
        t = _tracker()                              # aimd, Q_init = 2
        for owner in ("trainA#readplan", "trainB#readplan"):
            self.assertTrue(t.admit_flow(
                owner=owner, flow_class=FLOW_ELASTIC, links=[(0, 1)],
                port_id=5, now_ns=0).admitted)
        obs = t.observe_telemetry(1000, {(0, 1): 8.0}, _R_KV)   # 8 < 10
        record = obs["links"]["(0, 1)"]
        self.assertEqual(record["signal"], AIMD_SIGNAL_SHRINK)
        self.assertEqual(record["action"], AIMD_ACTION_SHRINK)
        self.assertEqual(record["quota_before"], 2)
        self.assertEqual(record["quota_after"], 1)   # MD 减半 2 → 1
        # grandfathered：在册 2 条长列车不逐出，余量 −1 封新准入。
        self.assertEqual(t.link_occupancy((0, 1)), 2)
        self.assertEqual(t.link_remaining((0, 1)), -1)
        v = t.admit_flow(
            owner="new#decode#0", flow_class=FLOW_REALTIME, links=[(0, 1)],
            port_id=5, r_hat_kv_bytes_per_ns=_R_KV, now_ns=1000)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_LINK)
        self.assertEqual(v.remaining, -1)
        deferred = quota_deferred_requeue(
            "req-1", {"remote-read": v}, t.quota_retry_key())
        self.assertTrue(deferred.requeue)            # 永不 raise
        # 端口平价门与 N_bulk 在 aimd 下常开（不随链路 Q 收缩）。
        self.assertEqual(t.n_bulk, t.q_init)
        self.assertTrue(t.admit_flow(
            owner="pp#decode#0", flow_class=FLOW_REALTIME,
            links=[(9, 10)], port_id=6,
            r_hat_kv_bytes_per_ns=60.0, now_ns=1000).admitted)   # 100 ≥ 60
        vp = t.admit_flow(
            owner="pp2#decode#0", flow_class=FLOW_REALTIME,
            links=[(9, 11)], port_id=6,
            r_hat_kv_bytes_per_ns=60.0, now_ns=1000)
        self.assertFalse(vp.admitted)                # u=1: 50 < 60
        self.assertEqual(vp.wait_reason, WAIT_QUOTA_PORT)
        # 列车排空：余量恢复正、准入恢复——收缩不逐出、不塌陷（(a)）。
        t.release_flow("trainA#readplan", now_ns=6000)
        t.release_flow("trainB#readplan", now_ns=6000)
        self.assertEqual(t.link_remaining((0, 1)), 1)
        self.assertTrue(t.admit_flow(
            owner="new#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=5, r_hat_kv_bytes_per_ns=_R_KV,
            now_ns=6000).admitted)

    def test_shrink_floor_at_quota_one_records_event_without_adjustment(self):
        t = _tracker()
        # 供一个寿命样本（走旁链路，不占被测链路）使 T_expand 有定义
        # ——comfort 分支才累计 streak（冷启动下 comfort 不累计）。
        t.admit_flow(owner="svc#decode#0", flow_class=FLOW_REALTIME,
                     links=[(9, 10)], port_id=6,
                     r_hat_kv_bytes_per_ns=_R_KV, now_ns=0)
        t.release_flow("svc#decode#0", now_ns=1000)   # EWMA=1000
        t.observe_telemetry(1000, {(0, 1): 8.0}, _R_KV)   # shrink 2 → 1
        epoch_before = t.quota_retry_key()
        obs = t.observe_telemetry(2000, {(0, 1): 8.0}, _R_KV)  # 1 已下限
        record = obs["links"]["(0, 1)"]
        self.assertEqual(record["action"], AIMD_ACTION_SHRINK_FLOOR)
        self.assertEqual(record["quota_before"], 1)
        self.assertEqual(record["quota_after"], 1)
        # 无调整 = 信用可得性未变 = 无代数 bump（无虚假唤醒）。
        self.assertEqual(t.quota_retry_key(), epoch_before)
        # 收缩事件仍清零扩张 streak（舒适后须重新连续 T_expand）。
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 0)
        # 舒适自 0 重计：下一采样 dt=1000 → quiet=1000。
        t.observe_telemetry(3000, {(0, 1): 240.0}, _R_KV)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 1000)

    def test_flow_lifetime_ewma_feeds_expand_timescale(self):
        t = _tracker()
        self.assertIsNone(t.t_expand_ns)             # 冷启动
        t.admit_flow(owner="a#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=_R_KV, now_ns=0)
        t.release_flow("a#decode#0", now_ns=6000)
        self.assertEqual(t.flow_lifetime_ewma_ns, 6000.0)
        self.assertEqual(t.flow_lifetime_samples, 1)
        self.assertEqual(t.t_expand_ns, AIMD_EXPAND_K * 6000.0)
        t.admit_flow(owner="b#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=_R_KV, now_ns=0)
        t.release_flow("b#decode#0", now_ns=7000)
        # EWMA = 7/8·6000 + 1/8·7000 = 6125（α=1/8 冻结平滑）。
        self.assertAlmostEqual(t.flow_lifetime_ewma_ns, 6125.0, places=9)

    def test_lifetime_sample_edge_rules(self):
        t = _tracker()
        # admit 无时标 + release 带时标 → 无样本（配对须两端带时标）。
        t.admit_flow(owner="x#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=_R_KV)
        t.release_flow("x#decode#0", now_ns=500)
        self.assertEqual(t.flow_lifetime_samples, 0)
        # 零寿命（同时刻准入/释放）= 非正服务时长，不计样本。
        t.admit_flow(owner="y#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=_R_KV, now_ns=500)
        t.release_flow("y#decode#0", now_ns=500)
        self.assertEqual(t.flow_lifetime_samples, 0)
        self.assertIsNone(t.t_expand_ns)             # 仍冷启动

    def test_clock_discipline_fail_closed(self):
        t = _tracker()
        t.observe_telemetry(1000, {(0, 1): 240.0}, _R_KV)
        with self.assertRaises(LinkQuotaError):
            t.observe_telemetry(999, {(0, 1): 240.0}, _R_KV)   # 钟倒退
        with self.assertRaises(LinkQuotaError):
            t.observe_telemetry(-1, {(0, 1): 240.0}, _R_KV)
        with self.assertRaises(LinkQuotaError):
            t.observe_telemetry(True, {(0, 1): 240.0}, _R_KV)
        t.admit_flow(owner="c#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=_R_KV, now_ns=100)
        with self.assertRaises(LinkQuotaError):
            t.release_flow("c#decode#0", now_ns=50)   # release 早于 admit
        # 钟纪律失败不污染簿记：流仍在册，正常时标释放可行。
        t.release_flow("c#decode#0", now_ns=200)
        self.assertEqual(t.flow_lifetime_samples, 1)

    def test_flow_lifetime_ewma_unit_semantics(self):
        ewma = FlowLifetimeEwma()
        self.assertIsNone(ewma.value_ns)
        self.assertEqual(ewma.samples, 0)
        ewma.observe(800)
        self.assertEqual(ewma.value_ns, 800.0)
        ewma.observe(1200)
        self.assertAlmostEqual(ewma.value_ns, 850.0, places=9)  # 0.875/0.125
        self.assertEqual(ewma.samples, 2)
        for bad in (0, -5, float("nan"), float("inf")):
            with self.assertRaises(LinkQuotaError):
                ewma.observe(bad)


# ==================================================== 剧本 3：merge 挤兑 ==


class MergeStampedeScenarioTest(unittest.TestCase):
    """剧本 3：merge 挤兑——并发预留挤兑下 N_bulk/链路信用封顶 +
    借还配对零泄漏 + 交错裁决。"""

    def test_stampede_respects_caps_and_pairs_cleanly(self):
        t = _tracker(mode=QUOTA_STATIC)              # Q_init = N_bulk = 2
        verdicts = [t.reserve_merge(
            f"rid-{i}", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8) for i in range(6)]
        for i in range(len(verdicts)):
            # (c) 端口/链路双侧不变式逐笔（挤兑峰值点不过饱和）。
            self.assertLessEqual(t.bulk_used(7), t.n_bulk, msg=f"p7#{i}")
            self.assertLessEqual(t.bulk_used(8), t.n_bulk, msg=f"p8#{i}")
            self.assertLessEqual(
                t.link_reserved((0, 1)) + t.link_occupancy((0, 1)),
                t.q_init, msg=f"L01#{i}")
            self.assertLessEqual(
                t.link_reserved((1, 0)) + t.link_occupancy((1, 0)),
                t.q_init, msg=f"L10#{i}")
        self.assertTrue(verdicts[0].admitted)
        self.assertTrue(verdicts[1].admitted)        # 链路 (0,1) 容量恰尽
        self.assertFalse(any(v.admitted for v in verdicts[2:]))
        self.assertEqual({v.wait_reason for v in verdicts[2:]},
                         {WAIT_QUOTA_LINK})
        self.assertEqual(t.bulk_used(7), 2)          # N_bulk 挤兑封顶
        self.assertEqual(t.bulk_used(8), 2)
        # 交错裁决：败者侧即刻归还，胜者侧持有到 merge_done。
        t.adjudicate_merge_direction("rid-0", MERGE_DIRECTION_FORWARD)
        self.assertEqual(t.link_reserved((1, 0)), 1)
        self.assertEqual(t.bulk_used(8), 1)
        self.assertEqual(t.link_reserved((0, 1)), 2)
        t.release_merge("rid-0")
        self.assertEqual(t.link_reserved((0, 1)), 1)
        self.assertEqual(t.bulk_used(7), 1)
        # 挤兑队列中的 retry 预留因释放而可行（信用恢复即回队重入）。
        self.assertTrue(t.reserve_merge(
            "rid-5b", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8).admitted)
        # 全释放 → 零泄漏（借还配对闭合）。
        t.adjudicate_merge_direction("rid-1", MERGE_DIRECTION_REVERSE)
        t.release_merge("rid-1")
        t.release_merge("rid-5b")
        self.assertEqual(
            t.link_reserved((0, 1)) + t.link_reserved((1, 0)), 0)
        self.assertEqual(t.bulk_used(7) + t.bulk_used(8), 0)

    def test_stampede_with_oneshot_flows_and_fail_closed_pairing(self):
        t = _tracker(mode=QUOTA_STATIC)
        # 2 条 oneshot 写回流占满链路后再挤 3 笔预留 + 1 条实时流。
        for owner in ("w0#writeback", "w1#writeback"):
            self.assertTrue(t.admit_flow(
                owner=owner, flow_class=FLOW_ONESHOT, links=[(0, 1)],
                port_id=5).admitted)
        for rid in ("m-1", "m-2", "m-3"):
            v = t.reserve_merge(
                rid, links_forward=[(0, 1)], links_reverse=[(1, 0)],
                port_forward=7, port_reverse=8)
            self.assertFalse(v.admitted)             # 链路已满
            self.assertEqual(v.wait_reason, WAIT_QUOTA_LINK)
            self.assertLessEqual(
                t.link_reserved((0, 1)) + t.link_occupancy((0, 1)),
                t.q_init)
        v = t.admit_flow(owner="r#decode#0", flow_class=FLOW_REALTIME,
                         links=[(0, 1)], port_id=5,
                         r_hat_kv_bytes_per_ns=1.0)
        self.assertFalse(v.admitted)
        # 写回流 settle 归还 → 预留可行 → 再释放一条后实时流可行。
        t.release_flow("w0#writeback")
        self.assertTrue(t.reserve_merge(
            "m-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8).admitted)
        t.release_flow("w1#writeback")
        self.assertTrue(t.admit_flow(
            owner="r#decode#0", flow_class=FLOW_REALTIME, links=[(0, 1)],
            port_id=5, r_hat_kv_bytes_per_ns=1.0).admitted)
        # 挤兑下的双释放/未知释放 fail-closed（配对纪律不因挤兑放松）。
        t.release_merge("m-1")
        with self.assertRaises(LinkQuotaError):
            t.release_merge("m-1")
        with self.assertRaises(LinkQuotaError):
            t.release_merge("ghost")


# ===================================== 剧本 4：EWMA 服务因子与配额耦合 ==


class EwmaQuotaCouplingScenarioTest(unittest.TestCase):
    """剧本 4：EWMA 服务因子与配额耦合——(a) 不发散 / (b) 阈值带
    不重合下无极限环 / (e) 反馈回路不振荡。"""

    def test_frozen_constants(self):
        # D5 冻结钉：k=10、带上沿因子 1.2；C10 裁定披露钉：MD 减半、
        # EWMA α=1/8。任何漂移 = 冻结破坏，测试即红。
        self.assertEqual(AIMD_EXPAND_K, 10)
        self.assertEqual(AIMD_BAND_UPPER_FACTOR, 1.2)
        self.assertEqual(AIMD_SHRINK_DIVISOR, 2)
        self.assertEqual(FLOW_LIFETIME_EWMA_ALPHA, 0.125)

    def test_threshold_band_partition_and_disjointness(self):
        # (b) 结构性前提：收缩触发集 {rate < r_KV} 与扩张触发集
        # {rate >= 1.2·r_KV} 不相交——采样网格上逐点验证两集合的
        # 互斥分离（signal=shrink ⟹ rate < r_KV；signal=comfort
        # ⟹ rate >= 1.2·r_KV；带内为死区 hold）。
        grid = [0.05 * i for i in range(1, 641)]    # 0.05 .. 32.0
        saw_shrink = saw_comfort = saw_hold = False
        for rate in grid:
            signal = aimd_band_signal(rate, _R_KV)
            if signal == AIMD_SIGNAL_SHRINK:
                saw_shrink = True
                self.assertLess(rate, _R_KV)
            elif signal == AIMD_SIGNAL_HOLD:
                saw_hold = True
                self.assertGreaterEqual(rate, _R_KV)
                self.assertLess(rate, 1.2 * _R_KV)
            else:
                saw_comfort = True
                self.assertGreaterEqual(rate, 1.2 * _R_KV)
            # 互斥：同一 rate 决不兼具收缩与扩张触发资格。
            # （F4：原恒真式 assertNotEqual(rate < _R_KV and
            # rate >= 1.2 * _R_KV, True) 已删——复审发现；上方的分支
            # 分域断言已逐点钉死互斥性。）
        self.assertTrue(saw_shrink and saw_comfort and saw_hold)
        # 边界语义：等值线本身不收缩（严格小于）；上沿含等号。
        self.assertEqual(aimd_band_signal(10.0, _R_KV), AIMD_SIGNAL_HOLD)
        self.assertEqual(aimd_band_signal(12.0, _R_KV), AIMD_SIGNAL_COMFORT)
        for bad_rate in (0.0, -1.0, float("nan")):
            with self.assertRaises(LinkQuotaError):
                aimd_band_signal(bad_rate, _R_KV)
        with self.assertRaises(LinkQuotaError):
            aimd_band_signal(5.0, 0.0)

    def test_shrink_law_and_ceiling_derivation(self):
        # MD 减半链：24→12→6→3→1→1（下限 1 防结构性死锁）。
        chain = []
        quota = _CEILING
        while quota > 1:
            quota = aimd_shrink_quota(quota)
            chain.append(quota)
        self.assertEqual(chain, [12, 6, 3, 1])
        self.assertEqual(aimd_shrink_quota(1), 1)
        # 扩张上界 = 收缩等值线：floor(B/r_KV)；极端配置保 1。
        self.assertEqual(aimd_expand_ceiling(_B_NOC, _R_KV), _CEILING)
        self.assertEqual(aimd_expand_ceiling(_B_NOC, 300.0), 1)
        for bad in (0, -1, True, 2.0):
            with self.assertRaises(LinkQuotaError):
                aimd_shrink_quota(bad)
        for noc, r in ((_B_NOC, 0.0), (0.0, _R_KV)):
            with self.assertRaises(LinkQuotaError):
                aimd_expand_ceiling(noc, r)

    def test_telemetry_mode_gating_and_validation(self):
        # static/off 下遥测消费 fail-closed（与 set_link_quota 同律）。
        for mode in (QUOTA_STATIC, QUOTA_OFF):
            t = _tracker(mode=mode)
            with self.assertRaises(LinkQuotaError):
                t.observe_telemetry(0, {(0, 1): 240.0}, _R_KV)
        t = _tracker()
        for telemetry in ({(0, 1): 0.0}, {(0, 1): -5.0},
                          {(0, 1): float("nan")}):
            with self.assertRaises(LinkQuotaError):
                t.observe_telemetry(0, telemetry, _R_KV)
        with self.assertRaises(LinkQuotaError):
            t.observe_telemetry(0, {(0, 1): 240.0}, -1.0)

    def test_cold_start_suppresses_expansion_until_first_lifetime(self):
        t = _tracker()
        # 冷启动：无寿命样本 → 舒适区不累计 streak、不扩张（无论
        # 持续多久）——T_expand 不可定义时不做时标分离外的决策。
        for now in range(1000, 100000, 1000):
            obs = t.observe_telemetry(now, {(0, 1): 240.0}, _R_KV)
            record = obs["links"]["(0, 1)"]
            self.assertEqual(record["action"],
                             AIMD_ACTION_COMFORT_COLD_START)
            self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 0)
        self.assertEqual(t.link_quota((0, 1)), t.q_init)
        # 首个寿命样本（5000）落地 → T_expand = 50000 定义；streak
        # 自 0 重新累计，恰在连续舒适 ≥ 50000 后扩张。
        t.admit_flow(owner="cs#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=_R_KV, now_ns=0)
        t.release_flow("cs#decode#0", now_ns=5000)
        self.assertEqual(t.t_expand_ns, 50000.0)
        expanded_at = None
        for i in range(1, 12):
            now = 100000 + i * 10000
            obs = t.observe_telemetry(now, {(0, 1): 240.0}, _R_KV)
            record = obs["links"]["(0, 1)"]
            if record["action"] == AIMD_ACTION_EXPAND:
                expanded_at = i
                break
            self.assertEqual(record["action"], AIMD_ACTION_COMFORT)
            self.assertLess(t.aimd_link_state((0, 1))["quiet_ns"], 50000)
        self.assertIsNotNone(expanded_at)           # streak 首达 50000
        self.assertEqual(expanded_at, 5)
        self.assertEqual(t.link_quota((0, 1)), t.q_init + 1)

    def test_closed_loop_settles_without_oscillation(self):
        """闭环全弧：保载爬升 → 卸载 → 回载 → 外生挤压 → 恢复。

        断言：(a) Q 全程 ∈ [1, 24] 且收敛到不动点；(b) 死区/上界下
        无自持极限环（各常参数段内动作方向单一，扰动后单调恢复）；
        (e) EWMA→决策→负载反馈回路不振荡（全程方向翻转恰 2 次，均
        落在 bg 扰动切换点，无一例自发翻转）。
        """
        t = _tracker(hbm=_B_HBM_CLOSED)     # aimd, Q_init=1, 平价门不绑定
        sim = _SyntheticLinkLoad(t)

        # 阶段 1（保载爬升）：深需求 → 舒适 → 每 T_expand +1 至上界。
        phase1 = sim.run(1200)
        self.assertEqual(phase1[-1][1], _CEILING)          # 收敛到上界
        p1_actions = {a for _, a in sim.actions}
        self.assertLessEqual(p1_actions, {
            AIMD_ACTION_COMFORT, AIMD_ACTION_COMFORT_COLD_START,
            AIMD_ACTION_HOLD, AIMD_ACTION_EXPAND,
            AIMD_ACTION_EXPAND_CAPPED})
        self.assertIn(AIMD_ACTION_EXPAND_CAPPED, p1_actions)  # 上界触到
        q1 = [q for _, q, _ in phase1]
        self.assertTrue(all(a <= b for a, b in zip(q1, q1[1:])))
        # 长尾稳定：末段无任何 Q 变化（(b) 不动点；段长 > 3×T_expand）。
        self.assertEqual(set(q1[-200:]), {_CEILING})

        # 阶段 2（卸载）：无需求 → 在册排空 → 无测量（无在册流），
        # Q 与 streak 冻结（不进不出）。
        phase2 = sim.run(80, demand=False)
        self.assertEqual({q for _, q, _ in phase2}, {_CEILING})
        self.assertEqual(t.link_occupancy((0, 1)), 0)
        frozen = t.aimd_link_state((0, 1))
        sim.run(20, demand=False)                          # 纯空闲段
        self.assertEqual(t.aimd_link_state((0, 1)), frozen)

        # 阶段 3（回载）：占用回升至 Q → 实测速率落入死区/舒适 →
        # 无收缩、无新增扩张（至多 capped）——负载整循环零振荡。
        n3 = len(sim.actions)
        phase3 = sim.run(120)
        self.assertEqual({q for _, q, _ in phase3}, {_CEILING})
        p3_actions = {a for _, a in sim.actions[n3:]}
        self.assertLessEqual(p3_actions, {
            AIMD_ACTION_HOLD, AIMD_ACTION_COMFORT,
            AIMD_ACTION_EXPAND_CAPPED})

        # 阶段 4（外生挤压）：背景除数注入使实测速率跌破 r_KV →
        # 连续收缩至下限 1（shrink_floor）；在册 grandfathered 不逐出。
        n4 = len(sim.actions)
        phase4 = sim.run(16, demand=True, bg_divisor=40.0)
        p4_actions = {a for _, a in sim.actions[n4:]}
        self.assertLessEqual(
            p4_actions, {AIMD_ACTION_SHRINK, AIMD_ACTION_SHRINK_FLOOR})
        self.assertEqual(min(q for _, q, _ in phase4), 1)
        # grandfathered 证据：存在 occupancy > Q 的采样点（不逐出）。
        self.assertTrue(any(occ > q for _, q, occ in phase4))

        # 阶段 5（恢复）：解除挤压 → 单调爬回上界并稳定。
        n5 = len(sim.actions)
        phase5 = sim.run(1400)
        p5_actions = {a for _, a in sim.actions[n5:]}
        self.assertLessEqual(p5_actions, {
            AIMD_ACTION_COMFORT, AIMD_ACTION_HOLD,
            AIMD_ACTION_EXPAND, AIMD_ACTION_EXPAND_CAPPED})
        self.assertEqual(phase5[-1][1], _CEILING)
        q5 = [q for _, q, _ in phase5]
        self.assertTrue(all(a <= b for a, b in zip(q5, q5[1:])))
        self.assertEqual(set(q5[-200:]), {_CEILING})

        # (a) 全程不发散：Q ∈ [1, floor(B/r_KV)]，占用有界。
        for now, q, occ in sim.quota_history:
            self.assertGreaterEqual(q, 1, msg=f"now={now}")
            self.assertLessEqual(q, _CEILING, msg=f"now={now}")
            self.assertLessEqual(occ, _CEILING, msg=f"now={now}")
        # (e) 方向翻转计数：全程恰 2 次（阶段4起、阶段5起），全部
        # 落在 bg 扰动边界——零自发振荡。
        direction_events = [a for _, a in sim.actions
                            if a in (AIMD_ACTION_EXPAND,
                                     AIMD_ACTION_SHRINK)]
        reversals = sum(
            1 for a, b in zip(direction_events, direction_events[1:])
            if (a == AIMD_ACTION_EXPAND) != (b == AIMD_ACTION_EXPAND))
        self.assertEqual(reversals, 2)
        # EWMA 服务因子收敛：寿命总体 (800,1200) 交替、均值 1000。
        # EWMA 是样本的凸组合 → 恒在 [800, 1200]（平滑规则的结构
        # 不变量）；重采样后落在均值 ±150 内（同刻批量结算使奇偶
        # 不严格逐样交替，波纹宽于理想逐样交替的 ±13.3，按凸包 +
        # 15% 收敛带断言）；T_expand = k×EWMA 恒等。
        ewma = t.flow_lifetime_ewma_ns
        self.assertGreaterEqual(ewma, 800.0)
        self.assertLessEqual(ewma, 1200.0)
        self.assertLess(abs(ewma - 1000.0), 150.0)
        self.assertEqual(t.t_expand_ns, AIMD_EXPAND_K * ewma)
        self.assertGreater(t.flow_lifetime_samples, 100)
        # 收缩总数冗余钉：24→12→6→3→1 恰 4 次（与阶段 4 一致）。
        self.assertEqual(
            sum(1 for _, a in sim.actions if a == AIMD_ACTION_SHRINK), 4)


# ==================================== 剧本 5：quota_deferred 重试链 ==


class DeferredRetryChainScenarioTest(unittest.TestCase):
    """剧本 5：quota_deferred 重试链——(d) 释放必唤醒 deferred、
    无漏唤醒/重复唤醒（模块级"唤醒"契约 = 配额代数 bump，C11 接线
    到 SH:1775-1802 epoch 键重试门）。"""

    def test_chain_wakes_exactly_once_per_release_side_event(self):
        t = _tracker()                          # aimd, Q_init = 2
        for owner in ("a#decode#0", "b#decode#0"):
            self.assertTrue(t.admit_flow(
                owner=owner, flow_class=FLOW_REALTIME, links=[(0, 1)],
                port_id=5, r_hat_kv_bytes_per_ns=_R_KV, now_ns=0).admitted)
        # 突发第 3 笔 → quota_deferred；失败准入不 bump（无虚假唤醒）。
        epoch0 = t.quota_retry_key()[0]
        v = t.admit_flow(
            owner="c#decode#0", flow_class=FLOW_REALTIME, links=[(0, 1)],
            port_id=5, r_hat_kv_bytes_per_ns=_R_KV, now_ns=0)
        self.assertFalse(v.admitted)
        self.assertEqual(t.quota_retry_key()[0], epoch0)
        deferred = quota_deferred_requeue(
            "req-c", {"remote-read": v}, t.quota_retry_key(),
            base_retry_key=(3, ((0, 17),)))
        self.assertTrue(deferred.requeue)            # 永不 raise
        self.assertEqual(deferred.retry_key, (3, ((0, 17),), epoch0))
        self.assertEqual(deferred.wait_reason, WAIT_QUOTA_LINK)
        # 流 settle：恰 +1 → 重试门重开（键变化）→ 再准入闭合。
        t.release_flow("a#decode#0", now_ns=1000)
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 1)
        self.assertNotEqual(deferred.retry_key[-1],
                            t.quota_retry_key()[0])
        self.assertTrue(t.admit_flow(
            owner="c#decode#0", flow_class=FLOW_REALTIME, links=[(0, 1)],
            port_id=5, r_hat_kv_bytes_per_ns=_R_KV, now_ns=1000).admitted)
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 1)   # 准入不 bump
        # 释放 → 预留（不 bump）→ 挤兑第 2 笔 deferred → 裁决/释放
        # 各恰 +1，且释放唤醒后 retry 预留可行。
        t.release_flow("b#decode#0", now_ns=2000)
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 2)
        self.assertTrue(t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8).admitted)
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 2)   # 预留不 bump
        v2 = t.reserve_merge(
            "rid-2", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8)
        self.assertFalse(v2.admitted)                # 链路余量 0
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 2)
        t.adjudicate_merge_direction("rid-1", MERGE_DIRECTION_FORWARD)
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 3)   # 裁决 +1
        t.release_merge("rid-1")
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 4)   # merge_done
        self.assertTrue(t.reserve_merge(
            "rid-2", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8).admitted)
        t.release_merge("rid-2")
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 5)
        # AIMD 调整恰 +1；无调整动作（floor/hold）不 bump。
        t.observe_telemetry(3000, {(0, 1): 8.0}, _R_KV)   # shrink 2→1
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 6)
        t.observe_telemetry(4000, {(0, 1): 8.0}, _R_KV)   # shrink_floor
        t.observe_telemetry(5000, {(0, 1): 11.0}, _R_KV)  # hold 死区
        self.assertEqual(t.quota_retry_key()[0], epoch0 + 6)

    def test_requeue_never_raises_and_classifies_wait_reasons(self):
        t = _tracker()
        for owner in ("a#decode#0", "b#decode#0"):
            t.admit_flow(owner=owner, flow_class=FLOW_REALTIME,
                         links=[(0, 1)], port_id=5,
                         r_hat_kv_bytes_per_ns=_R_KV)
        link_deferred = t.admit_flow(
            owner="L#decode#0", flow_class=FLOW_REALTIME, links=[(0, 1)],
            port_id=5, r_hat_kv_bytes_per_ns=_R_KV)
        port_deferred = t.admit_flow(
            owner="P#decode#0", flow_class=FLOW_REALTIME, links=[(5, 6)],
            port_id=5, r_hat_kv_bytes_per_ns=95.0)   # u=2: 100/3 < 95
        self.assertFalse(link_deferred.admitted)
        self.assertFalse(port_deferred.admitted)
        try:
            deferred = quota_deferred_requeue(
                "req-1", {"remote-read": link_deferred,
                          "copy": port_deferred}, t.quota_retry_key())
        except Exception:  # noqa: BLE001 - 永不 raise 合同
            self.fail("quota_deferred_requeue must never raise")
        self.assertTrue(deferred.requeue)
        # wait_reason 按 WAIT_REASONS 枚举序取首个非 None（确定性）。
        self.assertEqual(deferred.wait_reason, WAIT_QUOTA_LINK)
        empty = quota_deferred_requeue("req-2", {}, t.quota_retry_key())
        self.assertTrue(empty.requeue)
        self.assertEqual(empty.wait_reason, WAIT_CAPACITY)


if __name__ == "__main__":
    unittest.main()
