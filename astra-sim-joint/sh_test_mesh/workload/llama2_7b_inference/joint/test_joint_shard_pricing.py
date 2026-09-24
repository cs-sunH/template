#!/usr/bin/env python3
"""test_joint_shard_pricing.py -- C1（WP1a）逐 shard 三腿 min 计价 +
D1 读放大因果口径的零后端单测（F1 冻结形态的解析锚点）。

覆盖（卡 C1 测试清单）：
  1. 平价锚点：d ≤ ρ 空链路单流下 remote-read 每步代价 ≡ 本地读平价
     （t_r = max(t_l, c·d/B_D2D) 的 d≤ρ 分支退化为 t_l）——注意**仅
     历史扫描项平价，非整轮平价保证**（设计文档 §1.2：供给条件不能
     单独保证完整轮次与本地执行平价）；
  2. 三腿 min 三夹具（noc 快/HBM 慢、noc 慢/HBM 快、双慢）：断言取
     max 不取和；
  3. TP 并行：链路不相交路径 wall = 单 shard 时间（非 N× 串行放大）；
     共链时自身重叠计费保留（divisor 含 self）；零字节 shard 无流
     不争用；
  4. 读放大因果（D1）：decode 每步远读基数 = 仍位于远端的基础历史
     （H，非 H+I）；prefill 按实际消费遍数计（两遍扫描夹具遍数 = 2）；
  5. credit 逐 shard 切块：skewed rank 字节下 first/remaining 由最大
     rank 主导；K ≥ read_passes 单 credit 退化不变量；
  6. 代表路径广播锚（route_paths_fn 未注入）：字节均衡、空链路时
     与旧聚合口径（_transfer_ns）同 wall 值（C4 金值重推导的桥接锚）；
  7. 池腿保真边界（F3）：_pool_transfer_ns 五消费点口径不变
     （copy 混合腿的池后缀腿数值回归）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_shard_pricing.py   （或 pytest 同路径）
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
    ACTION_COPY,
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
    _transfer_ns_shards,
    _shard_stream_ns_shards,
)


# ============================================================ 夹具构造 ==

def _rates(noc=10.0, hbm=100.0, lat=10, pool=5.0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=pool, local_hbm_gbps=hbm,
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


def _request(input_tokens=50, decode=10, input_bytes=(25, 25),
             prefill_scan_passes=None):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes,
        prefill_scan_passes=prefill_scan_passes)


def _disjoint_paths(source, target):
    """tp=2 逐 rank 不相交路径（rank 对 (2s,2t)/(2s+1,2t+1)，各 1 跳）。"""
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


def _route(source, target):
    return ((source, target), 1)


def _model(loads=None, *, rates=None, route_paths_fn="disjoint",
           credit_iters="auto"):
    return JointCostModel(
        rates=rates if rates is not None else _rates(),
        loads=loads if loads is not None else {0: _load(0), 1: _load(1)},
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        route_paths_fn=(
            _disjoint_paths if route_paths_fn == "disjoint"
            else route_paths_fn),
        remote_credit_iters=credit_iters)


# ==================================================== 1. 平价锚点（§1.2）==


class ParityAnchorTest(unittest.TestCase):
    """d ≤ ρ 空链路单流：remote-read 每步历史扫描项 ≡ 本地读平价。"""

    def test_d_le_rho_degrades_to_local_scan(self):
        # 基座锚点：B_D2D=4050、B_HBM=1640 → ρ≈2.47；d=2 ≤ ρ、空链路、
        # 单流。三腿 max = max(c/4050, c/1640) = c/1640 = t_l（本地扫描
        # 项）——noc 腿（404.9）被 home 读腿（1000）遮蔽，与解析锚
        # t_r = max(t_l, c·d/B_D2D) 的 d≤ρ 分支退化为 t_l 同结论。
        # 注意：仅历史扫描项平价，非整轮平价保证（设计文档 §1.2——
        # 首块启动/流水填充/容量/同步另计，供给条件不单独保证整轮平价）。
        rates = _rates(noc=4050.0, hbm=1640.0, lat=5)
        shard_bytes = 1_640_000
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1, 2),),),
            bytes_by_rank=(shard_bytes,),
            include_self=True, kind="read")
        local_scan_ns = int(shard_bytes / rates.local_hbm_bytes_per_ns)
        self.assertEqual(local_scan_ns, 1000)
        self.assertLess(
            int(shard_bytes / rates.noc_link_bytes_per_ns), local_scan_ns)
        # 跳时延剥离后 ≡ 本地扫描项（hop 启动另计，不进扫描项平价）。
        self.assertEqual(wall - 2 * rates.d2d_latency_ns, local_scan_ns)

    def test_estimate_action_per_step_parity(self):
        # estimate_action 层：per-step =（全流 wall − 跳时延）/ 遍数 ≡
        # 本地逐 shard 扫描（数值整除夹具：1640000 B/rank、4 遍）。
        rates = _rates(noc=4050.0, hbm=1640.0, lat=5)
        model = _model(rates=rates)
        candidate = model.estimate_action(
            session=_session(history_bytes=(1_640_000, 1_640_000),
                             history_tokens=100),
            request=_request(decode=3), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertIn("remote_read_passes=4", candidate.breakdown.notes)
        per_step = (
            candidate.breakdown.remote_read_ns - rates.d2d_latency_ns) // 4
        self.assertEqual(per_step, 1_640_000 // 1640)

    def test_sub_parity_config_noc_leg_degrades(self):
        # 对照：ρ<1 配置（noc=1200 < hbm=1640、d=1 > ρ=0.73）——noc 腿
        # 主导，扫描项高于本地（锚点的 d>ρ 方向；不排除系统级收益，
        # 仅扫描项口径）。
        rates = _rates(noc=1200.0, hbm=1640.0, lat=5)
        shard_bytes = 1_640_000
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),),
            bytes_by_rank=(shard_bytes,),
            include_self=True, kind="read")
        self.assertEqual(
            wall - rates.d2d_latency_ns, int(shard_bytes / 1200))
        self.assertGreater(
            wall - rates.d2d_latency_ns,
            int(shard_bytes / rates.local_hbm_bytes_per_ns))


# ==================================================== 2. 三腿 min 夹具 ==


class ThreeLegMaxTest(unittest.TestCase):
    """三腿 max 不取和：noc 快/HBM 慢、noc 慢/HBM 快、双慢三夹具。

    注：u_home/u_exec 本卡恒 0（port_registry=None，C2 接线后生效），
    copy 的 home/exec 两端点腿同率；三腿区分度由 noc 腿与端点腿给出。
    """

    BYTES = 6000

    def _wall(self, rates):
        return _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),),
            bytes_by_rank=(self.BYTES,),
            include_self=True, kind="copy")

    def test_noc_fast_hbm_slow(self):
        # noc 腿 6 ns、home/exec 腿各 600 ns → max=600（和=1206）。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        noc_ns = self.BYTES / 1000.0
        hbm_ns = self.BYTES / 10.0
        self.assertEqual(self._wall(rates), int(max(noc_ns, hbm_ns)))
        self.assertNotEqual(self._wall(rates), int(noc_ns + 2 * hbm_ns))

    def test_noc_slow_hbm_fast(self):
        # noc 腿 6000 ns、端点腿各 6 ns → max=6000（和=6012）。
        rates = _rates(noc=1.0, hbm=1000.0, lat=0)
        self.assertEqual(self._wall(rates), self.BYTES)
        self.assertNotEqual(self._wall(rates), self.BYTES + 12)

    def test_both_slow(self):
        # noc 腿 3000、端点腿各 2000 → max=3000（和=7000）。
        rates = _rates(noc=2.0, hbm=3.0, lat=0)
        self.assertEqual(self._wall(rates), 3000)
        self.assertNotEqual(self._wall(rates), 7000)

    def test_read_kind_three_legs_exec_write_leg_at_u0(self):
        # C2 移交（C4 落地，2026-09-22）：A4' 补价 rider 后 kind="read"
        # 恒三腿（增执行端 HBM 写腿——镜像执行侧 remote-read 每 credit
        # 块到达的 COMM_WRITE 服务；F14 容量半边不动：读流直达消费、
        # 在途字节不占执行端 HBM 容量）。本夹具 u_home=u_exec=0
        # （port_registry=None 离线口径）：exec 写腿与 home 读腿同率
        # （各 6000/10 = 600），max 腿不变 → read 与 copy 同值 600
        #（断言锚不漂——C1 交付时两动作同值的成因自"read 无写腿"变为
        # "u=0 时写腿与读腿同值"）；kind 枚举外的值 fail-closed。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        read_wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
            include_self=True, kind="read")
        self.assertEqual(read_wall, 600)
        copy_wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
            include_self=True, kind="copy")
        self.assertEqual(read_wall, copy_wall)
        with self.assertRaises(Exception):
            _transfer_ns_shards(
                rates, LinkFlowRegistry(), None,
                paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
                include_self=True, kind="prefetch")


# ==================================================== 3. TP 并行口径 ==


class TpParallelismTest(unittest.TestCase):
    """TP6 不相交路径 wall = 单 shard 时间；共链自身重叠计费保留。"""

    BYTES_PER_RANK = 60

    def test_disjoint_paths_wall_is_single_shard(self):
        # 不相交：除数并集每链仅 1 条自身流 → divisor=1，各 shard 6 ns
        # 并行 → wall = 6 = 单 shard 时间；旧聚合口径（总 360 B ÷ 单
        # 链路率 10）= 36 = 6×（串行放大，即本卡废除的口径）。
        rates = _rates(noc=10.0, hbm=1e9, lat=0)
        paths = tuple(((rank, rank + 6),) for rank in range(6))
        bytes6 = (self.BYTES_PER_RANK,) * 6
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths, bytes_by_rank=bytes6,
            include_self=True, kind="read")
        self.assertEqual(wall, 6)
        legacy = _transfer_ns(
            total_bytes=sum(bytes6), path_hops=1, divisor=1,
            rates=rates, per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(legacy, 36)
        self.assertNotEqual(wall, legacy)

    def test_shared_link_keeps_self_overlap_billing(self):
        # 共链（6 条 TP 流同走 (0,6)）：divisor_multi 自身份额 = 6（divisor
        # 含 self）→ 每 shard 60/(10/6) = 36 ns → wall = 36 = 6× 单 shard
        # ——共享争用如实计费（自身重叠计费保留，divisor_multi 语义不动）。
        rates = _rates(noc=10.0, hbm=1e9, lat=0)
        registry = LinkFlowRegistry()
        paths = tuple(((0, 6),) for _ in range(6))
        bytes6 = (self.BYTES_PER_RANK,) * 6
        self.assertEqual(
            registry.divisor_multi(
                [path for rank_paths in paths for path in rank_paths],
                include_self=True), 6)
        wall = _transfer_ns_shards(
            rates, registry, None,
            paths_by_rank=paths, bytes_by_rank=bytes6,
            include_self=True, kind="read")
        self.assertEqual(wall, 36)
        # include_self=False 时不计自身流（除数回落 1）——参数语义对照。
        wall_no_self = _transfer_ns_shards(
            rates, registry, None,
            paths_by_rank=paths, bytes_by_rank=bytes6,
            include_self=False, kind="read")
        self.assertEqual(wall_no_self, 6)

    def test_zero_byte_shard_streams_nothing(self):
        # 零字节 shard 无流、不进除数并集：单 6000 B shard + 零字节
        # shard 共链 → wall = 600（若误计零字节自身份额则为 1200）。
        rates = _rates(noc=10.0, hbm=1e9, lat=0)
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 6),), ((0, 6),)),
            bytes_by_rank=(6000, 0),
            include_self=True, kind="read")
        self.assertEqual(wall, 600)

    def test_copy_disjoint_through_estimate_action(self):
        # estimate_action 层（copy NoC 前缀腿）：disjoint 逐 rank 路径
        # → 每 rank 3000/10 = 300 → history_prep = 10 + 300；旧聚合口径
        # = 10 + 600（总 6000 ÷ 10）。
        model = _model(rates=_rates(noc=10.0, hbm=1e9))
        candidate = model.estimate_action(
            session=_session(history_bytes=(3000, 3000)),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.history_prep_ns, 10 + 300)

    def test_endpoint_leg_binds_through_estimate_action(self):
        # 端点腿正式消费（E12）：noc=4050/hbm=1640（ρ≈2.47）下 home 读
        # 腿 500 ns 主导（noc 腿 202.5）→ history_prep = 10 + 500 > 旧
        # 聚合口径 10 + 404。
        model = _model(rates=_rates(noc=4050.0, hbm=1640.0))
        candidate = model.estimate_action(
            session=_session(history_bytes=(820_000, 820_000)),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        self.assertEqual(candidate.breakdown.history_prep_ns, 10 + 500)
        legacy = _transfer_ns(
            total_bytes=1_640_000, path_hops=1, divisor=1,
            rates=model.rates, per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(legacy, 10 + 404)
        self.assertGreater(candidate.breakdown.history_prep_ns, legacy)


# ==================================================== 4. 读放大因果（D1）==


class ReadAmplificationCausalTest(unittest.TestCase):
    """H=100k 基础 + I=20k 增量：decode 远读基数 = 100k（非 120k）。"""

    def _model(self):
        # noc=1000、hbm=1e9、lat=0、tp=2 disjoint → 纯 noc 腿可读数：
        # per-step per-rank = base_rank/1000。
        return _model(rates=_rates(noc=1000.0, hbm=1e9, lat=0))

    def test_decode_remote_base_is_history_only(self):
        # H = (50000, 50000) = 100k；I = (10000, 10000) = 20k（input +
        # 因果 decode 增长均属执行端本地读取，不计远读）。passes = 1
        # （prefill 单遍缺省）+ 10（decode 步）= 11 → 全流 wall = 50000
        # × 11 / 1000 = 550。若误用 H+I = 120k 基数则为 660。
        model = self._model()
        candidate = model.estimate_action(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=10,
                             input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.remote_read_ns, 550)
        self.assertNotEqual(candidate.breakdown.remote_read_ns, 660)
        self.assertIn("remote_read_passes=11", candidate.breakdown.notes)
        self.assertIn("prefill_scan_passes=1", candidate.breakdown.notes)

    def test_increment_size_does_not_inflate_remote_stream(self):
        # 同 H、I 放大 5×（100k 增量）：远读流不变（增量本地读取）。
        model = self._model()
        small = model.estimate_action(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        large = model.estimate_action(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=250, decode=10,
                             input_bytes=(50_000, 50_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertEqual(small.breakdown.remote_read_ns,
                         large.breakdown.remote_read_ns)

    def test_prefill_two_scan_passes_equals_two(self):
        # prefill 两遍扫描夹具：prefill_scan_passes=2、decode=0 → 遍数
        # = 2（prefill 对基础历史的实际消费遍数，非固定 1 遍）。
        model = self._model()
        candidate = model.estimate_action(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=0,
                             input_bytes=(10_000, 10_000),
                             prefill_scan_passes=2),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertIn("prefill_scan_passes=2", candidate.breakdown.notes)
        self.assertIn("remote_read_passes=2", candidate.breakdown.notes)
        # 两遍 × 50k/1000/rank → 100 ns。
        self.assertEqual(candidate.breakdown.remote_read_ns, 100)

    def test_default_prefill_scan_derivation(self):
        # 因果缺省：input > 0 → 单遍；input == 0 且 decode == 0 → 无读流
        #（遍数 0，credit 拆分恒 0、无 K 披露）。
        model = self._model()
        candidate = model.estimate_action(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=0,
                             input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertIn("remote_read_passes=1", candidate.breakdown.notes)
        empty = model.estimate_action(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=0, decode=0, input_bytes=(0, 0)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(empty.applicable)
        self.assertEqual(empty.breakdown.remote_read_ns, 0)
        self.assertEqual(empty.breakdown.remote_read_first_credit_ns, 0)
        self.assertEqual(empty.breakdown.remote_read_stream_ns, 0)

    def test_merge_legs_per_shard_disjoint(self):
        # merge 双腿逐 shard（disjoint）：exec 增量 (12000,12000)（input
        # (10000,10000) + 增长 (2000,2000)）→ 前向 = 12000/1000 = 12；
        # home 保留 (50000,50000) → 反向 = 50 → merge = min = 12（旧
        # 聚合前向 = 24000/1000 = 24，TP 并行减半）。
        model = self._model()
        candidate = model.estimate_action(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=10,
                             input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertEqual(candidate.breakdown.merge_ns, 12)
        self.assertIn("merge_v2_forward_ns=12", candidate.breakdown.notes)


# ==================================================== 5. credit 逐 shard ==


class CreditShardSplitTest(unittest.TestCase):
    """读流按 shard 字节分布切 credit 块：first/remaining 逐 shard 取 max。"""

    def test_skewed_ranks_driven_by_max_rank(self):
        # history (90000, 10000)、passes = 1+3 = 4、auto K = ceil(4/8) = 1
        # → credit1 = (90000, 10000)：first = max(90, 10) = 90；其余 3 遍
        # = (270000, 30000)：stream = max(270, 30) = 270；全流 = 360。
        model = _model(rates=_rates(noc=1000.0, hbm=1e9, lat=0))
        candidate = model.estimate_action(
            session=_session(history_bytes=(90_000, 10_000)),
            request=_request(input_tokens=20, decode=3,
                             input_bytes=(2000, 2000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        breakdown = candidate.breakdown
        self.assertIn("remote_credit_k=1", breakdown.notes)
        self.assertEqual(breakdown.remote_read_first_credit_ns, 90)
        self.assertEqual(breakdown.remote_read_stream_ns, 270)
        self.assertEqual(breakdown.remote_read_ns, 360)

    def test_single_credit_degenerate_invariant(self):
        # K ≥ read_passes（单 credit）：first == 全流总时延、remaining == 0
        # （不变量在逐 shard 口径下保持）。
        model = _model(
            rates=_rates(noc=1000.0, hbm=1e9, lat=0), credit_iters="4")
        candidate = model.estimate_action(
            session=_session(history_bytes=(90_000, 10_000)),
            request=_request(input_tokens=20, decode=3,
                             input_bytes=(2000, 2000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        breakdown = candidate.breakdown
        self.assertEqual(
            breakdown.remote_read_first_credit_ns,
            breakdown.remote_read_ns)
        self.assertEqual(breakdown.remote_read_stream_ns, 0)

    def test_hop_latency_only_in_first_credit(self):
        # 逐跳时延只进首块暴露段（其余读流纯流送段无时延）：lat=10、
        # disjoint 1 跳 → first = 10 + 90 = 100、stream = 270（无时延）。
        # _shard_stream_ns_shards 直接口径：与 _transfer_ns_shards 同
        # 腿源，仅剥离 startup/逐跳时延（650/350 = 首/余 per shard）。
        rates = _rates(noc=1000.0, hbm=1e9, lat=10)
        model = _model(rates=rates)
        candidate = model.estimate_action(
            session=_session(history_bytes=(90_000, 10_000)),
            request=_request(input_tokens=20, decode=3,
                             input_bytes=(2000, 2000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.remote_read_first_credit_ns, 100)
        self.assertEqual(breakdown.remote_read_stream_ns, 270)
        paths = (_disjoint_paths(0, 1),)
        paths_by_rank = tuple((path,) for path in paths[0])
        stream_only = _shard_stream_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths_by_rank,
            bytes_by_rank=(240_000, 240_000),
            include_self=True, kind="read")
        with_latency = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths_by_rank,
            bytes_by_rank=(240_000, 240_000),
            include_self=True, kind="read")
        self.assertEqual(stream_only, 240)
        self.assertEqual(with_latency, 240 + 10)


# ============================================ 6. 代表路径广播锚（C4 桥接）==


class RepresentativeBroadcastParityTest(unittest.TestCase):
    """route_paths_fn 未注入（单测/离线代表路径）：字节均衡、空链路时
    逐 shard 广播口径与旧聚合口径同 wall 值——C4 金值重推导的桥接锚。"""

    def test_balanced_bytes_match_legacy_aggregate(self):
        # 代表路径广播 → 除数 = 2（自身重叠按 rank 数计）→ 每 shard
        # 3000/(10/2) = 600 → wall = 10 + 600 ≡ 旧聚合
        # _transfer_ns(6000, hops=1, divisor=1) = 10 + 600。
        model = _model(rates=_rates(noc=10.0, hbm=1e5), route_paths_fn=None)
        candidate = model.estimate_action(
            session=_session(history_bytes=(3000, 3000)),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        legacy = _transfer_ns(
            total_bytes=6000, path_hops=1, divisor=1,
            rates=model.rates, per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(candidate.breakdown.history_prep_ns, legacy)
        self.assertEqual(candidate.breakdown.history_prep_ns, 610)

    def test_pool_suffix_leg_unchanged(self):
        # F3 池腿保真：copy 混合腿的池后缀（missing 守卫）口径不变。
        model = _model(rates=_rates(noc=10.0, hbm=1e5), route_paths_fn=None)
        candidate = model.estimate_action(
            session=_session(history_bytes=(3000, 3000),
                             missing=(1000, 1000),
                             location="partial_hbm_remote", prefix=3),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        expected_pool = _pool_transfer_ns(
            total_bytes=2000, divisor=1, rates=model.rates)
        self.assertEqual(
            candidate.breakdown.history_prep_ns, 610 + expected_pool)
        self.assertEqual(expected_pool, 100 + 400)


if __name__ == "__main__":
    unittest.main()
