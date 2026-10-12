"""predictor_cost_accounting.py -- 预测器自身开销统计接口（ACC）。

条目：joint修改文档「验收纪律（v2.0 §19 三类，全部纳入）」第 3 类
（仿真可行性，v2.0 §19.3）的**代码侧统计接口**：

  - 预测器自身决策 wall time 逐次样本（P 准入 / D 边界两阶段分列），
    汇总 P50/P95/P99（最近秩法，整数纳秒、确定性）与 total/mean/max；
  - 候选面宽度（candidates_max）——供"4N 变 8N 不推定总仿真只慢固定
    倍数"（§19.3）的逐决策证据；
  - 摘要维护费：D11 容量驱逐摘要冻结快照（capacity_summary_snapshot）
    的查询墙时样本；
  - 后端事件量：每已应用交付的图批节点/边/watch/未来 alarm 计数
    （幂等重放不经过 build_graph_batch，天然不重复计数）；
  - run 末快照：预测器合计、宿主总墙时与总占比（整数 ppm）、峰值
    RSS（ru_maxrss；平台相关口径见 ``_peak_rss``）、决策日志大小
    （由调用方在日志落定后以字节回填；缺省 None = 未读）。

设计红线：
  - **恒开、无开关**：§19.3 是验收必需项，不是可选机制——不设 env/
    CLI/配置键，宿主侧热路径只增加 perf_counter 差分与整数累加；
  - 纯宿主 Python 墙时不自动等于真实片上决策时延（§19.3 末段），
    快照字段只如实披露测量口径，不外推任何部署结论；
  - 首轮工程目标（两次决策额外宿主时间 ≲ 原固定动作回放 10%）是
    **待验证预算**，本接口只供数，不代下结论。

契约测试见 ``joint/test_acc_predictor_cost_accounting.py``（本模块
无自检主体、无 ``__main__`` 入口，不可直接运行）。
"""

from __future__ import annotations

import sys

# 峰值 RSS 源：resource.ru_maxrss（Linux = KiB，darwin = 字节）。
# Windows 无 resource 模块 → source="unavailable"、bytes=0（如实披露，
# 不用近似值冒充测量）。
try:  # pragma: no cover - 平台分支
    import resource as _resource

    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - 平台分支
    _resource = None
    _HAS_RESOURCE = False

_RU_MAXRSS_UNIT_BYTES = 1024 if sys.platform == "linux" else 1

# 决策阶段枚举：P = 初始映射准入（estimate 全候选遍历），D = P→D
# 边界二次决策（estimate_D_remaining 全候选遍历）。fail-closed：
# 未知阶段拒绝（新增阶段必须显式扩枚举，不静默归并）。
DECISION_STAGE_P = "P"
DECISION_STAGE_D = "D"
DECISION_STAGES = (DECISION_STAGE_P, DECISION_STAGE_D)

# 总占比单位：ppm（百万分比，整数）——避免在验收证据面引入浮点。
_SHARE_PPM_SCALE = 1_000_000

# 汇总分位档（v2.0 §19.3：P50/P95/P99）。
_PERCENTILES = ((50, "p50_ns"), (95, "p95_ns"), (99, "p99_ns"))


def _require_uint(value, name: str) -> int:
    """非负整数校验（bool 显式拒绝——Python bool 是 int 子类）。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be a non-negative int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


def _nearest_rank_percentile(sorted_samples, percentile: int) -> int:
    """最近秩法分位：升序第 ceil(p/100·n) 个样本（1 基）。空集 = 0。

    确定性：纯整数索引，无插值无浮点——同一批样本恒得同一分位。"""

    count = len(sorted_samples)
    if count == 0:
        return 0
    rank = -(-percentile * count // 100)  # ceil
    if rank < 1:
        rank = 1
    elif rank > count:
        rank = count
    return sorted_samples[rank - 1]


def _stage_stats_block(samples, candidates_max: int) -> dict:
    """单阶段决策墙时汇总块（count/total/mean/分位/max + 候选面宽）。"""

    total = sum(samples)
    count = len(samples)
    return {
        "count": count,
        "total_ns": total,
        "mean_ns": total // count if count else 0,
        "p50_ns": _nearest_rank_percentile(samples, 50),
        "p95_ns": _nearest_rank_percentile(samples, 95),
        "p99_ns": _nearest_rank_percentile(samples, 99),
        "max_ns": samples[-1] if samples else 0,
        "candidates_max": candidates_max,
    }


def _peak_rss() -> tuple[int, str]:
    """进程峰值 RSS（字节）与测量源披露。"""

    if not _HAS_RESOURCE:  # pragma: no cover - 平台分支
        return 0, "unavailable"
    # ru_maxrss：Linux 记 KiB、darwin 记字节（getrusage(2) 口径）。
    raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    return int(raw) * _RU_MAXRSS_UNIT_BYTES, "ru_maxrss"


class PredictorCostAccounting:
    """预测器自身开销账本（ACC 第 3 类统计接口；恒开，无开关）。"""

    def __init__(self) -> None:
        self._decision_samples: dict[str, list[int]] = {
            stage: [] for stage in DECISION_STAGES}
        self._candidates_max: dict[str, int] = {
            stage: 0 for stage in DECISION_STAGES}
        self._summary_maintenance_samples: list[int] = []
        self._backend_events = {
            "batches": 0, "nodes": 0, "parent_edges": 0,
            "watches": 0, "future_alarms": 0}
        self._log_bytes_total = 0

    # ------------------------------------------------------------ 记账 --

    def note_decision(self, stage: str, wall_ns: int, *,
                      candidates: int = 0) -> None:
        """记一次预测器决策墙时样本（纳秒；candidates = 本次遍历的候
        选面宽度，缺省 0 = 调用方未披露）。"""

        if stage not in self._decision_samples:
            raise ValueError(
                f"unknown predictor decision stage {stage!r} "
                f"(expected one of {DECISION_STAGES})")
        _require_uint(wall_ns, "wall_ns")
        _require_uint(candidates, "candidates")
        self._decision_samples[stage].append(int(wall_ns))
        if candidates > self._candidates_max[stage]:
            self._candidates_max[stage] = candidates

    def note_summary_maintenance(self, wall_ns: int) -> None:
        """记一次 D11 容量驱逐摘要快照的维护/查询墙时样本（纳秒）。"""

        _require_uint(wall_ns, "wall_ns")
        self._summary_maintenance_samples.append(int(wall_ns))

    def note_backend_batch(self, *, nodes: int, parent_edges: int,
                           watches: int, future_alarms: int) -> None:
        """记一批后端事件量（图批节点/边/watch/未来 alarm 计数；
        与 graph_batch_digests 行的四个 count 键同源同刻）。"""

        for name, value in (("nodes", nodes),
                            ("parent_edges", parent_edges),
                            ("watches", watches),
                            ("future_alarms", future_alarms)):
            _require_uint(value, name)
            self._backend_events[name] += value
        self._backend_events["batches"] += 1

    def note_log_bytes(self, size_bytes: int) -> None:
        """累计决策日志字节（流式 sink 消费方逐行回填口径；run 末
        文件级回填用 ``snapshot(decision_log_bytes=...)``，二者不混
        用——先到先记，snapshot 传值优先）。"""

        _require_uint(size_bytes, "size_bytes")
        self._log_bytes_total += int(size_bytes)

    # ---------------------------------------------------------- 快照 --

    def snapshot(self, *, host_wall_ns: int,
                 decision_log_bytes: int | None = None) -> dict:
        """run 末汇总快照（确定性整数面；host_wall_ns = 宿主全程墙
        时锚差，决策日志大小缺省 None = 未读，不用猜测值冒充）。"""

        _require_uint(host_wall_ns, "host_wall_ns")
        if decision_log_bytes is not None:
            _require_uint(decision_log_bytes, "decision_log_bytes")
        stages = {}
        for stage in DECISION_STAGES:
            samples = sorted(self._decision_samples[stage])
            stages[stage] = _stage_stats_block(
                samples, self._candidates_max[stage])
        decision_total = sum(
            sum(samples) for samples in self._decision_samples.values())
        decision_count = sum(
            len(samples) for samples in self._decision_samples.values())
        summary_total = sum(self._summary_maintenance_samples)
        summary_samples = sorted(self._summary_maintenance_samples)
        predictor_total = decision_total + summary_total
        peak_bytes, peak_source = _peak_rss()
        return {
            "decision_stages": stages,
            "decision_wall_ns": {
                "count": decision_count,
                "total_ns": decision_total,
            },
            "summary_maintenance": {
                "count": len(summary_samples),
                "total_ns": summary_total,
                "max_ns": summary_samples[-1] if summary_samples else 0,
            },
            "predictor_total_ns": predictor_total,
            "host_wall_ns": host_wall_ns,
            # 总占比（整数 ppm）：预测器合计（决策 + 摘要维护）/ 宿主
            # 全程墙时。宿主锚差为 0（构造即快照的退化夹具）→ None，
            # 不造 0/0。
            "host_share_ppm": (
                predictor_total * _SHARE_PPM_SCALE // host_wall_ns
                if host_wall_ns > 0 else None),
            "peak_rss": {"bytes": peak_bytes, "source": peak_source},
            "decision_log_bytes": decision_log_bytes,
            "backend_events": dict(self._backend_events),
        }
