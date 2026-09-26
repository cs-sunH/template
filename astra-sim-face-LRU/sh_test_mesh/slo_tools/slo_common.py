#!/usr/bin/env python3
"""SLO 离线后处理工具集——共享框架。

本模块集中存放 slo_tools 各脚本共用的框架代码：

* ``slo_params_manifest.json`` 的 fail-closed 装载（B 类参数 value=null 即
  拒绝执行，绝不内置示例值）；
* ``request_metrics.csv`` 的冻结列序（EXECUTION_PLAN §2）与严格校验读取；
* 整数纳秒/NA 语义、nearest-rank 分位（与 C++ MetricCollector 同法）；
* run_dir 事实装载（cpp.log init 行 / decision log / train ledger /
  metrics_manifest.json）。

语义映射的逐仓差异不放本模块，放各脚本的 REPO_VARIANTS 表或显式
per-repo 分支。所有工具仅用标准库。
"""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

# ---------------------------------------------------------------------------
# 退出码与错误
# ---------------------------------------------------------------------------

EXIT_FAIL_CLOSED = 2  # 输入缺失/格式错/参数未推导 → 明确报错后非零退出


class SloToolError(RuntimeError):
    """fail-closed 错误：消息面向用户，携带定位与处置提示。"""


def fail(message: str) -> None:
    raise SloToolError(message)


def run_main(main_fn) -> int:
    """统一入口：SloToolError → stderr + EXIT_FAIL_CLOSED。"""
    try:
        return main_fn()
    except SloToolError as exc:
        print(f"[slo-tools] FAIL-CLOSED: {exc}", file=sys.stderr)
        return EXIT_FAIL_CLOSED


# ---------------------------------------------------------------------------
# slo_params_manifest.json（B 类参数，fail-closed）
# ---------------------------------------------------------------------------

SLO_MANIFEST_FILENAME = "slo_params_manifest.json"


def default_manifest_path() -> Path:
    """默认 manifest = 本模块所在目录（sh_test_mesh/slo_tools/）下的清单。"""
    return Path(__file__).resolve().parent / SLO_MANIFEST_FILENAME


def load_slo_manifest(path: Optional[Path] = None) -> dict:
    manifest_path = Path(path) if path is not None else default_manifest_path()
    if not manifest_path.is_file():
        fail(
            f"SLO 参数清单不存在：{manifest_path}（--manifest 可显式指定；"
            f"仓内默认位置 sh_test_mesh/slo_tools/{SLO_MANIFEST_FILENAME}）")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"SLO 参数清单不是合法 JSON：{manifest_path}: {exc}")
    if manifest.get("schema_version") != 1:
        fail(
            f"SLO 参数清单 schema_version 必须为 1，实得 "
            f"{manifest.get('schema_version')!r}（{manifest_path}）")
    if not isinstance(manifest.get("params"), dict):
        fail(f"SLO 参数清单缺少 params 对象：{manifest_path}")
    return manifest


def require_param(manifest: dict, name: str) -> Any:
    """读取 B 类参数；null/缺失/类型不符一律 fail-closed。

    manifest 中的数值参数在 B4 批次推导完成前均为 null——此时任何依赖
    该参数的统计/判定都必须拒绝执行，禁止代入示例值。
    """
    params = manifest.get("params") or {}
    entry = params.get(name)
    if entry is None:
        fail(
            f"参数未推导（manifest 缺条目）：{name}——按主规格推导程序"
            f"（见 slo_params_manifest.json.derivation_program）推导后填入")
    if not isinstance(entry, dict) or "value" not in entry:
        fail(f"参数条目结构错误：{name}（需为 {{value, unit, derivation_program, evidence, rationale}}）")
    value = entry.get("value")
    if value is None:
        fail(
            f"参数未推导（value=null）：{name}——B4 批次按推导程序填充前，"
            f"本命令拒绝执行（fail-closed，禁止内置示例值）。推导程序："
            f"{entry.get('derivation_program')}")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        fail(f"参数值非法（NaN/Inf）：{name}")
    return value


def require_param_number(manifest: dict, name: str) -> float:
    value = require_param(manifest, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"参数必须为数值：{name}（实得 {value!r}）")
    return float(value)


def require_param_int(manifest: dict, name: str) -> int:
    value = require_param_number(manifest, name)
    if value != int(value):
        fail(f"参数必须为整数：{name}（实得 {value!r}）")
    return int(value)


# 分桶边界（bucket_percentiles）的结构化读取。
def require_bucket_edges(manifest: dict) -> tuple[list[float], list[float]]:
    """返回 (prefill_edges_tokens, decode_edges_tokens)。

    value 结构（B4 填充后）：{"percentiles": [...],
    "prefill_edges_tokens": [...], "decode_edges_tokens": [...]}，
    edges 含首尾哨兵；interior edges 为各左桶闭上界（桶 i 覆盖
    edges[i] < x <= edges[i+1] 的整数域右闭区间，首桶左端含 edges[0]），
    末桶无上限（x > edges[-1] 归末桶）——2026-09-05 口径裁决，与
    campaign_common.BucketGrid 语义统一；edges 数组值未变。
    """
    value = require_param(manifest, "bucket_percentiles")
    if not isinstance(value, dict):
        fail("bucket_percentiles.value 必须为对象：{percentiles, prefill_edges_tokens, decode_edges_tokens}")
    prefill_edges = value.get("prefill_edges_tokens")
    decode_edges = value.get("decode_edges_tokens")
    for name, edges in (("prefill_edges_tokens", prefill_edges),
                        ("decode_edges_tokens", decode_edges)):
        if not isinstance(edges, list) or len(edges) < 2:
            fail(f"bucket_percentiles.{name} 必须为长度>=2 的数值数组（含首尾哨兵）")
        if any(isinstance(e, bool) or not isinstance(e, (int, float)) for e in edges):
            fail(f"bucket_percentiles.{name} 含非数值元素")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            fail(f"bucket_percentiles.{name} 必须严格递增")
    return list(prefill_edges), list(decode_edges)


def bucket_index(edges: Sequence[float], x: float) -> int:
    """interior edges 为各左桶闭上界，末桶无上限。

    桶 i（非末桶）覆盖 edges[i] < x <= edges[i+1]——interior 边界值归左桶
    （如 decode edges [1,91,489,1785,32000] 下 91→桶 0、92→桶 1、
    1785→桶 2、1786→桶 3；prefill [1,415,2361,16470,950002] 下 415→桶 0）；
    末桶吸收一切越界值（x > edges[-1] 不再 fail-closed，归末桶）；
    x < edges[0] 仍报错。

    2026-09-05 口径裁决：与 campaign_common.BucketGrid（interior bounds
    为各桶闭上界、末桶延伸至 +inf）完全等价；edges 数组值一律未变，
    仅重解释语义（此前为左闭右开 edges[i] <= x < edges[i+1] 且
    x > edges[-1] 报错）。
    """
    if len(edges) < 2:
        fail("分桶边界至少需要两个哨兵值")
    if x < edges[0]:
        fail(f"长度 {x} 小于分桶边界下哨兵 {edges[0]}——"
             f"分桶边界与数据不匹配（边界以 manifest 冻结值为准）")
    lo, hi = 0, len(edges) - 2
    # 最小 i 使 x <= edges[i+1]（interior 边界归左桶）；均不满足则归末桶
    while lo < hi:
        mid = (lo + hi) // 2
        if x <= edges[mid + 1]:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# request_metrics.csv（WP1 产物，列序冻结，EXECUTION_PLAN §2）
# ---------------------------------------------------------------------------

REQUEST_METRICS_COLUMNS: tuple[str, ...] = (
    "queue_index", "request_id", "session_id", "turn_index", "request_type",
    "terminal_status", "arrival_ns", "prefill_start_ns", "prefill_end_ns",
    "decode_start_ns", "first_token_ns", "first_token_source",
    "completion_ns", "queue_ns", "prefill_ns", "prefill_decode_gap_ns",
    "decode_ns", "e2e_ns", "kv_hit_state", "restore_start_ns",
    "restore_complete_ns", "pre_prefill_restore_ns", "hidden_restore_ns",
    "exposed_restore_stall_ns", "hidden_ratio", "prefill_length",
    "decode_length", "prefix_len", "instructions",
)

REQUEST_METRICS_FILENAME = "request_metrics.csv"

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "rejected", "dropped", "timed_out", "failed"})
REQUEST_TYPES: frozenset[str] = frozenset({"human", "tool", "unknown"})
KV_HIT_STATES: frozenset[str] = frozenset(
    {"full", "partial", "miss", "no_history", "not_supported"})
FIRST_TOKEN_SOURCES: frozenset[str] = frozenset(
    {"exact", "train_interpolated", "NA"})

NA = "NA"


def is_na(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == NA)


def parse_ns(value: Any, field: str, where: str) -> Optional[int]:
    """整数纳秒解析：'NA'→None；其余必须为非负整数（fail-closed）。"""
    if is_na(value):
        return None
    text = str(value).strip()
    try:
        number = int(text, 10)
    except ValueError:
        fail(f"{where}: 字段 {field} 不是整数纳秒也不是 'NA'（实得 {value!r}）")
    if number < 0:
        fail(f"{where}: 字段 {field} 为负数（{number}）——整数纳秒时间戳不允许")
    return number


def parse_int(value: Any, field: str, where: str) -> Optional[int]:
    if is_na(value):
        return None
    text = str(value).strip()
    try:
        return int(text, 10)
    except ValueError:
        fail(f"{where}: 字段 {field} 不是整数也不是 'NA'（实得 {value!r}）")


def parse_float(value: Any, field: str, where: str) -> Optional[float]:
    if is_na(value):
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        fail(f"{where}: 字段 {field} 不是数值也不是 'NA'（实得 {value!r}）")


def read_request_metrics(path: Path) -> Iterator[dict[str, str]]:
    """流式读取并校验 request_metrics.csv（单遍契约；列序/列集与 §2
    冻结值逐一相等）。

    返回逐行记录的迭代器，不物化整表（2026-08-30 阶段2加固 §3.3 一档：
    旧实现 rows/records/typed 三份全量物化降为一份 typed）。失败时点：
    文件缺失与表头列序/列集校验在调用时即 fail（消息原文不变，调用方
    无需迭代即可见到输入错误）；数据行在迭代中逐行产出——列数校验
    沿用旧实现 ``enumerate(rows, start=2)`` 对"过滤空行后"的行序列编号
    的行号语义（空行被跳过后行号随之偏移的既有怪癖原样保留，列数错误
    消息里的行号才能与历史产物对上）；数据行计数为 0 在迭代耗尽时
    fail（无数据行）。迭代器只可消费一次，消费方应单遍处理完毕。
    """
    if not path.is_file():
        fail(
            f"缺少 {REQUEST_METRICS_FILENAME}: {path}——WP1（B1 批次）产物；"
            f"run_dir 需为含 request_metrics.csv/cpp.log/manifest 的运行目录")
    handle = path.open(newline="", encoding="utf-8")
    reader = csv.reader(handle)
    try:
        try:
            header = next(reader)
        except StopIteration:
            fail(f"{path}: 空文件（无表头）")
        if tuple(header) != REQUEST_METRICS_COLUMNS:
            missing = [c for c in REQUEST_METRICS_COLUMNS if c not in header]
            extra = [c for c in header if c not in REQUEST_METRICS_COLUMNS]
            fail(
                f"{path}: request_metrics.csv 列序/列集与冻结 schema 不符"
                f"（EXECUTION_PLAN §2）；缺失={missing} 多余={extra}")
    except BaseException:
        handle.close()
        raise

    def _records() -> Iterator[dict[str, str]]:
        # 生成器体内持句柄单遍流式（耗尽/异常/放弃时 finally 关句柄）；
        # 旧实现的两份中间物化（rows 列表 + records dict 列表）就此取消。
        lineno = 1  # 表头为第 1 行；数据行沿用"过滤空行后连续编号"语义
        count = 0
        try:
            for row in reader:
                if not row:
                    continue
                lineno += 1
                if len(row) != len(header):
                    fail(f"{path}:{lineno}: 列数 {len(row)} != {len(header)}")
                count += 1
                yield dict(zip(header, row))
        finally:
            handle.close()
        if count == 0:
            fail(f"{path}: 无数据行")

    return _records()


def validated_request_rows(records: Iterable[dict[str, str]],
                           path: Path) -> list[dict[str, Any]]:
    """逐行做枚举/数值校验，返回类型化行（ns 字段转 int 或 None）。

    records 放宽为 Iterable（含 read_request_metrics 的流式迭代器，用完
    即弃）。行 dict 就地写入类型化字段后 append——不再 dict(record) 复制
    （上游迭代器逐行产出独立 dict，无共享引用；与 read_request_metrics
    组合后峰值 = 1 份 typed 行 + 一个 queue_index int 集合）。queue_index
    查重改为循环内维护 seen 集合 + dup 标志，全部行处理完才统一 fail
    （保持旧实现"先逐行校验后查重"的时序与消息原文）。
    """
    typed: list[dict[str, Any]] = []
    seen: set[int] = set()
    dup = False
    for record in records:
        where = f"{path}:queue_index={record.get('queue_index', '?')}"
        queue_index = parse_int(record.get("queue_index"), "queue_index", where)
        if queue_index is None:
            fail(f"{where}: queue_index 不允许为 NA")
        status = record.get("terminal_status", "")
        if status not in TERMINAL_STATUSES:
            fail(f"{where}: terminal_status 非法（{status!r}），合法值 "
                 f"{sorted(TERMINAL_STATUSES)}")
        request_type = record.get("request_type", "")
        if request_type not in REQUEST_TYPES:
            fail(f"{where}: request_type 非法（{request_type!r}）")
        kv_state = record.get("kv_hit_state", "")
        if kv_state not in KV_HIT_STATES and not is_na(kv_state):
            fail(f"{where}: kv_hit_state 非法（{kv_state!r}）")
        ft_source = record.get("first_token_source", "")
        if ft_source not in FIRST_TOKEN_SOURCES:
            fail(f"{where}: first_token_source 非法（{ft_source!r}）")
        for column in ("arrival_ns", "prefill_start_ns", "prefill_end_ns",
                       "decode_start_ns", "first_token_ns", "completion_ns",
                       "queue_ns", "prefill_ns", "prefill_decode_gap_ns",
                       "decode_ns", "e2e_ns", "restore_start_ns",
                       "restore_complete_ns", "pre_prefill_restore_ns",
                       "hidden_restore_ns", "exposed_restore_stall_ns",
                       "prefill_length", "decode_length", "prefix_len"):
            record[column] = parse_ns(record.get(column), column, where)
        record["hidden_ratio"] = parse_float(
            record.get("hidden_ratio"), "hidden_ratio", where)
        record["turn_index"] = parse_int(
            record.get("turn_index"), "turn_index", where)
        record["_queue_index"] = queue_index
        typed.append(record)
        if queue_index in seen:
            dup = True
        else:
            seen.add(queue_index)
    if dup:
        fail(f"{path}: queue_index 存在重复（manifest 连接要求唯一）")
    return typed


# ---------------------------------------------------------------------------
# 统计原语（整数纳秒、先分位后转单位）
# ---------------------------------------------------------------------------

def nearest_rank_percentile(values: Sequence[int], p: float) -> int:
    """nearest-rank 分位：index = ceil(p*N) - 1（与 C++ MetricCollector
    ``nearest_rank_percentile`` 同法，doc sec.3.5）；整数纳秒上直接取值，
    之后才允许转换单位（先分位后转单位，主规格 §1.5-A）。"""
    n = len(values)
    if n == 0:
        fail("分位数计算：样本为空")
    if not 0.0 < p <= 1.0:
        fail(f"分位数 p 非法：{p}")
    rank = math.ceil(p * n)
    index = min(max(rank - 1, 0), n - 1)
    ordered = sorted(values)
    return ordered[index]


# ---------------------------------------------------------------------------
# 有界内存排序（A4，2026-08-29）
# ---------------------------------------------------------------------------

# 内存合同（A4_DESIGN §2.4）：
#   * chunk 容量 C = env SH_SLO_SORT_CHUNK（缺省 65536 元素；<=0 按 1 计）；
#   * 排序期驻留 = 恰一个未满 chunk（<=C 元素）+ spill 文件句柄；归并期
#     额外持有每路 spill 的有界读缓冲（heapq.merge 惰性迭代，不整读）；
#   * spill 文件 = pickle 序列化的已排序 chunk，落在系统临时目录，
#     sorted_iter 耗尽后删除（进程退出兜底再删一次）；
#   * 只用于纯整数/整数元组的全序排序（比较语义与 sorted() 逐元素相同，
#     外部归并输出序列 ≡ sorted()；相等元素不可区分 ⇒ 无稳定性可见差）。
#   * 需要稳定序（比较键并列时按输入序）的调用方必须自带单调序号列
#     （如 (tick, seq, payload)）——见 slo_stats/hbm_watermark 接入点。
SORT_CHUNK_ENV = "SH_SLO_SORT_CHUNK"
SORT_CHUNK_DEFAULT = 65536


def bounded_sort_chunk_size() -> int:
    raw = os.environ.get(SORT_CHUNK_ENV, "")
    try:
        size = int(raw) if raw else SORT_CHUNK_DEFAULT
    except ValueError:
        size = SORT_CHUNK_DEFAULT
    return size if size > 0 else 1


class BoundedSorter:
    """有界内存排序通道：add 累积、sorted_iter 一次性升序迭代。

    小数据量（总元素数 <= chunk 容量）时等价于内存 sorted()，零行为差；
    超出后自动排序+spill+多路归并。数据通道替换，不改任何比较键/公式
    （A4 红线 R9）。sorted_iter 只可消费一次。
    """

    def __init__(self, chunk_size: Optional[int] = None) -> None:
        self._chunk_size = (chunk_size if chunk_size is not None
                            else bounded_sort_chunk_size())
        if self._chunk_size < 1:
            self._chunk_size = 1
        self._buffer: list = []
        self._spills: list[Path] = []

    def add(self, item: Any) -> None:
        self._buffer.append(item)
        if len(self._buffer) >= self._chunk_size:
            self._spill_buffer()

    def _spill_buffer(self) -> None:
        if not self._buffer:
            return
        self._buffer.sort()
        fd, name = tempfile.mkstemp(prefix="slo_bounded_sort_",
                                    suffix=".pkl")
        os.close(fd)
        path = Path(name)
        with path.open("wb") as handle:
            pickle.dump(self._buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._spills.append(path)
        self._buffer = []

    @staticmethod
    def _iter_spill(path: Path) -> Iterator:
        with path.open("rb") as handle:
            while True:
                try:
                    yield from pickle.load(handle)
                except EOFError:
                    break

    def sorted_iter(self) -> Iterator:
        """升序迭代（一次性）。spill 文件在迭代结束后删除。"""
        if not self._spills:
            return iter(sorted(self._buffer))
        self._spill_buffer()
        spills = list(self._spills)
        self._spills = []
        iters = [self._iter_spill(path) for path in spills]

        def _merged():
            try:
                yield from heapq.merge(*iters)
            finally:
                for path in spills:
                    try:
                        path.unlink()
                    except OSError:
                        pass
        return _merged()


def _nearest_rank_index(p: float, n: int) -> int:
    """nearest-rank 索引（与 nearest_rank_percentile 逐字符同式）。"""
    rank = math.ceil(p * n)
    return min(max(rank - 1, 0), n - 1)


def nearest_rank_percentile_many(values: Sequence[int],
                                 p_list: Sequence[float]) -> list[int]:
    """多个 nearest-rank 分位共享一次有界排序（A4）。

    公式与 nearest_rank_percentile 完全一致（index = ceil(p·N)−1，在
    sorted(values) 上取值）；多个 p 按目标索引升序单遍推进游标，结果按
    p_list 原顺序返回。失败语义（空样本/非法 p）也逐字符一致。
    """
    n = len(values)
    if n == 0:
        fail("分位数计算：样本为空")
    for p in p_list:
        if not 0.0 < p <= 1.0:
            fail(f"分位数 p 非法：{p}")
    order = sorted(range(len(p_list)),
                   key=lambda i: _nearest_rank_index(p_list[i], n))
    sorter = BoundedSorter()
    for value in values:
        sorter.add(value)
    ordered = sorter.sorted_iter()
    results: list[Optional[int]] = [None] * len(p_list)
    cursor = -1
    current: Optional[int] = None
    for i in order:
        index = _nearest_rank_index(p_list[i], n)
        while cursor < index:
            current = next(ordered)
            cursor += 1
        results[i] = current
    assert all(value is not None for value in results)
    return [value for value in results if value is not None]  # type: ignore[ruff]


def fmt_ratio(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return NA
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# run_dir 事实装载
# ---------------------------------------------------------------------------

DECISION_LOG_RELPATH = Path("results") / "online_decision_log.jsonl"
TRAIN_LEDGER_RELPATH = Path("results") / "train_ledger.jsonl"
CPP_LOG_NAME = "cpp.log"
METRICS_LOG_NAME = "metrics.log"
CPP_LOG_GZ_NAME = "cpp.log.gz"


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        fail(f"缺少{what}：{path}")
    return path


def resolve_cpp_metric_log(run_dir: Path) -> Path:
    """cpp.log 的统一回退解析（P1，治归档后断点）。

    探测顺序：cpp.log（未归档 / SH_ARCHIVE_RUN=0）→ metrics.log（归档后
    常驻，archive_run_outputs.sh 对 cpp.log [METRIC] 行的无损抽取）→
    cpp.log.gz（归档后全量原文）。三条路径读到的 [METRIC] 记录集合同一，
    信息完备；slo_tools 所有读 cpp.log 的入口（read_init_record /
    restore_decomposition.collect 等）一律经本函数取路径。
    """
    for name in (CPP_LOG_NAME, METRICS_LOG_NAME, CPP_LOG_GZ_NAME):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    fail(
        f"{run_dir}: 找不到 {CPP_LOG_NAME}/{METRICS_LOG_NAME}/{CPP_LOG_GZ_NAME}"
        f" 中任一（未归档 run 应有 cpp.log；归档 run 应有 metrics.log 与 "
        f"{CPP_LOG_GZ_NAME}）——无法读取 [METRIC] 记录")


def iter_jsonl(path: Path) -> Iterator[dict]:
    require_file(path, f"JSONL 文件")
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{lineno}: 非法 JSON（{exc}）")
            if not isinstance(record, dict):
                fail(f"{path}:{lineno}: JSONL 行必须是对象")
            yield record


def read_cpp_metric_records(cpp_log: Path) -> Iterator[dict]:
    """逐行解析 cpp.log 的 ``[METRIC] {...}`` 行（summary/full 档均适用）。

    路径可为 cpp.log / metrics.log / cpp.log.gz：后缀 ``.gz`` 自动以 gzip
    文本模式读（回退顺序见 resolve_cpp_metric_log）。
    """
    require_file(cpp_log, "cpp.log（或归档回退 metrics.log/cpp.log.gz）")
    if cpp_log.name.endswith(".gz"):
        handle = gzip.open(cpp_log, "rt", encoding="utf-8", errors="replace")
    else:
        handle = cpp_log.open(encoding="utf-8", errors="replace")
    with handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped.startswith("[METRIC]"):
                continue
            payload = stripped[len("[METRIC]"):].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                fail(f"{cpp_log}:{lineno}: [METRIC] 行 JSON 非法（{exc}）")
            if not isinstance(record, dict):
                fail(f"{cpp_log}:{lineno}: [METRIC] 行必须是 JSON 对象")
            yield record


def read_init_record(run_dir: Path) -> dict:
    """首个 type=init 的 [METRIC] 记录（repo_variant/manifest_path 等事实）。

    读入路径经 resolve_cpp_metric_log 统一回退（cpp.log → metrics.log →
    cpp.log.gz），归档后的 run_dir 同样可用。
    """
    cpp_log = resolve_cpp_metric_log(run_dir)
    for record in read_cpp_metric_records(cpp_log):
        if record.get("type") == "init":
            return record
    fail(f"{cpp_log}: 未找到 type=init 的 [METRIC] 行"
         f"（无法识别 repo_variant 与 per-request manifest）")


def detect_repo_variant(run_dir: Path, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    init = read_init_record(run_dir)
    variant = init.get("repo_variant")
    if not variant:
        fail(f"{run_dir}: cpp.log init 行缺 repo_variant（用 --repo-variant 显式指定）")
    return str(variant)


def load_request_manifest(run_dir: Path,
                          explicit: Optional[Path] = None) -> dict:
    """per-request manifest（WP1/WP2 连接与 session 字段透传来源）。

    优先级：--request-manifest 显式 > run_dir/metrics_manifest.json >
    cpp.log init 行 manifest_path 指向的 generated 副本。
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        local = run_dir / "metrics_manifest.json"
        if local.is_file():
            candidates.append(local)
        try:
            init = read_init_record(run_dir)
        except SloToolError:
            init = {}
        manifest_path = init.get("manifest_path")
        if manifest_path:
            candidates.append(Path(str(manifest_path)))
    for candidate in candidates:
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                fail(f"per-request manifest 非法 JSON：{candidate}: {exc}")
            if not isinstance(data, dict):
                fail(f"per-request manifest 必须是对象：{candidate}")
            return data
    fail(
        f"找不到 per-request manifest（尝试：{[str(c) for c in candidates]}）——"
        f"用 --request-manifest 指定（metrics_manifest.json，含 requests[]."
        f"session_id/turn_index/arrival 及 WP2 透传的 human_time_ns/"
        f"tool_time_ns）")


def manifest_requests(manifest: dict) -> list[dict]:
    requests = manifest.get("requests")
    if not isinstance(requests, list):
        fail("per-request manifest 缺 requests 数组")
    return [r for r in requests if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def open_output(path_arg: str, default_name: str,
                run_dir: Optional[Path] = None) -> tuple[Any, bool]:
    """返回 (stream, should_close)。

    '-' → stdout；''（CLI 缺省）→ run_dir/<default_name>（无 run_dir 则
    stdout）。这样缺省时产物落在 run_dir 旁，显式 '-' 恒为管道友好的
    stdout。
    """
    if path_arg == "-":
        return sys.stdout, False
    if path_arg == "":
        if not run_dir:
            return sys.stdout, False
        out = run_dir / default_name
        return out.open("w", newline="", encoding="utf-8"), True
    target = Path(path_arg)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("w", newline="", encoding="utf-8"), True


def write_csv(stream, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    writer = csv.writer(stream)
    writer.writerow(list(header))
    for row in rows:
        writer.writerow([NA if cell is None else cell for cell in row])


def emit_json(stream, payload: dict) -> None:
    json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
    stream.write("\n")


# SLO 判定禁用列（主规格 §1.2-A：合约仅含 E2E；TTFT 与任何 proxy 字段
# 不进入请求级 SLO 判定）。violation 等判定路径调用本函数自证清白。
SLO_FORBIDDEN_JUDGMENT_COLUMNS: frozenset[str] = frozenset(
    {"first_token_ns", "first_token_source"})


def assert_no_proxy_columns(columns: Iterable[str],
                            context: str = "SLO 判定") -> None:
    """显式断言：proxy/TTFT 字段不得进入 SLO 判定输入。

    任何试图把 first_token_ns / first_token_source 喂给判定路径的输入
    （如 T_isolated 表附带的 proxy 列）都会被拒绝，退出码 EXIT_FAIL_CLOSED。
    """
    offending = SLO_FORBIDDEN_JUDGMENT_COLUMNS.intersection(columns)
    if offending:
        fail(
            f"{context}：输入携带禁用 proxy/TTFT 列 {sorted(offending)}——"
            f"主规格 §1.2-A 规定 SLO 合约仅含 E2E，TTFT 与 proxy 字段"
            f"不得进入请求级判定")
