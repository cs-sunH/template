#!/usr/bin/env python3
"""SLO 后处理单遍合并 driver（A4，2026-08-29；标准库实现）。

背景（读放大，A4_DESIGN §1.4）：run_slo_postprocess.sh 原以 9 个独立
python3 子命令串行提取 SLO 产物——request_metrics.csv 被整读+校验 4 次、
results/online_decision_log.jsonl 全量流 4 次（kv/load/hop/watermark）、
[METRIC] init 探测 4 次、manifest 族多次，另有 9 次解释器启动。

本 driver 把 9 步合并为单进程单遍驱动（顺序与产物集不变）：

  * RunContext 一次装载共享输入（rows / request_manifest / slo_manifest /
    repo_variant 惰性首用缓存；[METRIC] 流只在 restore 步全量读一次）；
  * decision log 单遍流，每条记录按固定次序喂给四个消费者
    （kv_cache_adapter → load_imbalance → hopbytes → hbm_watermark；
    watermark 的会话跟踪态在单循环内保序——记录内「逐出→恢复/迁移→
    增长」次序与独立遍历逐语句等价）。watermark 必须最后：其对记录
    注入 session_hint 键（其余 sink 不读该键）；
  * 各步 stderr 整块捕获后按步序回放——run_slo_postprocess.sh 把本进程
    stdout+stderr 追加进 slo_postprocess.log，块序=链序、块内=工具原生
    序，与九次子命令的 stderr 串联逐字节相同（红线 R5/R8）；
  * 每步复刻 run_step 的 "run:/ok:/FAIL:" 行与 slo_postprocess.FAIL
    条目语义；任一步失败 → 本进程退出码 1（shell 据此置 warn 态）；
    SloToolError → 该步 exit=2（与独立 CLI 的 run_main 一致）、
    hbm_watermark 容量违规 → exit=3（P1 起仅 per_rank_total_hbm_
    certified 层的逐 rank physical 认证；旧 decision-log 上界层超限
    只诊断，exit=0）；
  * 需排序分位数/事件的工具经 slo_common.BoundedSorter 有界外部排序
    （比较键与公式不动，红线 R9）。

门控（与 run_slo_postprocess.sh 双保险）：G1 输入解析 / G2 无 [METRIC]
行整步跳过 / G3 无 request_metrics 跳步 1-4+7 / G4 无 train_ledger 跳步
6。shell 已探测时经 SH_SLO_HAVE_RM /
SH_SLO_HAVE_TL / SH_SLO_SKIP_HBM 传入（不重复打跳过说明行）；裸调用时
本进程自探测并复刻同款 warn 行。

工具 CLI 零破坏（红线 R10）：七个工具的命令行入口与独立运行行为不变，
本 driver 只 import 复用其 prepare/consume/emit 拆分面。
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hbm_watermark  # noqa: E402
import hopbytes  # noqa: E402
import kv_cache_adapter  # noqa: E402
import load_imbalance  # noqa: E402
import restore_decomposition  # noqa: E402
import slo_stats  # noqa: E402
from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, REQUEST_METRICS_FILENAME, SloToolError,
    default_manifest_path, iter_jsonl, load_request_manifest,
    load_slo_manifest, read_init_record, read_request_metrics, run_main,
    validated_request_rows,
)

FAIL_MARKER_NAME = "slo_postprocess.FAIL"
FAIL_LINE_SUFFIX = (" — marked in slo_postprocess.FAIL "
                    "(simulation result NOT overturned)")


def emit(text: str) -> None:
    print(text, file=sys.stderr)


def warn(message: str) -> None:
    emit(f"[slo-postprocess] {message}")


# ---------------------------------------------------------------------------
# RunContext：共享输入一次装载（惰性首用缓存；失败不缓存——各步重复报错
# 与独立 CLI 逐工具自装载的失败形态一致，红线 R7）
# ---------------------------------------------------------------------------

class RunContext:
    """单遍 driver 的共享输入上下文。"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._init_record: Optional[dict] = None
        self._rows: Optional[list] = None
        self._request_manifest: Optional[dict] = None
        self._slo_manifest: Optional[dict] = None
        self._repo_variant: Optional[str] = None

    @property
    def init_record(self) -> dict:
        if self._init_record is None:
            # 与 read_init_record 独立调用同法：[METRIC] 流读到首条 init。
            self._init_record = read_init_record(self.run_dir)
        return self._init_record

    @property
    def repo_variant(self) -> str:
        if self._repo_variant is None:
            variant = self.init_record.get("repo_variant")
            if not variant:
                raise SloToolError(
                    f"{self.run_dir}: cpp.log init 行缺 repo_variant"
                    f"（用 --repo-variant 显式指定）")
            self._repo_variant = str(variant)
        return self._repo_variant

    @property
    def rows(self) -> list:
        if self._rows is None:
            path = self.run_dir / REQUEST_METRICS_FILENAME
            records = read_request_metrics(path)
            self._rows = validated_request_rows(records, path)
        return self._rows

    @property
    def request_manifest(self) -> dict:
        if self._request_manifest is None:
            self._request_manifest = load_request_manifest(self.run_dir, None)
        return self._request_manifest

    @property
    def slo_manifest(self) -> dict:
        if self._slo_manifest is None:
            self._slo_manifest = load_slo_manifest(default_manifest_path())
        return self._slo_manifest


# ---------------------------------------------------------------------------
# 步骤执行壳：复刻 run_slo_postprocess.sh run_step 的日志/失败语义
# ---------------------------------------------------------------------------

def _record_failure(fail_marker: Path, label: str, rc: int) -> None:
    with fail_marker.open("a", encoding="utf-8") as handle:
        handle.write(f"{label} (exit={rc})\n")


def _emit_step_result(label: str, block: str, rc: int,
                      fail_marker: Path) -> bool:
    """输出一步的 stderr 块 + ok/FAIL 行（顺序=shell run_step 同构）。"""
    sys.stderr.write(block)
    if rc == 0:
        emit(f"[slo-postprocess] ok: {label}")
        return False
    warn(f"FAIL: {label} (exit={rc}){FAIL_LINE_SUFFIX}")
    _record_failure(fail_marker, label, rc)
    return True


def run_step(label: str, body: Callable[[], int], fail_marker: Path) -> bool:
    """单发步骤：stderr 整块捕获 → 回放 → ok/FAIL。返回 True=有失败。"""
    emit(f"[slo-postprocess] run: {label}")
    buffer = io.StringIO()
    rc = 0
    try:
        with contextlib.redirect_stderr(buffer):
            rc = int(body() or 0)
    except SloToolError as exc:
        # 与独立 CLI 的 run_main 同款出口（exit=EXIT_FAIL_CLOSED）。
        print(f"[slo-tools] FAIL-CLOSED: {exc}", file=buffer)
        rc = 2
    except Exception:
        traceback.print_exc(file=buffer)
        rc = 1
    return _emit_step_result(label, buffer.getvalue(), rc, fail_marker)


class Sink:
    """decision-log 单遍消费的步骤壳（prepare/consume/finalize 三段）。

    stderr 一律进本 sink 的缓冲（含 prepare 期的参数/容量警告），最终
    按步序整块回放；任一段 fail-closed → 本步死亡，余下记录只喂其他
    sink（与独立链"前步失败不拦后步"的形态一致）。
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.buffer = io.StringIO()
        self.alive = True
        self.rc: Optional[int] = None
        self.state: Optional[dict] = None

    def guard(self, body: Callable) -> object:
        if not self.alive:
            return None
        try:
            with contextlib.redirect_stderr(self.buffer):
                return body()
        except SloToolError as exc:
            print(f"[slo-tools] FAIL-CLOSED: {exc}", file=self.buffer)
            self.alive = False
            self.rc = 2
            return None
        except Exception:
            traceback.print_exc(file=self.buffer)
            self.alive = False
            self.rc = 1
            return None

    def die(self, exc: BaseException) -> None:
        """共享迭代器失败（如 JSONL 损坏/缺文件）：本步以同错死亡。"""
        print(f"[slo-tools] FAIL-CLOSED: {exc}", file=self.buffer)
        self.alive = False
        self.rc = 2

    def prepare(self, body: Callable[[], dict]) -> None:
        self.state = self.guard(body)

    def finish(self, finalize: Callable[[], int]) -> None:
        if self.alive:
            result = self.guard(finalize)
            if self.alive:
                self.rc = int(result or 0)
        if self.rc is None:
            self.rc = 0

    def flush(self, fail_marker: Path) -> bool:
        assert self.rc is not None
        emit(f"[slo-postprocess] run: {self.label}")
        return _emit_step_result(self.label, self.buffer.getvalue(),
                                 self.rc, fail_marker)


# ---------------------------------------------------------------------------
# 门控（G1-G5；shell 已探测时经 env 传入，否则自探测+复刻 warn 行）
# ---------------------------------------------------------------------------

def gate_metric_source(run_dir: Path, fail_marker: Path) -> Optional[Path]:
    """G1：[METRIC] 来源探测（顺序=resolve_cpp_metric_log/shell 同序）。"""
    for name in ("cpp.log", "metrics.log", "cpp.log.gz"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    warn(f"FAIL: no cpp.log/metrics.log/cpp.log.gz under {run_dir}")
    _record_failure(fail_marker,
                    "input-resolve: no cpp.log/metrics.log/cpp.log.gz", 1)
    return None


def has_metric_line(metric_src: Path) -> bool:
    """G2：至少一条 '^[METRIC] ' 行（早停读；shell grep 同型）。"""
    opener = (gzip.open if metric_src.name.endswith(".gz") else open)
    with opener(metric_src, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[METRIC] "):
                return True
    return False


def _env_gate(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value == "1":
        return 1
    if value == "0":
        return 0
    return None


def resolve_gates(run_dir: Path) -> tuple[int, int, int]:
    """G3/G4/G5 → (have_rm, have_tl, skip_hbm)。"""
    have_rm = _env_gate("SH_SLO_HAVE_RM")
    if have_rm is None:
        have_rm = 1 if (run_dir / REQUEST_METRICS_FILENAME).is_file() else 0
        if not have_rm:
            warn("request_metrics.csv absent (metrics detail=summary/off) — "
                 "e2e-stats/backlog/session/warmup/restore_decomposition "
                 "skipped (by design)")
    have_tl = _env_gate("SH_SLO_HAVE_TL")
    if have_tl is None:
        have_tl = 1 if (run_dir / "results" / "train_ledger.jsonl").is_file() \
            else 0
        if not have_tl:
            warn("results/train_ledger.jsonl absent — load_imbalance.py "
                 "skipped (by design)")
    skip_hbm = _env_gate("SH_SLO_SKIP_HBM")
    if skip_hbm is None:
        # A.5/2026-09-05：legacy 与 relevant 两个历史变体已清除，
        # session-KV 水位重放对全部产物适用；缺省不跳过。
        skip_hbm = 0
    return have_rm, have_tl, skip_hbm


# ---------------------------------------------------------------------------
# 主流程（步骤序 = 现链 1→9；产物/行序/公式全部由工具函数承载）
# ---------------------------------------------------------------------------

def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def run(run_dir: Path) -> int:
    fail_marker = run_dir / FAIL_MARKER_NAME
    metric_src = gate_metric_source(run_dir, fail_marker)
    if metric_src is None:
        return 1
    if not has_metric_line(metric_src):
        warn("no [METRIC] lines (metrics detail=off?) — SLO extraction "
             "skipped (by design)")
        return 0
    have_rm, have_tl, skip_hbm = resolve_gates(run_dir)
    ctx = RunContext(run_dir)
    any_fail = False

    # -- 步 1-4：共享 ctx.rows / ctx.slo_manifest（单发步骤） --------------
    if have_rm:
        e2e_args = _ns(run_dir=run_dir, output="", extra_pct="90,95")
        any_fail = run_step(
            "slo_stats.py e2e-stats --extra-pct 90,95",
            lambda: slo_stats.cmd_e2e_stats(e2e_args, ctx), fail_marker) \
            or any_fail
        backlog_args = _ns(run_dir=run_dir, output="", bucket_ns=None)
        any_fail = run_step(
            "slo_stats.py backlog (per-event, no --bucket-ns)",
            lambda: slo_stats.cmd_backlog(backlog_args, ctx), fail_marker) \
            or any_fail
        session_args = _ns(run_dir=run_dir, output="", request_manifest=None)
        any_fail = run_step(
            "slo_stats.py session",
            lambda: slo_stats.cmd_session(session_args, ctx), fail_marker) \
            or any_fail
        warmup_args = _ns(run_dir=run_dir, output="", manifest=None)
        any_fail = run_step(
            "slo_stats.py warmup",
            lambda: slo_stats.cmd_warmup(warmup_args, ctx), fail_marker) \
            or any_fail

    # -- 步 5/6/8/9：decision log 单遍 + sink 扇出 ------------------------
    kv_sink = Sink("kv_cache_adapter.py")
    load_sink = Sink("load_imbalance.py")
    hop_sink = Sink("hopbytes.py")
    wm_sink = Sink("hbm_watermark.py")

    kv_args = _ns(run_dir=run_dir, output="", hit_states="", json="",
                  reconcile=False, repo_variant=None, request_manifest=None)
    load_args = _ns(run_dir=run_dir, manifest=None, output="", json="",
                    repo_variant=None)
    hop_args = _ns(run_dir=run_dir, output="", per_request="", json="",
                   repo_variant=None)
    wm_args = _ns(run_dir=run_dir, manifest=None, token_manifest=None,
                  request_manifest=None, trace_config=None,
                  hardware_config=None, intervals_csv="", output="",
                  instances_csv="", json="", repo_variant=None)

    def kv_prepare() -> dict:
        repo_variant = ctx.repo_variant
        manifest = ctx.request_manifest
        variant, turns, state = kv_cache_adapter.adapter_prepare(
            repo_variant, manifest)
        return {"repo_variant": repo_variant, "manifest": manifest,
                "variant": variant, "turns": turns, "state": state}

    def load_prepare() -> dict:
        # 参数读取先于 variant 解析（与独立 CLI 的步骤内顺序一致）。
        bucket_ns = load_imbalance.load_imbalance_bucket_ns(ctx.slo_manifest)
        repo_variant = ctx.repo_variant
        return {"bucket_ns": bucket_ns, "repo_variant": repo_variant,
                "state": load_imbalance.li_prepare(repo_variant)}

    def hop_prepare() -> dict:
        repo_variant = ctx.repo_variant
        source, acc, per_request = hopbytes.hopbytes_prepare(repo_variant)
        return {"repo_variant": repo_variant, "source": source, "acc": acc,
                "per_request": per_request}

    def wm_prepare() -> dict:
        repo_variant = ctx.repo_variant
        prep = hbm_watermark.watermark_prepare(
            wm_args, repo_variant,
            request_manifest_loader=lambda: ctx.request_manifest)
        # P1（2026-08-30）：decision-log 重放的容量参数恒传（诊断计数）；
        # 正式判决只由 journal 路径（per_rank_total_hbm_certified 层）给出。
        scan = hbm_watermark.WatermarkScan(
            run_dir, repo_variant, prep["mapping"], prep["tokens"],
            prep["coef"], prep["capacity"])
        return {"repo_variant": repo_variant, "prep": prep, "scan": scan}

    kv_sink.prepare(kv_prepare)
    if have_tl:
        load_sink.prepare(load_prepare)
    else:
        load_sink.alive = False  # G4：跳过（说明行由 shell/门控负责）
    hop_sink.prepare(hop_prepare)
    if skip_hbm:
        wm_sink.alive = False  # G5：跳过（说明行由 shell/门控负责）
    else:
        wm_sink.prepare(wm_prepare)

    def kv_consume(record: dict) -> None:
        state = kv_sink.state
        kv_cache_adapter.adapter_consume(
            record, state["variant"], state["turns"], state["state"])

    def load_consume(record: dict) -> None:
        state = load_sink.state
        load_imbalance.li_consume_decision(
            record, state["repo_variant"], state["state"])

    def hop_consume(record: dict) -> None:
        state = hop_sink.state
        state["source"]["collector"](record, state["acc"],
                                     state["per_request"])

    def wm_consume(record: dict) -> None:
        wm_sink.state["scan"].consume(record)

    # sink 次序固定（红线 R8）：kv → load → hop → watermark（watermark
    # 注入 session_hint，必须最后；其余 sink 互不读对方注键）。
    sinks: list[tuple[Sink, Callable[[dict], None]]] = [
        (kv_sink, kv_consume), (load_sink, load_consume),
        (hop_sink, hop_consume), (wm_sink, wm_consume)]

    log_path = run_dir / DECISION_LOG_RELPATH
    records = iter_jsonl(log_path)
    while True:
        try:
            record = next(records)
        except StopIteration:
            break
        except SloToolError as exc:
            # 迭代器自身失败（缺文件/JSON 损坏）：与独立链"每个工具都在
            # 同一行失败"同构——所有存活 sink 以同一错误死亡。
            for sink, _ in sinks:
                if sink.alive:
                    sink.die(exc)
            break
        for sink, consume in sinks:
            if sink.alive:
                sink.guard(lambda: consume(record))

    def kv_finalize() -> int:
        state = kv_sink.state
        return kv_cache_adapter.adapter_emit(
            kv_args, state["repo_variant"], state["variant"],
            state["manifest"], state["turns"], state["state"])

    def load_finalize() -> int:
        state = load_sink.state
        return load_imbalance.load_imbalance_finish(
            load_args, state["repo_variant"], state["bucket_ns"],
            state["state"])

    def hop_finalize() -> int:
        state = hop_sink.state
        return hopbytes.hopbytes_emit(
            hop_args, state["repo_variant"], state["source"], state["acc"],
            state["per_request"])

    def wm_finalize() -> int:
        state = wm_sink.state
        replay = state["scan"].finish()
        return hbm_watermark.watermark_emit(
            wm_args, state["repo_variant"], state["prep"], replay)

    # -- 按链序发射：5 → 6 → 7 → 8 → 9 -----------------------------------
    kv_sink.finish(kv_finalize)
    any_fail = kv_sink.flush(fail_marker) or any_fail
    if have_tl:
        load_sink.finish(load_finalize)
        any_fail = load_sink.flush(fail_marker) or any_fail

    if have_rm:
        restore_args = _ns(run_dir=run_dir, output="", json="")

        def restore_body() -> int:
            precollected = restore_decomposition.scan_metric_anchors_requests(
                metric_src)
            return restore_decomposition.cmd_restore(restore_args,
                                                     precollected)
        any_fail = run_step("restore_decomposition.py (detail=full)",
                            restore_body, fail_marker) or any_fail

    hop_sink.finish(hop_finalize)
    any_fail = hop_sink.flush(fail_marker) or any_fail
    if not skip_hbm:
        wm_sink.finish(wm_finalize)
        any_fail = wm_sink.flush(fail_marker) or any_fail

    return 1 if any_fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="slo_postprocess_driver.py",
        description="SLO 后处理单遍合并 driver（9 步链的单进程等价实现；"
                    "产物/行序/公式与独立工具逐字节一致，读放大收敛为每共享"
                    "输入恰读一次）")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（同 run_slo_postprocess.sh <run_dir>）")
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        print(f"[slo-postprocess] FAIL: run dir not found: {args.run_dir}",
              file=sys.stderr)
        return 1
    return run(args.run_dir)


if __name__ == "__main__":
    sys.exit(run_main(main))
