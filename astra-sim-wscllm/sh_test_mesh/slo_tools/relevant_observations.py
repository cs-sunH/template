#!/usr/bin/env python3
"""relevant_distributed 变体三个观测后处理（总文档 §4 裁决 #24；T3/2026-09-02）。

run 报告附属脚本——**不进契约工具**（不接入 slo_postprocess_driver 的
sink 链，不参与 SLO 判定），只从既有产物做只读提取；观测不改变任何
决策（主规格第一原则 b，native 日志保持原样）。

输入 = run_dir/results/{online_decision_log.jsonl, kv_delta_journal.jsonl}
（journal 为可选：KV_DELTA_JOURNAL=off 的 run 跳过占用时序段）。

输出（缺省落 run_dir，-o 重定向）：
  * ``relevant_kv_occupancy.csv`` —— 每实例 resident/remote KV 占用时序：
    journal 逐事件（按 planner_time_ns 与文件序）后的实例级累计值。
    resident_local = 宿主实例 == 请求 decode 实例的 KV（decode 段 + D
    吸收的 prefill 段，decode 本地读）；resident_remote = 宿主实例 !=
    decode 实例的 KV（P 暂存 stay + 中间 die/溢出 piece，3300 远程读的
    服务侧）；reserved_staging = P 的散布暂存 scratch（3100 完成即释放）。
  * ``relevant_read_edges.csv`` —— 读边（3300）字节分布：逐路由行
    （列车×成员×源×rank 粒度）+ 汇总统计（逐源/逐 D/逐请求聚合、
    min/max/mean）。
  * ``relevant_backpressure.csv`` —— 背压持续时长（裁决 #19 观测字段：
    kv_placement 行 backpressure_duration_ns，None = 无阻塞片段）。

汇总 JSON 走 stderr（emit_json）；run_header 姿态（policy/kv_remote_read/
d2d_to_hbm_bandwidth_ratio）原样回显供报告 provenance。非
relevant_distributed 的 run fail-closed（本工具仅服务该变体）。

用法：python3 relevant_observations.py <run_dir> [-o DIR] [--selftest]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, NA, emit_json, fail, iter_jsonl, run_main,
    write_csv,
)


def _warn(message: str) -> None:
    print(f"[relevant_observations] {message}", file=sys.stderr)

JOURNAL_RELPATH = Path("results") / "kv_delta_journal.jsonl"

OCCUPANCY_COLUMNS = (
    "instance_index", "time_ns", "cause", "request_id",
    "resident_local_bytes_after", "resident_remote_bytes_after",
    "reserved_staging_bytes_after",
)
READ_EDGE_COLUMNS = (
    "train_id", "request_id", "decode_instance_index",
    "source_instance_index", "relative_shard", "source_rank", "decode_rank",
    "bytes", "participation", "noc_hops",
)
BACKPRESSURE_COLUMNS = ("request_id", "admitted_at_ns",
                        "backpressure_duration_ns")

# journal cause → 处理类（model_weight_preload 不进 KV 时序；未知 cause
# fail-closed，防新 cause 静默漏账）。
_CAUSE_RESIDENT = ("relevant_placement", "relevant_release")
_CAUSE_RESERVED = ("staging_scratch",)
_CAUSE_IGNORE = ("model_weight_preload",)


class _InstanceTotals:
    """单实例三类占用累计（local/remote/staging）。"""

    __slots__ = ("resident_local", "resident_remote", "reserved_staging",
                 "peak_resident", "peak_reserved")

    def __init__(self) -> None:
        self.resident_local = 0
        self.resident_remote = 0
        self.reserved_staging = 0
        self.peak_resident = 0
        self.peak_reserved = 0

    def apply(self, *, local_delta: int, remote_delta: int,
              reserved_delta: int) -> None:
        self.resident_local += local_delta
        self.resident_remote += remote_delta
        self.reserved_staging += reserved_delta
        resident = self.resident_local + self.resident_remote
        if resident > self.peak_resident:
            self.peak_resident = resident
        if self.reserved_staging > self.peak_reserved:
            self.peak_reserved = self.reserved_staging


def _read_run_header(run_dir: Path) -> dict:
    """决策日志首行 run_header（B3 起恒为首行）；非 relevant 变体拒绝。"""
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        if record.get("kind") != "run_header":
            break
        decision = record.get("decision") or {}
        if decision.get("kv_cache_policy") != "relevant_distributed":
            fail("relevant_observations 仅服务 relevant_distributed 变体"
                 f"（run_header.kv_cache_policy="
                 f"{decision.get('kv_cache_policy')!r}）")
        return decision
    fail("决策日志缺 run_header 首行（relevant_distributed 产物应有）")


def _decode_instance_map(run_dir: Path) -> dict[str, int]:
    """request_id → decode_instance_index（kv_placement 行，准入时冻结）。"""
    mapping: dict[str, int] = {}
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        if record.get("kind") != "kv_placement":
            continue
        request_id = record.get("request_id")
        decode = (record.get("decision") or {}).get("decode_instance_index")
        if not isinstance(request_id, str) or isinstance(decode, bool) \
                or not isinstance(decode, int):
            fail(f"kv_placement 行缺 request_id/decode_instance_index："
                 f"{record!r}")
        if request_id in mapping and mapping[request_id] != decode:
            fail(f"请求 {request_id!r} 的 decode_instance_index 跨行漂移："
                 f"{mapping[request_id]} != {decode}")
        mapping[request_id] = decode
    return mapping


def collect_occupancy(run_dir: Path, decode_map: dict[str, int]
                      ) -> Optional[dict]:
    """观测 1：每实例 resident/remote 占用时序（journal 可选输入）。"""
    journal_path = run_dir / JOURNAL_RELPATH
    if not journal_path.is_file():
        _warn(f"{JOURNAL_RELPATH} 缺失（KV_DELTA_JOURNAL=off 的 run）——"
              "resident/remote 占用时序段跳过（by design）")
        return None
    totals: dict[int, _InstanceTotals] = {}
    rows: list[tuple] = []
    journal_rows = 0
    # 同一实例事件的逐 rank 行合并为一行输出（键含 anchor_kind 防不同
    # 边界的同 cause 事件并串；保持文件首见序）。
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    for record in iter_jsonl(journal_path):
        journal_rows += 1
        cause = record.get("cause")
        if cause in _CAUSE_IGNORE:
            continue
        if cause not in _CAUSE_RESIDENT and cause not in _CAUSE_RESERVED:
            fail(f"kv_delta_journal 行带未知 cause {cause!r}（观测不臆造"
                 "归类，请在 _CAUSE_* 表登记）")
        request_id = record.get("request_id")
        instance = record.get("instance_index")
        if not isinstance(request_id, str) or isinstance(instance, bool) \
                or not isinstance(instance, int):
            fail(f"journal {cause} 行缺 request_id/instance_index："
                 f"{record!r}")
        if request_id not in decode_map:
            fail(f"journal 请求 {request_id!r} 无 kv_placement 行"
                 f"（{cause}）——决策日志与 journal 不同源")
        resident_delta = record.get("resident_kv_delta_bytes") or 0
        reserved_delta = record.get("reserved_kv_delta_bytes") or 0
        key = (record.get("planner_time_ns"), cause, request_id,
               record.get("anchor_kind"), instance)
        slot = groups.get(key)
        if slot is None:
            slot = {"resident": 0, "reserved": 0}
            groups[key] = slot
            order.append(key)
        slot["resident"] += int(resident_delta)
        slot["reserved"] += int(reserved_delta)
    for key in order:
        time_ns, cause, request_id, _anchor, instance = key
        slot = groups[key]
        instance_totals = totals.setdefault(instance, _InstanceTotals())
        if cause in _CAUSE_RESIDENT:
            if instance == decode_map[request_id]:
                local_delta, remote_delta = slot["resident"], 0
            else:
                local_delta, remote_delta = 0, slot["resident"]
            reserved_delta = 0
        else:  # staging_scratch：P 暂存 scratch，reserved 通道
            local_delta = remote_delta = 0
            reserved_delta = slot["reserved"]
        instance_totals.apply(local_delta=local_delta,
                              remote_delta=remote_delta,
                              reserved_delta=reserved_delta)
        rows.append((
            instance, time_ns, cause, request_id,
            instance_totals.resident_local,
            instance_totals.resident_remote,
            instance_totals.reserved_staging))
    instances = {
        instance: {
            "final_resident_local_bytes": state.resident_local,
            "final_resident_remote_bytes": state.resident_remote,
            "final_reserved_staging_bytes": state.reserved_staging,
            "peak_resident_bytes": state.peak_resident,
            "peak_reserved_staging_bytes": state.peak_reserved,
        }
        for instance, state in sorted(totals.items())
    }
    return {
        "journal_rows": journal_rows,
        "occupancy_rows": rows,
        "instances": instances,
        "final_all_released": all(
            state.resident_local == 0 and state.resident_remote == 0
            and state.reserved_staging == 0
            for state in totals.values()),
    }


def collect_read_edges(run_dir: Path) -> dict:
    """观测 2：读边（3300）字节分布（kv_remote_reads 决策行逐路由行）。"""
    rows: list[tuple] = []
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        if record.get("kind") != "kv_remote_reads":
            continue
        decision = record.get("decision") or {}
        train_id = decision.get("train_id")
        for route in decision.get("routes") or []:
            if not isinstance(route, dict):
                fail(f"kv_remote_reads 路由行非对象：{route!r}")
            for field in ("request_id", "source_instance_index",
                          "decode_instance_index", "bytes"):
                if route.get(field) is None:
                    fail(f"读边路由行缺 {field}：{route!r}")
            rows.append((
                train_id, route.get("request_id"),
                route.get("decode_instance_index"),
                route.get("source_instance_index"),
                route.get("relative_shard", NA),
                route.get("source_rank", NA),
                route.get("decode_rank", NA),
                int(route["bytes"]),
                route.get("participation", NA),
                route.get("noc_hops", NA),
            ))
    by_source: dict[int, int] = {}
    by_decode: dict[int, int] = {}
    by_request: dict[str, int] = {}
    for row in rows:
        nbytes = row[7]
        by_source[row[3]] = by_source.get(row[3], 0) + nbytes
        by_decode[row[2]] = by_decode.get(row[2], 0) + nbytes
        by_request[row[1]] = by_request.get(row[1], 0) + nbytes
    byte_values = [row[7] for row in rows]
    total = sum(byte_values)
    return {
        "read_edge_rows": rows,
        "n_routes": len(rows),
        "total_bytes": total,
        "min_route_bytes": min(byte_values) if byte_values else None,
        "max_route_bytes": max(byte_values) if byte_values else None,
        "mean_route_bytes": (total / len(byte_values)) if byte_values else None,
        "bytes_by_source_instance": dict(sorted(by_source.items())),
        "bytes_by_decode_instance": dict(sorted(by_decode.items())),
        "bytes_by_request": dict(sorted(by_request.items())),
    }


def collect_backpressure(run_dir: Path) -> dict:
    """观测 3：背压持续时长（kv_placement 行 backpressure_duration_ns）。"""
    rows: list[tuple] = []
    n_placements = 0
    for record in iter_jsonl(run_dir / DECISION_LOG_RELPATH):
        if record.get("kind") != "kv_placement":
            continue
        n_placements += 1
        duration = (record.get("decision") or {}).get(
            "backpressure_duration_ns")
        if duration is None:
            continue
        if not isinstance(duration, int) or isinstance(duration, bool):
            fail(f"backpressure_duration_ns 非整数：{record!r}")
        rows.append((record.get("request_id"), record.get("tick"), duration))
    durations = [row[2] for row in rows]
    return {
        "backpressure_rows": rows,
        "n_placements": n_placements,
        "episode_count": len(rows),
        "total_ns": sum(durations),
        "max_ns": max(durations) if durations else None,
    }


def _write_outputs(args: argparse.Namespace, occupancy, read_edges,
                   backpressure) -> None:
    out_dir = Path(args.output) if args.output else args.run_dir
    if args.output:
        out_dir.mkdir(parents=True, exist_ok=True)
    occupancy_path = out_dir / "relevant_kv_occupancy.csv"
    with occupancy_path.open("w", newline="", encoding="utf-8") as handle:
        write_csv(handle, OCCUPANCY_COLUMNS,
                  occupancy["occupancy_rows"] if occupancy else [])
    edges_path = out_dir / "relevant_read_edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        write_csv(handle, READ_EDGE_COLUMNS, read_edges["read_edge_rows"])
    backpressure_path = out_dir / "relevant_backpressure.csv"
    with backpressure_path.open("w", newline="", encoding="utf-8") as handle:
        write_csv(handle, BACKPRESSURE_COLUMNS,
                  backpressure["backpressure_rows"])
    if not occupancy:
        occupancy_path.unlink()
        occupancy_path = None
    args._paths = (occupancy_path, edges_path, backpressure_path)  # type: ignore


def cmd_observations(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    run_header = _read_run_header(run_dir)
    decode_map = _decode_instance_map(run_dir)
    occupancy = collect_occupancy(run_dir, decode_map)
    read_edges = collect_read_edges(run_dir)
    backpressure = collect_backpressure(run_dir)
    _write_outputs(args, occupancy, read_edges, backpressure)
    summary = {
        "command": "relevant_observations",
        "run_header": run_header,
        "note": "观测后处理（裁决 #24）：只读提取，不改变任何决策；"
                "不进契约工具（SLO 判定链零接触）",
        "kv_occupancy": (
            {"journal_present": True,
             "journal_rows": occupancy["journal_rows"],
             "instances": occupancy["instances"],
             "final_all_released": occupancy["final_all_released"]}
            if occupancy else {"journal_present": False}),
        "read_edges": {
            "n_routes": read_edges["n_routes"],
            "total_bytes": read_edges["total_bytes"],
            "min_route_bytes": read_edges["min_route_bytes"],
            "max_route_bytes": read_edges["max_route_bytes"],
            "mean_route_bytes": read_edges["mean_route_bytes"],
            "bytes_by_source_instance": read_edges["bytes_by_source_instance"],
            "bytes_by_decode_instance":
                read_edges["bytes_by_decode_instance"],
            "bytes_by_request": read_edges["bytes_by_request"],
        },
        "backpressure": {
            "n_placements": backpressure["n_placements"],
            "episode_count": backpressure["episode_count"],
            "total_ns": backpressure["total_ns"],
            "max_ns": backpressure["max_ns"],
        },
        "outputs": [str(path) for path in args._paths if path],  # type: ignore
    }
    emit_json(sys.stderr, summary)
    return 0


# ---------------------------------------------------------------------------
# --selftest：合成 fixture 手算断言（不跑仿真）
# ---------------------------------------------------------------------------

def _selftest_fixture(run_dir: Path) -> None:
    """手算 fixture：r0（P=1→D=0，含 40B→D0 + 7B→d4 散布、背压 500ns）
    + r1（turn-1，拉回 25B@源3 + 5B@源2）。占用时序期望值见 assert。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    decision_log = [
        {"kind": "run_header", "request_id": "", "tick": 1,
         "decision": {"kv_cache_policy": "relevant_distributed",
                      "kv_remote_read": "physical",
                      "d2d_to_hbm_bandwidth_ratio": 2.5,
                      "sh_train_max_iter": 8,
                      "sh_first_token_split": "off"}},
        {"kind": "kv_placement", "request_id": "r0", "tick": 100,
         "decision": {"decode_instance_index": 0,
                      "backpressure_duration_ns": 500}},
        {"kind": "prefill", "request_id": "r0", "tick": 100,
         "decision": {"history_canonical_hit_state": None,
                      "history_pull_routes": []}},
        {"kind": "kv_scatter", "request_id": "r0", "tick": 200,
         "decision": {"routes": [
             {"category": 3100, "source_instance_index": 1,
              "target_instance_index": 0, "bytes": 30},
             {"category": 3100, "source_instance_index": 1,
              "target_instance_index": 0, "bytes": 10},
             {"category": 3100, "source_instance_index": 1,
              "target_instance_index": 4, "bytes": 7}]}},
        {"kind": "kv_remote_reads", "request_id": "t0", "tick": 400,
         "decision": {"train_id": "t0", "instance_index": 0,
                      "routes": [
                          {"category": 3300, "train_id": "t0",
                           "request_id": "r0",
                           "decode_instance_index": 0,
                           "source_instance_index": 1,
                           "relative_shard": 0, "source_rank": 1,
                           "decode_rank": 0, "bytes": 999,
                           "participation": 8, "noc_hops": 1},
                          {"category": 3300, "train_id": "t0",
                           "request_id": "r0",
                           "decode_instance_index": 0,
                           "source_instance_index": 2,
                           "relative_shard": 1, "source_rank": 2,
                           "decode_rank": 0, "bytes": 50,
                           "participation": 8, "noc_hops": 2},
                          {"category": 3300, "train_id": "t0",
                           "request_id": "r1",
                           "decode_instance_index": 0,
                           "source_instance_index": 4,
                           "relative_shard": 2, "source_rank": 4,
                           "decode_rank": 0, "bytes": 7,
                           "participation": 4, "noc_hops": 3}],
                      "total_bytes": 1056}},
        {"kind": "kv_placement", "request_id": "r1", "tick": 300,
         "decision": {"decode_instance_index": 0,
                      "backpressure_duration_ns": None}},
        {"kind": "prefill", "request_id": "r1", "tick": 300,
         "decision": {"history_canonical_hit_state": "partial",
                      "history_pull_routes": [
                          {"category": 1000, "source_instance_index": 3,
                           "target_instance_index": 1, "bytes": 25},
                          {"category": 1000, "source_instance_index": 2,
                           "target_instance_index": 1, "bytes": 5}]}},
    ]
    journal = [
        # 权重预载（不进 KV 时序）。
        {"cause": "model_weight_preload", "request_id": None,
         "instance_index": 0, "rank": 0, "planner_time_ns": 0,
         "resident_kv_delta_bytes": 0, "reserved_kv_delta_bytes": 0},
        # r0 准入：inst0 本地 +180（decode 段 80 + D 吸收 prefill 段 100）；
        # inst1（P）remote +100（prefill_stay）；staging +260。
        {"cause": "relevant_placement", "request_id": "r0",
         "instance_index": 0, "rank": 0, "anchor_kind": "prefill_start",
         "planner_time_ns": 100, "resident_kv_delta_bytes": 180,
         "reserved_kv_delta_bytes": 0},
        {"cause": "relevant_placement", "request_id": "r0",
         "instance_index": 1, "rank": 0, "anchor_kind": "prefill_start",
         "planner_time_ns": 100, "resident_kv_delta_bytes": 100,
         "reserved_kv_delta_bytes": 0},
        {"cause": "staging_scratch", "request_id": "r0",
         "instance_index": 1, "rank": 0, "anchor_kind": "prefill_start",
         "planner_time_ns": 100, "resident_kv_delta_bytes": 0,
         "reserved_kv_delta_bytes": 260},
        # drain：staging 释放（3100 完成）。
        {"cause": "staging_scratch", "request_id": "r0",
         "instance_index": 1, "rank": 0, "anchor_kind": "transfer_complete",
         "planner_time_ns": 200, "resident_kv_delta_bytes": 0,
         "reserved_kv_delta_bytes": -260},
        # r1 准入：inst0 本地 +60。
        {"cause": "relevant_placement", "request_id": "r1",
         "instance_index": 0, "rank": 0, "anchor_kind": "prefill_start",
         "planner_time_ns": 300, "resident_kv_delta_bytes": 60,
         "reserved_kv_delta_bytes": 0},
        # 终轮释放：r0（inst0 −180 / inst1 −100）、r1（inst0 −60）。
        {"cause": "relevant_release", "request_id": "r0",
         "instance_index": 0, "rank": 0, "anchor_kind": "completion",
         "planner_time_ns": 500, "resident_kv_delta_bytes": -180,
         "reserved_kv_delta_bytes": 0},
        {"cause": "relevant_release", "request_id": "r0",
         "instance_index": 1, "rank": 0, "anchor_kind": "completion",
         "planner_time_ns": 500, "resident_kv_delta_bytes": -100,
         "reserved_kv_delta_bytes": 0},
        {"cause": "relevant_release", "request_id": "r1",
         "instance_index": 0, "rank": 0, "anchor_kind": "completion",
         "planner_time_ns": 600, "resident_kv_delta_bytes": -60,
         "reserved_kv_delta_bytes": 0},
    ]
    import json
    (run_dir / "results" / "online_decision_log.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in decision_log),
        encoding="utf-8")
    (run_dir / "results" / "kv_delta_journal.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in journal),
        encoding="utf-8")


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, [row for row in reader]


def run_selftest() -> int:
    import tempfile

    from slo_common import SloToolError

    with tempfile.TemporaryDirectory(prefix="relevant_obs_selftest_") as tmp:
        run_dir = Path(tmp) / "run"
        _selftest_fixture(run_dir)
        out_dir = Path(tmp) / "out"
        args = argparse.Namespace(run_dir=run_dir, output=str(out_dir))
        import contextlib
        import io
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = cmd_observations(args)
        assert exit_code == 0

        # 观测 1：占用时序（逐实例事件后的累计；inst0 本地、inst1 remote
        # +staging 瞬态、终态全零）。
        header, rows = _read_csv_rows(out_dir / "relevant_kv_occupancy.csv")
        assert header == list(OCCUPANCY_COLUMNS), header
        assert rows == [
            ["0", "100", "relevant_placement", "r0", "180", "0", "0"],
            ["1", "100", "relevant_placement", "r0", "0", "100", "0"],
            ["1", "100", "staging_scratch", "r0", "0", "100", "260"],
            ["1", "200", "staging_scratch", "r0", "0", "100", "0"],
            ["0", "300", "relevant_placement", "r1", "240", "0", "0"],
            ["0", "500", "relevant_release", "r0", "60", "0", "0"],
            ["1", "500", "relevant_release", "r0", "0", "0", "0"],
            ["0", "600", "relevant_release", "r1", "0", "0", "0"],
        ], rows

        # 观测 2：读边字节分布（3 行 × 逐列）。
        header, rows = _read_csv_rows(out_dir / "relevant_read_edges.csv")
        assert header == list(READ_EDGE_COLUMNS), header
        assert [row[7] for row in rows] == ["999", "50", "7"]
        assert rows[0][:4] == ["t0", "r0", "0", "1"]

        # 观测 3：背压持续时长（唯一 episode r0=500ns）。
        header, rows = _read_csv_rows(out_dir / "relevant_backpressure.csv")
        assert header == list(BACKPRESSURE_COLUMNS), header
        assert rows == [["r0", "100", "500"]], rows

        # 汇总 JSON：手算分布统计 + 终态守恒报告位。
        import json as _json
        summary = _json.loads(stderr.getvalue())
        assert summary["run_header"]["d2d_to_hbm_bandwidth_ratio"] == 2.5
        read_edges = summary["read_edges"]
        assert read_edges["n_routes"] == 3
        assert read_edges["total_bytes"] == 1056
        assert read_edges["min_route_bytes"] == 7
        assert read_edges["max_route_bytes"] == 999
        assert read_edges["bytes_by_source_instance"] == {"1": 999, "2": 50,
                                                          "4": 7}
        assert read_edges["bytes_by_request"] == {"r0": 1049, "r1": 7}
        assert summary["backpressure"] == {
            "n_placements": 2, "episode_count": 1, "total_ns": 500,
            "max_ns": 500}
        occupancy_summary = summary["kv_occupancy"]
        assert occupancy_summary["journal_present"] is True
        assert occupancy_summary["final_all_released"] is True
        assert occupancy_summary["instances"]["1"] == {
            "final_resident_local_bytes": 0,
            "final_resident_remote_bytes": 0,
            "final_reserved_staging_bytes": 0,
            "peak_resident_bytes": 100,
            "peak_reserved_staging_bytes": 260}

        # 负例 1：journal 请求无 kv_placement 映射 → fail-closed。
        (run_dir / "results" / "kv_delta_journal.jsonl").write_text(
            '{"cause": "relevant_placement", "request_id": "ghost",'
            ' "instance_index": 0, "rank": 0, "anchor_kind": "prefill_start",'
            ' "planner_time_ns": 1, "resident_kv_delta_bytes": 1,'
            ' "reserved_kv_delta_bytes": 0}\n', encoding="utf-8")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cmd_observations(argparse.Namespace(run_dir=run_dir,
                                                    output=str(out_dir)))
        except SloToolError as exc:
            assert "ghost" in str(exc)
        else:
            raise AssertionError("journal 未知请求未 fail-closed")
    print("relevant_observations --selftest: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="relevant_observations.py",
        description="relevant_distributed 三个观测后处理（裁决 #24；"
                    "run 报告附属，不进契约工具，观测不改决策）")
    parser.add_argument("run_dir", type=Path, nargs="?", default=None,
                        help="运行目录（含 results/online_decision_log"
                             ".jsonl；kv_delta_journal.jsonl 可选）")
    parser.add_argument("-o", "--output", default="",
                        help="输出目录（缺省写 run_dir）")
    parser.add_argument("--selftest", action="store_true",
                        help="合成 fixture 手算断言（不跑仿真）")
    args = parser.parse_args()
    if args.selftest:
        return run_selftest()
    if args.run_dir is None:
        parser.error("run_dir 必填（或用 --selftest）")
    return int(cmd_observations(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
