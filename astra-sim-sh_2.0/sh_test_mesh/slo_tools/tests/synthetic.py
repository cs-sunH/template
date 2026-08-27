"""合成 run_dir fixture 构造器（T0/T2 测试共用）。

所有样本为手算可验的合成数据（仿真侧 fixture 留待 B3 批次真实运行后
补充）；本模块只落临时目录，不触碰仓内任何已有文件。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slo_common import REQUEST_METRICS_COLUMNS  # noqa: E402


def make_run_dir(tag: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"slo_t_{tag}_"))
    return root


def write_request_metrics(run_dir: Path, rows: Sequence[dict]) -> Path:
    """rows: 列名→值（缺省列填 NA）。列序冻结为 EXECUTION_PLAN §2。"""
    import csv
    path = run_dir / "request_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(REQUEST_METRICS_COLUMNS)
        for row in rows:
            writer.writerow([row.get(col, "NA") for col in
                             REQUEST_METRICS_COLUMNS])
    return path


def request_row(**overrides: Any) -> dict:
    row = {col: "NA" for col in REQUEST_METRICS_COLUMNS}
    row.update({
        "queue_index": "0",
        "request_id": "session_0_request_0",
        "session_id": "session_0",
        "turn_index": "0",
        "request_type": "human",
        "terminal_status": "completed",
        "kv_hit_state": "no_history",
        "first_token_source": "NA",
        "prefill_length": "500",
        "decode_length": "100",
    })
    row.update(overrides)
    return row


def write_metrics_manifest(run_dir: Path, requests: Sequence[dict],
                           repo_variant: str = "astra-sim-face") -> Path:
    payload = {
        "schema_version": 1,
        "repo_variant": repo_variant,
        "manifest_source": "synthetic-test",
        "requests": list(requests),
    }
    path = run_dir / "metrics_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def manifest_request(request_id: str, session_id: str, turn_index: int,
                     queue_index: int, **extra: Any) -> dict:
    entry = {
        "queue_index": queue_index,
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "arrival": {"kind": "absolute", "value_ns": 0},
    }
    entry.update(extra)
    return entry


def write_cpp_log(run_dir: Path, records: Sequence[dict],
                  repo_variant: str = "astra-sim-face") -> Path:
    """records 直接为 [METRIC] JSON 对象；init 行自动前置。"""
    init = {
        "schema": 1, "type": "init", "source": "simulator",
        "repo_variant": repo_variant, "run_id": "", "run_mode": "service",
        "detail_level": "full", "input_requests": 0,
        "manifest_path": str(run_dir / "metrics_manifest.json"),
        "schema_version": 1,
    }
    path = run_dir / "cpp.log"
    lines = ["[METRIC] " + json.dumps(init, sort_keys=True)]
    lines.extend("[METRIC] " + json.dumps(r, sort_keys=True)
                 for r in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_jsonl(run_dir: Path, name: str,
                records: Sequence[dict]) -> Path:
    path = run_dir / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8")
    return path


def write_slo_manifest(root: Path, params: dict) -> Path:
    """带值的测试用 slo_params_manifest.json（生产仓内清单 value=null）。"""
    base = json.loads(
        (Path(__file__).resolve().parent.parent /
         "slo_params_manifest.json").read_text(encoding="utf-8"))
    for name, value in params.items():
        if name not in base["params"]:
            raise KeyError(f"unknown param {name}")
        base["params"][name]["value"] = value
    path = root / "slo_params_manifest.json"
    path.write_text(json.dumps(base, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


TEST_BUCKET_EDGES = {
    "percentiles": [50],
    "prefill_edges_tokens": [0, 1000, 2000],
    "decode_edges_tokens": [0, 200, 400],
}


def write_t_isolated(root: Path, rows: Sequence[Sequence[int]]) -> Path:
    import csv
    path = root / "t_isolated.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prefill_bucket_idx", "decode_bucket_idx",
                         "t_isolated_ns"])
        writer.writerows(rows)
    return path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)
