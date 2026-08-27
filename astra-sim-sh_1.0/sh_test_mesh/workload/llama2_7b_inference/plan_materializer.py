#!/usr/bin/env python3
"""plan_materializer.py -- 路径③④ 的 plan 目录物化入口（唯一输入物化器）。

在线③④需要三类输入，本脚本以**不跑规划器、不写任何 .et** 的方式产出：

  1. runtime_config 四小件（system/comm_group/remote_memory/network.yml）
     —— 由 load_load_face_trace_config() 装载配置时经 config_resolver.materialize_runtime_configs
     副产（路径不变，零改动）；
  2. manifest.json —— 仅含在线侧实际消费的**队列派生 9 字段**
     （request_id/session_id/turn_index/queue_index/prefill_length/
     decode_length/history_tokens_before/prefill_context_tokens/
     final_context_tokens；推导规则与《sh_3.0的120档问题分析报告.md》§4
     受控实验（20 档真实 manifest 2091/2091 行 0 mismatch）同式：
     turn-0 history=0；turn>0 history=上一请求 final；
     prefill_context=折入后的 prefill_length（recompute 单口径：turn-0
     队列 prefill_length 已折入 prefix）；final=context+decode）；
     WP2（SLO B1，2026-08-26）追加透传字段 human_time_ns/tool_time_ns/
     request_type（来自队列旁 canonical sidecar；缺行回退 B1 推导规则，
     计数上报 stdout）——只加键不删不改既有字段；
  3. metrics_manifest.json —— 主 agent 裁决 (i) 的合成口径：
       - schema_version=1 + requests[]（arrival：turn0=absolute session
         arrival；turn>0=after_request(同 session 上一 queue_index, interval)
         ——C++ MetricCollector.cc 只要求
         schema_version==1 + requests[]）；
       - **显式合成标记**：repo_variant 原样 + manifest_source=
         "synthetic-prerun"（裁决 i 条件 a）；
       - **占位决策字段**：prefill/decode instance 恒 0、ranks 恒
         instance-0 ranks——静态 rank 归因维度**不可信**；请求级指标
         （e2e/完成/sim_end/tput）可信（裁决 i 条件 b，口径登记于实录）；
       - 对账工具（ledger_reconcile 族）不读本 manifest——只消费归档
         jsonl（裁决 i 条件 c，已核实维持）。

输出目录：sh_test_mesh/generated/llama2_7b_inference_54npus_plan_<cfg8>/（保留 54npus
前缀以过 GEN_MATCH；<cfg8> = trace_config 内容摘要 8 位 hex）。幂等：重跑
覆盖同目录。fail-closed：请求队列为空/占位（request-neutral 占位 csv）时
exit 1 并说明。

在线调度直接根据当前硬件、模型和队列状态计算 Roofline 代价；物化目录不再
携带任何预计算成本表。
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
WORKLOAD_DIR = MODULE_DIR
SH_TEST_DIR = MODULE_DIR.parents[1]
GENERATED_ROOT = SH_TEST_DIR / "generated"
# SLO B2-A：本仓 B 类参数清单（值+证据+理由；B4 批次填充推导值）。
SLO_PARAMS_MANIFEST_PATH = SH_TEST_DIR / "slo_tools" / "slo_params_manifest.json"

for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from generate_face_trace import load_face_trace_config  # noqa: E402  (READ-ONLY import)

PREFIX = "llama2_7b_inference"
REPO_VARIANT = "astra-sim-sh_1.0"

# WP2 (SLO B1, 2026-08-26): the canonical sidecar written next to the queue
# by traces/derive_20_first_30_seconds.py carries the source-row trigger
# fields; they are folded into manifest.json per-request entries as
# human_time_ns / tool_time_ns / request_type (decode_length and session_id
# were already present).  request_type rule: human_time non-empty -> "human";
# tool_time non-empty -> "tool"; both empty -> turn 0 "human", otherwise
# "unknown" (count reported on stdout).
SIDECAR_REQUEST_TYPE_VALUES = frozenset({"human", "tool", "unknown"})


def _sidecar_path_for(queue_csv: Path) -> Path:
    """Canonical sidecar sibling of the queue CSV (per-repo naming)."""
    name = queue_csv.name
    if name.endswith("_request_queue_recompute.csv"):
        sibling = name[: -len("_request_queue_recompute.csv")] + "_canonical_sidecar.csv"
    else:
        sibling = name + ".canonical_sidecar.csv"
    return queue_csv.parent / sibling


def _derive_request_type(human_text: str, tool_text: str, turn_index: int):
    if human_text:
        return "human", False
    if tool_text:
        return "tool", False
    if turn_index == 0:
        return "human", False
    return "unknown", True


def _load_sidecar_rows(queue_csv: Path) -> dict:
    """request_id -> raw sidecar row dict; empty dict when no sidecar."""
    sidecar = _sidecar_path_for(queue_csv)
    if not sidecar.is_file():
        print(
            "[plan_materializer] WARNING: canonical sidecar not found next "
            f"to the queue ({sidecar}); manifest request_type falls back to "
            "turn-0=human / otherwise=unknown",
            file=sys.stderr,
        )
        return {}
    with sidecar.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        if "request_id" not in columns:
            raise RuntimeError(f"sidecar {sidecar} lacks a request_id column")
        rows: dict[str, dict] = {}
        for row in reader:
            request_id = (row.get("request_id") or "").strip()
            if not request_id or request_id.startswith("#"):
                continue
            if request_id in rows:
                raise RuntimeError(
                    f"sidecar {sidecar} has duplicate request_id {request_id}")
            rows[request_id] = row
    return rows


def _config_digest8(config_csv: Path) -> str:
    return hashlib.sha256(config_csv.read_bytes()).hexdigest()[:8]


def _derive_manifest_requests(config, sidecar_rows: dict) -> tuple[list, dict]:
    """manifest.json 的 requests[]（队列派生 9 字段 + WP2 透传字段）。

    Returns (requests, request_type_counts)."""
    requests = []
    last_final_by_session = {}
    type_counts = {"human": 0, "tool": 0, "unknown": 0}
    sidecar_missing = 0
    for index, spec in enumerate(config.request_queue):
        if spec.turn_index == 0:
            history = 0
            # recompute 单口径:turn-0 队列 prefill 已折入 prefix
            context = int(spec.prefill_length)
        else:
            history = last_final_by_session.get(spec.session_id, 0)
            # recompute 单口径后续 turn:context = 驻留 history + 新 prefill
            context = history + int(spec.prefill_length)
        final = context + int(spec.decode_length)

        # WP2 (SLO B1): sidecar 透传 human_time_ns/tool_time_ns/request_type。
        # 只追加键，不删不改既有字段；sidecar 缺行/缺列时按 B1 规则回退。
        row = sidecar_rows.get(spec.request_id)
        if row is None:
            sidecar_missing += 1
        human_text = ((row or {}).get("human_time_ns") or "").strip()
        tool_text = ((row or {}).get("tool_time_ns") or "").strip()
        request_type = ((row or {}).get("request_type") or "").strip()
        if request_type not in SIDECAR_REQUEST_TYPE_VALUES:
            request_type = ""
        if not request_type:
            request_type, _unknown = _derive_request_type(
                human_text, tool_text, int(spec.turn_index))
        type_counts[request_type] += 1

        requests.append({
            "request_id": spec.request_id,
            "session_id": spec.session_id,
            "turn_index": int(spec.turn_index),
            "queue_index": index,
            "prefill_length": int(spec.prefill_length),
            "decode_length": int(spec.decode_length),
            "history_tokens_before": history,
            "prefill_context_tokens": context,
            "final_context_tokens": final,
            "human_time_ns": int(human_text) if human_text else None,
            "tool_time_ns": int(tool_text) if tool_text else None,
            "request_type": request_type,
        })
        last_final_by_session[spec.session_id] = final
    if sidecar_rows and sidecar_missing:
        print(
            "[plan_materializer] WARNING: "
            f"{sidecar_missing} queue requests have no sidecar row; their "
            "request_type used the fallback rule",
            file=sys.stderr,
        )
    extra = set(sidecar_rows) - {
        spec.request_id for spec in config.request_queue}
    if extra:
        print(
            "[plan_materializer] WARNING: sidecar carries "
            f"{len(extra)} request_id(s) absent from the queue (e.g. "
            f"{sorted(extra)[:3]}); ignored",
            file=sys.stderr,
        )
    return requests, type_counts


def _slo_sampling_section() -> dict:
    """SLO B2-A（2026-08-26）：组装 metrics_manifest.json 的 slo_sampling 节。

    值取本仓 sh_test_mesh/slo_tools/slo_params_manifest.json 的
    watermark_sample_period_ns / link_bucket_ns 的 value；value 为 null/
    缺失/非法，或清单不可读时，回退文档化临时锚点（5,000,000 ns；1 ms 草案锚点超出 WP6/WP8 磁盘预算，主控裁决 2026-08-26）并置
    provisional=true（B4 批次将替换为推导值；C++ 侧每条相关记录回显实际
    使用周期）。任何情况下不使运行失败。
    """
    anchor_ns = 5_000_000
    try:
        loaded = json.loads(
            SLO_PARAMS_MANIFEST_PATH.read_text(encoding="utf-8"))
        params = loaded.get("params", {})
    except (OSError, ValueError, AttributeError):
        params = {}
    if not isinstance(params, dict):
        params = {}
    section: dict = {}
    used_anchor = False
    for param_name, key in (
        ("watermark_sample_period_ns", "watermark_period_ns"),
        ("link_bucket_ns", "link_bucket_ns"),
    ):
        value = params.get(param_name)
        if (isinstance(value, int) and not isinstance(value, bool)
                and value > 0):
            section[key] = value
        else:
            section[key] = anchor_ns
            used_anchor = True
    section["provisional"] = used_anchor
    return section


def _derive_metrics_requests(config):
    """metrics_manifest.json 的 requests[]（合成口径，占位 rank）。"""
    group_by_index = dict(enumerate(config.inference_groups))
    placeholder_ranks = list(group_by_index[0].ranks)
    queue_by_session_turn = {
        (spec.session_id, spec.turn_index): index
        for index, spec in enumerate(config.request_queue)
    }
    records = []
    for index, spec in enumerate(config.request_queue):
        if spec.turn_index == 0:
            if spec.session_arrival_time_ns is None:
                raise RuntimeError(
                    f"request {spec.request_id} (turn-0) lost its session "
                    "arrival time; materialize the request queue first")
            arrival = {"kind": "absolute",
                       "value_ns": int(spec.session_arrival_time_ns)}
        else:
            parent = queue_by_session_turn.get(
                (spec.session_id, spec.turn_index - 1))
            if parent is None or spec.inter_request_interval_ns is None:
                raise RuntimeError(
                    f"request {spec.request_id} lost its previous-turn "
                    "parent")
            arrival = {"kind": "after_request",
                       "parent_queue_index": parent,
                       "interval_ns": int(spec.inter_request_interval_ns)}
        records.append({
            "queue_index": index,
            "request_id": spec.request_id,
            "session_id": spec.session_id,
            "turn_index": int(spec.turn_index),
            "arrival": arrival,
            # SLO B2-C（WP9）：decode_length 透传（C++ 侧可选解析，用于
            # decode_length==1 的 first_token==completion 不变量；缺失则
            # 该不变量跳过并注明 manifest_decode_length_missing）。
            "decode_length": int(spec.decode_length),
            # 裁决 i 条件 b：占位决策字段（instance 0 / instance-0 ranks）
            "prefill_instance": 0,
            "prefill_ranks": placeholder_ranks,
            "decode_instance": 0,
            "decode_ranks": placeholder_ranks,
        })
    return records


def main() -> int:
    config = load_face_trace_config()
    if not config.request_queue:
        print(
            "[plan_materializer] request queue is empty (request-neutral "
            "placeholder); materialize the 30s/10s input first and point "
            "trace_config.csv request_queue_csv at it",
            file=sys.stderr)
        return 1
    sidecar_rows = _load_sidecar_rows(config.request_queue_csv)
    cfg8 = _config_digest8(config.config_csv)
    output_dir = GENERATED_ROOT / f"{PREFIX}_54npus_plan_{cfg8}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_requests, type_counts = _derive_manifest_requests(
        config, sidecar_rows)
    manifest = {
        "requests": manifest_requests,
        "selected_request_count": len(config.request_queue),
        "selected_session_count": len({
            spec.session_id for spec in config.request_queue}),
        "manifest_source": "synthetic-prerun",
        "repo_variant": REPO_VARIANT,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    metrics_manifest = {
        "schema_version": 1,
        "repo_variant": REPO_VARIANT,
        "manifest_source": "synthetic-prerun",
        "requests": _derive_metrics_requests(config),
        # SLO B2-A：采样参数透传（WP8 水位线周期 / WP6 链路时间桶）。
        "slo_sampling": _slo_sampling_section(),
    }
    (output_dir / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, separators=(",", ":")) + "\n",
        encoding="utf-8")

    runtime_note = (
        "runtime_config four files re-materialized by the config loader "
        "under generated/runtime_config/ (unchanged side effect)")
    print(json.dumps({
        "plan_dir": str(output_dir),
        "requests": len(manifest["requests"]),
        "sessions": manifest["selected_session_count"],
        "request_type_counts": type_counts,
        "note": runtime_note,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
