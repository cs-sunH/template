#!/usr/bin/env python3
"""plan_materializer.py -- 路径③④ 的 plan 目录物化入口（唯一输入物化器）。

在线③④需要三类输入，本脚本以**不跑规划器、不写任何 .et** 的方式产出：

  1. runtime_config 四小件（system/comm_group/remote_memory/network.yml）
     —— 由 load_load_wsc_llm_trace_config() 装载配置时经 config_resolver.materialize_runtime_configs
     副产（路径不变，零改动）；
  2. manifest.json —— 含在线侧实际消费的**队列派生 9 字段**
     （request_id/session_id/turn_index/queue_index/prefill_length/
     decode_length/history_tokens_before/prefill_context_tokens/
     final_context_tokens；推导规则与《sh_3.0的120档问题分析报告.md》§4
     受控实验（20 档真实 manifest 2091/2091 行 0 mismatch）同式：
     turn-0 history=0；turn>0 history=上一请求 final；
     prefill_context=折入后的 prefill_length（turn-0）或 history+
     prefill_length（turn>0，recompute 口径）；final=context+decode）
     + WP2 透传字段（human_time_ns/tool_time_ns（非空者）/
     request_type/prefix_len，源自队列旁 canonical sidecar 的
     human_time_ns/tool_time_ns/request_type/raw_prefix_tokens 列；
     sidecar 缺失/行缺失时对应字段不写，消费方回退 unknown/NA 并注明，
     不猜测填充）；
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

输出目录：sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_plan_<cfg8>/（保留 54npus
前缀以过 GEN_MATCH；<cfg8> = trace_config 内容摘要 8 位 hex）。幂等：重跑
覆盖同目录。fail-closed：请求队列为空/占位（request-neutral 占位 csv）时
exit 1 并说明。

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

for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_wsc_llm_trace import load_wsc_llm_trace_config  # noqa: E402  (READ-ONLY import)

PREFIX = "llama2_7b_wsc_llm_inference"
REPO_VARIANT = "astra-sim-wscllm"


def _config_digest8(config_csv: Path) -> str:
    return hashlib.sha256(config_csv.read_bytes()).hexdigest()[:8]


QUEUE_SUFFIX = "_request_queue_recompute.csv"
SIDECAR_SUFFIX = "_canonical_sidecar.csv"

# SLO pipeline B2 (CPP_SPEC §A) provisional sampling anchor. Used when the
# per-repo slo_params_manifest.json carries no derived value yet (all null
# until batch B4); the value follows the coordinator ruling 2026-08-26
# (1 ms measured 31,895 bucket records on the S3 2s window -- over budget).
SLO_PROVISIONAL_SAMPLING_NS = 5_000_000


def _slo_sampling_node():
    """slo_sampling 节（CPP_SPEC §A）：WP8 水位线周期 + WP6 链路桶长。

    值来源：sh_test_mesh/slo_tools/slo_params_manifest.json 的
    watermark_sample_period_ns / link_bucket_ns 的 value；value 为 null 或
    文件缺失时用文档化临时锚点（SLO_PROVISIONAL_SAMPLING_NS）并置
    provisional=true。C++ 侧缺节同样回退临时锚点（不 fail 运行），每条
    相关记录都回显实际使用的周期。
    """
    params = {}
    manifest_path = SH_TEST_DIR / "slo_tools" / "slo_params_manifest.json"
    try:
        params = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "params", {}
        )
    except (OSError, ValueError):
        params = {}

    def _param_ns(name):
        entry = params.get(name)
        if not isinstance(entry, dict):
            return None
        value = entry.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return int(value)

    watermark = _param_ns("watermark_sample_period_ns")
    link_bucket = _param_ns("link_bucket_ns")
    return {
        "watermark_period_ns": (
            watermark if watermark is not None else SLO_PROVISIONAL_SAMPLING_NS
        ),
        "link_bucket_ns": (
            link_bucket if link_bucket is not None else SLO_PROVISIONAL_SAMPLING_NS
        ),
        "provisional": watermark is None or link_bucket is None,
    }


def _sidecar_path_for_queue(queue_csv: Path):
    """Canonical sidecar sibling of a materialized request queue (or None)."""
    name = queue_csv.name
    if not name.endswith(QUEUE_SUFFIX):
        return None
    sidecar = queue_csv.parent / (name[: -len(QUEUE_SUFFIX)] + SIDECAR_SUFFIX)
    return sidecar if sidecar.is_file() else None


def _load_sidecar_rows(queue_csv: Path) -> dict:
    """Sidecar rows by request_id (WP2 passthrough channel; absence tolerated)."""
    sidecar = _sidecar_path_for_queue(queue_csv)
    if sidecar is None:
        return {}
    with sidecar.open(newline="", encoding="utf-8") as source:
        return {row["request_id"]: row for row in csv.DictReader(source)}


def _derive_manifest_requests(config, sidecar_rows):
    """manifest.json 的 requests[]（队列派生 9 字段）。"""
    requests = []
    last_final_by_session = {}
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
        entry = {
            "request_id": spec.request_id,
            "session_id": spec.session_id,
            "turn_index": int(spec.turn_index),
            "queue_index": index,
            "prefill_length": int(spec.prefill_length),
            "decode_length": int(spec.decode_length),
            "history_tokens_before": history,
            "prefill_context_tokens": context,
            "final_context_tokens": final,
        }
        # WP2 passthrough (append-only; existing fields above are frozen):
        # sidecar columns only; absent row/column -> field omitted, never
        # guessed.
        sidecar_row = sidecar_rows.get(spec.request_id)
        if sidecar_row is not None:
            human_text = (sidecar_row.get("human_time_ns") or "").strip()
            tool_text = (sidecar_row.get("tool_time_ns") or "").strip()
            if human_text:
                entry["human_time_ns"] = int(human_text)
            if tool_text:
                entry["tool_time_ns"] = int(tool_text)
            request_type = (sidecar_row.get("request_type") or "").strip()
            if request_type:
                entry["request_type"] = request_type
            prefix_text = (sidecar_row.get("raw_prefix_tokens") or "").strip()
            if prefix_text:
                entry["prefix_len"] = int(prefix_text)
        requests.append(entry)
        last_final_by_session[spec.session_id] = final
    return requests


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
            # SLO B2-C（WP9，WP9_CONTRACT §6 2026-08-27 放宽后口径）：
            # decode_length 透传（C++ 侧可选解析，用于 decode_length==1 的
            # 信息性 first_token_completion_skew_ns 字段；缺失则该字段
            # 不可评估并注明 manifest_decode_length_missing）。
            "decode_length": int(spec.decode_length),
            # 裁决 i 条件 b：占位决策字段（instance 0 / instance-0 ranks）
            "prefill_instance": 0,
            "prefill_ranks": placeholder_ranks,
            "decode_instance": 0,
            "decode_ranks": placeholder_ranks,
        })
    return records


def main() -> int:
    config = load_wsc_llm_trace_config()
    if not config.request_queue:
        print(
            "[plan_materializer] request queue is empty (request-neutral "
            "placeholder); materialize the 30s/10s input first and point "
            "trace_config.csv request_queue_csv at it",
            file=sys.stderr)
        return 1
    cfg8 = _config_digest8(config.config_csv)
    output_dir = GENERATED_ROOT / f"{PREFIX}_54npus_plan_{cfg8}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sidecar_rows = _load_sidecar_rows(config.request_queue_csv)
    manifest = {
        "requests": _derive_manifest_requests(config, sidecar_rows),
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
        # SLO pipeline B2（CPP_SPEC §A）：C++ 观测器采样锚点（WP8 水位线
        # 周期 / WP6 链路桶长）；null 值 -> 临时锚点 + provisional=true。
        "slo_sampling": _slo_sampling_node(),
    }
    (output_dir / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, separators=(",", ":")) + "\n",
        encoding="utf-8")

    runtime_note = (
        "runtime_config four files re-materialized by the config loader "
        "under generated/runtime_config/ (unchanged side effect)")
    request_type_counts = {}
    for entry in manifest["requests"]:
        key = entry.get("request_type", "<absent>")
        request_type_counts[key] = request_type_counts.get(key, 0) + 1
    print(json.dumps({
        "plan_dir": str(output_dir),
        "requests": len(manifest["requests"]),
        "sessions": manifest["selected_session_count"],
        "sidecar": str(_sidecar_path_for_queue(config.request_queue_csv)),
        "sidecar_rows_joined": len(sidecar_rows),
        "request_type_counts": request_type_counts,
        "note": runtime_note,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
