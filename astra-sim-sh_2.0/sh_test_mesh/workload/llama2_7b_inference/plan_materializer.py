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
     prefill_context=input_tokens_total（sidecar 变体）或折入后的
     prefill_length（recompute 变体/无 sidecar）；final=context+decode）；
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

sh_1.0 附加：face_lut.csv（合同⑨冻结 LUT）——由 face_scheduler 的
build_instances + FaceLut 直接构建（与请求规划无关），口径 =
plan.lut.export_csv（FaceLut 构建导出）。
"""

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

from generate_face_trace import load_face_trace_config  # noqa: E402  (READ-ONLY import)

PREFIX = "llama2_7b_inference"
REPO_VARIANT = "astra-sim-sh_2.0"
WRITE_FACE_LUT = False


def _config_digest8(config_csv: Path) -> str:
    return hashlib.sha256(config_csv.read_bytes()).hexdigest()[:8]


def _derive_manifest_requests(config):
    """manifest.json 的 requests[]（队列派生 9 字段）。"""
    requests = []
    last_final_by_session = {}
    for index, spec in enumerate(config.request_queue):
        if spec.turn_index == 0:
            history = 0
        else:
            history = last_final_by_session.get(spec.session_id, 0)
            prefix = getattr(spec, "prefix_tokens", None)
            if prefix is not None:
                # sidecar_restore 变体:history = min(prefix, 上一请求 final)
                #（《120档问题分析报告》§4 已验证同式)
                history = min(int(prefix), history)
        if getattr(spec, "input_tokens_total", None) is not None:
            # sidecar_restore 变体:context = prefix + new(sidecar 给全量)
            context = int(spec.input_tokens_total)
        elif spec.turn_index == 0:
            # recompute 变体:turn-0 队列 prefill 已折入 prefix
            context = int(spec.prefill_length)
        else:
            # recompute 变体后续 turn:context = 驻留 history + 新 prefill
            context = history + int(spec.prefill_length)
        final = context + int(spec.decode_length)
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
        })
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
            # 裁决 i 条件 b：占位决策字段（instance 0 / instance-0 ranks）
            "prefill_instance": 0,
            "prefill_ranks": placeholder_ranks,
            "decode_instance": 0,
            "decode_ranks": placeholder_ranks,
        })
    return records


def _write_face_lut(config, output_dir: Path) -> None:
    """sh_1.0 合同⑨冻结 LUT（FaceLut 构建，与请求规划无关）。"""
    from face_scheduler import FaceInstanceSpec, build_instances  # noqa: E402
    specs = tuple(
        FaceInstanceSpec(group.name, group.pg_name, group.ranks)
        for group in config.inference_groups
    )
    instances = build_instances(config.hardware, config.model, specs)
    lut = instances.lut
    lut.export_csv(output_dir / "face_lut.csv")


def main() -> int:
    config = load_face_trace_config()
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

    manifest = {
        "requests": _derive_manifest_requests(config),
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
    }
    (output_dir / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, separators=(",", ":")) + "\n",
        encoding="utf-8")

    if WRITE_FACE_LUT:
        _write_face_lut(config, output_dir)

    runtime_note = (
        "runtime_config four files re-materialized by the config loader "
        "under generated/runtime_config/ (unchanged side effect)")
    print(json.dumps({
        "plan_dir": str(output_dir),
        "requests": len(manifest["requests"]),
        "sessions": manifest["selected_session_count"],
        "face_lut": WRITE_FACE_LUT,
        "note": runtime_note,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
