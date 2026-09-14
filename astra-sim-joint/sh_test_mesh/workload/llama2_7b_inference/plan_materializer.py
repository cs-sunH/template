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
     turn-0 history=0、prefill_context=折入后的 prefill_length；
     turn>0 history=上一请求 final、prefill_context=history+prefill_length
     （recompute 单口径，turn-0 前缀已折入队列 prefill_length）；
     final=context+decode）；
     B1/WP2 追加（SLO request_metrics.csv 连接用，不删不改既有字段）：
     每请求 human_time_ns / tool_time_ns / request_type——触发本请求的
     gap（上一队列行 next_trigger_type × 本行 inter_request_interval_ns，
     0-gap 无类型延续记 None/None）；request_type 规则：human 非空→human、
     tool 非空→tool、皆空 turn0→human、皆空 turn>0→unknown（stdout 计数）。
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
         jsonl（裁决 i 条件 c，已核实维持）；
       - B4/WP2 透传补丁（2026-08-26，S3 异常③）：requests[] 每条附加
         human_time_ns / tool_time_ns / request_type（与 manifest.json 同
         源同值，_trigger_gap_ns/_request_type 推导）——slo_stats session
         的 T_session 依赖这两个透传字段，缺省（此前）只能 NA 或靠
         --request-manifest 指到 manifest.json。附加键，不删不改既有键；
         C++ MetricCollector 只按需读键，多余键无影响。

输出目录：sh_test_mesh/generated/llama2_7b_inference_54npus_plan_<cfg8>/（保留 54npus
前缀以过 GEN_MATCH；<cfg8> = trace_config 内容摘要 8 位 hex）。幂等：重跑
覆盖同目录。fail-closed：请求队列为空/占位（request-neutral 占位 csv）时
exit 1 并说明。

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
REPO_VARIANT = "astra-sim-joint"

# B2/WP6+WP8 (SLO 指标改造): slo_sampling 注入参数的文档化临时锚点——
# sh_test_mesh/slo_tools/slo_params_manifest.json 对应 value 为 null（B4
# 批次才推导）或文件缺失时使用，并置 provisional=true（下游 C++ 记录回显）。
# 2026-08-26 主控修正：锚点 1ms -> 5ms（60s 窗口桶记录量级控制）。
SLO_SAMPLING_PROVISIONAL_NS = 5_000_000


def _slo_sampling_section():
    """B2: metrics_manifest.json 的 ``slo_sampling`` 节。

    值来源：本仓 ``sh_test_mesh/slo_tools/slo_params_manifest.json`` 的
    ``params.watermark_sample_period_ns.value`` / ``params.link_bucket_ns.value``；
    value 为 null / 参数缺失 / 文件缺失 / 非正整数时回落临时锚点
    (5,000,000 ns) 并置 ``provisional: true``。只读、fail-open（缺参数不阻断
    物化；B4 推导后由 C++ 记录回显实际使用值）。
    """
    section = {}
    provisional = False
    manifest_path = SH_TEST_DIR / "slo_tools" / "slo_params_manifest.json"
    params = {}
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("params"), dict):
            params = loaded["params"]
    except (OSError, ValueError):
        params = {}
    for key, field in (
        ("watermark_period_ns", "watermark_sample_period_ns"),
        ("link_bucket_ns", "link_bucket_ns"),
    ):
        raw = params.get(field)
        value = raw.get("value") if isinstance(raw, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            section[key] = SLO_SAMPLING_PROVISIONAL_NS
            provisional = True
        else:
            section[key] = value
    section["provisional"] = provisional
    return section


def _config_digest8(config_csv: Path) -> str:
    return hashlib.sha256(config_csv.read_bytes()).hexdigest()[:8]


def _scan_max_session_span(queue_csv):
    """P0-2 (2026-08-31, 总文档 §4 P0-2.1) provenance audit: the queue CSV's
    max consecutive same-session data-row span.

    流式单遍（逐行、不驻留行内容），与 sh/face 系 C++ 启动预检
    WindowedTraceReader::scan_max_session_span 同一判定口径：第 1 列
    session_id、跳过表头（首个非空行）、空行既不算数据行也不断开同
    session 连续段；行号口径 = 1-based CSV 文件行号（表头=第 1 行，
    首个数据行=第 2 行）。并列取首个最大段（确定性）。

    wscllm 裁决（sync-A16 批次4，合同 §2.1/§3.1）：本仓 window 为
    advisory（calendar reader），不设 C++ span 预检/拒绝门——本字段
    仅 provenance（campaign 复核与根因归档），无任何运行期强制的
    拒绝点。
    """
    max_span = 0
    span_session = ""
    span_first_line = 0
    span_last_line = 0
    data_rows = 0
    run_session = None
    run_length = 0
    run_first_line = 0
    run_last_line = 0
    header_seen = False
    with open(queue_csv, newline="", encoding="utf-8") as source:
        for file_line, raw in enumerate(source, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue  # 空行：非数据行，也不断开 session 连续段
            if not header_seen:
                header_seen = True  # 首个非空行 = 表头
                continue
            session_id = line.split(",", 1)[0]
            data_rows += 1
            if run_length > 0 and session_id == run_session:
                run_length += 1
                run_last_line = file_line
            else:
                # 严格大于：并列保留首个最大段
                if run_length > max_span:
                    max_span = run_length
                    span_session = run_session or ""
                    span_first_line = run_first_line
                    span_last_line = run_last_line
                run_session = session_id
                run_length = 1
                run_first_line = file_line
                run_last_line = file_line
    # 收尾段（EOF 结束的最大段，如单 session 文件）
    if run_length > max_span:
        max_span = run_length
        span_session = run_session or ""
        span_first_line = run_first_line
        span_last_line = run_last_line
    return {
        "max_same_session_span": max_span,
        "span_session_id": span_session,
        "span_row_range": [span_first_line, span_last_line],
        "span_row_range_convention":
            "1-based csv file lines (header=line 1; mirrors C++ "
            "WindowedTraceReader::scan_max_session_span)",
        "scanned_data_rows": data_rows,
    }


def _trigger_gap_ns(spec, prev_spec):
    """(human_time_ns, tool_time_ns) of the gap that TRIGGERED ``spec``.

    B1/WP2: the queue stores, per row, the interval to its successor turn
    (``inter_request_interval_ns``) plus the successor's trigger type
    (``next_trigger_type`` -- "human"/"tool", frozen materializer
    semantics).  A request's own trigger gap is therefore read off the
    PREVIOUS queue row of the same session: interval>0 attributes exactly
    (next_trigger=human => human_time non-empty = interval;
    next_trigger=tool & interval>0 => tool_time non-empty = interval);
    a zero interval is the untyped 0-gap continuation (both stay empty).
    Turn-0 rows have no preceding gap (both None).
    """

    human_ns = None
    tool_ns = None
    if spec.turn_index > 0 and prev_spec is not None:
        interval = spec.inter_request_interval_ns
        if interval is not None:
            interval = int(interval)
            trigger = prev_spec.next_trigger_type
            if trigger == "human":
                human_ns = interval
            elif trigger == "tool" and interval > 0:
                tool_ns = interval
    return human_ns, tool_ns


def _request_type(human_ns, tool_ns, turn_index):
    """human 非空→human；tool 非空→tool；皆空 turn0→human、否则 unknown."""

    if human_ns is not None:
        return "human"
    if tool_ns is not None:
        return "tool"
    return "human" if turn_index == 0 else "unknown"


def _derive_manifest_requests(config):
    """manifest.json 的 requests[]（队列派生 9 字段 + B1/WP2 追加字段）。

    B1/WP2 追加（不删不改既有字段）：human_time_ns / tool_time_ns /
    request_type（session_id / decode_length 已在 9 字段内，缺则补）。
    """

    requests = []
    last_final_by_session = {}
    prev_spec_by_session = {}
    request_type_counts = {"human": 0, "tool": 0, "unknown": 0}
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
        human_ns, tool_ns = _trigger_gap_ns(spec, prev_spec_by_session.get(spec.session_id))
        request_type = _request_type(human_ns, tool_ns, int(spec.turn_index))
        request_type_counts[request_type] = request_type_counts.get(request_type, 0) + 1
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
            # ---- B1/WP2 追加字段（SLO request_metrics 连接用） ----
            "human_time_ns": human_ns,
            "tool_time_ns": tool_ns,
            "request_type": request_type,
        })
        last_final_by_session[spec.session_id] = final
        prev_spec_by_session[spec.session_id] = spec
    return requests, request_type_counts


def _derive_metrics_requests(config):
    """metrics_manifest.json 的 requests[]（合成口径，占位 rank）。

    B4/WP2 透传补丁：每条附加 human_time_ns / tool_time_ns /
    request_type（与 manifest.json 同源同值——slo_stats session 的
    T_session 输入；附加键，不删不改既有键）。
    """
    group_by_index = dict(enumerate(config.inference_groups))
    placeholder_ranks = list(group_by_index[0].ranks)
    queue_by_session_turn = {
        (spec.session_id, spec.turn_index): index
        for index, spec in enumerate(config.request_queue)
    }
    prev_spec_by_session = {}
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
        human_ns, tool_ns = _trigger_gap_ns(
            spec, prev_spec_by_session.get(spec.session_id))
        prev_spec_by_session[spec.session_id] = spec
        records.append({
            "queue_index": index,
            "request_id": spec.request_id,
            "session_id": spec.session_id,
            "turn_index": int(spec.turn_index),
            "arrival": arrival,
            # B2/WP9-C++: decode_length 带入（code-8 首 token 的
            # decode_length==1 不变量检查用；缺失时 C++ 侧跳过该不变量）。
            "decode_length": int(spec.decode_length),
            # 裁决 i 条件 b：占位决策字段（instance 0 / instance-0 ranks）
            "prefill_instance": 0,
            "prefill_ranks": placeholder_ranks,
            "decode_instance": 0,
            "decode_ranks": placeholder_ranks,
            # ---- B4/WP2 附加键（S3 异常③：session 统计透传） ----
            "human_time_ns": human_ns,
            "tool_time_ns": tool_ns,
            "request_type": _request_type(
                human_ns, tool_ns, int(spec.turn_index)),
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
    # P0-2 (2026-08-31, 总文档 §4 P0-2.1): max consecutive same-session span
    # ——与 sh/face 系启动预检同口径的 provenance 审计字段。wscllm 裁决
    # （批次4）：本仓无 C++ 拒绝门（window advisory），字段仅 provenance。
    span_scan = _scan_max_session_span(config.request_queue_csv)
    cfg8 = _config_digest8(config.config_csv)
    output_dir = GENERATED_ROOT / f"{PREFIX}_54npus_plan_{cfg8}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_requests, request_type_counts = _derive_manifest_requests(config)
    manifest = {
        "requests": manifest_requests,
        "selected_request_count": len(config.request_queue),
        "selected_session_count": len({
            spec.session_id for spec in config.request_queue}),
        "manifest_source": "synthetic-prerun",
        "repo_variant": REPO_VARIANT,
        # P0-2 provenance: 同 session 连续段审计字段（wscllm 无 C++ 拒绝门，
        # window advisory——字段仅供 campaign 复核，见函数 docstring 裁决）。
        "max_same_session_span": span_scan["max_same_session_span"],
        "span_session_id": span_scan["span_session_id"],
        "span_row_range": span_scan["span_row_range"],
        "span_row_range_convention": span_scan["span_row_range_convention"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    metrics_manifest = {
        "schema_version": 1,
        "repo_variant": REPO_VARIANT,
        "manifest_source": "synthetic-prerun",
        # B2/WP6+WP8: 采样参数注入（值见 _slo_sampling_section；C++ 记录回显）
        "slo_sampling": _slo_sampling_section(),
        "requests": _derive_metrics_requests(config),
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
        # P0-2 (2026-08-31): span 审计字段随 stdout 权威 provenance 记录
        # 一并输出（与 manifest.json 持久字段同源同值）。
        "max_same_session_span": span_scan["max_same_session_span"],
        "span_session_id": span_scan["span_session_id"],
        "span_row_range": span_scan["span_row_range"],
        "request_type_counts": request_type_counts,
        "note": runtime_note,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
