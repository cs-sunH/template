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
     每请求 human_time_ns / tool_time_ns / request_type。
     H5②（2026-09-29）起 human_time_ns/tool_time_ns 为 **initiator 口径**
     （冻结 sidecar 语义：本行等待 = 本行之后的 gap，物理落在
     [completion(j), arrival(j+1)]，取自同 session 下一队列行的
     inter_request_interval_ns × 本行 next_trigger_type；会话末行无后继
     记 None/None）——与 slo_stats session H5① 修正（Σ 只累加除末轮外
     的 turns）配套，消费者仅 slo_stats session。request_type 保留到达侧
     触发口径（触发本请求的 gap 类型：human 非空→human、tool 非空→tool、
     皆空 turn0→human、皆空 turn>0→unknown，stdout 计数；metrics_
     postprocess type_counts 语义不变）——同一记录内 request_type 与
     human/tool 描述两个不同 gap（见 _initiator_gap_ns docstring）。
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
         human_time_ns / tool_time_ns / request_type——slo_stats session
         的 T_session 依赖这两个透传字段。附加键，不删不改既有键；
         C++ MetricCollector 只按需读键，多余键无影响。H5②（2026-09-29）
         起 human/tool 为 initiator 口径（_initiator_gap_ns）、request_type
         保留到达侧触发口径（语义不变），见上 B1/WP2 追加段。

输出目录：sh_test_mesh/generated/llama2_7b_inference_54npus_plan_<cfg8>/（保留 54npus
前缀以过 GEN_MATCH；<cfg8> = trace_config 内容摘要 8 位 hex）。幂等：重跑
覆盖同目录。fail-closed：请求队列为空/占位（request-neutral 占位 csv）时
exit 1 并说明。

"""

import hashlib
import json
import os
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
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
            # R3-J5（2026-09-29 face 移植）：空行判定对齐 C++ 口径——
            # strip 后判空（WindowedTraceReader 的行处理在剥除行尾符后
            # 判空）。旧写法 rstrip("\n").rstrip("\r") 与 C++ 对 CRLF
            # 空行的处理相反（C++ 视其为非空数据行、断开 session 连续
            # 段；Python 跳过），advisory 字段两套口径会对同一文件给出
            # 不同 span。
            line = raw.strip()
            if not line:
                continue  # 空行(含 CRLF/纯空白):非数据行,也不断开 session 连续段
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

    H5②（2026-09-29）：本函数保留为 request_type 推导口径（到达侧触发
    gap = 本请求自身类型的事实来源，metrics_postprocess 的 type_counts
    消费，语义不变）。human_time_ns/tool_time_ns 字段面自本日起改用
    _initiator_gap_ns（initiator 口径，slo_stats session H5① 修正的
    数据侧配套）——同一记录内 request_type 与 human/tool 字段描述的
    是两个不同 gap，见 _initiator_gap_ns docstring。
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


def _initiator_gap_ns(spec, next_spec):
    """(human_time_ns, tool_time_ns) recorded ON ``spec``'s own row.

    H5②（2026-09-29，initiator 口径）：冻结 sidecar 语义 = "the gap after
    turn j is the human_time or tool_time recorded on turn j's own row"
    （derive_20_first_30_seconds.py），物理上落在 [completion(j),
    arrival(j+1)]——slo_stats session（H5① 修正后）只把它从除末轮外的
    行上累加。取值自同 session 的**下一**队列行：interval =
    next_spec.inter_request_interval_ns，类型 = 本行 next_trigger_type；
    0-gap 延续（interval==0）双侧留空。会话末行在队列里没有后继行，
    其行后等待（在会话窗口外、对 T_session 无贡献）无从取数 ->
    (None, None)。消费者仅 slo_stats session（manifest human/tool 字段
    面无其他读者）。
    """

    human_ns = None
    tool_ns = None
    if (next_spec is not None
            and next_spec.session_id == spec.session_id):
        interval = next_spec.inter_request_interval_ns
        if interval is not None:
            interval = int(interval)
            if interval > 0:
                trigger = spec.next_trigger_type
                if trigger == "human":
                    human_ns = interval
                elif trigger == "tool":
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
    H5②（2026-09-29）：human_time_ns/tool_time_ns 改 initiator 口径
    （_initiator_gap_ns：本行等待=本行之后的 gap，与冻结 sidecar 语义
    对齐）；request_type 保留到达侧触发口径（_trigger_gap_ns，本请求
    自身类型的事实来源，metrics_postprocess type_counts 语义不变）。
    """

    requests = []
    seen_request_ids: set = set()
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
        human_ns, tool_ns = _trigger_gap_ns(
            spec, prev_spec_by_session.get(spec.session_id))
        request_type = _request_type(human_ns, tool_ns, int(spec.turn_index))
        request_type_counts[request_type] = request_type_counts.get(request_type, 0) + 1
        initiator_human_ns, initiator_tool_ns = _initiator_gap_ns(
            spec,
            config.request_queue[index + 1]
            if index + 1 < len(config.request_queue) else None)
        # 4.3 fix (2026-09-29, same as the wscllm materializer): manifest
        # 级 request_id 唯一性 fail-closed（与 token manifest 同强度）。
        # （2026-10-01 R3 订正：上游队列装载器实际不查重复 id——本仓
        # 的唯一性防线就是这道 manifest 边界复核，非纵深第二层。）manifest
        # 是冻结产物、被下游逐 id 消费（last-wins/双计均静默），在清单
        # 边界独立复核。
        if spec.request_id in seen_request_ids:
            raise RuntimeError(
                f"manifest requests[] has duplicate request_id "
                f"{spec.request_id!r} (fail-closed); downstream id-keyed "
                "consumers cannot disambiguate it"
            )
        seen_request_ids.add(spec.request_id)
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
            # H5②：human/tool = initiator 口径（本行之后的 gap）；
            # request_type = 到达侧触发类型（语义不变）。
            "human_time_ns": initiator_human_ns,
            "tool_time_ns": initiator_tool_ns,
            "request_type": request_type,
        })
        last_final_by_session[spec.session_id] = final
        prev_spec_by_session[spec.session_id] = spec
    return requests, request_type_counts


def _derive_metrics_requests(config):
    """metrics_manifest.json 的 requests[]（合成口径，占位 rank）。

    B4/WP2 透传补丁：每条附加 human_time_ns / tool_time_ns /
    request_type——slo_stats session 的 T_session 输入；附加键，不删
    不改既有键。H5②（2026-09-29）：human_time_ns/tool_time_ns 改
    initiator 口径（_initiator_gap_ns，本行等待=本行之后的 gap，与冻结
    sidecar 语义及 slo_stats H5① 修正对齐）；request_type 保留到达侧
    触发口径（语义不变）。
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
        trigger_human_ns, trigger_tool_ns = _trigger_gap_ns(
            spec, prev_spec_by_session.get(spec.session_id))
        prev_spec_by_session[spec.session_id] = spec
        initiator_human_ns, initiator_tool_ns = _initiator_gap_ns(
            spec,
            config.request_queue[index + 1]
            if index + 1 < len(config.request_queue) else None)
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
            # H5②：human/tool = initiator 口径（本行之后的 gap）；
            # request_type = 到达侧触发类型（语义不变）。
            "human_time_ns": initiator_human_ns,
            "tool_time_ns": initiator_tool_ns,
            "request_type": _request_type(
                trigger_human_ns, trigger_tool_ns, int(spec.turn_index)),
        })
    return records


def _atomic_write_text(path: Path, payload: str) -> None:
    """L14（2026-09-29）：tmp + os.replace 原子落盘（单文件粒度）。

    manifest.json 与 metrics_manifest.json 双写之间仍非事务（无跨文件
    原子性/代际绑定），但消除满盘/SIGKILL 场景的半写撕裂——runner 消费
    的要么是旧完整文件、要么是新完整文件，不再有半截 JSON。tmp 与目标
    同目录（同文件系统，os.replace 原子；覆盖同目录既存 .tmp 幂等）。
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp, path)


def _resolved_backend_keys(config) -> dict:
    """manifest 的 die 后端键（契约 §7 裁定 2；第一期）。

    从 load_face_trace_config 副产的 resolved system.json（config.
    system_config）读 compute-cost-model 与 die-model-config：前者声明
    backend（die/roofline/replay），后者是 resolved_die_model.json 绝对
    路径（resolver 注入的 managed 字段）。legacy 配置无这些键 → 返回空
    dict（manifest 形状不变，向后兼容）；die-model-config 必须是已存在
    的绝对路径文件（fail-closed，不静默丢键）。"""
    keys = {}
    with open(config.system_config, "r", encoding="utf-8") as source:
        system = json.load(source)
    cost_model = system.get("compute-cost-model")
    if cost_model is not None:
        keys["compute-cost-model"] = str(cost_model)
    die_model_config = system.get("die-model-config")
    if die_model_config is not None:
        path = Path(str(die_model_config))
        if not path.is_absolute():
            raise RuntimeError(
                f"resolved system.json die-model-config must be an absolute "
                f"path, got {die_model_config!r}")
        if not path.is_file():
            raise RuntimeError(
                f"resolved system.json die-model-config file is missing: "
                f"{path}")
        keys["die-model-config"] = str(path)
    return keys


def main() -> int:
    config = load_face_trace_config()
    # L14（2026-09-29）显式守卫：request_queue_session_limit>0 时 loader
    # 只把前 N 个 session 截进 Python 侧 config.request_queue，而 C++
    # WindowedTraceReader 按物化队列 CSV 全量枚举（该旋钮无 C++/runner
    # 消费面；两仓 trace_config.csv 恒 0、全工作区无 >0 的合法使用）——
    # manifest/metrics_manifest 与 C++ 请求枚举系统性分歧（先静默指标
    # 错配、后整跑崩溃）。旋钮事实上不可用，fail-closed 钉死。
    if config.request_queue_session_limit > 0:
        print(
            "[plan_materializer] request_queue_session_limit="
            f"{config.request_queue_session_limit} (>0) is not supported "
            "in this repo: the loader would truncate only the Python-side "
            "request_queue while the C++ reader enumerates the materialized "
            "queue CSV in full (silent metrics mismatch then hard failure); "
            "materialize a subset queue file instead",
            file=sys.stderr)
        return 1
    # (2026-09-29 face 移植)原"空队列占位"分支已删:load_request_queue 对
    # 0 数据行直接 ValueError("must contain at least one request"),
    # config.request_queue 到此处不可能为空,分支不可达。
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
    # 第一期（契约 §7 裁定 2）：die 后端键（compute-cost-model +
    # die-model-config 绝对路径）；legacy 配置无键，manifest 形状不变。
    manifest.update(_resolved_backend_keys(config))
    # L14：两 manifest 先写 tmp 再 os.replace 原子落盘（见
    # _atomic_write_text docstring；两写间仍非事务，半写撕裂已消除）。
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2) + "\n")

    # (2026-10-01 审查修复) 工作负载身份 digest 三件：metrics_postprocess
    # 的 auto-group 以 (repo_variant, run_mode, trace_digest,
    # request_mapping_digest) 为组键（sec.11.4 "绝不静默混组"守卫），本
    # manifest 此前一件都不写 ⇒ 键退化为 ("astra-sim-joint","service",
    # "","")——任意两个不同工作负载的 run 被静默归入同组互相归一化，
    # digest-mismatch 失败条件永远不可触发。trace_digest = trace_config
    # 全文、request_mapping_digest = 请求队列 CSV 全文的 sha256（与
    # 物化输入唯一允许源一致；kv_event_digest 留空——本仓 kv 事件面经
    # 决策日志/侧车，无独立 kv 事件文件可摘要）。
    queue_digest = hashlib.sha256(
        config.request_queue_csv.read_bytes()).hexdigest()
    trace_digest = hashlib.sha256(
        config.config_csv.read_bytes()).hexdigest()
    metrics_manifest = {
        "schema_version": 1,
        "repo_variant": REPO_VARIANT,
        "manifest_source": "synthetic-prerun",
        # B2/WP6+WP8: 采样参数注入（值见 _slo_sampling_section；C++ 记录回显）
        "slo_sampling": _slo_sampling_section(),
        "trace_digest": trace_digest,
        "request_mapping_digest": queue_digest,
        "kv_event_digest": "",
        "requests": _derive_metrics_requests(config),
    }
    # 第一期：die 后端键同步进 metrics_manifest（与 manifest.json 同源，
    # _resolved_backend_keys）。归档 run 目录只保证携带本文件——把
    # compute-cost-model 放进来让 metrics_postprocess 的 die 利用率管线
    # 断言（active_kernel_die_* 非 null 且 ∈[0,1+ε]）离线可判；legacy 配置
    # 无键，manifest 形状不变。C++ load_manifest 与 slo_common 均只取
    # 自己认识的键，未知顶层键被忽略。
    metrics_manifest.update(_resolved_backend_keys(config))
    _atomic_write_text(
        output_dir / "metrics_manifest.json",
        json.dumps(metrics_manifest, separators=(",", ":")) + "\n")

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
