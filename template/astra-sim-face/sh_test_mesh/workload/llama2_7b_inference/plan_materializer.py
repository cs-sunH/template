#!/usr/bin/env python3
"""plan_materializer.py -- 路径③④ 的 plan 目录物化入口（唯一输入物化器）。

在线③④需要三类输入，本脚本以**不跑规划器、不写任何 .et** 的方式产出：

  1. runtime_config 三件（system/comm_group/network.yml）
     —— 由 load_face_trace_config() 装载配置时经 config_resolver.materialize_runtime_configs
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
         不猜测填充）
         + 跨 run 归因标签（model.name/hardware.label/
         kv_management.policy/request_queue_csv；metrics_postprocess
         的 _labels 据此填 raw_metrics.csv 的 model/dataset/hardware/
         kv_policy 列）；
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
覆盖同目录。fail-closed：请求队列输入（空队列由 load_request_queue 的
ValueError 兜底、占位 csv 缺失由 loader 的 sys.exit 物化提示兜底）非零退出；
实例布局（重叠/漏覆盖/非矩形/非等尺寸）自 2026-10-01 起物化即拒（main 开头
预跑 build_instances，原唯一校验点在调度器构造、时点过晚）。

"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

MODULE_DIR = Path(__file__).resolve().parent
SH_TEST_DIR = MODULE_DIR.parents[1]
GENERATED_ROOT = SH_TEST_DIR / "generated"

for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import FaceInstanceSpec, build_instances  # noqa: E402  (READ-ONLY import)
from generate_face_trace import load_face_trace_config  # noqa: E402  (READ-ONLY import)

PREFIX = "llama2_7b_inference"
REPO_VARIANT = "astra-sim-face"


def _config_digest8(config_csv: Path) -> str:
    return hashlib.sha256(config_csv.read_bytes()).hexdigest()[:8]


def _write_text_atomic(path: Path, text: str) -> None:
    """R3-J3 (2026-09-29): 原子发布（临时文件 + os.replace），对齐仓内既有
    原子写范式（config_resolver._write_text_atomically 同款）——并发读者
    （C++ 门禁 / runner / 对账工具）绝不读到半截 manifest。S-5 (2026-10-01):
    临时名唯一化（NamedTemporaryFile 唯一名 + 异常 unlink），取代原先
    确定性的 `<name>.tmp`——两个物化进程并发写同一输出目录时旧名必然互踩
    （B 进程 write_text 的 open("w") 截断发生在 A 进程 replace 之前时，A 可
    把 B 的半截内容发布成"完整" manifest，B 随后对已换名 inode 继续写完、
    其 replace 报 FileNotFoundError）。唯一名后写者间不再共享临时 inode，
    读者侧 os.replace 原子性不变；异常路径 unlink 半截临时文件，优于旧实现
    （旧实现写半途失败会残留 .tmp 垃圾）。输出目录由 main 先 mkdir，本函数
    不负责建目录。"""
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                            prefix=f".{path.name}.", delete=False) as tmp:
        try:
            tmp.write(text)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise
        tmp_name = tmp.name
    try:
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


QUEUE_SUFFIX = "_request_queue_recompute.csv"
SIDECAR_SUFFIX = "_canonical_sidecar.csv"

# SLO pipeline B2 (CPP_SPEC §A) provisional sampling anchor. Fallback only:
# batch B4 has since filled the per-repo slo_params_manifest.json with
# derived values (watermark_sample_period_ns/link_bucket_ns), so this
# constant is used just when the manifest is missing or its entries are
# null/non-positive; the value follows the coordinator ruling 2026-08-26
# (1 ms measured 31,895 bucket records on the S3 2s window -- over budget).
SLO_PROVISIONAL_SAMPLING_NS = 5_000_000


def _slo_sampling_node():
    """slo_sampling 节（CPP_SPEC §A）：WP8 水位线周期 + WP6 链路桶长。

    值来源：sh_test_mesh/slo_tools/slo_params_manifest.json 的
    watermark_sample_period_ns / link_bucket_ns 的 value；value 为 null、
    错型（float/str/bool，_param_ns 一律按"无有效值"处理）或文件缺失时
    用文档化临时锚点（SLO_PROVISIONAL_SAMPLING_NS）并置
    provisional=true——错型与 null 同回退面是有意容忍（采样锚点不 fail
    运行；经 slo_echo / link observer 行回显 provisional=true 可观测），
    与 C++ 侧 MetricCollector 的 never-fail 契约对称。每条相关记录都
    回显实际使用的周期。
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


def _scan_max_session_span(queue_csv):
    """P0-2 (2026-08-31, 总文档 §4 P0-2.1) provenance audit: the queue CSV's
    max consecutive same-session data-row span.

    流式单遍（逐行、不驻留行内容），判定口径：第 1 列 session_id、
    跳过表头（首个非空行）、空行既不算数据行也不断开同 session 连续段；
    行号口径 = 1-based CSV 文件行号（表头=第 1 行，首个数据行=第 2 行）。
    并列取首个最大段（确定性）。

    对齐 wscllm 裁决（sync-A16 批次4，合同 §2.1/§3.1）：本仓 window 为
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
            # R3-J5 (2026-09-29): 空行判定对齐 C++ 口径——strip 后判空
            # (WindowedTraceReader 的行处理在剥除行尾符后判空)。旧写法
            # rstrip("\n").rstrip("\r") 与 C++ 对 CRLF 空行的处理相反
            # (C++ 视其为非空数据行、断开 session 连续段;Python 跳过),
            # advisory 字段两套口径会对同一文件给出不同 span。
            line = raw.strip()
            if not line:
                continue  # 空行(含 CRLF/纯空白):非数据行,也不断开 session 连续段
            if not header_seen:
                header_seen = True  # 首个非空行 = 表头
                continue
            # '#' 注释行（request_id 首字符为 '#'）与 C++ 索引趟/generate_trace
            # loader 同口径跳过——不计数、不断开 session 连续段。
            _fields = line.split(",")
            if len(_fields) > 2 and _fields[2][:1] == "#":
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
            "1-based csv file lines (header=line 1; computed by this "
            "materializer's max-session-span scan)",
        "scanned_data_rows": data_rows,
    }


def _load_sidecar_rows(queue_csv: Path) -> dict:
    """Sidecar rows by request_id (WP2 passthrough channel; absence tolerated)."""
    sidecar = _sidecar_path_for_queue(queue_csv)
    if sidecar is None:
        # r5 勘误护栏（F5-P1）：join 落空最常见的静默死因是队列文件名
        # 不带 _request_queue_recompute.csv 后缀（官方命名约定是
        # load-bearing 的：run_golden_live 与指标建议文档都按它取数）。
        # 同目录若存在任何 canonical sidecar，几乎必然是命名错位——
        # WARN 提示（不改退出码：sidecar 缺失本身是合法容忍态）。
        if any(queue_csv.parent.glob("*" + SIDECAR_SUFFIX)):
            print(
                "[plan_materializer] WARN: queue %s does not follow the "
                "'*_request_queue_recompute.csv' naming convention, so its "
                "canonical sidecar (WP2 request_type/prefix_len/timing "
                "passthrough) was NOT joined; a *_canonical_sidecar.csv "
                "exists next to it -- rename the queue/sidecar pair to the "
                "convention to enable the join." % queue_csv.name,
                file=sys.stderr,
            )
        return {}
    with sidecar.open(newline="", encoding="utf-8") as source:
        return {row["request_id"]: row for row in csv.DictReader(source)}


def _derive_manifest_requests(config, sidecar_rows):
    """manifest.json 的 requests[]（队列派生 9 字段）。

    fail-closed：turn>0 行之前必须已按文件序处理过同 session 的
    turn-1 行（正式队列口径：session 块连续、块内 turn 严格 +1），
    否则「turn>0 history=上一请求 final」无法推导——显式 RuntimeError，
    与 _derive_metrics_requests 的 parent 缺失处理一致；绝不静默回退
    history=0（那会把整条 prefill/final 链推到错误基线上）。
    """
    requests = []
    final_by_session_turn = {}
    for index, spec in enumerate(config.request_queue):
        if spec.turn_index == 0:
            history = 0
            # recompute 单口径:turn-0 队列 prefill 已折入 prefix
            context = int(spec.prefill_length)
        else:
            parent_final = final_by_session_turn.get(
                (spec.session_id, spec.turn_index - 1))
            if parent_final is None:
                raise RuntimeError(
                    f"request {spec.request_id} (turn-{spec.turn_index}) "
                    "has no preceding turn-"
                    f"{spec.turn_index - 1} row of session "
                    f"{spec.session_id!r} before it in queue file order; "
                    "history_tokens_before cannot be derived (fail-closed; "
                    "the queue must keep each session in one contiguous "
                    "block with strictly increasing turn indices)")
            history = parent_final
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
        final_by_session_turn[(spec.session_id, spec.turn_index)] = final
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


def _resolved_backend_keys(config) -> dict:
    """manifest 的 die 后端键(契约 §7 裁定 2;阶段 C)。

    从 load_face_trace_config 副产的 resolved system.json(config.
    system_config)读 compute-cost-model 与 die-model-config:前者声明
    backend(die/roofline/replay),后者是 resolved_die_model.json 绝对
   路径(resolver 注入的 managed 字段)。legacy 配置无这些键 → 返回空
    dict(manifest 形状不变,向后兼容);die-model-config 必须是已存在
    的绝对路径文件(fail-closed,不静默丢键)。"""
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
    # (2026-10-01) 实例布局硬校验前移到物化期:原唯一校验点是
    # face_online_scheduler 构造(build_instances 的矩形性/不重叠/全覆盖/
    # 等尺寸硬校验),此时 C++ 已启动、桥已建;重叠或漏覆盖的布局现在
    # 物化即拒(与调度器构造同参:require_equal_size=True),plan 产物
    # 不落盘。保留对象只读复用,策略语义零改动。
    build_instances(
        config.hardware,
        tuple(
            FaceInstanceSpec(
                name=group.name,
                pg_name=group.pg_name,
                ranks=group.ranks,
            )
            for group in config.inference_groups
        ),
        require_equal_size=True,
    )
    # (2026-09-29)原"空队列占位"分支已删:load_request_queue 对 0 数据行
    # 直接 ValueError("must contain at least one request"),config.request_
    # queue 到此处不可能为空,分支不可达。
    cfg8 = _config_digest8(config.config_csv)
    output_dir = GENERATED_ROOT / f"{PREFIX}_54npus_plan_{cfg8}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sidecar_rows = _load_sidecar_rows(config.request_queue_csv)
    # P0-2 (2026-08-31, 总文档 §4 P0-2.1): max consecutive same-session span
    # ——同 session 连续段 provenance 审计字段。对齐 wscllm 裁决（批次4）：
    # 本仓无 C++ 拒绝门（window advisory），字段仅 provenance。
    span_scan = _scan_max_session_span(config.request_queue_csv)
    manifest = {
        "requests": _derive_manifest_requests(config, sidecar_rows),
        "selected_request_count": len(config.request_queue),
        "selected_session_count": len({
            spec.session_id for spec in config.request_queue}),
        "manifest_source": "synthetic-prerun",
        "repo_variant": REPO_VARIANT,
        # 跨 run 归因标签：metrics_postprocess._labels 从本 manifest.json
        # 读 model.name / hardware.label / kv_management.policy /
        # request_queue_csv，填 raw_metrics.csv 的 model/dataset/hardware/
        # kv_policy 列（此前不写这些键，四列恒空串）。
        "model": {"name": config.model_name},
        "hardware": {
            "label": config.hardware.label,
            "capacity_profile": config.hardware_capacity_profile,
        },
        "kv_management": {"policy": config.kv_cache_policy},
        "request_queue_csv": str(config.request_queue_csv),
        "kv_reserve_context_tokens": config.kv_reserve_context_tokens,
        # P0-2 provenance: 同 session 连续段审计字段（本仓无 C++ 拒绝门，
        # window advisory——字段仅供 campaign 复核，见函数 docstring 裁决）。
        "max_same_session_span": span_scan["max_same_session_span"],
        "span_session_id": span_scan["span_session_id"],
        "span_row_range": span_scan["span_row_range"],
        "span_row_range_convention": span_scan["span_row_range_convention"],
    }
    # 阶段 C(契约 §7 裁定 2):die 后端键(compute-cost-model +
    # die-model-config 绝对路径);legacy 配置无键,manifest 形状不变。
    manifest.update(_resolved_backend_keys(config))
    _write_text_atomic(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2) + "\n")

    metrics_manifest = {
        "schema_version": 1,
        "repo_variant": REPO_VARIANT,
        "manifest_source": "synthetic-prerun",
        "requests": _derive_metrics_requests(config),
        # SLO pipeline B2（CPP_SPEC §A）：C++ 观测器采样锚点（WP8 水位线
        # 周期 / WP6 链路桶长）；null 值 -> 临时锚点 + provisional=true。
        "slo_sampling": _slo_sampling_node(),
    }
    # 阶段 E2：die 后端键同步进 metrics_manifest（与 manifest.json 同源，
    # _resolved_backend_keys）。归档 run 目录只保证携带本文件——把
    # compute-cost-model 放进来让 metrics_postprocess 的 die 利用率管线
    # 断言（active_kernel_die_* 非 null 且 ∈[0,1+ε]）离线可判；legacy 配置
    # 无键，manifest 形状不变。C++ load_manifest 与 slo_common 均只取
    # 自己认识的键，未知顶层键被忽略。
    metrics_manifest.update(_resolved_backend_keys(config))
    _write_text_atomic(
        output_dir / "metrics_manifest.json",
        json.dumps(metrics_manifest, separators=(",", ":")) + "\n")

    runtime_note = (
        "runtime_config three files (system/comm_group/network.yml) "
        "re-materialized by the config loader "
        "under generated/runtime_config/ (unchanged side effect)")
    request_type_counts = {}
    for entry in manifest["requests"]:
        key = entry.get("request_type", "<absent>")
        request_type_counts[key] = request_type_counts.get(key, 0) + 1
    print(json.dumps({
        "plan_dir": str(output_dir),
        # (2026-09-30, R2 audit) read the persisted manifest keys (not a
        # recomputed len()) so the stdout provenance record and manifest.json
        # can never disagree; selected_request_count was previously written
        # but never read anywhere.
        "requests": manifest["selected_request_count"],
        "sessions": manifest["selected_session_count"],
        # P0-2 (2026-08-31): span 审计字段随 stdout 权威 provenance 记录
        # 一并输出（与 manifest.json 持久字段同源同值）。
        "max_same_session_span": span_scan["max_same_session_span"],
        "span_session_id": span_scan["span_session_id"],
        "span_row_range": span_scan["span_row_range"],
        "sidecar": str(_sidecar_path_for_queue(config.request_queue_csv)),
        "sidecar_rows_joined": len(sidecar_rows),
        "request_type_counts": request_type_counts,
        "note": runtime_note,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
