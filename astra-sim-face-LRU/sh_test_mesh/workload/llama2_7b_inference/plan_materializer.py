#!/usr/bin/env python3
"""plan_materializer.py -- 路径③④ 的 plan 目录物化入口（唯一输入物化器）。

在线③④需要三类输入，本脚本以**不跑规划器、不写任何 .et** 的方式产出：

  1. runtime_config 四小件（system/comm_group/remote_memory/network.yml）
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
         kv_policy 列——2026-09-30 自 face 仓同步，模型值固定
         "llama2_7b"：本仓 config 已无 model_name 键（X10 删除），该
         workload 身份由模块目录/PREFIX 承载）；
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
覆盖同目录。fail-closed：请求队列为空由 loader 承担
（generate_trace.load_request_queue 对零请求队列 raise ValueError）。

"""

import csv
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
# 本仓溯源标识。该值是 slo 工具的注册表键（经 metrics_manifest → cpp.log
# init 行 → slo_common.detect_repo_variant 查表）；astra-sim-face-LRU 已在
# kv_cache_adapter/hbm_watermark/hopbytes/load_imbalance 四张表登记，映射
# 复制 astra-sim-face 条目（本仓产物即其条目注释所述 -LRU 新产物形态）。
REPO_VARIANT = "astra-sim-face-LRU"
# kv_cache_policy 唯一合法值（2026-09-25 session 级 Tiered-LRU 命名卫生，
# 与 generate_face_trace._parse_config_value 值域同源；旧值别名与本值同
# 调度器（无档位分支），已随命名卫生直接清除）。
# loader 已做值域校验，此处为防御性断言（防账本漂移同族语义）；
# stdout 权威 provenance 记录增携 kv_cache_policy 字段
# （物化 stdout vs 决策日志 run 头的核对锚点）。
# 机制零改动：不写 manifest（既有字段冻结），仅 stdout 增字段。
_KNOWN_KV_POLICIES = frozenset({"session_lru_tiered"})


def _assert_policy_passthrough(config) -> dict:
    """policy 透传断言（轻校验，fail-closed；机制零改动）。"""
    policy = config.kv_cache_policy
    if policy not in _KNOWN_KV_POLICIES:
        raise RuntimeError(
            f"kv_cache_policy {policy!r} not in the known set "
            f"{sorted(_KNOWN_KV_POLICIES)} (loader value domain drifted?)")
    return {
        "kv_cache_policy": policy,
    }


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
    """Canonical sidecar sibling of a materialized request queue (or None).

    配对规则:队列名以 ``_request_queue_recompute.csv`` 结尾 → 同前缀
    ``_canonical_sidecar.csv``。队列名不合该规则、但按同族命名
    (``*request_queue<X>.csv`` ↔ ``_canonical_sidecar<X>.csv``)能在同
    目录找到疑似配对的 sidecar 物化件时 fail-closed raise——命名不合
    约定会让 sidecar join 静默缺失,WP2 字段整链丢失、request_type 全
    unknown,宁可拒绝物化也不静默降级。同目录确无疑似配对文件时返回
    None(合法无 sidecar 场景;_load_sidecar_rows 容忍缺失)。
    """
    name = queue_csv.name
    if name.endswith(QUEUE_SUFFIX):
        sidecar = queue_csv.parent / (name[: -len(QUEUE_SUFFIX)] + SIDECAR_SUFFIX)
        return sidecar if sidecar.is_file() else None
    stem = name[: -len(".csv")] if name.endswith(".csv") else name
    marker = "request_queue"
    position = stem.find(marker)
    if position >= 0:
        tail = stem[position + len(marker):]
        suspected = queue_csv.parent / ("_canonical_sidecar" + tail + ".csv")
        if suspected.is_file():
            raise RuntimeError(
                f"request queue {queue_csv} does not follow the canonical "
                f"naming ('{QUEUE_SUFFIX}' suffix) but a suspected sidecar "
                f"pair exists: {suspected}; rename the pair to the "
                f"canonical '<prefix>{QUEUE_SUFFIX}' / "
                f"'<prefix>{SIDECAR_SUFFIX}' names -- silently skipping the "
                f"sidecar would drop the WP2 fields and mark every "
                f"request_type unknown")
    return None


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
            "1-based csv file lines (header=line 1; Python-side "
            "provenance scan of _scan_max_session_span, no C++ "
            "counterpart)",
        "scanned_data_rows": data_rows,
    }


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
    # P4(face-LRU 审计,2026-09-29):history 必须取"该 (session, turn-1)
    # 的 final",而非"文件序上一行"的 final。旧实现按会话只记最后一个
    # final,守卫又只查 parent 存在不查位置——同会话 turn 乱序的输入
    # (turn2 行先于 turn1 行)会静默把 turn0 的 final 记成 turn2 的
    # history,manifest 的 history/context 全链错账。改为按 (session,
    # turn) 精确键取值:合法文件序(同会话 turn 升序)结果逐字节不变,
    # 乱序输入在缺失 parent final 处 fail-closed。
    final_by_session_turn = {}
    # 与 _derive_metrics_requests 同一门:load_request_queue 不校验 turn
    # 连续性,turn>0 行缺同 session 上一 turn 行(会话首行即 turn>0、或
    # 中间 turn 缺行)时在此 fail-closed,绝不静默按 history=0 记账——
    # manifest / metrics 两函数对同形状输入口径一致。
    queue_by_session_turn = {
        (spec.session_id, spec.turn_index): index
        for index, spec in enumerate(config.request_queue)
    }
    for index, spec in enumerate(config.request_queue):
        if spec.turn_index == 0:
            history = 0
            # recompute 单口径:turn-0 队列 prefill 已折入 prefix
            context = int(spec.prefill_length)
        else:
            parent_key = (spec.session_id, spec.turn_index - 1)
            parent = queue_by_session_turn.get(parent_key)
            history = final_by_session_turn.get(parent_key)
            if parent is None or history is None:
                raise RuntimeError(
                    f"request {spec.request_id} lost its previous-turn "
                    "parent")
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
        final_by_session_turn[(spec.session_id, int(spec.turn_index))] = final
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
    """manifest 的 die 后端键(契约 §7 裁定 2;阶段 C;face-LRU 随迁
    2026-10-03)。

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
    policy_provenance = _assert_policy_passthrough(config)
    # (2026-10-01) The "54npus" in the plan directory name is pinned to the
    # Case-5 9x6 mesh (the five runners' GEN_MATCH glob and run_golden_live
    # probe that literal; the CMake fixture staging keys on the
    # runtime_config slug__profile__pool dir name instead, which carries no
    # NPU count). Fail closed on any other mesh instead of materializing a
    # misnamed plan directory the runners cannot discover.
    npus_total = getattr(
        config.hardware, "npus_count",
        getattr(config.hardware, "mesh_rows", 0)
        * getattr(config.hardware, "mesh_columns",
                  getattr(config.hardware, "mesh_cols", 0)))
    if npus_total != 54:
        print(
            f"hardware mesh declares {npus_total} NPUs but the plan "
            "directory name (and the runner GEN_MATCH glob) is pinned to "
            "54npus; regenerate the naming convention before materializing",
            file=sys.stderr)
        return 1
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
        # sync-py-E3 (2026-09-30, ported from the face repo): 跨 run 归因
        # 标签——metrics_postprocess._labels 从本 manifest.json 读
        # model.name / hardware.label / kv_management.policy /
        # request_queue_csv，填 raw_metrics.csv 的 model/dataset/hardware/
        # kv_policy 列（此前不写这些键，四列恒空串）。适配差异：本仓
        # config 已无 model_name 键（X10 删除），model 值固定为该
        # workload 的模块 PREFIX（与 workload 目录同名）；
        # kv_management.policy 取 config.kv_cache_policy（loader 值域
        # 已收敛为 session_lru_tiered，与 KV_POLICY_LABELS 枚举一致，
        # 产物可装载）。
        "model": {"name": PREFIX},
        "hardware": {"label": config.hardware.label},
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
    # L14：两 manifest 先写 tmp 再 os.replace 原子落盘（见
    # _atomic_write_text docstring；两写间仍非事务，半写撕裂已消除）。
    _atomic_write_text(
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
    # 阶段 E2(face-LRU 随迁):die 后端键同步进 metrics_manifest(与
    # manifest.json 同源,_resolved_backend_keys)。归档 run 目录只保证
    # 携带本文件——把 compute-cost-model 放进来让 metrics_postprocess 的
    # die 利用率管线断言(active_kernel_die_* 非 null 且 ∈[0,1+ε])离线
    # 可判;legacy 配置无键,manifest 形状不变。C++ load_manifest 与
    # slo_common 均只取自己认识的键,未知顶层键被忽略。
    metrics_manifest.update(_resolved_backend_keys(config))
    _atomic_write_text(
        output_dir / "metrics_manifest.json",
        json.dumps(metrics_manifest, separators=(",", ":")) + "\n")

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
        # stdout 权威 provenance 增携 kv_cache_policy（与 WL 仓同构的核对
        # 锚点；物化时点值域防御断言见 _assert_policy_passthrough）。
        "kv_cache_policy": policy_provenance["kv_cache_policy"],
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
