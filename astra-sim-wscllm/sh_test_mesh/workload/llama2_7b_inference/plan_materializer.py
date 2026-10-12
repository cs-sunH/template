#!/usr/bin/env python3
"""plan_materializer.py -- 路径③④ 的 plan 目录物化入口（唯一输入物化器）。

在线③④需要三类输入，本脚本以**不跑规划器、不写任何 .et** 的方式产出：

  1. runtime_config 三件（system/comm_group/network.yml）
     —— 由 load_wsc_llm_trace_config() 装载配置时经 config_resolver.materialize_runtime_configs
     副产（路径不变，零改动）；
  2. manifest.json —— 含在线侧实际消费的**队列派生 9 字段**
     （request_id/session_id/turn_index/queue_index/prefill_length/
     decode_length/history_tokens_before/prefill_context_tokens/
     final_context_tokens；推导规则与《sh_3.0的120档问题分析报告.md》§4
     受控实验（20 档真实 manifest 2091/2091 行 0 mismatch）同式：
     turn-0 history=0；turn>0 history=上一请求 final；
     prefill_context=折入后的 prefill_length（turn-0）或 history+
     prefill_length（turn>0，recompute 口径）；final=context+decode）
     + WP2 透传字段（request_type/prefix_len，源自队列旁 canonical
     sidecar 的 request_type/raw_prefix_tokens 列；sidecar 缺失/行缺失时
     对应字段不写，消费方回退 unknown/NA 并注明，不猜测填充）；
  3. metrics_manifest.json —— 主 agent 裁决 (i) 的合成口径：
       - schema_version=1 + requests[]（arrival：turn0=absolute session
         arrival；turn>0=after_request(同 session 上一 queue_index, interval)
         ——C++ MetricCollector.cc 只要求
         schema_version==1 + requests[]）；
       - WP2 透传字段（human_time_ns/tool_time_ns（非空者），源自队列旁
         canonical sidecar 同名列）——P2-13（2026-09-28）修正写位：消费链
         （load_request_manifest 候选链、run_online_strategy.sh 拷贝进
         run_dir 的）是 metrics_manifest.json，slo_stats session 从它的
         requests[] 取这两字段算 T_session；manifest.json 副本零消费方，
         属写错文件（test_driver_parity 契约同证），已删除；
       - **显式合成标记**：repo_variant 原样 + manifest_source=
         "synthetic-prerun"（裁决 i 条件 a）；
       - **占位决策字段**：prefill/decode instance 恒 0、ranks 恒
         instance-0 ranks——静态 rank 归因维度**不可信**；请求级指标
         （e2e/完成/sim_end/tput）可信（裁决 i 条件 b，口径登记于实录）；
       - 对账工具（ledger_reconcile 族）不读本 manifest——只消费归档
         jsonl（裁决 i 条件 c，已核实维持）。

输出目录：sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_plan_<cfg8>/（保留 54npus
前缀以过 GEN_MATCH；<cfg8> = trace_config 内容摘要 8 位 hex）。幂等：重跑
覆盖同目录。D3（2026-09-28）目录发布语义：全部产物先写入点前缀暂存目录
（`.<目录名>.staging`，GEN_MATCH 的 `llama2_7b_wsc_llm_inference_54npus_*`
glob 不可见），两份 manifest 均完整后才原子换名上台——中断残留的只有
glob 不可见的暂存目录，不再产生「恰一闸误命中」的半成品/空目录；换名
采用 (旧→.previous, 新→台前) 两步，中途崩溃的最坏状态是 GEN_MATCH 找到
**零**个目录（runner 以明确报文 exit 1），而不是新旧两代 manifest 混居
一台。发布成功后清扫其它 <cfg8> 陈旧 plan 目录与残留暂存目录——
trace_config 任何字节变化都生成新 <cfg8> 目录，旧目录对 GEN_MATCH 是
永久假命中，由物化器自身维持「恰一」不变量。
fail-closed：请求队列为空/占位（request-neutral 占位 csv）在装载层
（generate_trace.load_request_queue）即 raise，本入口收到的
config.request_queue 不可能为空（装载先于本入口、空队列不构造 config）。
D3（2026-09-28）provenance 增强字段：metrics_manifest.json 新写
npus_count（C++ MetricCollector 一致性门
`manifest npus_count != simulated systems count` 此前因键缺席恒不触发）
与 request_queue_csv_sha256/bytes（两份 manifest 同写）——同路径重派生
后，陈旧 plan 配新 CSV 的三处口径不一致可被对账发现，不再静默；stdout
`sidecar_rows_joined` 改为真实 join 命中数（此前写 sidecar 总行数，虚报
透传覆盖率）。

"""

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SH_TEST_DIR = MODULE_DIR.parents[1]
GENERATED_ROOT = SH_TEST_DIR / "generated"

for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_wsc_llm_trace import load_wsc_llm_trace_config  # noqa: E402  (READ-ONLY import)

PREFIX = "llama2_7b_wsc_llm_inference"
REPO_VARIANT = "astra-sim-wscllm"

# A.5（2026-09-05，legacy 与 relevant 两个历史变体清除）policy 取值集：
# 与 generate_wsc_llm_trace._parse_config_value 的 kv_cache_policy 值域
# 同源。loader 已做值域校验，此处为防御性断言（防账本漂移同族语义）；
# stdout 权威 provenance 记录增携 kv_cache_policy 字段
# （物化 stdout vs 决策日志 run 头的核对锚点）。
# 机制零改动：不写 manifest（既有字段冻结），仅 stdout 增字段。
_KNOWN_KV_POLICIES = frozenset({"session_lru_recompute"})


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


def _queue_digest(queue_csv: Path) -> dict:
    """D3 (2026-09-28): byte digest of the request queue this plan was built
    from; both manifests carry it verbatim.

    A same-path re-derivation swaps the CSV under an unchanged path string,
    so `manifest.request_queue_csv` alone cannot tell a stale plan from a
    fresh one -- without the digest, an old plan paired with the new CSV is
    silently inconsistent across the three ports (manifest.json requests /
    metrics_manifest.json requests / the queue rows themselves). With it,
    the mismatch is one sha256 comparison away (audit-level detection; the
    C++ MetricCollector ignores unknown top-level keys, so this is
    provenance-only by design)."""
    data = queue_csv.read_bytes()
    return {
        "request_queue_csv_sha256": hashlib.sha256(data).hexdigest(),
        "request_queue_csv_bytes": len(data),
    }


def _publish_plan_dir(output_dir: Path, manifest: dict,
                      metrics_manifest: dict) -> None:
    """D3 (2026-09-28): stage both manifests, then publish the plan dir.

    The former code mkdir'd the final dir up front and wrote the two
    manifests sequentially straight into it: an interrupt left (a) a
    mixed-generation dir (new manifest.json next to old metrics_manifest.json)
    or (b) an empty dir from an even earlier crash -- both count as a
    legitimate GEN_MATCH hit (`llama2_7b_wsc_llm_inference_54npus_*` glob),
    and any byte drift in trace_config created a NEW cfg8 dir while the old
    one stayed behind, so the exactly-one gate exit-1'd until manual cleanup.

    Now: everything is staged in a dot-prefixed sibling (invisible to the
    GEN_MATCH glob), and published by two renames. Worst crash state is
    GEN_MATCH finding ZERO dirs (the runner fails loudly with its
    "run plan_materializer" message) -- never a half-written or stale
    generation sitting in the hot seat. After a successful publish the
    materializer sweeps other-cfg8 plan dirs and any crash leftovers, so the
    exactly-one invariant is maintained by the tool that creates the dirs.
    (Single-materializer ops model, same as the runners themselves.)"""
    stage_dir = GENERATED_ROOT / f".{output_dir.name}.staging"
    prev_dir = GENERATED_ROOT / f".{output_dir.name}.previous"
    for leftover in (stage_dir, prev_dir):
        if leftover.exists():
            shutil.rmtree(leftover)
    stage_dir.mkdir(parents=True)
    try:
        (stage_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (stage_dir / "metrics_manifest.json").write_text(
            json.dumps(metrics_manifest, separators=(",", ":")) + "\n",
            encoding="utf-8")
        renamed_prev = False
        if output_dir.exists():
            os.rename(output_dir, prev_dir)
            renamed_prev = True
        try:
            os.rename(stage_dir, output_dir)
        except OSError:
            if renamed_prev:
                # Restore the previous complete plan rather than leaving
                # nothing behind.
                os.rename(prev_dir, output_dir)
            raise
    except BaseException:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    if prev_dir.exists():
        shutil.rmtree(prev_dir)
    # Stale sibling plan dirs (older <cfg8> generations) are permanent
    # GEN_MATCH false hits once a new generation is published.
    for stale in GENERATED_ROOT.glob(f"{PREFIX}_54npus_plan_*"):
        if stale.is_dir() and stale.resolve() != output_dir.resolve():
            shutil.rmtree(stale)
    # Crash leftovers of any interrupted publication (dot-prefixed, invisible
    # to GEN_MATCH, never legitimate).
    for pattern in (f".{PREFIX}_54npus_plan_*.staging",
                    f".{PREFIX}_54npus_plan_*.previous"):
        for leftover in GENERATED_ROOT.glob(pattern):
            shutil.rmtree(leftover, ignore_errors=True)


QUEUE_SUFFIX = "_request_queue_recompute.csv"
SIDECAR_SUFFIX = "_canonical_sidecar.csv"

# SLO pipeline B2 (CPP_SPEC §A) provisional sampling anchors. Used when the
# per-repo slo_params_manifest.json carries no derived value (null/missing;
# B4b has since filled derived values -- watermark 5 ms / link bucket 20 ms
# at the time of writing, manifest is authoritative).  Each config keeps its
# own documented anchor so the two fallbacks can drift independently; the
# values follow the coordinator ruling 2026-08-26 (1 ms measured 31,895
# bucket records on the S3 2s window -- over budget).
SLO_PROVISIONAL_WATERMARK_PERIOD_NS = 5_000_000
SLO_PROVISIONAL_LINK_BUCKET_NS = 5_000_000


def _slo_sampling_node():
    """slo_sampling 节（CPP_SPEC §A）：WP8 水位线周期 + WP6 链路桶长。

    值来源：sh_test_mesh/slo_tools/slo_params_manifest.json 的
    watermark_sample_period_ns / link_bucket_ns 的 value；value 为 null 或
    文件缺失时用各自文档化临时锚点（SLO_PROVISIONAL_WATERMARK_PERIOD_NS /
    SLO_PROVISIONAL_LINK_BUCKET_NS）并置 provisional=true。C++ 侧缺节同样
    回退临时锚点（不 fail 运行），每条相关记录都回显实际使用的周期。
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
            watermark if watermark is not None
            else SLO_PROVISIONAL_WATERMARK_PERIOD_NS
        ),
        "link_bucket_ns": (
            link_bucket if link_bucket is not None
            else SLO_PROVISIONAL_LINK_BUCKET_NS
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


def _reconcile_queue_provenance(queue_csv: Path) -> None:
    """F2_PY (2026-09-29, RV4-F3): reconcile the queue against its
    provenance record before any sidecar join.

    A1 made each derive artifact atomic, but publication is still three
    sequential renames (queue -> sidecar -> provenance): a crash between
    them leaves a MIXED-GENERATION set on disk (e.g. new queue + old
    sidecar + old provenance).  The C++ path is closed by the provenance
    gate, but this materializer never read the provenance and the sidecar
    carries no digest binding, so the stale sidecar rows silently joined
    the new queue (passthrough fields silently omitted, both manifests
    shaped "legitimately", SLO T_session degraded to NA, zero errors).
    Comparing the recorded ``csv_sha256`` against the actual queue bytes
    turns that window into an explicit failure.  A queue without a
    provenance record (e.g. unit-test synthetic queues) has nothing to
    reconcile and is tolerated -- the C++ online reader still enforces
    provenance presence for real runs.
    """
    provenance_path = Path(str(queue_csv) + ".provenance.json")
    if not provenance_path.is_file():
        return
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"queue provenance {provenance_path} is unreadable "
            f"(fail-closed): {error}"
        ) from None
    if not isinstance(provenance, dict):
        raise RuntimeError(
            f"queue provenance {provenance_path} is not a JSON object "
            "(fail-closed)")
    recorded = provenance.get("csv_sha256")
    actual = _queue_digest(queue_csv)["request_queue_csv_sha256"]
    if not isinstance(recorded, str) or recorded != actual:
        raise RuntimeError(
            f"queue provenance {provenance_path} does not match the queue "
            f"bytes (fail-closed): recorded csv_sha256={recorded!r} != "
            f"actual {actual} -- mixed-generation artifacts (queue and "
            "sidecar/provenance from different derivations); re-run "
            "traces/derive_20_first_30_seconds.py before materializing")


def _load_sidecar_rows(queue_csv: Path) -> dict:
    """Sidecar rows by request_id (WP2 passthrough channel; absence tolerated)."""
    sidecar = _sidecar_path_for_queue(queue_csv)
    if sidecar is None:
        return {}
    with sidecar.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "request_id" not in reader.fieldnames:
            raise RuntimeError(
                f"canonical sidecar {sidecar} lacks a request_id column; "
                "regenerate the sidecar next to the request queue")
        rows: dict = {}
        for row in reader:
            request_id = row["request_id"]
            if request_id in rows:
                raise RuntimeError(
                    f"canonical sidecar {sidecar} has duplicate request_id "
                    f"{request_id!r}; WP2 passthrough cannot disambiguate it")
            rows[request_id] = row
        return rows


def _derive_manifest_requests(config, sidecar_rows):
    """manifest.json 的 requests[]（队列派生 9 字段）。"""
    spec_by_session_turn = {
        (spec.session_id, int(spec.turn_index)): spec
        for spec in config.request_queue
    }
    finals_by_session_turn: dict[tuple[str, int], int] = {}

    def _final_for(session_id: str, turn_index: int) -> int:
        """该轮 final_context_tokens（history 链式依赖上一轮，与 CSV 行序
        解耦，按 (session_id, turn_index) 精确父轮查找）。"""
        key = (session_id, turn_index)
        if key in finals_by_session_turn:
            return finals_by_session_turn[key]
        spec = spec_by_session_turn.get(key)
        if spec is None:
            raise RuntimeError(
                f"session {session_id!r} lost its turn-{turn_index} "
                "request; cannot derive the history-chain final")
        history = 0 if turn_index == 0 else _final_for(
            session_id, turn_index - 1)
        final = (history + int(spec.prefill_length)
                 + int(spec.decode_length))
        finals_by_session_turn[key] = final
        return final

    requests = []
    seen_request_ids: set = set()
    # P2-21（2026-09-28）：queue_index 与 C++ 同源——按源 CSV 数据行序
    # （config.request_queue_indices，WindowedTraceReader 以未过滤全量行
    # 序编号），不按 session_limit 过滤后序枚举（过滤态下两端错位）。
    for index, spec in zip(config.request_queue_indices,
                           config.request_queue):
        turn_index = int(spec.turn_index)
        if turn_index == 0:
            history = 0
            # recompute 单口径:turn-0 队列 prefill 已折入 prefix
            context = int(spec.prefill_length)
        else:
            if (spec.session_id, turn_index - 1) not in spec_by_session_turn:
                # 与 metrics 派生同策略的 fail-closed：历史长度宁可抛错，
                # 不静默取 0（G5-F12/M7-B2 修复）。
                raise RuntimeError(
                    f"request {spec.request_id} lost its previous-turn "
                    "parent; manifest history derivation is fail-closed")
            history = _final_for(spec.session_id, turn_index - 1)
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
        # guessed. P2-13 (2026-09-28): human_time_ns/tool_time_ns moved out
        # of this manifest -- their only consumers (load_request_manifest
        # chain / slo_stats session T_session) read metrics_manifest.json,
        # so the copy here was write-only (see _derive_metrics_requests).
        sidecar_row = sidecar_rows.get(spec.request_id)
        if sidecar_row is not None:
            request_type = (sidecar_row.get("request_type") or "").strip()
            if request_type:
                entry["request_type"] = request_type
            prefix_text = (sidecar_row.get("raw_prefix_tokens") or "").strip()
            if prefix_text:
                entry["prefix_len"] = int(prefix_text)
        # 4.3 fix (2026-09-29): manifest 级 request_id 唯一性 fail-closed
        # (与 token manifest 同强度)。上游队列装载器已拒绝重复 id,但
        # manifest 是冻结产物、被下游逐 id 消费(last-wins/双计均静默),
        # 在清单边界独立复核。wsc_llm_scheduler 的 anchor/rank 账本同样
        # 按 request_id 键控。
        if entry["request_id"] in seen_request_ids:
            raise RuntimeError(
                f"manifest requests[] has duplicate request_id "
                f"{entry['request_id']!r} (fail-closed); downstream id-keyed "
                "consumers cannot disambiguate it"
            )
        seen_request_ids.add(entry["request_id"])
        requests.append(entry)
    return requests


def _derive_metrics_requests(config, sidecar_rows):
    """metrics_manifest.json 的 requests[]（合成口径，占位 rank）。

    P2-13（2026-09-28）：WP2 透传的 human_time_ns/tool_time_ns 从
    manifest.json 移入本 manifest——消费链（slo_common.load_request_manifest
    候选链与 run_online_strategy.sh 的 run_dir 拷贝）只读
    metrics_manifest.json，slo_stats session 从 requests[] 取这两字段算
    T_session；manifest.json 副本无消费方（写错文件）。口径与原实现一致：
    仅 sidecar 非空文本转 int 写入；行/列缺席不写（不猜测填充）。
    """
    group_by_index = dict(enumerate(config.inference_groups))
    placeholder_ranks = list(group_by_index[0].ranks)
    # P2-21（2026-09-28）：queue_index/parent_queue_index 与 C++ 同源——
    # 源 CSV 数据行序（config.request_queue_indices），不按过滤后序枚举
    # （after_request 的 parent_queue_index 由 C++ MetricCollector 按同
    # 一序号解析）。
    queue_by_session_turn = {
        (spec.session_id, spec.turn_index): index
        for index, spec in zip(config.request_queue_indices,
                               config.request_queue)
    }
    records = []
    for index, spec in zip(config.request_queue_indices,
                           config.request_queue):
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
        record = {
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
        }
        # P2-13（2026-09-28）：WP2 透传（append-only；字段口径与原
        # manifest.json 写位一致——非空 sidecar 文本转 int，缺席不写）。
        sidecar_row = sidecar_rows.get(spec.request_id)
        if sidecar_row is not None:
            human_text = (sidecar_row.get("human_time_ns") or "").strip()
            tool_text = (sidecar_row.get("tool_time_ns") or "").strip()
            if human_text:
                record["human_time_ns"] = int(human_text)
            if tool_text:
                record["tool_time_ns"] = int(tool_text)
        records.append(record)
    return records


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


def _resolved_backend_keys(config) -> dict:
    """manifest 的 die 后端键(契约 §7 裁定 2;wscllm die 推广)。

    从 load_wsc_llm_trace_config 副产的 resolved system.json(config.
    system_config)读 compute-cost-model 与 die-model-config:前者声明
    backend(die/roofline/replay),后者是 resolved_die_model.json 绝对
    路径(resolver 注入的 managed 字段)。legacy 配置无这些键 → 返回空
    dict(manifest 形状不变,向后兼容);die-model-config 必须是已存在
    的绝对路径文件(fail-closed,不静默丢键)。"""
    keys = {}
    # getattr 容缺省:duck-type 测试夹具(_Config)不带 system_config ——
    # 视为 legacy(无键),manifest 形状不变;真实 loader 产物恒有该路径。
    system_config = getattr(config, "system_config", None)
    if system_config is None:
        return keys
    with open(system_config, "r", encoding="utf-8") as source:
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
    config = load_wsc_llm_trace_config()
    # D3 (2026-09-28): the former `if not config.request_queue: return 1`
    # guard was unreachable dead code -- load_wsc_llm_trace_config ->
    # load_request_queue raises on an empty/placeholder queue before any
    # config object exists, and select_first_session_requests only ever
    # returns a non-empty subset of a non-empty queue. The loader's own
    # fail-closed error is the empty-queue surface now.
    # A.5（2026-09-05）：policy 透传断言 + provenance（见函数 docstring）。
    policy_provenance = _assert_policy_passthrough(config)
    cfg8 = _config_digest8(config.config_csv)
    output_dir = GENERATED_ROOT / f"{PREFIX}_54npus_plan_{cfg8}"

    # F2_PY (2026-09-29, RV4-F3): fail-closed before any cross-generation
    # data (the sidecar) is consumed -- see _reconcile_queue_provenance.
    _reconcile_queue_provenance(config.request_queue_csv)
    sidecar_rows = _load_sidecar_rows(config.request_queue_csv)
    # P0-2 (2026-08-31, 总文档 §4 P0-2.1): max consecutive same-session span
    # ——与 sh/face 系启动预检同口径的 provenance 审计字段。wscllm 裁决
    # （批次4）：本仓无 C++ 拒绝门（window advisory），字段仅 provenance。
    span_scan = _scan_max_session_span(config.request_queue_csv)
    queue_digest = _queue_digest(config.request_queue_csv)
    # D3 (2026-09-28): stdout authoritative provenance must count REAL join
    # hits (queue rows that actually got a sidecar row), not the sidecar's
    # total row count -- the two differ exactly when the sidecar covers rows
    # the queue does not carry (windowing/limit), and the old number
    # overstated the passthrough coverage.
    sidecar_rows_joined = sum(
        1 for spec in config.request_queue if spec.request_id in sidecar_rows)
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
        # kv_policy 列（此前不写这些键，四列恒空串、A.5 KV_POLICY_LABELS
        # fail-closed 校验被旁路）。
        "model": {"name": config.model_name},
        "hardware": {
            "label": config.hardware.label,
            "capacity_profile": config.hardware_capacity_profile,
        },
        "kv_management": {"policy": config.kv_cache_policy},
        "request_queue_csv": str(config.request_queue_csv),
        # D3 (2026-09-28): queue byte digest (see _queue_digest).
        **queue_digest,
        # P0-2 provenance: 同 session 连续段审计字段（wscllm 无 C++ 拒绝门，
        # window advisory——字段仅供 campaign 复核，见函数 docstring 裁决）。
        "max_same_session_span": span_scan["max_same_session_span"],
        "span_session_id": span_scan["span_session_id"],
        "span_row_range": span_scan["span_row_range"],
        "span_row_range_convention": span_scan["span_row_range_convention"],
    }
    # 契约 §7 裁定 2:die 后端键(compute-cost-model + die-model-config
    # 绝对路径);legacy 配置无键,manifest 形状不变。
    manifest.update(_resolved_backend_keys(config))

    metrics_manifest = {
        "schema_version": 1,
        # D3 (2026-09-28): MetricCollector's consistency gate
        # ("manifest npus_count != simulated systems count") only runs when
        # the key is present -- its absence since forever kept the gate
        # structurally dormant. config.npus_count is the authoritative
        # topology size the run simulates.
        "npus_count": config.npus_count,
        "repo_variant": REPO_VARIANT,
        "manifest_source": "synthetic-prerun",
        "requests": _derive_metrics_requests(config, sidecar_rows),
        # SLO pipeline B2（CPP_SPEC §A）：C++ 观测器采样锚点（WP8 水位线
        # 周期 / WP6 链路桶长）；null 值 -> 临时锚点 + provisional=true。
        "slo_sampling": _slo_sampling_node(),
        # D3 (2026-09-28): queue byte digest (see _queue_digest); the C++
        # side ignores unknown top-level keys, this is audit provenance.
        **queue_digest,
    }
    # die 后端键同步进 metrics_manifest(与 manifest.json 同源,
    # _resolved_backend_keys):归档 run 目录只保证携带本文件——把
    # compute-cost-model 放进来让 metrics_postprocess 的 die 利用率管线
    # 断言(active_kernel_die_* 非 null 且 ∈[0,1+ε])离线可判;legacy
    # 配置无键,manifest 形状不变。
    metrics_manifest.update(_resolved_backend_keys(config))

    _publish_plan_dir(output_dir, manifest, metrics_manifest)

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
        "requests": len(manifest["requests"]),
        "sessions": manifest["selected_session_count"],
        # A.5 (2026-09-05): policy 透传 provenance（物化后 run 头/配置
        # 一致的核对锚点）。
        **policy_provenance,
        # P0-2 (2026-08-31): span 审计字段随 stdout 权威 provenance 记录
        # 一并输出（与 manifest.json 持久字段同源同值）。
        "max_same_session_span": span_scan["max_same_session_span"],
        "span_session_id": span_scan["span_session_id"],
        "span_row_range": span_scan["span_row_range"],
        "sidecar": str(_sidecar_path_for_queue(config.request_queue_csv)),
        "sidecar_rows_joined": sidecar_rows_joined,
        "sidecar_rows_total": len(sidecar_rows),
        "request_type_counts": request_type_counts,
        "note": runtime_note,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
