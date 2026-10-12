#!/usr/bin/env python3
"""plan_materializer.py -- 路径③④ 的 plan 目录物化入口（唯一输入物化器）。

在线③④需要三类输入，本脚本以**不跑规划器、不写任何 .et** 的方式产出：

  1. runtime_config 四小件（system/comm_group/remote_memory/network.yml）
     —— 由 load_wsc_llm_trace_config() 装载配置时经 config_resolver.materialize_runtime_configs
     副产（路径不变，零改动）；
  2. manifest.json —— 含在线侧实际消费的**队列派生 9 字段**
     （request_id/session_id/turn_index/queue_index/prefill_length/
     decode_length/history_tokens_before/prefill_context_tokens/
     final_context_tokens；推导规则与《sh_3.0的120档问题分析报告.md》§4
     受控实验（20 档真实 manifest 2091/2091 行 0 mismatch）同式：
     turn-0 history=0；turn>0 history=上一请求 final；
     prefill_context=折入后的 prefill_length（turn-0）或 history+
     prefill_length（turn>0，恢复取代重算口径）；final=context+decode）
     + WP2 透传字段 request_type/prefix_len（源自队列旁 canonical
     sidecar 的 request_type/raw_prefix_tokens 列；
     sidecar 缺失/行缺失时对应字段不写，消费方回退 unknown/NA 并注明，
     不猜测填充）；human_time_ns/tool_time_ns 的唯一消费链
     （slo_common.load_request_manifest -> slo_stats session T_session）
     读 metrics_manifest.json，故这两个字段只写 metrics_manifest
     （P2-13 对齐 2026-10-01：此前写在 manifest.json 属写错文件，
     消费侧恒拿不到 -> T_session 全 NA）；
  3. metrics_manifest.json —— 主 agent 裁决 (i) 的合成口径：
       - schema_version=1 + requests[]（arrival：turn0=absolute session
         arrival；turn>0=after_request(同 session 上一 queue_index, interval)
         ——C++ MetricCollector.cc 只要求
         schema_version==1 + requests[]）；
       - **显式合成标记**：repo_variant 原样 + manifest_source=
         "synthetic-prerun"（裁决 i 条件 a）+ npus_count（config 权威
         拓扑规模，喂 C++ MetricCollector 的 "manifest npus_count !=
         simulated systems count" 一致性门——该门只在该键在场时生效，
         缺席即结构性休眠，P2-13 对齐 2026-10-01 补写）；
       - **占位决策字段**：prefill/decode instance 恒 0、ranks 恒
         instance-0 ranks——静态 rank 归因维度**不可信**；请求级指标
         （e2e/完成/sim_end/tput）可信（裁决 i 条件 b，口径登记于实录）；
       - 对账工具（ledger_reconcile 族）不读本 manifest——只消费归档
         jsonl（裁决 i 条件 c，已核实维持）。

输出目录：sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_plan_<cfg8>/（保留 54npus
前缀以过 GEN_MATCH；<cfg8> = trace_config 内容摘要 8 位 hex）。发布纪律（D3 对齐
2026-10-01）：两份 manifest 先写进点前缀 staging 兄弟目录、再经两次 rename 原子发布，
发布成功后清扫其它 <cfg8> 世代的陈旧 plan 目录与任何中断残留——"恰一个 plan 目录"
的 GEN_MATCH 不变量由本物化器自身维持（trace_config 字节变化产生新 cfg8 目录时，
旧目录不再滞留成热闸）。物化中途崩溃的最坏状态是 GEN_MATCH 找到零个目录
（runner 大声失败），不再是半成品/混代目录。fail-closed 在 loader 侧：
request_queue_csv 缺失/空队列
（load_request_queue 要求至少一个请求行）时 load_wsc_llm_trace_config
直接退出，本脚本因此拿到的请求队列恒非空。

关于目录名中硬编码的"54npus"（2026-10-01 审查登记）：它是一个**冻结的
GEN_MATCH 兼容产物名前缀**，与实际 NPU 规模无关——实际 NPU 数由
hardware_config 解析（load_wsc_llm_trace_config -> config.hardware /
npus_count），勿据目录名推断规模。保留硬编码的原因：该名字的消费面广且
是字节级产物名依赖（通配/前缀字面量），并非经统一常量获取——README 的
GEN_MATCH 说明、sh_test_mesh/run_scripts/ 下 run_online_strategy.sh /
run_online_strategy_sensing.sh / run_online_idle_fixture.sh /
run_online_wakeup_guard_fixture.sh / run_online_same_tick_milestone.sh 的
`llama2_7b_wsc_llm_inference_54npus_*` glob、online/verify/
train_a1_eviction_fixture.py 的 PLAN_PREFIX、以及 slo_tools/tests/
run_golden_live.py 的陈旧目录清理 glob 都按该字面量匹配；参数化目录名需
同步全部消费方并作废既有归档产物名，收益低、破坏面大，故保留行为并在此
登记。

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
# 唯一取值 session_lru_tiered（session 级二态冷热管理：完整本地/完整远端、
# 整体 LRU 逐出 + 远端池全量恢复；2026-09-25 session 级 Tiered-LRU 批起
# 原 PARTIAL 半层化三态与两段式逐出已物理移除），与
# generate_wsc_llm_trace._parse_config_value 的
# kv_cache_policy 值域同源；旧值别名与本值同调度器（无档位分支），已随
# 2026-09-25 命名卫生直接清除。
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


def _queue_digest(queue_csv: Path) -> dict:
    """D3 对齐（2026-10-01）：本 plan 由以构建的 request queue 字节摘要；
    两份 manifest 原样携带。

    同路径再物化会在不变的路径串下换掉 CSV 文件，仅凭
    manifest.request_queue_csv 无法区分陈旧 plan 与新鲜 plan——没有
    摘要时"旧 plan 配新队列"在三个出口（manifest.json requests /
    metrics_manifest.json requests / 队列行本身）静默不一致。带上摘要
    后，失配只是一次 sha256 比对（审计级探测；C++ MetricCollector 忽略
    未知顶层键，故本字段 by design 仅为 provenance）。"""
    data = queue_csv.read_bytes()
    return {
        "request_queue_csv_sha256": hashlib.sha256(data).hexdigest(),
        "request_queue_csv_bytes": len(data),
    }


def _publish_plan_dir(output_dir: Path, manifest: dict,
                      metrics_manifest: dict) -> None:
    """D3 对齐（2026-10-01）：两份 manifest 先暂存，再发布 plan 目录。

    旧实现先 mkdir 最终目录再顺序直写两份 manifest：中断会留下
    (a) 混代目录（新 manifest.json 配旧 metrics_manifest.json）或
    (b) 更早崩溃留下的空目录——两者都算 GEN_MATCH
    （`llama2_7b_wsc_llm_inference_54npus_*` glob）的合法命中；且
    trace_config 任何字节漂移产生新 cfg8 目录、旧目录滞留，
    "恰一个"闸一直 exit-1 到手工清理为止。

    现在：一切先暂存在点前缀兄弟目录（GEN_MATCH glob 不可见），
    经两次 rename 发布。最坏崩溃状态是 GEN_MATCH 找到零个目录
    （runner 以其 "run plan_materializer" 消息大声失败）——绝无
    半成品或陈旧世代滞留热座。发布成功后物化器清扫其它 cfg8 的
    plan 目录与任何崩溃残留，"恰一"不变量由建目录的工具自身维持。
    （单物化器运维模型，与 runner 自身相同。）"""
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
                # 恢复上一份完整 plan，而不是什么都不留。
                os.rename(prev_dir, output_dir)
            raise
    except BaseException:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    if prev_dir.exists():
        shutil.rmtree(prev_dir)
    # 陈旧兄弟 plan 目录（旧 <cfg8> 世代）在新世代发布后就是永久的
    # GEN_MATCH 假命中。
    for stale in GENERATED_ROOT.glob(f"{PREFIX}_54npus_plan_*"):
        if stale.is_dir() and stale.resolve() != output_dir.resolve():
            shutil.rmtree(stale)
    # 任何中断发布的崩溃残留（点前缀，GEN_MATCH 不可见，永不合法）。
    for pattern in (f".{PREFIX}_54npus_plan_*.staging",
                    f".{PREFIX}_54npus_plan_*.previous"):
        for leftover in GENERATED_ROOT.glob(pattern):
            shutil.rmtree(leftover, ignore_errors=True)


def _reconcile_queue_provenance(queue_csv: Path) -> None:
    """F2_PY 对齐（2026-10-01）：任何 sidecar join 之前，先对账队列与其
    provenance 记录。

    物化器发布三件套仍是三次顺序 rename（queue -> sidecar ->
    provenance）：中途崩溃留下混代组合（如新 queue + 旧 sidecar +
    旧 provenance）。C++ 侧由 provenance 门闭合，但本物化器此前不读
    provenance、sidecar 又不携带摘要绑定，陈旧 sidecar 行会静默 join
    新队列（透传字段静默缺席、两份 manifest 形似"合法"、SLO T_session
    退化为 NA、零报错）。把记录的 ``csv_sha256`` 与队列实际字节比对，
    把该窗口变成显式失败。无 provenance 记录的队列（如单测合成队列）
    无可对账、容忍之——真实 run 的 C++ 在线读取器仍强制 provenance
    在场。"""
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


QUEUE_SUFFIX = "_request_queue_recompute.csv"
SIDECAR_SUFFIX = "_canonical_sidecar.csv"

# SLO pipeline B2 (CPP_SPEC §A) provisional sampling anchor. Fallback for a
# missing/null derived value in the per-repo slo_params_manifest.json only
# (batch B4b filled every value; the normal path no longer hits this anchor);
# the value follows the coordinator ruling 2026-08-26
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
    requests = []
    seen_request_ids: set = set()
    last_final_by_session = {}
    last_turn_by_session = {}
    for index, spec in enumerate(config.request_queue):
        if spec.turn_index == 0:
            # 同 session 重复 turn-0 fail-closed（2026-09-30）：此前静默
            # 覆盖该 session 的 history/turn 游标，物化看似成功、运行期
            # 请求准入才崩（报错远离根因）。装载器不查 turn-0 唯一性，
            # 在物化边界按本仓 fail-closed 纪律拦截。
            if spec.session_id in last_turn_by_session:
                raise RuntimeError(
                    f"request {spec.request_id} is a second turn-0 row for "
                    f"session {spec.session_id!r} (the session already has "
                    "queue rows); every session must have exactly one "
                    "turn-0 row (fail-closed)")
            history = 0
            # 恢复取代重算单口径:turn-0 队列 prefill 已折入 prefix
            context = int(spec.prefill_length)
        else:
            # 畸形队列 fail-closed（与同文件 _derive_metrics_requests 的
            # parent 缺失 raise 同口径）：turn>0 却无紧邻前序 turn 在队列
            # 中时不得静默 history=0（窗口物化保证 turn-0 先行、turn 连续）。
            if last_turn_by_session.get(spec.session_id) != spec.turn_index - 1:
                raise RuntimeError(
                    f"request {spec.request_id} (turn {spec.turn_index}) has "
                    "no immediately preceding turn in the queue; "
                    "materialize a well-formed request queue (turn-0 first, "
                    "turns in order per session)")
            history = last_final_by_session[spec.session_id]
            # 恢复取代重算单口径后续 turn:context = 驻留 history + 新 prefill
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
        # guessed. P2-13 对齐（2026-10-01）：human_time_ns/tool_time_ns
        # 不写本 manifest——其唯一消费链（load_request_manifest ->
        # slo_stats session T_session）读 metrics_manifest.json，写在这里
        # 是写后不读（见 _derive_metrics_requests）。
        sidecar_row = sidecar_rows.get(spec.request_id)
        if sidecar_row is not None:
            request_type = (sidecar_row.get("request_type") or "").strip()
            if request_type:
                entry["request_type"] = request_type
            prefix_text = (sidecar_row.get("raw_prefix_tokens") or "").strip()
            if prefix_text:
                entry["prefix_len"] = int(prefix_text)
        # manifest 级 request_id 唯一性 fail-closed（4.3 对齐 2026-10-01）：
        # manifest 是冻结产物、被下游逐 id 消费（last-wins/双计均静默），
        # 在清单边界独立复核（上游装载器 + metrics 侧 (session,turn) 门
        # 之外再一道门）。wsc_llm_scheduler 的 anchor/rank 账本同样按
        # request_id 键控。
        if entry["request_id"] in seen_request_ids:
            raise RuntimeError(
                f"manifest requests[] has duplicate request_id "
                f"{entry['request_id']!r} (fail-closed); downstream "
                "id-keyed consumers cannot disambiguate it"
            )
        seen_request_ids.add(entry["request_id"])
        requests.append(entry)
        last_final_by_session[spec.session_id] = final
        last_turn_by_session[spec.session_id] = spec.turn_index
    return requests


def _derive_metrics_requests(config, sidecar_rows):
    """metrics_manifest.json 的 requests[]（合成口径，占位 rank）。"""
    group_by_index = dict(enumerate(config.inference_groups))
    placeholder_ranks = list(group_by_index[0].ranks)
    queue_by_session_turn = {
        (spec.session_id, spec.turn_index): index
        for index, spec in enumerate(config.request_queue)
    }
    records = []
    seen_session_turns: set = set()
    for index, spec in enumerate(config.request_queue):
        # (session, turn) 唯一性 fail-closed（2026-09-30，与
        # _derive_manifest_requests 的重复 turn-0 门同口径）：重复键此前
        # 被 queue_by_session_turn 静默折叠、父链查表二义，物化看似成功。
        session_turn = (spec.session_id, spec.turn_index)
        if session_turn in seen_session_turns:
            raise RuntimeError(
                f"request {spec.request_id} repeats (session_id="
                f"{spec.session_id!r}, turn_index={spec.turn_index!r}); "
                "(session, turn) must be unique in the queue (fail-closed)")
        seen_session_turns.add(session_turn)
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
        # P2-13 对齐（2026-10-01）：WP2 透传的 human_time_ns/tool_time_ns
        # 写本 manifest（其唯一消费链 load_request_manifest ->
        # slo_stats session T_session 从这里读；字段口径与原误写位一致
        # ——非空 sidecar 文本转 int，缺席不写）。
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
    WindowedTraceReader 的索引遍逐字节同口径（M28 修复 2026-09-29；
    2026-10-01 追齐 C++ 的行尾处理）：C++ 按 '\n' 切分原始字节、
    剥行尾单个 '\r' 后再做空行判定（process_indexed_row 的
    line.empty() 门）——CRLF 文件的裸 CR 行是空行、不计数据行。
    Python 侧据此剥 '\n' 后再剥单个 '\r'，
    与 C++ data_rows 口径一致，manifest 审计字段 scanned_data_rows 不再
    可能与 C++ 静默分叉；第 1 列 session_id、跳过表头（首个非空行）、
    空行既不算数据行也不断开同 session 连续段；行号口径 = 1-based
    CSV 文件行号（表头=第 1 行，首个数据行=第 2 行）。并列取首个最大段
    （确定性）。

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
            # 对齐 C++（2026-10-01 追齐）：剥 '\n' 后再剥行尾单个 '\r'，
            # CRLF 空行按空行处理（与
            # WindowedTraceReader.process_indexed_row 的终止符/空行语义
            # 逐字节一致）。
            line = raw.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                continue  # 空行：非数据行，也不断开 session 连续段
            if not header_seen:
                header_seen = True  # 首个非空行 = 表头
                continue
            # '#' 注释行（request_id 第 3 列前缀）与 C++ data_rows 同口径
            # 在计数前跳过（C++ M11 分类点；loader 侧对同一形状拒绝，
            # 本审计字段与 C++ 计数保持逐字节一致）。
            fields_head = line.split(",", 2)
            if len(fields_head) > 2 and fields_head[2].startswith("#"):
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
            "1-based csv file lines (header=line 1; produced by "
            "plan_materializer._scan_max_session_span, aligned with the "
            "C++ WindowedTraceReader line-splitting/blank-line semantics)",
        "scanned_data_rows": data_rows,
    }


def _resolved_backend_keys(config) -> dict:
    """manifest 的 die 后端键(契约 §7 裁定 2;阶段 C)。

    从 load_wsc_llm_trace_config 副产的 resolved system.json(config.
    system_config)读 compute-cost-model 与 die-model-config:前者声明
    backend(die/roofline/replay),后者是 resolved_die_model.json 绝对
    路径(resolver 注入的 managed 字段)。legacy 配置无这些键 → 返回空
    dict(manifest 形状不变,向后兼容);die-model-config 必须是已存在
    的绝对路径文件(fail-closed,不静默丢键)。"""
    keys = {}
    system_config = getattr(config, "system_config", None)
    if system_config is None:
        # 最小测试替身无 resolved system.json(legacy 形状):manifest
        # 保持无键,行为与旧配置一致。
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
    # A.5（2026-09-05）：policy 透传断言 + provenance（见函数 docstring）。
    policy_provenance = _assert_policy_passthrough(config)
    cfg8 = _config_digest8(config.config_csv)
    # "54npus" 为冻结的 GEN_MATCH 兼容产物名前缀(勿据以推断规模,消费方
    # 清单与保留原因见模块 docstring 的 2026-10-01 审查登记)。
    output_dir = GENERATED_ROOT / f"{PREFIX}_54npus_plan_{cfg8}"

    # F2_PY 对齐（2026-10-01）：任何跨代数据（sidecar）被消费之前先对账
    # 队列 provenance（见 _reconcile_queue_provenance）。
    _reconcile_queue_provenance(config.request_queue_csv)
    sidecar_rows = _load_sidecar_rows(config.request_queue_csv)
    # P0-2 (2026-08-31, 总文档 §4 P0-2.1): max consecutive same-session span
    # ——与 sh/face 系启动预检同口径的 provenance 审计字段。wscllm 裁决
    # （批次4）：本仓无 C++ 拒绝门（window advisory），字段仅 provenance。
    span_scan = _scan_max_session_span(config.request_queue_csv)
    queue_digest = _queue_digest(config.request_queue_csv)
    # stdout 权威 provenance 须计真实 join 命中数（拿到 sidecar 行的
    # 队列行），不是 sidecar 全量行数（D3 对齐 2026-10-01）。
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
        # 队列字节摘要（见 _queue_digest；D3 对齐 2026-10-01）。
        **queue_digest,
        # P0-2 provenance: 同 session 连续段审计字段（wscllm 无 C++ 拒绝门，
        # window advisory——字段仅供 campaign 复核，见函数 docstring 裁决）。
        "max_same_session_span": span_scan["max_same_session_span"],
        "span_session_id": span_scan["span_session_id"],
        "span_row_range": span_scan["span_row_range"],
        "span_row_range_convention": span_scan["span_row_range_convention"],
    }
    # 阶段 C(契约 §7 裁定 2):die 后端键(compute-cost-model +
    # die-model-config 绝对路径);legacy 配置无键,manifest 形状不变。
    manifest.update(_resolved_backend_keys(config))

    metrics_manifest = {
        "schema_version": 1,
        # P2-13 对齐（2026-10-01）：C++ MetricCollector 的 "manifest
        # npus_count != simulated systems count" 一致性门只在该键在场时
        # 生效——缺席使该门自始结构性休眠。config.npus_count 是本 run
        # 仿真的权威拓扑规模。
        "npus_count": config.npus_count,
        "repo_variant": REPO_VARIANT,
        "manifest_source": "synthetic-prerun",
        "requests": _derive_metrics_requests(config, sidecar_rows),
        # SLO pipeline B2（CPP_SPEC §A）：C++ 观测器采样锚点（WP8 水位线
        # 周期 / WP6 链路桶长）；null 值 -> 临时锚点 + provisional=true。
        "slo_sampling": _slo_sampling_node(),
        # 队列字节摘要（见 _queue_digest；C++ 侧忽略未知顶层键，审计
        # provenance；D3 对齐 2026-10-01）。
        **queue_digest,
    }
    # 阶段 E2：die 后端键同步进 metrics_manifest（与 manifest.json 同源，
    # _resolved_backend_keys）。归档 run 目录只保证携带本文件——把
    # compute-cost-model 放进来让 metrics_postprocess 的 die 利用率管线
    # 断言（active_kernel_die_* 非 null 且 ∈[0,1+ε]）离线可判；legacy 配置
    # 无键，manifest 形状不变。C++ load_manifest 与 slo_common 均只取
    # 自己认识的键，未知顶层键被忽略。
    metrics_manifest.update(_resolved_backend_keys(config))

    _publish_plan_dir(output_dir, manifest, metrics_manifest)

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
