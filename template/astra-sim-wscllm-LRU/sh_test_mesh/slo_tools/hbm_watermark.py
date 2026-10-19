#!/usr/bin/env python3
"""WP8 补充主数据源：离线 HBM KV 水位线重建（标准库实现）。

同步基线注记（2026-09-26）：本文件与 face/wscllm 家族同名脚本已按仓分叉
（本仓承载 B4 -LRU wscllm tiered 的 report-only/对账增量），不再与其它仓
逐字节相同；跨仓同步仅覆盖下方"manager 原函数（逐字拷贝）"节与公共框架
导入，其余差异以各仓 REPO_VARIANTS 注记为准。

在线模式 C++ 内存账本为空是既有现状（C++ 侧 hbm_watermark 记录全零属预
期）；真实 KV 占用时序在 python 侧产物中。经五仓 B0 基线逐仓实测核对：

  * ``bridge/ledger.jsonl``（sensing 档的分层请求账本）只在感知模式落盘，
    且逐 request 记 admitted/committed/issued 层、**不含 bytes**——不是
    KV 占用数据源；正式跑（sensing off）没有该文件。
  * 真正的 KV 动作时序在 ``results/online_decision_log.jsonl``（每请求
    prefill/decode/completion 三条决策，附 restore 迁移 bytes、逐出条目
    bytes/victim、P→D 迁移 bytes）；token 事实（prefill_context_tokens/
    final_context_tokens/history_tokens_before）在物化 plan manifest
    （manifest.json）。本脚本把这两者合称"python 侧 ledger"，逐仓字段
    映射登记于 REPO_VARIANTS（先实际检查五仓字段再登记，不猜测）。

四层可信度（P1 工具语义改造，2026-08-30；tier 由 run_dir 内容自动判定，
log 一行说明判定依据，无需新 CLI 参数）：

  * ``per_rank_total_hbm_certified``——``results/kv_delta_journal.jsonl``
    （manager mutation 提交点的 append-only 逐 rank delta 权威账本）与
    ``results/kv_delta_journal_checksum.json``（run 末守恒证书）都在场，
    且四道门全过：journal sha256 == checksum.sha256、行级链自洽
    （before == 前行 after、after == before + 三类 delta）、重放终态 ==
    checksum 终态、证书四项 checks 全 true。流式重放 journal 得逐 rank
    physical=weight+resident+reserved 时序，对行内 capacity_bytes 逐 rank
    认证——**正式容量判决只在本层给出**（违规 → 退出码 3，fail-loud）。
  * ``resident_kv_exact``——journal 在场、行级链自洽，但 checksum 缺失：
    逐 rank resident/weight/reserved 时序精确（行级 before/after 自证），
    缺 run 末守恒证书（journal 覆盖完整性无证明）→ 逐 rank 容量检查如实
    报告（violation 计数标注 tier），但不作正式判决（不出退出码 3）。
  * ``lifecycle_replay_exact``——journal + checksum 都在场、链自洽、终态
    一致，但证书四项 checks 有 false（run 末守恒未通过，如残差非零），
    或证书缺必需项（sha256/line_count/ranks 缺失或类型不符，HBM-01：
    完整性证明不完整，不得 certified）：
    会话生命周期逐事务精确（manager_state_match 仍真），无守恒证书 →
    同上仅报告；log 显著警告守恒失败项。
  * ``upper_bound_only``——journal 缺失（阶段2 前的全部旧 run）：只能走
    decision-log 重放，其结果对"真实占用"是**上界**（terminal 退休/decode
    准入逐出两缺口不落盘 → 17.24TB 级幻影）。本层**不得输出物理违规认证**：
    occupancy_valid 恒 false、退出码 3 废除（超限只作诊断计数并标注
    tier，退出码不再因"聚合占用>容量"为 3）。旧 run 重跑后处理得本层 +
    诊断属预期语义，不是回归。
  * journal 在场时 decision-log 重放照常执行，作为**对照列**输出（上界
    vs 权威，差异 = 账本缺口的直接可视化）；journal 缺失时 decision-log
    重放即权威（上界口径）。journal sha256 不匹配 / 行级链断裂 / 终态与
    证书矛盾 → fail-closed 退出码 2（账本损坏不得静默降级）。

算法（重放，非策略复刻）：
  按 (文件顺序=seq) 重放每条决策记录（journal 路径按 sequence 流式重放），
  记录内固定次序「逐出 → 恢复/迁移 → 增长」。会话状态（当前所在实例、
  当前本地 bytes）由本脚本自行跟踪，恢复/迁移的方向由跟踪态判定（本地
  跨实例=搬移、远端/同实例=只增），已落盘 bytes 与 f(tokens)=2·layers·
  hidden_size·bytes_per_elem·tokens 逐条对账（f 已在五仓基线数据上核对：
  restore 比值全 1.0，S3 另有 0.5 分层半恢复）。增长按 manager 语义
  "长到 f(目标 tokens)"取增量。

  → 事件流 stats（P1-②）：peak/mean/residual/violation 全部在变点归并
    （RLE）时 O(动作数) 内存计算，与 span/桶长彻底解耦——同事件流换任意
    桶长，stats 逐字段不变（单测断言）；
  → 权威时序产物 ``slo_hbm_intervals.csv``（P1-④）：change-point/RLE
    区间（区间起止 tick、起止占用、区间内峰值、逐出叠加），无损、
    O(动作数) 行，可从它恢复任意桶长序列（单测断言无损恢复）；
  → 绘图产物 ``slo_hbm_plot_series.csv``（P1-④，取代旧
    slo_hbm_watermark_series.csv）：全局行预算 R（manifest
    watermark_series_row_budget）约束，B_eff = max(B_requested,
    ceil(S·N/(R−N)))（S=全局 span=末 KV 事件−首 KV 事件、N=有事件实例
    数；R≤N 时拒绝稠密输出只给 RLE，正常完成并 log 说明）；流式边算边
    写，桶行在游标推进时逐行写出，内存 O(实例数+活跃游标)。元数据
    resolution_adjusted/adjustment_reason/row_budget/span_ns/
    bucket_origin_ns 独立登记，**不复用 bucket_ns_provisional**（后者
    语义=manifest 缺正式桶长锚点）。

容量三口径（P1-③，summary 分列；逐 rank 剖面函数逐字拷贝自仓内
session_kv_manager.py，见"manager 原函数（逐字拷贝）"节）：
  * 正式认证：逐 rank physical = weight + resident + reserved ≤
    capacity_bytes（数据源 = journal 行；只在 per_rank_total_hbm_
    certified 层构成判决）；
  * resident 硬上限：reservation=0 时逐 rank ⌊(capacity − weight_i)/
    kv_i⌋ 最小 token 数 × Σkv_i（任意时刻成立；llama2_7b/swiglu/TP6/
    160GiB 锚定 1,723,864 token → 903,801,208,832 B）；
  * 水位目标：kv_reserve_context_tokens（1M/rank）扣减后同式（锚定
    723,864 token → 379,513,208,832 B）——水位目标非任意时刻上限，
    **仅报告不作判决**。

容量链（逐仓登记，不编造）：
  trace_config.csv config 行 local_hbm_capacity_profile → 仓内
  sh_test_mesh/hardware/*.json 的 local-hbm.capacity-profiles[profile].bytes
  （每 NPU 字节）× npus_per_instance（取 request manifest requests[].
  prefill_ranks 长度）。任一环节缺失 → capacity=NA，违规检查降级为
  "峰值记录"并注明，绝不代拟容量值。

覆盖度（decision-log 重放路径的口径；fail-closed 语义的对偶面，显式
降级、绝不静默）：
  * FACE/W/S1/S3：逐出条目带 bytes+victim → eviction_coverage=full，
  * S2：decision log 只落 *_eviction_count（无 victim/bytes），他人逐出
    无法归因 → occupancy 为上界（未归因逐出不扣减）。B3-6（2026-08-27）
    起新产物逐条序列化 *_evictions（victim/bytes）+ history_transfers
    （partial 两段式恢复逐段对象）——字段在场即升级 full 口径
    （upgrade_s2_mapping_if_fields_present），旧产物保底回退 count_only。

退出码：0 正常（含 upper_bound/resident/lifecycle 层的超限诊断——
tier 已标注，不构成物理违规认证）；2 fail-closed（缺文件/缺列/结构错/
重放不可续/journal 损坏）；3 容量违规——**仅 per_rank_total_hbm_
certified 层**的逐 rank physical > capacity_bytes。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, NA, SloToolError, default_manifest_path,
    detect_repo_variant, emit_json, fail, iter_jsonl,
    load_json_strict, load_request_manifest, load_slo_manifest,
    manifest_requests, open_output, read_init_record, require_param_int,
    run_main, write_csv,
)

def slo_json_constant_reject(token: str) -> None:
    """json.loads 的 parse_constant 钩子：NaN/Infinity 字面量 fail-closed
    （与 slo_common.load_json_strict 同纪律，供行级 json.loads 使用）。"""
    fail(f"journal 行含非有限常量字面量 {token}——NaN/Infinity 不进入"
         f"本工具链（fail-closed）")


EXIT_VIOLATION = 3  # 容量违规（与 fail-closed 的 2 区分；仅 certified 层）

# watermark_sample_period_ns 未推导（null）时的临时桶长锚点（ns）。
# 全输出（JSON/CSV/stderr）都会带 bucket_ns_provisional=true 标注。
PROVISIONAL_BUCKET_NS = 5_000_000

TOKEN_MANIFEST_FILENAME = "manifest.json"

# P1 权威账本（阶段2 manager mutation 提交点落盘；journal 开关 off 时
# → decision-log 重放上界路径）。
KV_DELTA_JOURNAL_RELPATH = Path("results") / "kv_delta_journal.jsonl"
KV_DELTA_JOURNAL_CHECKSUM_RELPATH = (
    Path("results") / "kv_delta_journal_checksum.json")

# 四层可信度（判定条件见模块 docstring；输出顺序=证据强度降序）。
TRUST_TIER_UPPER_BOUND = "upper_bound_only"
TRUST_TIER_LIFECYCLE = "lifecycle_replay_exact"
TRUST_TIER_RESIDENT = "resident_kv_exact"
TRUST_TIER_CERTIFIED = "per_rank_total_hbm_certified"

# 绘图 series（P1-④：受全局行预算约束的稠密产物；旧产物名
# slo_hbm_watermark_series.csv 退役）。
SERIES_COLUMNS = (
    "repo_variant", "instance_index", "bucket_index", "bucket_start_ns",
    "bucket_end_ns", "occupancy_end_bytes", "occupancy_peak_in_bucket_bytes",
    "evict_events", "evict_bytes",
)
# 权威 RLE 区间（P1-④：change-point 无损时序，O(动作数) 行）。
INTERVAL_COLUMNS = (
    "repo_variant", "instance_index", "interval_start_ns", "interval_end_ns",
    "occupancy_start_bytes", "occupancy_end_bytes",
    "occupancy_peak_in_interval_bytes", "evict_events", "evict_bytes",
)
INSTANCE_COLUMNS = (
    "repo_variant", "instance_index", "trust_tier", "eviction_coverage",
    "occupancy_valid",
    "capacity_bytes", "capacity_source", "bucket_ns",
    "bucket_ns_provisional", "effective_bucket_ns", "resolution_adjusted",
    "first_event_ns", "last_event_ns",
    "duration_ns", "peak_occupancy_bytes", "mean_occupancy_bytes",
    "residual_occupancy_bytes", "evict_events", "evict_bytes",
    "violation_events", "upper_bound_peak_occupancy_bytes",
)


# ---------------------------------------------------------------------------
# 逐仓字段映射（五仓 B0 基线 60s_summary 实测登记；改动需重新对基线核对）
# ---------------------------------------------------------------------------

def _sum_shard_bytes(entry: dict) -> Optional[int]:
    """FACE/W 逐出条目：shard_bytes 列表 → 实例合计。"""
    shards = entry.get("shard_bytes")
    if not isinstance(shards, list) or not shards:
        return None
    total = 0
    for value in shards:
        # bool 排除（2026-10-01 对齐）：isinstance(True, int) 为真，
        # shard_bytes:[true] 曾按 1 字节计入合计；bool → None → 调用方
        # _evict_action fail-closed，不静默当 1 字节。
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        total += value
    return total


def _safe_mean_float(area: int, duration: int, peak: int) -> float:
    """stats.mean 的溢出安全包装（见调用点注释）。"""
    try:
        return (area / duration) if duration > 0 else float(peak)
    except OverflowError as error:
        fail(f"占用面积/峰值超出浮点可表示范围（area={area}, "
             f"peak={peak}）：账本含畸形超大 delta，拒绝产出水位线"
             f"（{error}）")


def _scalar_bytes(entry: dict) -> Optional[int]:
    # bool 排除（2026-10-01 对齐，hopbytes.py D2 同口径）：
    # isinstance(True, int) 为真，total_bytes=true 曾按 1 字节放行；
    # bool → None → 调用方 fail-closed，不静默当 1 字节。
    value = entry.get("total_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _eviction_entry_bytes(entry: dict) -> Optional[int]:
    """legacy 逐出条目 bytes：shard_bytes 列表优先，缺则回退 total_bytes。

    wscllm 旧基线（B2 三态前）逐出行是 EvictionRecord（shard_bytes），
    B2 窗口产物逐出行已换 KVTransfer 契约行结构（total_bytes，无
    shard_bytes）——B2 窗口产物保持 legacy 逐出消费路径时必须两形通吃，
    否则 _evict_action 拒绝、B2 逐出整类不可消费。"""
    shard_total = _sum_shard_bytes(entry)
    if shard_total is not None:
        return shard_total
    return _scalar_bytes(entry)


def _evict_action(tick: int, entry: dict, bytes_of, where: str) -> dict:
    """逐出条目 → 规范动作。victim 实例缺省 None（用跟踪态归位）。"""
    nbytes = bytes_of(entry)
    if nbytes is None:
        fail(f"{where}: 逐出条目缺可解析 bytes（shard_bytes/total_bytes）")
    victim_session = entry.get("victim_session_id")
    if victim_session is None:
        victim_session = entry.get("session_id")
    if not isinstance(victim_session, str) or not victim_session:
        fail(f"{where}: 逐出条目缺 victim 会话标识"
             f"（victim_session_id/session_id）")
    victim_instance = entry.get("victim_instance_index")
    if victim_instance is None:
        victim_instance = entry.get("source_instance_index")
    if victim_instance is not None and (isinstance(victim_instance, bool)
                                        or not isinstance(victim_instance,
                                                          int)):
        # 类型谎报不再静默降级 None（丢失实例归位信息会错扣别的实例，
        # 幻影占用零异常）——fail-closed。
        fail(f"{where}: 逐出条目 victim_instance_index/"
             f"source_instance_index 非整数（实得 {victim_instance!r}，"
             f"类型 {type(victim_instance).__name__}）")
    time_ns = entry.get("time_ns")
    if time_ns is not None and (isinstance(time_ns, bool)
                                or not isinstance(time_ns, int)
                                or time_ns < 0):
        # bool True 是合法 int 子类：旧行为当 tick=1 参与变点序——生效
        # 时刻谎报，fail-closed；None/缺席才回落记录 tick。
        fail(f"{where}: 逐出条目 time_ns 非非负整数（实得 {time_ns!r}，"
             f"类型 {type(time_ns).__name__}）")
    eff_tick = time_ns if time_ns is not None else tick
    # 类型谎报 fail-closed（F4，2026-09-30）：与 time_ns/victim_instance
    # 同口径——layer_start/layer_end 在场而非非负 int（含 bool）时旧实现
    # 静默降 None，会把层域谎报漏判出 partial_evictions 判定。
    layer_start = entry.get("layer_start")
    if layer_start is not None and (isinstance(layer_start, bool)
                                    or not isinstance(layer_start, int)
                                    or layer_start < 0):
        fail(f"{where}: 逐出条目 layer_start 非非负整数（实得 "
             f"{layer_start!r}，类型 {type(layer_start).__name__}）")
    layer_end = entry.get("layer_end")
    if layer_end is not None and (isinstance(layer_end, bool)
                                  or not isinstance(layer_end, int)
                                  or layer_end < 0):
        fail(f"{where}: 逐出条目 layer_end 非非负整数（实得 "
             f"{layer_end!r}，类型 {type(layer_end).__name__}）")
    return {"type": "evict", "tick": eff_tick, "session": victim_session,
            "bytes": nbytes, "instance": victim_instance,
            "layer_start": layer_start, "layer_end": layer_end}


def _collect_eviction_lists(record: dict, mapping: dict, where: str) -> list:
    actions = []
    decision = record.get("decision") or {}
    for kind_key, field, bytes_of in mapping["eviction_lists"]:
        if record.get("kind") != kind_key:
            continue
        entries = decision.get(field)
        if entries is None:
            continue
        if not isinstance(entries, list):
            fail(f"{where}: {field} 必须是数组")
        for entry in entries:
            if not isinstance(entry, dict):
                fail(f"{where}: {field} 成员必须是对象")
            actions.append(_evict_action(record.get("tick", 0), entry,
                                         bytes_of, where))
    return actions


def _prefill_restore_actions(record: dict, mapping: dict, where: str) -> list:
    """prefill 历史恢复 → 规范动作（restore）。"""
    decision = record.get("decision") or {}
    spec = mapping["restore"]
    nbytes = 0
    kind = None
    logged_source = None
    state_before = None
    holder = decision

    # B3-6（2026-08-27）：restore_list_field——逐段恢复对象列表（S2 新产物
    # history_transfers：partial 前缀 noc_migrate + suffix remote_load 各一
    # 段）。每段独立成动作（kind/bytes/source 逐段），标量聚合会把跨实例
    # partial 迁移错记成整体搬移而击穿占用下界。local_hit 段不计 bytes
    # （信息量非搬移量，与标量口径 history_transfer_bytes 的排除法一致）。
    restore_list_field = spec.get("restore_list_field")
    if restore_list_field:
        rows = decision.get(restore_list_field)
        if rows is None:
            holders = []
        elif isinstance(rows, list):
            # HBM-04：非 dict 成员不得静默过滤——损坏行只会缩小恢复/占用
            # 分母，必须 fail-closed。
            for row in rows:
                if not isinstance(row, dict):
                    fail(f"{where}: {restore_list_field} 成员必须是对象"
                         f"（实得 {type(row).__name__}）")
            holders = rows
        else:
            fail(f"{where}: {restore_list_field} 必须是数组")
        location_field = spec.get("location_field")
        state_before = decision.get(location_field) if location_field \
            else None
        actions = []
        for row in holders:
            row_kind = row.get("kind")
            row_bytes = row.get("total_bytes")
            # R2-2(2026-09-30):与 joint 同款 bool 排除——本文件其余各臂
            # (_evict_action/restore_list_field 等)均排除,唯此臂漏;
            # bool True 经 isinstance(int) 混入会按 1 字节计账。
            if (isinstance(row_bytes, bool)
                    or not isinstance(row_bytes, int)
                    or row_bytes < 0):
                fail(f"{where}: {restore_list_field} 成员缺非负整数 "
                     f"total_bytes")
            if row_kind in ("local_hit", "LOCAL_HIT"):
                row_bytes = 0
            row_source = row.get("source_instance_index")
            if row_source is not None and not isinstance(row_source, int):
                row_source = None
            actions.append({
                "type": "restore", "tick": record.get("tick", 0),
                "session": record.get("session_hint"), "bytes": row_bytes,
                "kind": row_kind, "logged_source": row_source,
                "state_before": state_before,
                "target": decision.get("prefill_instance_index")})
        return actions

    for path_element in spec.get("bytes_path", []):
        if isinstance(holder, dict):
            holder = holder.get(path_element)
        else:
            holder = None
    if isinstance(holder, dict):
        value = holder.get("total_bytes")
        # 在场但类型谎报（float/str/bool）不得静默按 0 重放（恢复/迁移
        # bytes 被吞会击穿占用下界与对账；缺席=None 才是合法 0 语义）。
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, int)):
            fail(f"{where}: history 恢复 total_bytes 非整数（实得 "
                 f"{value!r}，类型 {type(value).__name__}）——拒绝静默"
                 f"按 0 重放")
        if isinstance(value, int):
            nbytes = value
        kind = holder.get("kind")
        source = holder.get("source_instance_index")
        if isinstance(source, int) and not isinstance(source, bool):
            logged_source = source
    elif holder is not None:
        # 标量形态同门：在场标量必须是非负整数（bool 排除）；负值由
        # 下方统一拒绝。
        if isinstance(holder, bool) or not isinstance(holder, int):
            fail(f"{where}: history 恢复 bytes 非整数（实得 {holder!r}，"
                 f"类型 {type(holder).__name__}）——拒绝静默按 0 重放")
        nbytes = holder
    if spec.get("kind_field"):
        value = decision.get(spec["kind_field"])
        if isinstance(value, str):
            kind = value
    if spec.get("source_field"):
        value = decision.get(spec["source_field"])
        if isinstance(value, int) and not isinstance(value, bool):
            logged_source = value
    if spec.get("state_before_field"):
        value = decision.get(spec["state_before_field"])
        if isinstance(value, str):
            state_before = value
    if nbytes < 0:
        fail(f"{where}: history 恢复 bytes 为负（{nbytes}）")
    return [{"type": "restore", "tick": record.get("tick", 0),
             "session": record.get("session_hint"), "bytes": nbytes,
             "kind": kind, "logged_source": logged_source,
             "state_before": state_before,
             "target": decision.get("prefill_instance_index")}]


def _decode_move_actions(record: dict, mapping: dict, where: str) -> list:
    """decode 段 P→D 迁移 → 规范动作（move）。S3 无此段（P==D 恒成立）。"""
    decision = record.get("decision") or {}
    spec = mapping.get("decode_move")
    if spec is None:
        return []
    nbytes = 0
    logged_source = None
    target = decision.get("decode_instance_index")
    holder = decision
    for path_element in spec.get("bytes_path", []):
        if isinstance(holder, dict):
            holder = holder.get(path_element)
        else:
            holder = None
    if isinstance(holder, dict):
        value = holder.get("total_bytes")
        # 在场但类型谎报（float/str/bool）不得静默按 0 重放（迁移 bytes
        # 被吞会把 P→D 交割记成 stay、源实例驻留成幻影；缺席=None 才是
        # 合法 0 语义——与 _prefill_restore_actions 标量分支同门）。
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, int)):
            fail(f"{where}: decode 迁移 total_bytes 非整数（实得 "
                 f"{value!r}，类型 {type(value).__name__}）——拒绝静默"
                 f"按 0 重放")
        if isinstance(value, int):
            nbytes = value
        source = holder.get("source_instance_index")
        if isinstance(source, int) and not isinstance(source, bool):
            logged_source = source
    elif holder is not None:
        # 标量形态同门：在场标量必须是非负整数（bool 排除）；负值由
        # 下方统一拒绝。
        if isinstance(holder, bool) or not isinstance(holder, int):
            fail(f"{where}: decode 迁移 bytes 非整数（实得 {holder!r}，"
                 f"类型 {type(holder).__name__}）——拒绝静默按 0 重放")
        nbytes = holder
    if spec.get("source_field"):
        value = decision.get(spec["source_field"])
        if isinstance(value, int) and not isinstance(value, bool):
            logged_source = value
    if nbytes < 0:
        fail(f"{where}: decode 迁移 bytes 为负（{nbytes}）")
    return [{"type": "move", "tick": record.get("tick", 0),
             "session": record.get("session_hint"), "bytes": nbytes,
             "logged_source": logged_source, "target": target}]


REPO_VARIANTS: dict[str, dict] = {
    # FACE：prefill 携带 history_action/history_transfer_bytes/history_
    # source_instance_index + admission_evictions/decode_target_evictions
    # （shard_bytes 列表）；decode 携带 prefill_decode_transfer 对象；
    # completion 携带 completion_evictions。逐出全量落盘，重放闭合。
    "astra-sim-face": {
        "eviction_lists": (
            ("prefill", "admission_evictions", _sum_shard_bytes),
            ("prefill", "decode_target_evictions", _sum_shard_bytes),
            ("completion", "completion_evictions", _sum_shard_bytes),
        ),
        "restore": {"bytes_path": ["history_transfer_bytes"],
                    "kind_field": "history_action",
                    "source_field": "history_source_instance_index"},
        "decode_move": {"bytes_path": ["prefill_decode_transfer"]},
        "completion_relocation_field": None,  # 逐出条目已覆盖自身释放
        "eviction_count_fields": (),
        "eviction_coverage": "full",
        "capacity_field_evidence":
            "trace_config: local_hbm_capacity_profile=validation-160gib → "
            "hardware/face_case5_config_c.json "
            "local-hbm.capacity-profiles.validation-160gib.bytes=171798691840"
            "（每 NPU）",
    },
    # W 同 FACE 字段族 + history_cache_state_before。基线实测账本缺口：decode
    # 准入期逐出（decode_target_evictions 在 prefill 记录落盘之后才累积，
    # decode 记录不含逐出列表）不落盘——full_tracelab 本 run 实测 7,923 例
    # RECOMPUTE(state_before=EVICTED) 隐含静默逐出（802 为 FACE 基线旧数，
    # 非 W 本 run 口径）；重放按账本断言在恢复点对账扣减
    # （silent_evictions_reconciled），逐出真实时刻 ∈ 上次可见事件与恢复
    # tick 之间，占用在该窗口为上界。P1（2026-08-30）起缺 journal 的 run
    # 一律标 upper_bound_only（上界，无物理违规认证）；journal 在场的 run
    # 以 kv_delta_journal 为权威、本重放仅作对照列。
    "astra-sim-wscllm": {
        "eviction_lists": (
            ("prefill", "admission_evictions", _eviction_entry_bytes),
            ("prefill", "decode_target_evictions", _eviction_entry_bytes),
            ("completion", "completion_evictions", _eviction_entry_bytes),
        ),
        "restore": {"bytes_path": ["history_transfer_bytes"],
                    "kind_field": "history_action",
                    "source_field": "history_source_instance_index",
                    "state_before_field": "history_cache_state_before"},
        "decode_move": {"bytes_path": ["prefill_decode_transfer"]},
        "completion_relocation_field": None,
        "eviction_count_fields": (),
        "eviction_coverage": "full_reconciled",
        "capacity_field_evidence":
            "同 FACE（hardware/face_case5_config_c.json，"
            "validation-160gib=171798691840 B/NPU）",
        # B4（-LRU wscllm tiered，2026-09-07）：新产物（B3 分段逐出字段
        # 在场）经 upgrade_s2_mapping_if_fields_present 升级契约口径（逐段
        # restore + 契约逐出行 + completion_evictions 契约行）；升级后保持
        # full_reconciled + evict_report_only + location_field（逐出条目到
        # 时序晚于 victim 其间事件，report-only + 恢复点对账，见
        # watermark_prepare 的 W 分支注记）。B2 窗口产物（history_transfers
        # 在场、分段字段缺席）不触发升级——本表 legacy 逐出提取器
        # _eviction_entry_bytes（shard_bytes→total_bytes 回退）两形通吃，
        # coverage 保持 full_reconciled 诚实口径；更旧基线（EvictionRecord
        # shard_bytes 形）同样走本表口径。
    },
    # S1：restore 为 history_transfer 对象（kind=noc_migrate/local_hit/
    # remote_load）；逐出条目在各阶段列表、带 total_bytes+source_instance
    # _index；completion 的 kv_location 释放已含于 completion_evictions
    # （基线核对 29/29 全列出）。
    "astra-sim-sh_1.0": {
        "eviction_lists": (
            ("prefill", "history_evictions", _scalar_bytes),
            ("prefill", "prefill_evictions", _scalar_bytes),
            ("decode", "decode_evictions", _scalar_bytes),
            ("completion", "completion_evictions", _scalar_bytes),
        ),
        "restore": {"bytes_path": ["history_transfer"]},
        "decode_move": {"bytes_path": ["prefill_decode_transfer"]},
        "completion_relocation_field": None,
        "eviction_count_fields": (),
        "eviction_coverage": "full",
        "capacity_field_evidence":
            "trace_config: local_hbm_capacity_profile（kv_reserve_context_"
            "tokens=1M 的 validation 档）→ hardware json 同 FACE",
    },
    # S2（分层 KV）：decision log 只落聚合 bytes 与 *_eviction_count，
    # 无逐出 victim/bytes → 他人逐出不可归因（上界重建）；completion 的
    # kv_location_after_completion 可归因自身会话去向。
    # B3-6（2026-08-27）：B3 起新产物的 decision log 逐条带 *_evictions
    # （victim/bytes，字段名与 S1 对齐）+ history_transfers（partial 两段
    # 式恢复逐段对象）——字段在场时 upgrade_s2_mapping_if_fields_present
    # 升级 full 口径；旧产物无字段 → 原样保底回退 count_only。
    "astra-sim-sh_2.0": {
        "eviction_lists": (),
        "restore": {"bytes_path": ["history_transfer_bytes"]},
        "decode_move": {"bytes_path": ["prefill_decode_transfer_bytes"],
                        "source_field": "prefill_instance_index"},
        "completion_relocation_field": "kv_location_after_completion",
        "eviction_count_fields": ("history_eviction_count",
                                  "decode_eviction_count",
                                  "completion_eviction_count"),
        "eviction_coverage": "count_only",
        "capacity_field_evidence":
            "trace_config: local_hbm_capacity_profile → hardware json 同 FACE",
    },
    # S3（三段式/分层）：restore 为标量 bytes + history_source_instance_
    # index（0.5 比值=半层恢复；G7-5 后该比值按 restore_bytes_mismatch
    # 计诊断，不再放行）；decode 恒 P==D（红线 #4）无迁移记录；
    # 逐出仅 completion_evictions（分层 layer 区间，带 total_bytes，实例由
    # 跟踪态归位）。基线实测账本缺口：completion 时 kv_location_after_
    # completion=partial_hbm_remote 的 suffix 半层释放不落盘——重放在下次
    # 恢复点按「恢复前本地 = f(h) − 恢复 bytes」对账扣减
    # （restore_prefix_reconciled）；会话若不再回归则该部分残留为上界。
    "astra-sim-sh_3.0": {
        "eviction_lists": (
            ("completion", "completion_evictions", _scalar_bytes),
        ),
        "restore": {"bytes_path": ["history_transfer_bytes"],
                    "source_field": "history_source_instance_index"},
        "decode_move": None,
        "completion_relocation_field": None,
        "eviction_count_fields": (),
        "eviction_coverage": "full_reconciled",
        "capacity_field_evidence":
            "trace_config: local_hbm_capacity_profile → hardware json 同 FACE",
    },
}


# ---------------------------------------------------------------------------
# manager 原函数（逐字拷贝）+ 容量三口径（P1-③）
# ---------------------------------------------------------------------------
# 以下八个函数逐字拷贝自仓内 workload/llama2_7b_inference/
# session_kv_manager.py（_require_nonnegative_int / partition_values_exact /
# attention_heads_by_tp_rank / estimate_model_weight_bytes /
# model_weight_shard_bytes_by_tp_rank / kv_cache_bytes_for_tokens /
# kv_cache_shard_bytes_for_tokens / kv_cache_shard_bytes_for_layer_range，
# 以函数名检索为准）——manager 是逐 rank HBM 记账的权威实现，本工具
# 不得另立口径。**同步义务**：session_kv_manager.py 上述函数任何改动
# 必须同步拷贝到本节（跨仓同改，md5 对齐）；函数是家族共性，模型参数
# 由 run_dir 自带的 trace_config/hardware 配置实例化（各仓 hardware
# 配置可不同）。

def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def partition_values_exact(total: int, partitions: int) -> tuple[int, ...]:
    """Split an integer without padding, putting remainders on low ranks."""

    _require_nonnegative_int(total, "total")
    if isinstance(partitions, bool) or not isinstance(partitions, int) or partitions <= 0:
        raise ValueError("partitions must be a positive integer")
    quotient, remainder = divmod(total, partitions)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(partitions))


def attention_heads_by_tp_rank(num_heads: int, tp_degree: int) -> tuple[int, ...]:
    """Return the whole-head ownership of each relative TP rank."""

    if (
        isinstance(num_heads, bool)
        or isinstance(tp_degree, bool)
        or not isinstance(num_heads, int)
        or not isinstance(tp_degree, int)
        or num_heads <= 0
        or tp_degree <= 0
    ):
        raise ValueError("num_heads and tp_degree must be positive integers")
    return partition_values_exact(num_heads, tp_degree)


def estimate_model_weight_bytes(model: Any) -> int:
    """LLaMA-family model-size estimate (manager twin).

    Known in-repo divergence (2026-09-30 audit, disclosed -- the old
    "Match the existing LLaMA-family model-size estimate exactly" claim
    was false): for the gelu variant this twin counts the final norm as 0
    elements while wsc_llm_scheduler.estimate_model_weight_bytes counts a
    final LayerNorm (weight + bias, 2*hidden_size).  The two agree exactly
    on the swiglu variant -- the production trace_config.csv model, and
    the shape every pinned test value assumes.  Converging either side is
    a frozen-formula family ruling (it would silently move HBM capacity
    accounting here / the W_bytes metric proxy in metrics_postprocess,
    which treats the scheduler twin as authoritative, and desync the
    sh/face lineage); left explicit for a user-level ruling instead of a
    silent "fix"."""

    mlp_matrices = 3 if getattr(model, "mlp_variant", "gelu") == "swiglu" else 2
    norm_elements = (
        2 * model.hidden_size
        if getattr(model, "mlp_variant", "gelu") == "swiglu"
        else 4 * model.hidden_size
    )
    per_layer_elements = (
        4 * model.hidden_size * model.hidden_size
        + mlp_matrices * model.hidden_size * model.ffn_size
        + norm_elements
    )
    embedding_elements = 2 * model.vocab_size * model.hidden_size
    final_norm_elements = (
        model.hidden_size if getattr(model, "mlp_variant", "gelu") == "swiglu" else 0
    )
    return (
        model.layers * per_layer_elements + embedding_elements + final_norm_elements
    ) * model.bytes_per_elem


def model_weight_shard_bytes_by_tp_rank(model: Any, tp_degree: int) -> tuple[int, ...]:
    """Return exact TP weight shards aligned with whole-head KV ownership."""

    if model.hidden_size % model.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    ffn_extents = partition_values_exact(model.ffn_size, tp_degree)
    vocab_extents = partition_values_exact(model.vocab_size, tp_degree)
    head_dim = model.hidden_size // model.num_heads
    mlp_matrices = 3 if getattr(model, "mlp_variant", "gelu") == "swiglu" else 2
    matrix_shards = tuple(
        (
            model.layers
            * (
                4 * model.hidden_size * head_count * head_dim
                + mlp_matrices * model.hidden_size * ffn_extent
            )
            + 2 * model.hidden_size * vocab_extent
        )
        * model.bytes_per_elem
        for head_count, ffn_extent, vocab_extent in zip(heads, ffn_extents, vocab_extents)
    )
    total = estimate_model_weight_bytes(model)
    residual = total - sum(matrix_shards)
    if residual < 0:
        raise RuntimeError("TP weight matrix shards exceed the model total")
    shards = tuple(
        shard + remainder
        for shard, remainder in zip(matrix_shards, partition_values_exact(residual, tp_degree))
    )
    if sum(shards) != total:
        raise RuntimeError("TP weight shards do not preserve the model total")
    return shards


def kv_cache_bytes_for_tokens(model: Any, tokens: int) -> int:
    """Total (all layers, all heads) KV bytes for ``tokens`` tokens."""

    _require_nonnegative_int(tokens, "tokens")
    return 2 * model.layers * tokens * model.hidden_size * model.bytes_per_elem


def kv_cache_shard_bytes_for_tokens(
    model: Any,
    tokens: int,
    tp_degree: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV shard bytes for a complete model cache."""

    return kv_cache_shard_bytes_for_layer_range(
        model,
        tokens,
        tp_degree,
        layer_start=0,
        layer_end=model.layers,
    )


# 注记(2026-09-30 审查):manager 侧本函数带 G5-11 的
# @functools.lru_cache(maxsize=4096)(manager 的 model 恒为可哈希
# frozen dataclass);本拷贝不带——本工具重放路径的 model 可为非可哈希
# 命名空间(测试夹具即如此),装饰器会 TypeError。纯函数两版结果严格
# 相等(语义一致),拷贝区同步义务以"函数体逐字一致、装饰器差异如上"
# 口径维护。
def kv_cache_shard_bytes_for_layer_range(
    model: Any,
    tokens: int,
    tp_degree: int,
    *,
    layer_start: int,
    layer_end: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV bytes for ``[layer_start, layer_end)``.

    Layer ranges are derived from ``model.layers``; bytes scale linearly with
    the number of layers so any layer-range split is byte-conservative by
    construction (blueprint sh_2.0 ``face_scheduler.py``)."""

    if tokens < 0:
        raise ValueError("KV token count must be non-negative")
    if (
        isinstance(layer_start, bool)
        or isinstance(layer_end, bool)
        or not isinstance(layer_start, int)
        or not isinstance(layer_end, int)
        or not 0 <= layer_start <= layer_end <= model.layers
    ):
        raise ValueError("KV layer range must be within the configured model")
    if model.hidden_size % model.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    bytes_per_head = (
        2
        * (layer_end - layer_start)
        * tokens
        * (model.hidden_size // model.num_heads)
        * model.bytes_per_elem
    )
    shards = tuple(head_count * bytes_per_head for head_count in heads)
    expected_total = (
        kv_cache_bytes_for_tokens(model, tokens)
        * (layer_end - layer_start)
        // model.layers
    )
    if sum(shards) != expected_total:
        raise RuntimeError(
            "whole-head layer-range KV partition does not preserve total bytes"
        )
    return shards
# ----- 逐字拷贝区结束（以上与 session_kv_manager.py 保持字节一致） -----


DEFAULT_KV_RESERVE_CONTEXT_TOKENS = 1_000_000


def build_model_spec(config: dict) -> Optional[SimpleNamespace]:
    """trace_config config 行 → manager 原函数可用的 model duck-typed 对象。

    与 workload 侧（wsc_llm_scheduler.WscLlmModel / metrics_postprocess._
    load_model_bytes）同一参数化：layers/hidden_size/ffn_size/num_heads/
    vocab_size/bytes_per_elem 必填，mlp_variant 缺省 gelu。任一必填缺失或
    非正 → None（调用方把三口径降级为 NA 并注明，不 fail——旧合成
    fixture/非 LLaMA 配置无这些行，聚合容量链不受影响）。
    """
    values = {}
    for field in ("layers", "hidden_size", "ffn_size", "num_heads",
                  "vocab_size", "bytes_per_elem"):
        raw = config.get(field)
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        values[field] = parsed
    variant = config.get("mlp_variant")
    values["mlp_variant"] = variant if isinstance(variant, str) and variant \
        else "gelu"
    return SimpleNamespace(**values)


def compute_capacity_calibers(model: SimpleNamespace, tp_degree: int,
                              capacity_bytes_per_npu: int,
                              reserve_context_tokens: int) -> dict:
    """三口径容量剖面（P1-③；summary 分列）。

    * 正式认证口径 per_rank_total_hbm：逐 rank physical=weight+resident+
      reserved ≤ capacity_bytes——判决性检查在 journal 重放（行级
      capacity_bytes/before/after）执行，本函数只给静态剖面。
    * resident 硬上限 resident_kv_hard_limit：reservation=0 时任意时刻
      成立的 resident 上限 = min_r ⌊(capacity − weight_r)/kv_r⌋ token ×
      Σkv_r（llama2_7b/swiglu/TP6/160GiB 锚定 1,723,864 → 903,801,208,832）。
    * 水位目标 watermark_reserve_target：enforce_watermark 的
      kv_reserve_context_tokens/rank 扣减后同式（锚定 723,864 →
      379,513,208,832）——水位目标非任意时刻上限，仅报告不作判决。
    """
    weight_shards = model_weight_shard_bytes_by_tp_rank(model, tp_degree)
    kv_shards_per_token = kv_cache_shard_bytes_for_tokens(model, 1, tp_degree)
    reserve_shards = kv_cache_shard_bytes_for_tokens(
        model, reserve_context_tokens, tp_degree)
    # 0 头 shard（num_heads < tp_degree 的退化切分）不持有 KV，对 resident
    # 上限不构成约束——min 只在 kv>0 的 rank 上取。
    hard_tokens = min(
        (capacity_bytes_per_npu - weight) // kv_per_token
        for weight, kv_per_token in zip(weight_shards, kv_shards_per_token)
        if kv_per_token > 0)
    target_tokens = min(
        (capacity_bytes_per_npu - weight - reserve) // kv_per_token
        for weight, reserve, kv_per_token in zip(
            weight_shards, reserve_shards, kv_shards_per_token)
        if kv_per_token > 0)
    # 坏配置（capacity 不足以容纳某 rank 的 weight shard / reserve 超容量）
    # 会使上式为负：负上限无物理意义。本函数是诊断剖面（判决层在 journal
    # 重放），各自独立置 None 并在输出注明，不让负值直通。
    hard_limit_defined = hard_tokens > 0
    target_defined = target_tokens > 0
    if not hard_limit_defined:
        hard_tokens = None
    if not target_defined:
        target_tokens = None
    coef = sum(kv_shards_per_token)
    hard_tokens_by_weight = {}
    ambiguous_weights = set()
    for weight, kv_per_token in zip(weight_shards, kv_shards_per_token):
        if kv_per_token <= 0 or hard_tokens is None:
            continue  # 0 头 shard 无 KV 上限语义；硬限未定义时不构造绑定表
        limit = kv_per_token * hard_tokens
        known = hard_tokens_by_weight.get(weight)
        if known is not None and known != limit:
            ambiguous_weights.add(weight)
        else:
            hard_tokens_by_weight[weight] = limit
    for weight in ambiguous_weights:
        hard_tokens_by_weight.pop(weight, None)
    return {
        "tp_degree": tp_degree,
        "capacity_bytes_per_rank": capacity_bytes_per_npu,
        "weight_shard_bytes_by_relative_rank": list(weight_shards),
        "weight_total_bytes_per_instance": sum(weight_shards),
        "kv_shard_bytes_per_token_by_relative_rank":
            list(kv_shards_per_token),
        "resident_kv_hard_limit": {
            "limit_tokens": hard_tokens,
            "limit_bytes_per_instance":
                None if hard_tokens is None else coef * hard_tokens,
            "limit_resident_bytes_by_relative_rank": [
                None if hard_tokens is None else kv_per_token * hard_tokens
                for kv_per_token in kv_shards_per_token],
            "limit_by_weight_bytes": hard_tokens_by_weight,
            "ambiguous_weight_binding": sorted(ambiguous_weights),
            "hard_limit_defined": hard_limit_defined,
            **({} if hard_limit_defined else {
                "note": "capacity below weight shard on some rank; hard "
                        "limit undefined (negative arithmetic reported as "
                        "null)"}),
            "semantics": "reservation=0 时逐 rank ⌊(capacity−weight_r)/"
                         "kv_r⌋ 最小 token 数 × Σkv_r——任意时刻成立的 "
                         "resident-KV 硬上限（诊断口径）",
        },
        "watermark_reserve_target": {
            "reserve_context_tokens": reserve_context_tokens,
            "reserve_shard_bytes_by_relative_rank": list(reserve_shards),
            "target_tokens": target_tokens,
            "target_bytes_per_instance":
                None if target_tokens is None else coef * target_tokens,
            **({} if target_defined else {
                "note": "reserve context exceeds capacity on some rank; "
                        "target undefined (negative arithmetic reported as "
                        "null)"}),
            "semantics": "kv_reserve_context_tokens/rank 水位目标（无 "
                         "active/reserved 消费者时冷态驻留池预算，非任意"
                         "时刻上限）——仅报告不作判决",
        },
        "per_rank_total_hbm": {
            "invariant": "逐 rank physical = weight + resident + reserved "
                         "≤ capacity_bytes（session_kv_manager 不变量）",
            "verdict_tier": TRUST_TIER_CERTIFIED,
        },
    }


# ---------------------------------------------------------------------------
# 输入装载（fail-closed）
# ---------------------------------------------------------------------------

# B3-6（2026-08-27）：S2 逐出 victim/bytes 序列化（sh20_online_scheduler
# 的 _transfer_entry_rows）落地后的 full 口径升级。判定字段 = 契约分段
# 逐出字段 history_evictions/prefill_evictions（与 hopbytes 单键取用同式
# ：分段字段在场即 B3+ 契约产物，两字段同批发射）。判定不能用
# history_transfers：该字段 B2 就在场（wsc_llm_online_scheduler 的 B2
# 三态契约），而逐出分段字段是 B3 才加——以 history_transfers 触发会把
# B2 窗口产物误升级（升级后只挂 B3 分段逐出字段，B2 的 legacy
# admission_evictions/decode_target_evictions 整类不消费 → 逐出字节全 0
# 且 coverage 谎报 full）；B2 窗口产物必须保底 legacy 路径（wscllm
# legacy 逐出提取器 _eviction_entry_bytes 两形通吃、coverage 保持
# full_reconciled 诚实口径；sh_2.0 保持 count_only 上界口径）。
# B4（-LRU wscllm，2026-09-07）：同一触发字段把升级扩到 wscllm 变体——
# -LRU 新产物逐出/恢复已切契约 §3 序列化（B3），legacy admission_
# evictions/decode_target_evictions 是契约行的同值镜像，两级并存消费会
# 双计；且标量聚合 history_transfer_bytes 在 tiered 语义下把 partial 前缀
# noc（move 段）与 suffix remote_load（add-only 段）混成一个数——按整体
# 搬移重放即击穿占用下界（负占用）。completion 契约行字段名与 sh_2.0
# 同为 completion_evictions（face 侧 B3 命名为 completion_eviction_
# transfers，故参数化区分）。
S2_FULL_TRIGGER_FIELDS = ("history_evictions", "prefill_evictions")


def upgrade_s2_mapping_if_fields_present(mapping: dict, run_dir: Path,
                                         completion_field: str =
                                         "completion_evictions") -> dict:
    """分段逐出字段在场 → full 口径；B3 前产物（含 B2 窗口）→ 原映射。

    full 口径语义（与 S1 对齐）：逐出条目可归因（victim session + bytes +
    source 实例）；restore 按 history_transfers 逐段对账（partial 前缀
    noc_migrate 搬移 + suffix remote_load 只增，标量聚合会把跨实例 partial
    迁移错记成整体搬移而击穿占用下界）；completion 逐出行（enforce_
    reserve 含自身逐出）→ 关闭 kv_location_after_completion 归因避免双重
    扣减；*_eviction_count 不再计入 unattributed（逐出已逐条归因）。
    completion_field 按 repo_variant 取契约行字段名（sh_2.0/wscllm=
    completion_evictions；face=-LRU 的 completion_eviction_transfers）。"""
    log_path = run_dir / DECISION_LOG_RELPATH
    present = False
    lacks_any = False
    for record in iter_jsonl(log_path):
        decision = record.get("decision")
        if record.get("kind") == "prefill" and isinstance(decision, dict):
            carries = any(field in decision
                          for field in S2_FULL_TRIGGER_FIELDS)
            if carries:
                present = True
            else:
                lacks_any = True
    # 混代守卫（2026-10-01 R3 补齐；R4 补双向序）：升级是全局一次性的，
    # 升级后 _collect_eviction_lists 对缺契约字段的旧形状行静默
    # continue——旧行的逐出/恢复双静默清零。单 run 的决策日志以截断
    # 模式单写者产出（单代），混代只可能来自人工拼装（任意拼接序）；
    # "有带字段行且有无字段行"即混代（无论先后），fail-closed 点名。
    mixed_generation = present and lacks_any
    if mixed_generation:
        fail(f"{log_path}: prefill 决策行混代（部分带 "
             f"{sorted(S2_FULL_TRIGGER_FIELDS)} 契约字段、部分不带）——"
             f"全局 S2 升级会静默丢弃旧形状行的逐出/恢复，拒绝重放"
             f"（fail-closed；单 run 产物应为单代，混代通常是人工拼装）")
    if not present:
        return mapping
    upgraded = dict(mapping)
    upgraded["eviction_lists"] = (
        ("prefill", "history_evictions", _scalar_bytes),
        ("prefill", "prefill_evictions", _scalar_bytes),
        ("decode", "decode_evictions", _scalar_bytes),
        ("completion", completion_field, _scalar_bytes),
    )
    upgraded["restore"] = {"restore_list_field": "history_transfers"}
    upgraded["completion_relocation_field"] = None
    upgraded["eviction_count_fields"] = ()
    upgraded["eviction_coverage"] = "full"
    return upgraded


def load_token_manifest(run_dir: Path, explicit: Optional[Path]) -> dict:
    """token 事实 manifest（plan_materializer 产物，requests[] 带
    prefill_context_tokens/final_context_tokens/history_tokens_before）。

    优先级：--token-manifest > run_dir/manifest.json > cpp.log init 行
    manifest_path 同目录的 manifest.json（正式跑在 generated/<plan>/ 下）。
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        local = run_dir / TOKEN_MANIFEST_FILENAME
        if local.is_file():
            candidates.append(local)
        try:
            init = read_init_record(run_dir)
        except SloToolError:
            init = {}
        manifest_path = init.get("manifest_path")
        if manifest_path:
            candidates.append(Path(str(manifest_path)).parent
                              / TOKEN_MANIFEST_FILENAME)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            # load_json_strict（2026-10-01 对齐）：token manifest 是
            # 外部文件，NaN/Infinity 字面量不进本工具链。
            data = load_json_strict(candidate.read_text(encoding="utf-8"),
                                    f"token manifest {candidate}")
        except SloToolError as exc:
            fail(f"token manifest 非法 JSON：{candidate}: {exc}")
        if not isinstance(data, dict) or not isinstance(
                data.get("requests"), list):
            fail(f"token manifest 缺 requests 数组：{candidate}")
        tokens: dict[str, dict] = {}
        for entry in data["requests"]:
            if not isinstance(entry, dict):
                fail(f"token manifest requests 成员必须是对象：{candidate}")
            request_id = entry.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                fail(f"token manifest requests 成员缺 request_id：{candidate}")
            if request_id in tokens:
                fail(f"token manifest request_id 重复：{request_id}")
            row = {}
            for field in ("prefill_context_tokens", "final_context_tokens",
                          "history_tokens_before"):
                value = entry.get(field)
                # bool 冒充门（2026-10-01 对齐，报告 3.4）：
                # isinstance(True, int) 为真，true 曾通过下方的 int 门按
                # 1 token 放行——token 事实被布尔值谎报。bool 显式拒绝，
                # 错误信息说明收到布尔值。
                if isinstance(value, bool):
                    fail(f"token manifest {request_id}.{field} 收到布尔值"
                         f"（{value!r}）——bool 是 int 子类但不是合法 "
                         f"token 数（{candidate}）")
                if not isinstance(value, int) or value < 0:
                    fail(f"token manifest {request_id} 缺非负整数 {field}"
                         f"（{candidate}）")
                row[field] = value
            row["session_id"] = entry.get("session_id")
            if not isinstance(row["session_id"], str):
                fail(f"token manifest {request_id} 缺 session_id"
                     f"（{candidate}）")
            tokens[request_id] = row
        if not tokens:
            fail(f"token manifest requests 为空：{candidate}")
        return {"path": str(candidate), "requests": tokens}
    fail(
        f"找不到 token manifest（尝试：{[str(c) for c in candidates]}）——"
        f"hbm_watermark 需要 manifest.json（plan_materializer 产物，含 "
        f"requests[].prefill_context_tokens/final_context_tokens/"
        f"history_tokens_before）；用 --token-manifest 显式指定")


def _read_trace_config(path: Path) -> dict:
    config: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0] != "config" or len(row) < 3:
                continue
            if row[1] in config:
                # 同名 config 行 last-wins 会让容量/模型参数静默取决于行序
                # ——fail-closed 拒绝（配置文件自身损坏，不得猜测取哪行）。
                fail(f"trace_config config 行重复键 {row[1]!r}：{path}"
                     f"——拒绝 last-wins 静默取尾")
            config[row[1]] = row[2]
    return config


def resolve_trace_config_path(run_dir: Path, explicit: Optional[Path],
                              allow_repo_default: bool = True) -> Path:
    """trace_config.csv 定位（load_trace_config / count_configured_
    instances 共用的同一候选链，单一来源）。

    优先级：--trace-config > run_dir/trace_config.csv >
    run_dir/trace_config.csv.snapshot（B0 归档名）> 脚本所在仓的
    sh_test_mesh/workload/llama2_7b_inference/trace_config.csv。

    allow_repo_default=False 时仓内默认回退不可用（缺 run_dir 本地副本
    即 fail-closed）——load_imbalance 的 CV 分母走此档：分母取自错误
    布局的默认配置会静默膨胀（CV 失真），宁可拒绝。
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        candidates.append(run_dir / "trace_config.csv")
        candidates.append(run_dir / "trace_config.csv.snapshot")
        if allow_repo_default:
            candidates.append(
                Path(__file__).resolve().parent.parent
                / "workload" / "llama2_7b_inference" / "trace_config.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    fail(f"找不到 trace_config.csv（尝试：{[str(c) for c in candidates]}）"
         f"——用 --trace-config 指定")


def count_configured_instances(run_dir: Path,
                               explicit: Optional[Path] = None,
                               allow_repo_default: bool = True) -> int:
    """配置实例总数：trace_config.csv 的 inference_group 组行数。

    plan_materializer 的 manifest 均不携带实例/组清单（manifest.json 只
    有队列派生 9 字段；metrics_manifest.json 的 prefill/decode_instance
    与 *_ranks 是 instance-0 占位——静态归因维度不可信），配置布局的
    权威来源是 trace_config.csv 的 inference_group 行（本仓标准布局
    9 组 × 6 rank）。禁止从 decision log 推导（空闲实例不出现在 log，
    从 log 推导会复现"分母丢失空闲实例"的旧错）。"""
    path = resolve_trace_config_path(run_dir, explicit,
                                     allow_repo_default=allow_repo_default)
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row and row[0] == "inference_group":
                count += 1
    if count <= 0:
        fail(f"trace_config 无 inference_group 组行（{path}）——"
             f"配置实例总数无法确定")
    return count


def load_trace_config(run_dir: Path, explicit: Optional[Path]) -> dict:
    """trace_config.csv 的 config 行（layers/hidden_size/bytes_per_elem/
    local_hbm_capacity_profile）。候选链见 resolve_trace_config_path。
    """
    candidate = resolve_trace_config_path(run_dir, explicit)
    config = _read_trace_config(candidate)
    for field in ("layers", "hidden_size", "bytes_per_elem"):
        if field not in config:
            fail(f"trace_config 缺 config 行 {field}：{candidate}")
        try:
            config[field] = int(config[field])
        except ValueError:
            fail(f"trace_config {field} 不是整数：{candidate}")
        if config[field] <= 0:
            fail(f"trace_config {field} 必须为正：{candidate}")
    return {"path": str(candidate), "values": config}


def resolve_hardware_capacity(profile: str, explicit: Optional[Path]) -> dict:
    """hardware json 的 capacity-profiles[profile].bytes（每 NPU）。

    优先级：--hardware-config > 脚本所在仓 sh_test_mesh/hardware/*.json
    （逐个找含该 profile 的 canonical hardware source）。
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        hardware_dir = (Path(__file__).resolve().parent.parent
                        / "hardware")
        if hardware_dir.is_dir():
            candidates.extend(sorted(hardware_dir.glob("*.json")))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            data = load_json_strict(candidate.read_text(
                encoding="utf-8"), str(candidate))
        except (json.JSONDecodeError, SloToolError):
            # NaN/Infinity 字面量（load_json_strict 拒绝）与坏 JSON 同等
            # 处理：候选文件跳过（2026-10-01 对齐——裸 json.loads 会把
            # NaN/Inf 静默放行进容量/档位解析）。
            continue
        if not isinstance(data, dict):
            continue
        local_hbm = data.get("local-hbm")
        if not isinstance(local_hbm, dict):
            continue
        profiles = local_hbm.get("capacity-profiles")
        if not isinstance(profiles, dict) or profile not in profiles:
            continue
        entry = profiles[profile]
        if not isinstance(entry, dict):
            fail(f"hardware 容量档 {profile!r} 结构错误：{candidate}")
        value = entry.get("bytes")
        # bool 冒充门（2026-10-01 对齐，报告 3.4）：isinstance(True, int)
        # 为真，capacity_bytes=true 曾按 1 字节容量放行——污染 calibers 与
        # decision-log 诊断（journal 行内 capacity 已排 bool，certified
        # 判决不受此影响）。bool 显式拒绝，错误信息说明收到布尔值。
        if isinstance(value, bool):
            fail(f"hardware 容量档 {profile!r}.bytes 收到布尔值（{value!r}）"
                 f"——bool 是 int 子类但不是合法容量，不放行：{candidate}")
        if not isinstance(value, int) or value <= 0:
            fail(f"hardware 容量档 {profile!r}.bytes 非正整数：{candidate}")
        return {"path": str(candidate), "profile": profile, "bytes": value}
    fail(f"hardware 配置中找不到容量档 {profile!r}"
         f"（尝试：{[str(c) for c in candidates]}）——用 --hardware-config "
         f"指定含 local-hbm.capacity-profiles 的 json；无法解析时本命令"
         f"拒绝执行（不编造容量）")


def load_npus_per_instance(run_dir: Path, explicit: Optional[Path],
                           manifest_loader=None) -> int:
    """每实例 NPU 数：request manifest requests[].prefill_ranks 长度。

    优先级：--request-manifest > run_dir/metrics_manifest.json >
    cpp.log init 行 manifest_path。manifest_loader（A4 driver 传入的
    惰性装载器，返回已装载 manifest）在场时复用（读放大收敛）；其
    SloToolError 与自装载同路处理（fail → 调用方降级警告），失败时点
    不变。
    """
    loader = manifest_loader if manifest_loader is not None else (
        lambda: load_request_manifest(run_dir, explicit))
    try:
        manifest = loader()
    except SloToolError as exc:
        fail(f"无法定位 request manifest（npus_per_instance 推导失败）："
             f"{exc}")
    requests = manifest_requests(manifest)
    sizes: set[int] = set()
    for entry in requests:
        ranks = entry.get("prefill_ranks")
        if isinstance(ranks, list) and ranks:
            sizes.add(len(ranks))
    if len(sizes) != 1:
        fail(f"request manifest prefill_ranks 长度不一致（{sorted(sizes)}）"
             f"——npus_per_instance 无法唯一确定")
    return sizes.pop()


def load_bucket_ns(manifest: dict) -> tuple[int, bool]:
    """桶长：manifest watermark_sample_period_ns；null/缺 → 临时锚点。

    返回 (bucket_ns, provisional)。锚点 5ms 为 WP8 开销预算未标定前的
    临时值，全输出标 provisional（B4 推导后自动转为正式值）。
    """
    params = manifest.get("params") or {}
    entry = params.get("watermark_sample_period_ns")
    value = entry.get("value") if isinstance(entry, dict) else None
    if value is None:
        print(
            f"[hbm-watermark] 警告：watermark_sample_period_ns 未推导"
            f"（null）——使用临时锚点 {PROVISIONAL_BUCKET_NS} ns，"
            f"全输出标 provisional=true（B4 填充后自动生效）",
            file=sys.stderr)
        return PROVISIONAL_BUCKET_NS, True
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or value != int(value) or int(value) <= 0:
        fail(f"watermark_sample_period_ns 必须为正整数 ns（实得 {value!r}）")
    return int(value), False


# ---------------------------------------------------------------------------
# 重放引擎
# ---------------------------------------------------------------------------

class _ChangePoint:
    """单变点：同 tick 的占用事件归并为一行（RLE 的原子；逐出标记在
    _UnitLog.evicts 独立成表，merged_change_points 归并时叠加同 tick 行）。"""

    __slots__ = ("tick", "start", "end", "peak")

    def __init__(self, tick: int, start: int) -> None:
        self.tick = tick
        self.start = start  # 本 tick 首事件前占用
        self.end = start
        self.peak = start  # 本 tick 内事件后占用的最大值（含进入值）


class _UnitLog:
    """单实例（journal 路径=journal instance；decision-log 路径=跟踪实例）
    的变点序列 + 事件流 stats（P1-②：与桶长/跨度彻底解耦）。"""

    __slots__ = ("points", "evicts", "open", "occupancy", "has_event",
                 "first_tick", "last_tick", "prev_tick", "peak", "area")

    def __init__(self) -> None:
        self.points: list[_ChangePoint] = []
        self.evicts: dict[int, list[int]] = {}  # tick -> [events, bytes]
        self.open: Optional[_ChangePoint] = None
        self.occupancy = 0
        self.has_event = False
        self.first_tick = -1
        self.last_tick = -1
        self.prev_tick = -1
        self.peak = 0
        self.area = 0  # Σ(occupancy × dt)，整数累计，展示层才转浮点

    def add(self, tick: int, delta: int) -> None:
        if self.open is None or self.open.tick != tick:
            if tick < self.prev_tick:
                # 单实例变点流必须单调不减（decision-log 记录级 tick 检查
                # 不覆盖逐出条目 time_ns 的生效 tick）：回退会使 area 累入
                # 负 dt、points 乱序破坏桶扫游标假设——fail-closed 不静默。
                fail(f"变点 tick 回退（{tick} < {self.prev_tick}）——同一"
                     f"实例事件序非单调（逐出条目 time_ns 早于已处理变点？）"
                     f"，拒绝产生负 dt 面积")
            if not self.has_event:
                self.has_event = True
                self.first_tick = tick
            else:
                self.area += self.occupancy * (tick - self.prev_tick)
            self.prev_tick = tick
            self.open = _ChangePoint(tick, self.occupancy)
            self.points.append(self.open)
            self.last_tick = tick
        self.occupancy += delta
        self.open.end = self.occupancy
        if self.occupancy > self.open.peak:
            self.open.peak = self.occupancy
        if self.occupancy > self.peak:
            self.peak = self.occupancy

    def add_evict(self, tick: int, nbytes: int) -> None:
        """逐出标记独立成表（tick 可早于/晚于事件 tick，统计与旧实现
        一致：计入总量、参与锚桶，不触碰事件流 stats 的首末/面积）。"""
        pair = self.evicts.get(tick)
        if pair is None:
            self.evicts[tick] = [1, nbytes]
        else:
            pair[0] += 1
            pair[1] += nbytes

    def add_evict_row(self, tick: int, nbytes: int,
                      new_event: bool) -> None:
        """journal 逐出行记账（M-31 会话口径）：事件计数仅在该行的会话
        是本事务内首次出现时 +1（同会话一次逐出跨 rank 落多行，其余行
        只累加字节）；decision-log 路径的 add_evict（一次调用=一个事件）
        不变。"""
        pair = self.evicts.get(tick)
        if pair is None:
            self.evicts[tick] = [1 if new_event else 0, nbytes]
        else:
            if new_event:
                pair[0] += 1
            pair[1] += nbytes

    def stats(self) -> dict:
        duration = self.last_tick - self.first_tick
        evict_events = sum(pair[0] for pair in self.evicts.values())
        evict_bytes = sum(pair[1] for pair in self.evicts.values())
        return {
            "first_event_ns": self.first_tick,
            "last_event_ns": self.last_tick,
            "duration_ns": duration,
            "peak_occupancy_bytes": self.peak,
            # int→float 溢出守卫（2026-10-01 对齐）：超大字节（畸形账本
            # delta）的真除溢出走 fail-closed 出口（SloToolError → 退出码
            # 2），不再裸 OverflowError 栈（run_main 不捕它、退出码语义被
            # 破坏为 1）。
            "mean_occupancy_bytes": _safe_mean_float(self.area, duration,
                                                     self.peak),
            "residual_occupancy_bytes": self.occupancy,
            "evict_events": evict_events,
            "evict_bytes": evict_bytes,
        }

    def merged_change_points(self):
        """事件变点 ∪ 逐出 tick 的有序归并流（生成器，不物化第二份）。

        逐出 tick 落在事件之间时占用 = 前一事件后占用（const 段），其
        delta=0；逐出 tick 与事件 tick 相同时并入该变点行（evict 列叠
        加）——保证从本流可无损恢复任意桶长序列（含逐出列）。
        """
        evict_items = sorted(self.evicts.items())
        evict_index = 0
        occupancy = 0
        point_index = 0
        points = self.points
        evict_count = len(evict_items)
        while point_index < len(points) or evict_index < evict_count:
            point_tick = points[point_index].tick \
                if point_index < len(points) else None
            evict_tick = evict_items[evict_index][0] \
                if evict_index < evict_count else None
            if point_tick is not None and (evict_tick is None
                                           or point_tick < evict_tick):
                point = points[point_index]
                point_index += 1
                occupancy = point.end
                yield (point.tick, point.start, point.end, point.peak,
                       0, 0)
            elif evict_tick is not None and (point_tick is None
                                             or evict_tick < point_tick):
                tick, pair = evict_items[evict_index]
                evict_index += 1
                yield (tick, occupancy, occupancy, occupancy,
                       pair[0], pair[1])
            else:
                # 同 tick：事件变点行叠加逐出列。
                point = points[point_index]
                pair = evict_items[evict_index][1]
                point_index += 1
                evict_index += 1
                occupancy = point.end
                yield (point.tick, point.start, point.end, point.peak,
                       pair[0], pair[1])


class ChangePointLog:
    """全部实例的变点日志（P1-④：series 行字典全量物化就此消灭）。

    add(tick, instance, delta) 要求 tick 对同一实例单调不减（_UnitLog.add
    内的 fail-closed 断言强制——记录级 tick 检查不覆盖逐出条目 time_ns
    的生效 tick；journal 路径另有 planner_time_ns 单调检查）；逐出标记
    走独立 per-unit 表。内存 O(变点数+实例数)。
    """

    def __init__(self) -> None:
        self.units: dict[int, _UnitLog] = {}

    def _unit(self, instance: int) -> _UnitLog:
        unit = self.units.get(instance)
        if unit is None:
            unit = _UnitLog()
            self.units[instance] = unit
        return unit

    def add(self, tick: int, instance: int, delta: int) -> None:
        self._unit(instance).add(tick, delta)

    def add_evict(self, tick: int, instance: int, nbytes: int) -> None:
        self._unit(instance).add_evict(tick, nbytes)

    def add_evict_row(self, tick: int, instance: int, nbytes: int,
                      new_event: bool) -> None:
        self._unit(instance).add_evict_row(tick, nbytes, new_event)

    def has_events(self) -> bool:
        return any(unit.has_event for unit in self.units.values())

    def eventful_units(self) -> list[int]:
        return sorted(unit for unit, log in self.units.items()
                      if log.has_event)

    def span(self) -> tuple[Optional[int], Optional[int]]:
        """全局事件跨度（首/末 KV 事件 tick；仅统计有事件实例）。"""
        first: Optional[int] = None
        last: Optional[int] = None
        for log in self.units.values():
            if not log.has_event:
                continue
            if first is None or log.first_tick < first:
                first = log.first_tick
            if last is None or log.last_tick > last:
                last = log.last_tick
        return first, last


def bucket_row_sweep(change_points_factory, origin: int, span_end: int,
                     bucket_ns: int):
    """变点流 → 桶行流（P1-④：游标推进时逐行 yield，不物化行字典）。

    change_points_factory：零参 callable，每次调用返回一份新鲜的变点迭代
    器（_UnitLog.merged_change_points 即是；单测的 intervals.csv 无损恢复
    同样喂本函数）——两遍消费（先锚桶后扫行）均流式，除锚桶索引集合外
    不新增整表物化。语义与旧 bucketize 逐行等价：桶 j 覆盖
    [origin + j·B, +B)；事件 tick 落在 [bucket_start, bucket_end) 计入
    桶 j，事件吸收窗的 bucket_end 钳到 span_end+1（含恰落在 span_end
    的事件；实现内局部变量即此钳制值）；产出的桶行 bucket_end_ns 列
    保持名义网格边界 bucket_start+bucket_ns 不钳（末桶可 > span_end+1
    ——与旧 bucketize 产出及逐行等价对拍测试一致，消费方按
    bucket_index/bucket_start 定位，不依赖末桶 end 列的物理意义）。
    锚桶 = 有变点（事件或逐出）的桶位，锚桶间无变点且占用>0 的桶位补
    占位行（占用保持段）；出界（负桶号）变点只进统计不产行。迭代器
    元素 = (tick, occ_start, occ_end, occ_peak, evict_events,
    evict_bytes)。
    """
    n_buckets = (span_end - origin) // bucket_ns + 1

    def bucket_index(tick: int) -> int:
        # 与旧实现对齐：逐出 tick 越上界的钳到末桶；事件 tick 自然在界内。
        return min((tick - origin) // bucket_ns, n_buckets - 1)

    anchors: list[int] = []
    seen: set[int] = set()
    for item in change_points_factory():
        index = bucket_index(item[0])
        if index >= 0 and index not in seen:
            seen.add(index)
            anchors.append(index)
    anchors.sort()
    cursor_item = None
    stream = iter(change_points_factory())

    def advance():
        nonlocal cursor_item
        cursor_item = next(stream, None)
        return cursor_item is not None

    advance()
    # 出界（tick 早于全局首事件）的变点只进统计不产行——在锚桶扫描前丢弃
    # （与旧实现的负桶号过滤一致；stats 的逐出总量在 _UnitLog 统计）。
    while cursor_item is not None and cursor_item[0] < origin:
        advance()
    occupancy = 0
    for pos, index in enumerate(anchors):
        bucket_start = origin + index * bucket_ns
        bucket_end = min(bucket_start + bucket_ns, span_end + 1)
        bucket_peak = occupancy
        evict_events = 0
        evict_bytes = 0
        while cursor_item is not None \
                and min(cursor_item[0], span_end) < bucket_end:
            _, _, occ_end, occ_peak, ev_n, ev_b = cursor_item
            occupancy = occ_end
            if occ_peak > bucket_peak:
                bucket_peak = occ_peak
            evict_events += ev_n
            evict_bytes += ev_b
            advance()
        if occupancy > 0 or bucket_peak > 0 or evict_events:
            yield (index, bucket_start, bucket_start + bucket_ns, occupancy,
                   bucket_peak, evict_events, evict_bytes)
        next_bound = anchors[pos + 1] if pos + 1 < len(anchors) \
            else n_buckets
        for fill in range(index + 1, next_bound):
            if occupancy <= 0:
                break
            fill_start = origin + fill * bucket_ns
            yield (fill, fill_start, fill_start + bucket_ns, occupancy,
                   occupancy, 0, 0)


class SessionState:
    """会话跟踪态：当前实例 + 当前本地 bytes（分层仓可为部分层）。"""

    __slots__ = ("instance", "bytes")

    def __init__(self) -> None:
        self.instance: Optional[int] = None
        self.bytes: int = 0


class ReplayReport:
    def __init__(self) -> None:
        self.actions = {
            "prefill_grow": 0, "restore_move": 0, "restore_remote_add": 0,
            "restore_local_add": 0, "restore_none": 0,
            "decode_move": 0, "decode_stay": 0, "decode_grow": 0,
            "evictions": 0, "evict_bytes": 0, "partial_evictions": 0,
            "own_relocation_remote": 0, "own_relocation_partial_unknown": 0,
            "unattributed_evictions": 0,
            "silent_evictions_reconciled": 0,
            "silent_eviction_bytes": 0,
            "restore_prefix_reconciled": 0,
            "restore_prefix_reconcile_bytes": 0,
        }
        self.anomalies = {
            "restore_bytes_mismatch": 0, "restore_source_mismatch": 0,
            "restore_kind_mismatch": 0, "move_bytes_mismatch": 0,
            "move_source_mismatch": 0, "local_hit_location_mismatch": 0,
            "evict_location_mismatch": 0,
            "evict_untracked_session": 0, "evict_exceeds_tracked": 0,
        }
        self.restore_ratio_hist: dict[str, int] = {}


class WatermarkReplay:
    """按文件顺序重放决策记录，产出逐实例 delta 事件流与违规计数。"""

    def __init__(self, coef_bytes_per_token: int,
                 capacity_per_instance: Optional[int],
                 model_layers: Optional[int] = None) -> None:
        self.coef = coef_bytes_per_token
        self.capacity = capacity_per_instance
        # trace_config 模型层数（report-only partial_evictions 层域判定的
        # 全量上界；缺席=None 时仅 layer_start>0 的后缀半层可判，见
        # report_evictions_only）。
        self.model_layers = model_layers
        self.sessions: dict[str, SessionState] = {}
        # P1-②/④：事件/逐出标记直入变点日志（RLE），不再物化裸事件表；
        # stats 从变点归并时 O(动作数) 计算，与桶长解耦。
        self.cplog = ChangePointLog()
        self.occupancy: dict[int, int] = {}
        self.violation_events = 0
        self.violation_by_instance: dict[int, int] = {}
        self.max_exceed_bytes = 0
        self.report = ReplayReport()

    # -- 基元 -----------------------------------------------------------

    def _apply(self, tick: int, instance: int, delta: int,
               evict_bytes: Optional[int] = None) -> None:
        where = f"tick={tick} instance={instance}"
        current = self.occupancy.get(instance, 0) + delta
        if current < 0:
            fail(f"重放出现负占用（{where}: {current}）——重建口径与账本"
                 f"不一致（多扣/漏加），拒绝输出错误水位线；请核对"
                 f"REPO_VARIANTS 映射与输入 ledger")
        self.occupancy[instance] = current
        self.cplog.add(tick, instance, delta)
        if evict_bytes is not None:
            self.cplog.add_evict(tick, instance, evict_bytes)
        if self.capacity is not None and current > self.capacity:
            # decision-log 路径的违规计数恒为诊断口径（P1-①：本路径属
            # upper_bound_only 层，不构成物理违规认证、不触发退出码 3）。
            self.violation_events += 1
            self.violation_by_instance[instance] = \
                self.violation_by_instance.get(instance, 0) + 1
            self.max_exceed_bytes = max(self.max_exceed_bytes,
                                        current - self.capacity)

    def _session(self, session_id: str) -> SessionState:
        return self.sessions.setdefault(session_id, SessionState())

    def _expected_bytes(self, tokens: int) -> int:
        return self.coef * tokens

    # -- 动作 ------------------------------------------------------------

    def reconcile_stale_resident(self, tick: int, session_id: str,
                                 local_bytes_expected: int) -> None:
        """B4（-LRU wscllm tiered，evict_report_only 配套）：恢复点对账。

        admission 时点的物理本地驻留 = local_bytes_expected（remote_memory
        → 0；partial_hbm_remote → 驻留前缀 = history_transfers 的 noc 段
        bytes）。逐出条目未扣占用（report-only，到时序晚于 victim 其间的
        生命周期事件），跟踪 bytes 超出部分属已逐出的旧代账——在恢复点
        从跟踪实例对账扣减（同 W 的 EVICTED 口径，占用上界收紧到物理
        值），会话 bytes 同步收敛；后续逐段 restore 按真实搬移重放。
        """
        session = self._session(session_id)
        if session.bytes > local_bytes_expected \
                and session.instance is not None:
            excess = session.bytes - local_bytes_expected
            self._apply(tick, session.instance, -excess)
            session.bytes = local_bytes_expected
            if local_bytes_expected == 0:
                # 收敛到 0 时同步清跟踪实例：后续 restore 段走远端回载
                # add-only 分支（不清则会按跨实例搬移在源实例二次扣减）。
                session.instance = None
            self.report.actions["silent_evictions_reconciled"] += 1
            self.report.actions["silent_eviction_bytes"] += excess

    def report_evictions_only(self, actions: list) -> None:
        """B4（-LRU wscllm tiered）：逐出条目 report-only 口径。

        净额准入重试把多次尝试的逐出累积到最终获准请求的行上，条目的
        到达时序（该请求的 decode 行）晚于 victim 会话其间的恢复/迁移
        事件——按时序消费会扣错实例（victim 已迁移）或双扣旧代账
        （恢复已隐式核销），决策日志单源无法区分。故条目只进诊断账
        （evict_events/evict_bytes，变点日志的 evict 标记按条目登记的
        source 实例），不动占用与会话跟踪态（占用=上界）；旧代驻留在
        victim 下次恢复点按 history_location_before=remote_memory 对账
        核销（apply_restore）。权威占用 = journal 重放（tier>=resident）。
        （R5-batch3，2026-10-01：第二形参 trigger 在函数体内零使用，
        与 apply_evict 不同——它只进诊断账不消费跟踪态，无 trigger 报错
        场景，死形参已删。）

        partial_evictions 判定（py-slo-E2，2026-09-26）：不再对比条目
        bytes 与跟踪 session.bytes——跟踪态在条目晚到时已是 victim 下一
        代际的数值，`0 < bytes < session.bytes` 对新运行整体逐出恒真
        （实证复现=1），违反"partial_evictions 新运行恒 0"契约。改用条
        目层域：层域子集（layer_start>0，或 layer_end<模型层数）才判
        partial；二态模型整会话逐出层域恒全量 [0,L) → 恒不判（契约恒
        0）；层域缺席（旧产物）宁缺勿造同样不判。
        """
        for action in actions:
            session = self.sessions.get(action["session"])
            instance = action["instance"]
            if not isinstance(instance, int):
                instance = session.instance if session is not None else None
            self.report.actions["evictions"] += 1
            self.report.actions["evict_bytes"] += action["bytes"]
            layer_start = action.get("layer_start")
            layer_end = action.get("layer_end")
            if (isinstance(layer_start, int) and layer_start > 0) or (
                    isinstance(layer_end, int)
                    and self.model_layers is not None
                    and layer_end < self.model_layers):
                self.report.actions["partial_evictions"] += 1
            self.cplog.add_evict(
                action["tick"], instance if isinstance(instance, int)
                else -1, action["bytes"])

    def apply_evict(self, action: dict, trigger: str) -> None:
        session = self._session(action["session"])
        if session.instance is None or session.bytes <= 0:
            # 逐出对象不在跟踪态：要么此前已全部离开（重复/漏记），要么
            # 重建漏了它的进入——都属结构不一致，宁可报错。
            self.report.anomalies["evict_untracked_session"] += 1
            fail(f"逐出对象 {action['session']!r} 不在跟踪态"
                 f"（tick={action['tick']}, trigger={trigger}, "
                 f"bytes={action['bytes']}）——重建漏记该会话的进入，"
                 f"或账本重复逐出；拒绝继续")
        victim_instance = action["instance"]
        if victim_instance is not None and victim_instance != session.instance:
            self.report.anomalies["evict_location_mismatch"] += 1
        if action["bytes"] > session.bytes:
            self.report.anomalies["evict_exceeds_tracked"] += 1
            fail(f"逐出 bytes {action['bytes']} 超过会话 "
                 f"{action['session']!r} 跟踪 bytes {session.bytes}"
                 f"（tick={action['tick']}）——账本与重放不一致")
        if action["bytes"] < session.bytes:
            # 部分逐出（旧变体 apply_evict 路径专用：face 的 shard 子集
            # 逐出/旧分层产物层域子集；本仓 B4 新产物走 report-only，
            # 不经此分支——其 partial 判定见 report_evictions_only）。
            self.report.actions["partial_evictions"] += 1
        self._apply(action["tick"], session.instance, -action["bytes"],
                    evict_bytes=action["bytes"])
        session.bytes -= action["bytes"]
        if session.bytes == 0:
            session.instance = None
        self.report.actions["evictions"] += 1
        self.report.actions["evict_bytes"] += action["bytes"]

    def apply_restore(self, action: dict, history_tokens: int) -> None:
        session = self._session(action["session"])
        target = action.get("target")
        if not isinstance(target, int):
            fail(f"restore 动作缺整数目标实例（tick={action['tick']}）")
        nbytes = action["bytes"]
        expected = self._expected_bytes(history_tokens)
        kind = action.get("kind")
        state_before = action.get("state_before")
        if nbytes:
            try:
                ratio = nbytes / expected if expected else None
            except OverflowError:
                # 溢出守卫（2026-10-01 对齐）：畸形超大 bytes 的比值计算
                # 走 fail-closed（同 stats 的 OverflowError 处置）。
                fail(f"restore bytes 超出浮点可表示范围（{nbytes}）——"
                     f"账本含畸形超大 delta，拒绝重放")
            bucket = (f"{ratio:.3f}" if ratio is not None else "no-tokens")
            self.report.restore_ratio_hist[bucket] = \
                self.report.restore_ratio_hist.get(bucket, 0) + 1
            # G7-5（2026-09-26）：移除旧分层模型的 0.5 半层恢复比值放行
            # 残片——本仓二态冷热管理下恢复段 bytes 恒等于 f(history)
            #（ratio=1.0）；仅 legacy 旧产物（sh_2.0/S3 半层分段）会出现
            # 0.5 比值，按 restore_bytes_mismatch 计诊断（重放占用数学
            # 不变，分段恢复/对账分支照常工作）。
            if ratio is None or not math.isclose(ratio, 1.0, rel_tol=1e-9):
                self.report.anomalies["restore_bytes_mismatch"] += 1
        source = session.instance
        logged_source = action.get("logged_source")

        # -- 账本断言对账（W/S3 静默释放缺口，见 REPO_VARIANTS 注记）------
        # RECOMPUTE 且账本断言 state_before=EVICTED：恢复前本地应为 0——
        # 若跟踪态仍有驻留，说明其间发生了未落盘的静默逐出（真实逐出时刻
        # ∈ (上次可见事件, 本 tick]，占用在该窗口内为上界），在此对账扣减。
        if kind in ("RECOMPUTE", "recompute") or \
                state_before == "EVICTED":
            if session.bytes > 0 and source is not None:
                self._apply(action["tick"], source, -session.bytes)
                self.report.actions["silent_evictions_reconciled"] += 1
                self.report.actions["silent_eviction_bytes"] += session.bytes
            elif session.bytes > 0:
                fail(f"会话 {action['session']!r} 跟踪 bytes>0 但无实例"
                     f"（tick={action['tick']}）——重放内部状态损坏")
            session.bytes = 0
            session.instance = None
            source = None

        # 显式 kind 优先（S1 的 local_hit.total_bytes 是"本地复用 KV 大小"
        # 信息量而非搬移量——基线实测 174 例 local_hit 携带 f(h) bytes，
        # 误当搬移会把占用翻倍）。
        if kind in ("LOCAL_HIT", "local_hit"):
            if nbytes and nbytes != session.bytes:
                self.report.anomalies["restore_bytes_mismatch"] += 1
            if source != target:
                self.report.anomalies["local_hit_location_mismatch"] += 1
                # HBM-05：location 矛盾不得裸转移会话归属——跟踪 bytes 仍
                # 挂在源实例时，改挂目标会让源残留幻影占用、后续增长/逐出
                # 在目标侧双计或漏计（负占用兜底来不及的结构缺口）。持有
                # 驻留即拒绝（账本与重放状态矛盾，fail-closed）；无驻留
                # （bytes==0 或未跟踪）仅记 anomaly 后照常转移。
                if source is not None and session.bytes > 0:
                    fail(f"local_hit 会话 {action['session']!r} 声称位于"
                         f"实例 {target}，跟踪态仍在实例 {source} 且持有 "
                         f"{session.bytes}B（tick={action['tick']}）——"
                         f"位置矛盾且占用未转移，拒绝继续")
            self.report.actions["restore_none"] += 1
            session.instance = target  # 会话归属目标实例（增长在目标发生）
            return
        if kind in ("NO_HISTORY", "RECOMPUTE", "recompute", "no_history") \
                and nbytes:
            self.report.anomalies["restore_kind_mismatch"] += 1
        # S3 半层恢复（bytes = suffix）：恢复前本地应恰持有 f(h) − bytes
        # （prefix 驻留）；跟踪态超出该值的部分即未落盘的静默 suffix 释放，
        # 在此对账扣减（字节守恒；时刻同样为上界窗口）。恢复 bytes 以
        # f(h) 为上界——suffix 不可能大于整体，越界即账本损坏；若放行，
        # expected − nbytes < 0 会把对账 excess 抬过 session.bytes、使其
        # 转负继续参与守恒（M6-04），故先断言边界，fail-closed。
        if nbytes and nbytes > expected:
            fail(f"会话 {action['session']!r} 恢复 bytes {nbytes} 超过 "
                 f"f(history)={expected}（tick={action['tick']}）——"
                 f"suffix 大于整体，账本损坏，拒绝继续")
        if nbytes and source == target and session.bytes > expected - nbytes:
            excess = session.bytes - (expected - nbytes)
            self._apply(action["tick"], target, -excess)
            session.bytes -= excess
            self.report.actions["restore_prefix_reconciled"] += 1
            self.report.actions["restore_prefix_reconcile_bytes"] += excess

        if nbytes == 0:
            self.report.actions["restore_none"] += 1
            session.instance = target  # 会话归属目标实例（增长在目标发生）
            return
        if source is None:
            # 远端回载（remote_load / S3 半层回载）：只增目标。
            self._apply(action["tick"], target, nbytes)
            session.bytes += nbytes
            session.instance = target
            self.report.actions["restore_remote_add"] += 1
            if kind in ("NOC_MIGRATE", "noc_migrate"):
                self.report.anomalies["restore_kind_mismatch"] += 1
            if logged_source is not None:
                self.report.anomalies["restore_source_mismatch"] += 1
            return
        if source == target:
            # 同实例：分层半层回载（S3 partial）——只增，不扣源。
            self._apply(action["tick"], target, nbytes)
            session.bytes += nbytes
            self.report.actions["restore_local_add"] += 1
            if kind in ("NOC_MIGRATE", "noc_migrate"):
                self.report.anomalies["restore_kind_mismatch"] += 1
            return
        # 跨实例搬移（noc_migrate 全量）。
        if logged_source is not None and logged_source != source:
            self.report.anomalies["restore_source_mismatch"] += 1
        self._apply(action["tick"], source, -nbytes)
        self._apply(action["tick"], target, nbytes)
        session.instance = target
        self.report.actions["restore_move"] += 1

    def apply_move(self, action: dict, prefill_tokens: int) -> None:
        session = self._session(action["session"])
        target = action.get("target")
        if not isinstance(target, int):
            fail(f"move 动作缺整数目标实例（tick={action['tick']}）")
        if session.instance is None:
            fail(f"decode 迁移时会话 {action['session']!r} 无跟踪实例"
                 f"（tick={action['tick']}）——重放次序或账本缺失")
        source = session.instance
        nbytes = action["bytes"]
        # D2 修复：跨实例零字节交割不再按 stay 静默放行——账本宣称没有
        # 迁移、但会话跟踪 bytes 仍挂在源实例且归属要切到目标实例，源
        # 实例占用从此永不回收（幻影占用零异常）。与同函数"无跟踪实例/
        # bytes 超跟踪"两处 fail 同姿态；会话本无驻留（bytes==0）时仅
        # 实例归属切换，仍按 stay 计。
        if nbytes == 0 and source != target and session.bytes > 0:
            fail(f"decode 零字节交割但会话 {action['session']!r} 在源实例 "
                 f"{source} 仍跟踪 {session.bytes} B（tick={action['tick']}，"
                 f"目标实例 {target}）——账本宣称无迁移而驻留未清，源实例"
                 f"占用将成幻影；拒绝重放")
        if source == target or nbytes == 0:
            # 同实例交割（S1/S2 P==D 时仍记 f(c)）：占用不变。
            self.report.actions["decode_stay"] += 1
            session.instance = target
            return
        if nbytes != self._expected_bytes(prefill_tokens):
            self.report.anomalies["move_bytes_mismatch"] += 1
        logged_source = action.get("logged_source")
        if logged_source is not None and logged_source != source:
            self.report.anomalies["move_source_mismatch"] += 1
        if nbytes > session.bytes:
            fail(f"decode 迁移 bytes {nbytes} 超过会话跟踪 bytes "
                 f"{session.bytes}（tick={action['tick']}）")
        self._apply(action["tick"], source, -nbytes)
        self._apply(action["tick"], target, nbytes)
        session.instance = target
        self.report.actions["decode_move"] += 1

    def apply_grow(self, tick: int, session_id: str, instance: int,
                   tokens: int, label: str) -> None:
        session = self._session(session_id)
        desired = self._expected_bytes(tokens)
        delta = desired - session.bytes
        if delta < 0:
            fail(f"{label} 增长为负（会话 {session_id!r}: "
                 f"目标 {desired} < 跟踪 {session.bytes}， tick={tick}）——"
                 f"上下文不允许收缩，重放口径与账本不一致")
        if delta > 0:
            self._apply(tick, instance, delta)
            session.bytes += delta
        session.instance = instance
        self.report.actions[label] = self.report.actions.get(label, 0) + 1

    def apply_own_relocation(self, tick: int, session_id: str,
                             location: Optional[str]) -> None:
        """S2：completion 的 kv_location_after_completion 归因自身会话。"""
        session = self._session(session_id)
        if location == "remote_memory":
            if session.instance is not None and session.bytes > 0:
                self._apply(tick, session.instance, -session.bytes,
                            evict_bytes=session.bytes)
                self.report.actions["evictions"] += 1
                self.report.actions["evict_bytes"] += session.bytes
            session.bytes = 0
            session.instance = None
            self.report.actions["own_relocation_remote"] += 1
        elif location == "partial_hbm_remote":
            # 分层部分去向：留下/离开比例账本未落——保留 bytes（上界），
            # 显式计数，不臆造比例。
            self.report.actions["own_relocation_partial_unknown"] += 1
        # local_hbm：保留，无事。


# ---------------------------------------------------------------------------
# 决策记录 → 动作流
# ---------------------------------------------------------------------------

class WatermarkScan:
    """A4 driver 复用面：单条决策记录一次 consume，扫完 finish。

    与独立 CLI cmd_hbm_watermark 的决策日志循环体逐语句等价（含 fail
    消息与记录内「逐出→恢复/迁移→增长」固定次序、session_hint 回写）。
    注意 consume 会向 record 注入 session_hint 键——driver 的 sink 次序中
    watermark 必须最后（kv/load/hop 不读该键，注入对其不可见）。
    """

    def __init__(self, run_dir: Path, mapping: dict,
                 tokens: dict, coef: int,
                 capacity: Optional[int],
                 model_layers: Optional[int] = None) -> None:
        self.replay = WatermarkReplay(coef, capacity,
                                      model_layers=model_layers)
        self.mapping = mapping
        self.tokens = tokens
        self.seen_kinds: dict[tuple[str, str], int] = {}
        self.log_path = run_dir / DECISION_LOG_RELPATH
        self.last_tick = -1
        self.last_seq = -1  # HBM-03：seq 单调（严格递增）游标

    def consume(self, record: dict) -> None:
        replay = self.replay
        mapping = self.mapping
        tokens = self.tokens
        where = f"{self.log_path}:seq={record.get('seq', '?')}"
        request_id = record.get("request_id")
        kind = record.get("kind")
        if not isinstance(request_id, str) or not request_id:
            fail(f"{where}: 决策记录缺 request_id")
        if kind not in ("prefill", "decode", "completion"):
            fail(f"{where}: 未知 kind={kind!r}")
        key = (request_id, kind)
        if key in self.seen_kinds:
            fail(f"{where}: 请求 {request_id} 的 {kind} 决策出现两次"
                 f"（先于 seq={self.seen_kinds[key]}）——账本次序异常")
        # HBM-03：seq 只用于判重不设防——补类型/非负/单调校验（文件顺序
        # 与 sequence 序必须一致，异常 seq 不得污染扫描状态）。
        seq = record.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            fail(f"{where}: 缺非负整数 seq（实得 {seq!r}）")
        if seq <= self.last_seq:
            fail(f"{where}: seq 回退/重复（{seq} <= {self.last_seq}）——"
                 f"文件顺序与 sequence 序不一致，无法安全重放")
        self.last_seq = seq
        self.seen_kinds[key] = seq
        tick = record.get("tick")
        # bool 冒充门（2026-10-01 对齐）：tick=true 曾按 tick=1 进入变点流。
        if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
            fail(f"{where}: 缺非负整数 tick")
        if tick < self.last_tick:
            fail(f"{where}: tick 回退（{tick} < {self.last_tick}）——文件顺序"
                 f"与时间顺序不一致，无法安全重放")
        self.last_tick = tick
        token_row = tokens["requests"].get(request_id)
        if token_row is None:
            fail(f"{where}: 请求 {request_id} 不在 token manifest"
                 f"（{tokens['path']}）——两源请求集必须一致")
        session_id = token_row["session_id"]
        record["session_hint"] = session_id

        # 1) 逐出（记录内先于恢复/迁移/增长——manager 语义：先腾后放）。
        evictions = _collect_eviction_lists(record, mapping, where)
        if mapping.get("evict_report_only"):
            # B4（-LRU wscllm tiered）：逐出条目到时序晚于 victim 其间的
            # 生命周期事件（净额重试累积），按时序扣占用会错实例/双扣——
            # report-only，占用上界 + 恢复点对账（见 report_evictions_only）。
            replay.report_evictions_only(evictions)
        else:
            for action in evictions:
                replay.apply_evict(action, request_id)

        decision = record.get("decision") or {}

        # 2) 按记录类型主动作。
        if kind == "prefill":
            target_instance = decision.get("prefill_instance_index")
            if not isinstance(target_instance, int):
                fail(f"{where}: prefill 决策缺整数 prefill_instance_index")
            restore_actions = _prefill_restore_actions(record, mapping,
                                                       where)
            # B4（-LRU wscllm tiered）：evict_report_only 下逐出条目不动
            # 跟踪账——恢复前先按 admission 时点 location 对账收敛旧代驻留
            #（remote_memory → 0；partial_hbm_remote → noc 段前缀 bytes），
            # 逐段 restore 随后按真实搬移重放。
            location_field = (mapping.get("restore") or {}).get(
                "location_field")
            if location_field:
                location = decision.get(location_field)
                if location == "remote_memory":
                    replay.reconcile_stale_resident(
                        tick, session_id, 0)
                elif location == "partial_hbm_remote":
                    prefix_bytes = sum(
                        # bool 排除（2026-10-01 对齐）：True or 0 → True
                        # 参与求和 +1，把 1 字节谎报进前缀 bytes。
                        row.get("total_bytes") if (
                            isinstance(row.get("total_bytes"), int)
                            and not isinstance(row.get("total_bytes"), bool)
                        ) else 0
                        for row in
                        decision.get("history_transfers") or []
                        if isinstance(row, dict)
                        and row.get("kind") == "noc_migrate")
                    replay.reconcile_stale_resident(
                        tick, session_id, prefix_bytes)
            for action in restore_actions:
                replay.apply_restore(action,
                                     token_row["history_tokens_before"])
            replay.apply_grow(tick, session_id, target_instance,
                              token_row["prefill_context_tokens"],
                              "prefill_grow")
        elif kind == "decode":
            target_instance = decision.get("decode_instance_index")
            if not isinstance(target_instance, int):
                fail(f"{where}: decode 决策缺整数 decode_instance_index")
            for action in _decode_move_actions(record, mapping, where):
                replay.apply_move(action,
                                  token_row["prefill_context_tokens"])
            replay.apply_grow(tick, session_id, target_instance,
                              token_row["final_context_tokens"],
                              "decode_grow")
        else:  # completion
            field = mapping.get("completion_relocation_field")
            if field is not None:
                replay.apply_own_relocation(
                    tick, session_id, decision.get(field))

        # 3) 计数型逐出（S2）：无 victim/bytes，只记事件数（上界重建）。
        for count_field in mapping.get("eviction_count_fields", ()):
            count = decision.get(count_field)
            if count is None:
                continue
            if not isinstance(count, int) or count < 0:
                fail(f"{where}: {count_field} 必须为非负整数")
            if count:
                replay.report.actions["unattributed_evictions"] += count
                hint = replay.sessions.get(session_id)
                instance = hint.instance if hint else None
                for _ in range(count):
                    replay.cplog.add_evict(
                        tick, instance if instance is not None else -1, 0)

    def finish(self) -> WatermarkReplay:
        tokens = self.tokens
        missing = []
        for request_id in tokens["requests"]:
            for kind in ("prefill", "decode", "completion"):
                if (request_id, kind) not in self.seen_kinds:
                    missing.append(f"{request_id}:{kind}")
        if missing:
            fail(f"{self.log_path}: token manifest 中的请求缺决策记录（前 5 例："
                 f"{missing[:5]}，共 {len(missing)}）——两源请求集不一致")
        if not self.replay.cplog.has_events():
            fail(f"{self.log_path}: 没有任何可重放的 KV 动作")
        return self.replay


# ---------------------------------------------------------------------------
# journal 权威重放（P1-①）与 tier 判定
# ---------------------------------------------------------------------------

JOURNAL_REQUIRED_FIELDS = (
    "schema_version", "sequence", "transaction_id", "planner_time_ns",
    "rank", "instance_index", "capacity_bytes", "weight_delta_bytes",
    "resident_kv_delta_bytes", "reserved_kv_delta_bytes", "before_bytes",
    "after_bytes", "cause",
)
# B4（-LRU wscllm，2026-09-07）：journal 行 schema v2（B2 remote 列）——
# v1 = 三类 delta + before/after 三分量；v2 追加 remote_delta_bytes 列与
# before/after.remote 分量（远端池账面，journal-only：MemoryDelta/manifest
# 冻结流零改动，watermark 只改读取口径）。remote 不计入逐 rank 判决和
# （w+r+s；术语对齐 metrics_schema：physical = weight+resident 两分量、
# committed = physical+reserved 三分量——本工具的判决和即 schema 的
# committed 口径，输出字段名 physical_* 为历史命名沿用）——远端字节
# 驻留共享池，不占本 rank 容量。
JOURNAL_SCHEMA_VERSIONS = (1, 2)
JOURNAL_V2_REQUIRED_FIELDS = ("remote_delta_bytes",)


class JournalReplay:
    """kv_delta_journal.jsonl 的流式权威重放（逐 rank 认证 + 逐实例 RLE）。

    每行是 manager mutation 提交点的权威记录（rank/instance/capacity_
    bytes/三类 delta/before/after）。三类检查：

    * 行级链自洽（fail-closed）：before == 该 rank 前行 after；
      after == before + 对应 delta（三类各自）；after 各分量非负。
    * 逐 rank 容量认证：physical = after.weight + after.resident +
      after.reserved > capacity_bytes → 违规计数（是否构成正式判决由
      tier 决定——仅 per_rank_total_hbm_certified 层 exit 3）。
    * resident 硬上限口径（诊断）：after.resident > kv_shard_r ×
      hard_limit_tokens → 计数报告（三口径之一，不作判决）。

    逐实例 resident 变点流喂 ChangePointLog（journal 的 instance 语义与
    decision-log 重放一致，可直接并到同一 instances/series/intervals 产物）。
    """

    def __init__(self, journal_path: Path,
                 hard_limit_by_weight: Optional[dict[int, int]]) -> None:
        self.journal_path = journal_path
        self.hard_limit_by_weight = hard_limit_by_weight or {}
        self.cplog = ChangePointLog()
        self.state: dict[int, dict] = {}  # rank -> {w, r, s[, m]}（当前）
        self.schema_version: Optional[int] = None  # 首行落定（单文件同版）
        self.rank_instance: dict[int, int] = {}
        self.rank_capacity: dict[int, int] = {}
        self.rows = 0
        self.sequence_first: Optional[int] = None
        self.sequence_last: Optional[int] = None
        self.transaction_max = -1
        self.causes: dict[str, int] = {}
        self.physical_violation_events = 0
        self.violation_by_rank: dict[int, int] = {}
        self.violation_by_instance: dict[int, int] = {}
        self.max_exceed_bytes = 0
        self.hard_limit_exceed_events = 0
        self.hard_limit_exceed_ranks: dict[int, int] = {}
        # journal 重放的逐出总量（单一来源：summary total / instances CSV
        # 的逐出列在 journal 路径激活时一律取此处，不再混用 decision-log
        # 重放的 report 计数与 decision coverage 门控）。
        self.evict_events_total = 0
        self.evict_bytes_total = 0
        # M-31 会话去重态：journal 行是逐 rank 提交点，同一会话的一次逐出
        # 按 TP 数落多行（TP=6 → 逐行计数 6×虚增）；decision-log 路径
        # 同名字段=会话数，两模式口径统一为会话数。writer 契约（metrics_
        # integration）保证同一 mutation 的行连续且 transaction_id 单调
        # ——事务切换即重置会话集；bytes 逐行求和（每 rank 只报自己分片
        # 的字节，总和本就不虚增）。（2026-10-01 R3 注记：本重放侧**有
        # 意**不重复校验 transaction_id 单调/连续——HBM-02 的 sequence
        # 从 0 连续已钉死行序，无 checksum 的崩溃残账中"sequence 连续
        # 而 transaction_id 回退"需要伪造序号而非截断；writer 侧
        # run-end 门（metrics_integration.replay_journal）才是该契约的
        # 执行点。）
        self._evict_tx_current: Optional[int] = None
        self._evict_sessions_tx: set[str] = set()
        self.sha256 = hashlib.sha256()
        self._prev_sequence: Optional[int] = None
        self._prev_time: Optional[int] = None
        # 当前行定位（journal_path:行号）——_require 错误消息消费（旧行为
        # 的 where 形参被丢弃、错误消息引用上一行的 sequence_last 指错行）。
        self._current_where: Optional[str] = None

    # -- 行校验与施加 -----------------------------------------------------

    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            # 定位前缀用当前行 where（journal_path:行号）——旧行为引用
            # self.sequence_last，那是前行序号：首行字段检查失败时它还是
            # None，任何"先行字段"失败都会指错行。
            fail(f"{self._current_where or self.journal_path}: {message}"
                 f"——kv_delta_journal 行结构/链不自洽（账本损坏，"
                 f"fail-closed 不降级）")

    def consume(self, row: dict, where: str) -> None:
        # 行号信息落地为错误定位（消费 where，不再留给死参数）。
        self._current_where = where
        for field in JOURNAL_REQUIRED_FIELDS:
            self._require(field in row, f"缺字段 {field}")
        schema_version = row.get("schema_version")
        # bool 显式排除（True==1 会混过 `in (1, 2)` 成员判断，与 replay_
        # journal 证书 line_count 的 bool 排除同纪律）。
        self._require(not isinstance(schema_version, bool)
                      and schema_version in JOURNAL_SCHEMA_VERSIONS,
                      f"schema_version 必须为 {' 或 '.join(str(v) for v in JOURNAL_SCHEMA_VERSIONS)}"
                      f"（实得 {schema_version!r}）")
        self._require(
            self.schema_version is None
            or self.schema_version == schema_version,
            f"schema_version 混版本（{self.schema_version} -> "
            f"{schema_version}）——单文件必须同版")
        if self.schema_version is None:
            self.schema_version = schema_version
        if schema_version == 2:
            for field in JOURNAL_V2_REQUIRED_FIELDS:
                self._require(field in row, f"缺字段 {field}（v2）")
        sequence = row["sequence"]
        tick = row["planner_time_ns"]
        self._require(isinstance(sequence, int) and not isinstance(
            sequence, bool) and sequence >= 0, "sequence 非非负整数")
        self._require(isinstance(tick, int) and not isinstance(tick, bool)
                      and tick >= 0, "planner_time_ns 非非负整数")
        # HBM-02：sequence 必须从 0 连续递增（与 writer 侧
        # metrics_integration.replay_journal 的 contiguity 契约同口径）
        # ——只要求"大于上一条"会接受删行/截断的账本，fail-closed。
        expected_sequence = 0 if self._prev_sequence is None \
            else self._prev_sequence + 1
        self._require(sequence == expected_sequence,
                      f"sequence 不连续（实得 {sequence}，应为 "
                      f"{expected_sequence}）——journal 序列必须从 0 连续"
                      f"递增（缺行/删行即账本损坏）")
        self._require(self._prev_time is None or tick >= self._prev_time,
                      f"planner_time_ns 回退（{tick} < {self._prev_time}）")
        self._prev_sequence = sequence
        self._prev_time = tick
        if self.sequence_first is None:
            self.sequence_first = sequence
        self.sequence_last = sequence
        transaction_id = row["transaction_id"]
        # HBM-07：transaction_id 必须为非负整数（bool 亦拒）——malformed
        # 值不得静默跳过 max 统计。
        self._require(isinstance(transaction_id, int)
                      and not isinstance(transaction_id, bool)
                      and transaction_id >= 0,
                      f"transaction_id 非非负整数（实得 {transaction_id!r}）")
        if transaction_id > self.transaction_max:
            self.transaction_max = transaction_id
        rank = row["rank"]
        instance = row["instance_index"]
        capacity = row["capacity_bytes"]
        self._require(isinstance(rank, int) and not isinstance(rank, bool)
                      and rank >= 0, "rank 非非负整数")
        self._require(isinstance(instance, int) and not isinstance(
            instance, bool) and instance >= 0, "instance_index 非非负整数")
        self._require(isinstance(capacity, int) and not isinstance(
            capacity, bool) and capacity > 0, "capacity_bytes 非正整数")
        deltas = {"m": 0}  # v1 无 remote 列：按 0 参与行级自洽校验
        delta_fields = (("weight_delta_bytes", "w"),
                        ("resident_kv_delta_bytes", "r"),
                        ("reserved_kv_delta_bytes", "s"))
        if schema_version == 2:
            delta_fields += (("remote_delta_bytes", "m"),)
        for field, key in delta_fields:
            value = row[field]
            self._require(isinstance(value, int) and not isinstance(
                value, bool), f"{field} 非整数")
            deltas[key] = value
        before = row["before_bytes"]
        after = row["after_bytes"]
        self._require(isinstance(before, dict) and isinstance(after, dict),
                      "before_bytes/after_bytes 必须是对象")
        parsed_before = {"m": 0}  # v1 无 remote 列：按 0 参与链校验
        parsed_after = {"m": 0}
        for source, target in ((before, parsed_before), (after, parsed_after)):
            component_fields = (("weight", "w"), ("resident", "r"),
                                ("reserved", "s"))
            if schema_version == 2:
                component_fields += (("remote", "m"),)
            for field, key in component_fields:
                value = source.get(field)
                self._require(isinstance(value, int) and not isinstance(
                    value, bool) and value >= 0,
                    f"{field if source is after else 'before.' + field}"
                    f" 非非负整数")
                target[key] = value
        current = self.state.get(rank)
        expected_before = current if current is not None \
            else {"w": 0, "r": 0, "s": 0, "m": 0}
        for key, label in (("w", "weight"), ("r", "resident"),
                           ("s", "reserved"), ("m", "remote")):
            self._require(parsed_before[key] == expected_before[key],
                          f"rank {rank} {label} 链断裂（before="
                          f"{parsed_before[key]}，前行 after="
                          f"{expected_before[key]}）")
            self._require(parsed_after[key] == parsed_before[key] +
                          deltas[key],
                          f"rank {rank} {label} 行不自洽（after="
                          f"{parsed_after[key]} != before+delta="
                          f"{parsed_before[key] + deltas[key]}）")
        known_instance = self.rank_instance.get(rank)
        if known_instance is not None:
            self._require(known_instance == instance,
                          f"rank {rank} 实例漂移（{known_instance} -> "
                          f"{instance}）")
        known_capacity = self.rank_capacity.get(rank)
        if known_capacity is not None:
            self._require(known_capacity == capacity,
                          f"rank {rank} capacity 漂移（{known_capacity} -> "
                          f"{capacity}）")
        self.rank_instance[rank] = instance
        self.rank_capacity[rank] = capacity
        self.state[rank] = parsed_after
        # v2 的 remote（m）不计入 physical：远端池字节不占本 rank 本地 HBM
        # 容量（判决口径不变，remote 仅链校验+终态报告）。
        physical_after = parsed_after["w"] + parsed_after["r"] + \
            parsed_after["s"]
        if physical_after > capacity:
            self.physical_violation_events += 1
            self.violation_by_rank[rank] = \
                self.violation_by_rank.get(rank, 0) + 1
            self.violation_by_instance[instance] = \
                self.violation_by_instance.get(instance, 0) + 1
            self.max_exceed_bytes = max(self.max_exceed_bytes,
                                        physical_after - capacity)
        hard_limit = self.hard_limit_by_weight.get(parsed_after["w"]) \
            if parsed_after["w"] else None
        if hard_limit is not None and parsed_after["r"] > hard_limit:
            self.hard_limit_exceed_events += 1
            self.hard_limit_exceed_ranks[rank] = \
                self.hard_limit_exceed_ranks.get(rank, 0) + 1
        cause = row.get("cause")
        cause_label = cause if isinstance(cause, str) else repr(cause)
        self.causes[cause_label] = self.causes.get(cause_label, 0) + 1
        if deltas["r"]:
            self.cplog.add(tick, instance, deltas["r"])
        # 逐出识别按 cause 前缀 "evict_"（F8，2026-09-30）：生产 cause
        # 词汇表里逐出恒为 evict_{reason}_session:layers0-{L}；非逐出的
        # 负 resident 路径（terminal_session_retire / *_source_remove /
        # reservation_release/extend）均不含该前缀。子串 "evict" 匹配
        # 依赖词汇表永不出现同子串的非逐出 cause，无测试钉住——前缀化
        # 收窄误纳面。
        if cause_label.lower().startswith("evict_") and deltas["r"] < 0:
            transaction = row["transaction_id"]
            if transaction != self._evict_tx_current:
                self._evict_tx_current = transaction
                self._evict_sessions_tx = set()
            # 会话键：session_id 可缺席（旧 schema/合成行）——repr 兜底
            # 保证 None/str 都可哈希且互不混并。
            session_key = repr(row.get("session_id"))
            new_event = session_key not in self._evict_sessions_tx
            if new_event:
                self._evict_sessions_tx.add(session_key)
                self.evict_events_total += 1
            self.evict_bytes_total += -deltas["r"]
            self.cplog.add_evict_row(tick, instance, -deltas["r"], new_event)
        self.rows += 1

    def final_rank_state(self) -> dict:
        return {rank: {
            "instance_index": self.rank_instance[rank],
            "capacity_bytes": self.rank_capacity[rank],
            "weight": state["w"],
            "resident": state["r"],
            "reserved": state["s"],
            # v2：远端池账面终态（v1 无此列，恒 0 报告位）。
            "remote": state.get("m", 0),
            "physical": state["w"] + state["r"] + state["s"],
        } for rank, state in sorted(self.state.items())}


def load_journal_checksum(path: Path) -> dict:
    try:
        # load_json_strict（2026-10-01 对齐）：checksum 证书是外部文件，
        # NaN/Infinity 字面量不进本工具链。
        data = load_json_strict(path.read_text(encoding="utf-8"),
                                f"kv_delta_journal_checksum.json {path}")
    except SloToolError:
        fail(f"kv_delta_journal_checksum.json 非法 JSON：{path}")
    if not isinstance(data, dict):
        fail(f"kv_delta_journal_checksum.json 必须是对象：{path}")
    return data


def replay_journal(journal_path: Path,
                   checksum_path: Optional[Path],
                   hard_limit_by_weight: Optional[dict[int, int]]) -> dict:
    """journal 单遍流式重放 + tier 判定（P1-① 的权威路径）。

    返回 {tier, replay, checksum, sha256_hex, checks, cert_gaps}。
    fail-closed 条件（退出码 2，不降级）：sha256 不匹配 / 行级链断裂 /
    行数与证书不符 / 重放终态与证书终态矛盾。checks 有 false，或证书
    缺必需项（sha256/line_count/ranks 缺失或类型不符，cert_gaps 非空，
    HBM-01）→ lifecycle_replay_exact（完整性证明不完整：journal 行级
    仍精确，但不作正式判决，emit 侧显式列出缺陷）。
    """
    replay = JournalReplay(journal_path, hard_limit_by_weight)
    with journal_path.open("rb") as handle:
        for lineno, raw in enumerate(handle, start=1):
            replay.sha256.update(raw)
            stripped = raw.strip()
            if not stripped:
                # HBM-06：空白行与 writer 侧（metrics_integration.
                # iter_journal_deltas）同口径判损坏，不再静默跳过——
                # 同一格式在不同入口不得有两套 fail 语义。
                fail(f"{journal_path}:{lineno}: 空白行（writer 侧同判损坏，"
                     f"fail-closed 不降级）")
            try:
                # parse_constant 门（2026-10-01 对齐）：journal 行是外部
                # 文件，NaN/Infinity 字面量不进重放（与 load_json_strict
                # 同一纪律；行级 sha256/错误语义保持不变）。
                row = json.loads(
                    stripped,
                    parse_constant=slo_json_constant_reject)
            except UnicodeDecodeError as exc:
                # F3（2026-09-30）：json.loads(bytes) 对坏 UTF-8 字节抛
                # UnicodeDecodeError（非 JSONDecodeError 子类），旧行为
                # 逃出本 except、退化为无行号定位的通用读错误——与 writer
                # 侧（metrics_integration 的显式 decode + 行号）对齐。
                fail(f"{journal_path}:{lineno}: 非法 UTF-8 字节（{exc}）")
            except json.JSONDecodeError as exc:
                fail(f"{journal_path}:{lineno}: 非法 JSON（{exc}）")
            if not isinstance(row, dict):
                fail(f"{journal_path}:{lineno}: journal 行必须是对象")
            replay.consume(row, f"{journal_path}:{lineno}")
    if replay.rows == 0:
        fail(f"{journal_path}: 没有任何可重放的 KV delta 行")
    sha256_hex = replay.sha256.hexdigest()
    checksum: Optional[dict] = None
    checks: dict = {}
    tier = TRUST_TIER_RESIDENT
    # HBM-01（fail-closed，2026-09-26）：sha256/line_count/ranks 是 certified
    # 的必需项（文件头契约），缺失或类型不符不得跳过对应校验、不得仅凭
    # checks 布尔给出 certified——完整性证明不完整即降级 lifecycle 并把
    # 缺陷清单显式带回 emit 侧（绝不静默）。
    cert_gaps: list[str] = []
    if checksum_path is not None:
        checksum = load_journal_checksum(checksum_path)
        expected_sha = checksum.get("sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            cert_gaps.append(
                f"证书缺合法 sha256（实得 {expected_sha!r}）")
        elif expected_sha != sha256_hex:
            fail(f"{journal_path}: journal sha256 与证书不符（journal="
                 f"{sha256_hex}，checksum={expected_sha}）——账本完整性"
                 f"破坏，fail-closed 不降级")
        expected_lines = checksum.get("line_count")
        if not isinstance(expected_lines, int) or \
                isinstance(expected_lines, bool):
            cert_gaps.append(
                f"证书缺整数 line_count（实得 {expected_lines!r}）")
        elif expected_lines != replay.rows:
            fail(f"{journal_path}: journal 行数与证书不符（journal="
                 f"{replay.rows}，checksum={expected_lines}）")
        ranks_block = checksum.get("ranks")
        if not isinstance(ranks_block, dict) or not ranks_block:
            cert_gaps.append(f"证书缺非空 ranks 终态（实得类型 "
                             f"{type(ranks_block).__name__}）")
        else:
            for rank_key, entry in sorted(ranks_block.items()):
                try:
                    rank = int(rank_key)
                except ValueError:
                    fail(f"{checksum_path}: ranks 键必须是 rank 整数"
                         f"（实得 {rank_key!r}）")
                state = replay.state.get(rank)
                if state is None:
                    fail(f"{checksum_path}: 证书含 rank {rank} 但 journal"
                         f"无该 rank 行——账本与证书矛盾")
                if not isinstance(entry, dict):
                    fail(f"{checksum_path}: ranks[{rank}] 必须是对象")
                # M-32：终态分量全字段严格类型校验。旧的
                # `isinstance(value, int) and ...` 形状对缺失/None/str/float
                # 静默空过、bool 冒充整数混过（False 当 0 混过）——证书
                # 残缺也能 certified，第三道门形同虚设。证书终态是完整性
                # 证明的一部分：分量字段缺失/类型不符 = 证书无法证明终态，
                # 与终态不符同罪，fail-closed 不降级。v2 证书多 remote
                # 分量（v1 无该键不要求）。
                component_fields = (("weight", "w"), ("resident", "r"),
                                    ("reserved", "s"))
                if replay.schema_version == 2:
                    component_fields += (("remote", "m"),)
                for field, key in component_fields:
                    value = entry.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) \
                            or value < 0:
                        fail(
                            f"{checksum_path}: rank {rank} 终态 {field} 缺失"
                            f"或非非负整数（实得 {value!r}）——证书残缺，"
                            f"fail-closed 不降级")
                    if value != state.get(key, 0):
                        fail(
                            f"{checksum_path}: rank {rank} 终态 {field} 与"
                            f" journal 重放不符（证书 {value}，重放 "
                            f"{state.get(key, 0)}）——账本与证书矛盾")
                # M-32：physical 字段纳入校验（此前完全不查）。口径与写侧
                # 一致（metrics_integration）：physical = weight + resident
                #（schema 的 physical 两分量口径，不含 reserved/remote）。
                expected_physical = entry.get("physical")
                if isinstance(expected_physical, bool) or \
                        not isinstance(expected_physical, int):
                    fail(
                        f"{checksum_path}: rank {rank} 终态 physical 缺失"
                        f"或非整数（实得 {expected_physical!r}）——证书"
                        f"残缺，fail-closed 不降级")
                replay_physical = state.get("w", 0) + state.get("r", 0)
                if expected_physical != replay_physical:
                    fail(
                        f"{checksum_path}: rank {rank} 终态 physical 与 "
                        f"journal 重放不符（证书 {expected_physical}，重放 "
                        f"{replay_physical}）——账本与证书矛盾")
                capacity = entry.get("capacity_bytes")
                if isinstance(capacity, bool) or not isinstance(capacity, int) \
                        or capacity <= 0:
                    fail(f"{checksum_path}: rank {rank} capacity_bytes 缺失"
                         f"或非正整数（实得 {capacity!r}）——证书残缺，"
                         f"fail-closed 不降级")
                if capacity != replay.rank_capacity.get(rank):
                    fail(f"{checksum_path}: rank {rank} capacity 与 journal"
                         f" 不符（证书 {capacity}，重放 "
                         f"{replay.rank_capacity.get(rank)}）")
            # 非整数键已在上方 int(rank_key) 处 fail-closed——此处无需再
            # 防御（旧行的 .isdigit() 过滤条件恒真，已删）。
            missing = sorted(set(replay.state)
                             - {int(key) for key in ranks_block})
            if missing:
                fail(f"{checksum_path}: journal 含证书未覆盖的 rank"
                     f"（{missing[:5]}，共 {len(missing)}）——账本与证书"
                     f"矛盾")
        checks = checksum.get("checks") if isinstance(
            checksum.get("checks"), dict) else {}
        # v2 证书多一项 remote_account_zero（B2 终态守恒新口径：retire 核销
        # 后远端池账面与会话账本同归零）——v2 journal 的 certified 层要求
        # 五项全过；v1 维持原四项。
        required_checks = ["manager_state_match", "physical_equals_weight",
                           "residual_reserved_zero", "residual_resident_zero"]
        if replay.schema_version == 2:
            required_checks.append("remote_account_zero")
        if checks and all(checks.get(name) is True
                          for name in required_checks) and not cert_gaps:
            tier = TRUST_TIER_CERTIFIED
        else:
            tier = TRUST_TIER_LIFECYCLE
    return {
        "tier": tier,
        "replay": replay,
        "checksum": checksum,
        "sha256_hex": sha256_hex,
        "checks": checks,
        "cert_gaps": cert_gaps,
    }


def resolve_effective_bucket_ns(requested_bucket_ns: int, span_ns: int,
                                eventful_units: int,
                                row_budget: int) -> tuple[int, bool, str]:
    """B_eff = max(B_requested, ceil(S·N/(R−N)))（P1-④）。

    S=全局 span（末 KV 事件−首 KV 事件）、N=有事件实例数、R=全局行预算。
    R≤N → 拒绝稠密输出（返回 (requested, True, "row_budget_exceeded_
    dense_refused")，调用方只写 RLE、log 说明、正常完成）。
    """
    if row_budget <= eventful_units:
        return requested_bucket_ns, True, "row_budget_exceeded_dense_refused"
    minimum = -(-span_ns * eventful_units
                // (row_budget - eventful_units))  # ceil(商)
    if minimum > requested_bucket_ns:
        return minimum, True, "row_budget_coarsened"
    return requested_bucket_ns, False, "none"



# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def watermark_prepare(args: argparse.Namespace, repo_variant: str,
                      request_manifest_loader=None) -> dict:
    """A4 driver 用：映射/参数/token/容量装载（stderr 警告时点原样）。

    request_manifest_loader：driver 传入的惰性 manifest 装载器（npus
    推导复用；缺省 None=自行装载=CLI 行为不变，失败时点同构）。
    """
    mapping = REPO_VARIANTS.get(repo_variant)
    if mapping is None:
        fail(f"未登记的 repo_variant：{repo_variant}"
             f"（REPO_VARIANTS 需扩表并附基线核对证据）")
    # B3-6：S2 新产物（逐出 victim/bytes 在场）升级 full 口径；旧产物
    # 保底回退 count_only 上界口径。
    if repo_variant == "astra-sim-sh_2.0":
        mapping = upgrade_s2_mapping_if_fields_present(mapping, args.run_dir)
    elif repo_variant == "astra-sim-wscllm":
        # B4（-LRU wscllm）：新产物同触发字段升级（legacy 逐出行是契约行
        # 的同值镜像，防双计）。与 S2/face 的三点差异：
        # ① coverage=full_reconciled——wscllm 净额准入重试会把多次尝试的
        #   逐出累积到最终获准请求的行上（B3 冒烟 r1b 实测：session_1 的
        #   逐出条目晚于其下一轮 restore 到达、session_2_request_2 行含
        #   同一 victim 两个代际的 KV 各一组逐出行），逐出条目到达时序
        #   ≠ 物理 mutation 时序，决策日志重放无法按正确实例扣减占用——
        #   沿用 W 变体 full_reconciled 口径：逐出条目 report-only（登记
        #   evict_events/evict_bytes 诊断账，不动占用/会话跟踪态，占用为
        #   上界），旧代驻留在 victim 下次恢复点按 admission 时点
        #   history_location_before=remote_memory 对账核销（apply_restore
        #   的 reconcile 分支）；
        # ② restore 逐段动作携带 state_before=history_location_before；
        # ③ completion 契约行字段名 = completion_evictions（sh_2.0 同名）。
        # journal 在场时以 journal 重放为权威（P1 设计，certified 层的
        # 容量判决不走本路径）。
        mapping = upgrade_s2_mapping_if_fields_present(mapping, args.run_dir)
        if mapping.get("eviction_coverage") == "full":
            mapping["eviction_coverage"] = "full_reconciled"
            mapping["evict_report_only"] = True
            mapping["restore"] = dict(mapping["restore"])
            mapping["restore"]["location_field"] = "history_location_before"

    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    bucket_ns, bucket_provisional = load_bucket_ns(manifest)

    tokens = load_token_manifest(args.run_dir, args.token_manifest)
    trace = load_trace_config(args.run_dir, args.trace_config)
    config = trace["values"]
    coef = 2 * config["layers"] * config["hidden_size"] * \
        config["bytes_per_elem"]

    # 容量链：profile → hardware bytes/NPU × npus/instance；任一环缺失
    # → capacity=NA，违规检查降级为峰值记录（不编造）。
    profile = config.get("local_hbm_capacity_profile")
    capacity: Optional[int] = None
    capacity_source = NA
    if not profile:
        print("[hbm-watermark] 警告：trace_config 无 "
              "local_hbm_capacity_profile 行——capacity=NA，违规检查降级"
              "为峰值记录", file=sys.stderr)
    else:
        try:
            hardware = resolve_hardware_capacity(profile,
                                                 args.hardware_config)
            try:
                npus = load_npus_per_instance(
                    args.run_dir, args.request_manifest,
                    request_manifest_loader)
            except SloToolError as exc:
                print(f"[hbm-watermark] 警告：{exc}——capacity=NA，违规"
                      f"检查降级为峰值记录", file=sys.stderr)
                npus = 0
            if npus:
                capacity = hardware["bytes"] * npus
                capacity_source = (
                    f"{hardware['path']}#local-hbm.capacity-profiles."
                    f"{profile}.bytes={hardware['bytes']} × "
                    f"npus_per_instance={npus}（request manifest "
                    f"prefill_ranks）")
        except SloToolError as exc:
            print(f"[hbm-watermark] 警告：{exc}——capacity=NA，违规检查"
                  f"降级为峰值记录", file=sys.stderr)

    # P1-③：三口径容量剖面（manager 原函数逐字拷贝；配置缺模型行 → NA 降
    # 级，不 fail——聚合容量链不受影响）。
    calibers: Optional[dict] = None
    calibers_note = NA
    if capacity is not None:
        model = build_model_spec(config)
        if model is None:
            print("[hbm-watermark] 警告：trace_config 缺模型行（ffn_size/"
                  "num_heads/vocab_size 任一）——三口径容量剖面=NA（聚合容"
                  "量链不受影响）", file=sys.stderr)
        else:
            raw_reserve = config.get("kv_reserve_context_tokens")
            try:
                reserve_tokens = int(raw_reserve)
            except (TypeError, ValueError):
                reserve_tokens = DEFAULT_KV_RESERVE_CONTEXT_TOKENS
            if reserve_tokens < 0:
                reserve_tokens = DEFAULT_KV_RESERVE_CONTEXT_TOKENS
            # capacity 仅在 load_npus_per_instance 成功(npus 真值)时赋值,
            # 进入本块 npus 必非零,直接复用(npus_for_calibers 的回取调用
            # 不可达,且其异常也不在下方 except 域内,已删——M6-08,与
            # face-LRU 姊妹仓同款清理)。
            try:
                calibers = compute_capacity_calibers(
                    model, npus, hardware["bytes"],
                    reserve_tokens)
            except (ValueError, RuntimeError, ZeroDivisionError) as exc:
                print(f"[hbm-watermark] 警告：三口径容量剖面计算失败"
                      f"（{exc}）——capacity_calibers=NA", file=sys.stderr)
            else:
                # resident 硬上限的逐 rank 诊断口径按 rank 当前 weight 绑定
                # （journal 只给绝对 rank 编号，相对位=拓扑属性不落盘；
                # weight 值唯一确定 (w_r, kv_r) 剖面项——值冲突且限值不同
                # 的退化配置该 rank 诊断跳过，summary 注记）。绑定表在
                # calibers["resident_kv_hard_limit"]["limit_by_weight_bytes"]。
                calibers_note = (
                    f"{trace['path']} 模型行 × manager 原函数"
                    f"（model_weight_shard_bytes_by_tp_rank / "
                    f"kv_cache_shard_bytes_for_tokens，逐字拷贝自 "
                    f"session_kv_manager.py）")

    # P1-①：tier 判定基础（run_dir 内容探测；最终 tier 在 emit 侧经
    # journal 重放 + 证书对账落定）。
    journal_path = args.run_dir / KV_DELTA_JOURNAL_RELPATH
    checksum_path = args.run_dir / KV_DELTA_JOURNAL_CHECKSUM_RELPATH
    journal_present = journal_path.is_file()
    checksum_present = checksum_path.is_file()
    if journal_present and checksum_present:
        violation_check = "per_rank_physical_gt_capacity"
    elif journal_present:
        violation_check = "per_rank_reported_no_formal_verdict"
    elif checksum_present:
        # M-33：孤儿证书（journal 缺失、checksum 在场）——诚实标注，不再
        # 与"阶段2 前旧 run"的 diagnostic_upper_bound_only 混同（该 run 走
        # 过 journal 模式、产物不完整，与"journal 机制存在前"是两回事）。
        violation_check = "orphan_checksum_upper_bound_only"
    else:
        violation_check = "diagnostic_upper_bound_only"
    if capacity is None and not journal_present and not checksum_present:
        # M-32：degraded 覆盖只适用于 decision-log 上界层——journal 路径
        # 的逐 rank 认证用行内 capacity_bytes（不依赖配置容量），capacity
        # 缺失只降级展示列。旧的无条件覆盖让 certified 层 exit 3 正式判决
        # 与 "degraded_peak_recorded" 标注同框自相矛盾。
        # checksum 在场时不得覆盖（2026-10-01 修正）：孤儿证书 run 恰好
        # 容量链缺失时，M-33 专设的 orphan_checksum_upper_bound_only 标注
        # 会被本分支静默覆盖回 degraded——两处口径（summary 的
        # violation_check 字段与 stderr trust_tier 依据行）自相矛盾。
        violation_check = "degraded_peak_recorded"

    # P1-④：绘图 series 全局行预算（manifest 新键，fail-closed）。
    row_budget = require_param_int(manifest, "watermark_series_row_budget")

    coverage = mapping["eviction_coverage"]
    return {
        "mapping": mapping,
        "bucket_ns": bucket_ns,
        "bucket_provisional": bucket_provisional,
        "row_budget": row_budget,
        "tokens": tokens,
        "trace": trace,
        "coef": coef,
        "model_layers": config["layers"],
        "capacity": capacity,
        "capacity_source": capacity_source,
        "calibers": calibers,
        "calibers_note": calibers_note,
        "coverage": coverage,
        "journal_path": journal_path,
        "checksum_path": checksum_path,
        "journal_present": journal_present,
        "checksum_present": checksum_present,
        "violation_check": violation_check,
    }


def cmd_hbm_watermark(args: argparse.Namespace) -> int:
    # CLI 入口（独立运行行为不变）。A4 driver 经 watermark_prepare /
    # WatermarkScan.consume / watermark_emit 组合复用同一逻辑。decision-log
    # 重放的容量参数恒为诊断口径（upper_bound_only 层不出物理违规认证）。
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    prep = watermark_prepare(args, repo_variant)
    scan = WatermarkScan(args.run_dir, prep["mapping"],
                         prep["tokens"], prep["coef"], prep["capacity"],
                         model_layers=prep.get("model_layers"))
    for record in iter_jsonl(scan.log_path):
        scan.consume(record)
    replay = scan.finish()
    return watermark_emit(args, repo_variant, prep, replay)


def _intervals_rows(cplog: ChangePointLog, repo_variant: str):
    """intervals.csv 的行流：per instance 变点区间（含逐出列）。"""
    for instance in cplog.eventful_units():
        unit = cplog.units[instance]
        pending: Optional[tuple] = None
        for item in unit.merged_change_points():
            if pending is not None:
                yield (repo_variant, instance, pending[0], item[0],
                       pending[1], pending[2], pending[3], pending[4],
                       pending[5])
            pending = item
        if pending is not None:
            # 末变点：后续段 [tick, ∞) 占用恒定，interval_end 记自身 tick
            # （退化段；恢复任意桶长只需变点 tick 序，无需后继）。
            yield (repo_variant, instance, pending[0], pending[0],
                   pending[1], pending[2], pending[3], pending[4],
                   pending[5])


def _plot_series_rows(cplog: ChangePointLog, repo_variant: str, origin: int,
                      span_end: int, effective_bucket_ns: int):
    """plot_series.csv 的行流：游标推进时逐行 yield（P1-④ 流式写出）。"""
    for instance in cplog.eventful_units():
        unit = cplog.units[instance]
        for row in bucket_row_sweep(unit.merged_change_points, origin,
                                    span_end, effective_bucket_ns):
            yield (repo_variant, instance, row[0], row[1], row[2], row[3],
                   row[4], row[5], row[6])


def watermark_emit(args: argparse.Namespace, repo_variant: str, prep: dict,
                   replay: WatermarkReplay) -> int:
    bucket_ns = prep["bucket_ns"]
    bucket_provisional = prep["bucket_provisional"]
    row_budget = prep["row_budget"]
    tokens = prep["tokens"]
    trace = prep["trace"]
    coef = prep["coef"]
    capacity = prep["capacity"]
    capacity_source = prep["capacity_source"]
    calibers = prep["calibers"]
    calibers_note = prep["calibers_note"]
    coverage = prep["coverage"]
    violation_check = prep["violation_check"]
    mapping = prep["mapping"]

    # -- P1-①：tier 判定 + journal 权威重放 --------------------------------
    journal_result = None
    journal_replay = None
    trust_tier = TRUST_TIER_UPPER_BOUND
    if prep["journal_present"]:
        hard_limit_by_weight = None
        if calibers is not None:
            hard_limit_by_weight = calibers["resident_kv_hard_limit"][
                "limit_by_weight_bytes"]
        journal_result = replay_journal(
            prep["journal_path"],
            prep["checksum_path"] if prep["checksum_present"] else None,
            hard_limit_by_weight)
        journal_replay = journal_result["replay"]
        trust_tier = journal_result["tier"]
        if trust_tier == TRUST_TIER_CERTIFIED:
            basis = ("kv_delta_journal.jsonl + kv_delta_journal_checksum."
                     "json 在场，且 sha256/行级链/重放终态/守恒四项全过")
        elif trust_tier == TRUST_TIER_LIFECYCLE:
            if journal_result.get("cert_gaps"):
                # HBM-01：证书缺必需项 → 完整性证明不完整，显式列出缺陷。
                basis = ("journal + checksum 在场、行级链与终态一致，但证书"
                         "缺必需项、完整性证明不完整（"
                         f"{'；'.join(journal_result['cert_gaps'])}）——"
                         f"lifecycle 精确、不作正式判决")
            else:
                basis = ("journal + checksum 在场、行级链与终态一致，但守恒"
                         f"checks 未全过（{journal_result['checks']}）——"
                         f"lifecycle 精确、无守恒证书")
        else:
            basis = ("kv_delta_journal.jsonl 在场、行级链自洽，但 "
                     "kv_delta_journal_checksum.json 缺失——resident 精确、"
                     "无 run 末守恒证书")
        print(f"[hbm-watermark] trust_tier={trust_tier}：依据 {basis}",
              file=sys.stderr)
    else:
        if prep["checksum_present"]:
            # M-33：孤儿证书诚实诊断——journal 丢失（归档/拷贝不完整）
            # 不是"阶段2 前旧 run"；判决面不变（无权威重放即无正式判决，
            # 退出码 3 废除），只修误导文案与标注。
            print(f"[hbm-watermark] trust_tier={trust_tier}：依据 results/"
                  f"kv_delta_journal.jsonl 缺失而 kv_delta_journal_checksum."
                  f"json 在场（孤儿证书）——journal 权威重放不可行（run "
                  f"产物不完整：journal 丢失/未随归档拷贝？），decision-log"
                  f" 重放为上界口径，occupancy_valid=false、退出码 3 废除"
                  f"（超限只作诊断计数；正式判决必须完整 journal+证书）",
                  file=sys.stderr)
        else:
            print(f"[hbm-watermark] trust_tier={trust_tier}：依据 run_dir 缺 "
                  f"results/kv_delta_journal.jsonl（阶段2 前旧 run）——"
                  f"decision-log 重放为上界口径，occupancy_valid=false、"
                  f"退出码 3 废除（超限只作诊断计数）", file=sys.stderr)

    # 权威变点日志：journal 在场=journal 重放（journal 为权威）；否则=
    # decision-log 重放（上界口径，即权威可得的最好结果）。
    cplog = journal_replay.cplog if journal_replay is not None \
        else replay.cplog
    occupancy_source = ("kv_delta_journal resident 重放（权威）"
                        if journal_replay is not None else
                        "decision-log KV 动作重放（上界）")
    occupancy_valid = journal_replay is not None

    span_start, span_end = cplog.span()
    if span_start is None or span_end is None:
        fail("无可重放的 KV 事件（decision-log 与 journal 两侧均无 KV 动作）"
             "——无法产出水位线")
    # 全局零跨度（单 tick 事件流）与逐实例 duration==0 同口径（stats 的
    # mean 退化取峰值，不 fail）：不再硬退码 2 误杀单事件输入——产出退化
    # RLE 区间/单桶行 + 明确 note。
    zero_span = span_end <= span_start
    span_ns = span_end - span_start
    units = cplog.eventful_units()
    n_units = len(units)

    # -- P1-④：B_eff 行预算 ------------------------------------------------
    effective_bucket_ns, resolution_adjusted, adjustment_reason = \
        resolve_effective_bucket_ns(bucket_ns, span_ns, n_units, row_budget)
    dense_refused = adjustment_reason == "row_budget_exceeded_dense_refused"
    if resolution_adjusted and not dense_refused:
        print(f"[hbm-watermark] 绘图 series 行预算调整：requested_bucket_ns"
              f"={bucket_ns} → effective_bucket_ns={effective_bucket_ns}"
              f"（row_budget={row_budget}，span_ns={span_ns}，"
              f"eventful_units={n_units}；B_eff=max(B, ceil(S·N/(R−N)))）",
              file=sys.stderr)
    if dense_refused:
        print(f"[hbm-watermark] 绘图 series 拒绝稠密输出：row_budget"
              f"={row_budget} ≤ eventful_units={n_units}——只写 RLE 权威"
              f"区间（slo_hbm_intervals.csv），plot_series 仅表头；"
              f"正常完成（非失败）", file=sys.stderr)
    if zero_span:
        print(f"[hbm-watermark] note：全局 KV 事件跨度为 0（span_start="
              f"{span_start}，单 tick 事件流）——与逐实例 duration==0 同"
              f"口径产出退化区间/单桶行，不 fail-closed", file=sys.stderr)
    worst_rows_requested = max(
        (cplog.units[u].last_tick - span_start) // bucket_ns for u in units) \
        + 1
    worst_rows_effective = max(
        (cplog.units[u].last_tick - span_start) // effective_bucket_ns
        for u in units) + 1

    # -- P1-④：行预算守恒先于落盘判定 --------------------------------------
    # 计数遍 + 写出遍均流式（与锚桶扫描的两遍消费同型）：预算破坏在写盘
    # 前退码，不留半截产物。
    dense_rows_total = 0
    if not dense_refused:
        for _ in _plot_series_rows(cplog, repo_variant, span_start, span_end,
                                   effective_bucket_ns):
            dense_rows_total += 1
        if dense_rows_total > row_budget:
            fail(f"绘图 series 行数 {dense_rows_total} 超全局行预算 "
                 f"{row_budget}（B_eff 公式失效）——fail-closed（不变量破坏，"
                 f"拒绝静默超预算交付；产物尚未落盘）")

    # -- 输出：intervals（权威 RLE）----------------------------------------
    interval_rows_total = 0
    istream, iclose = open_output(args.intervals_csv,
                                  "slo_hbm_intervals.csv", args.run_dir)
    try:
        writer_rows = _intervals_rows(cplog, repo_variant)
        def counted_intervals():
            nonlocal interval_rows_total
            for row in writer_rows:
                interval_rows_total += 1
                yield row
        write_csv(istream, INTERVAL_COLUMNS, counted_intervals())
    finally:
        if iclose:
            istream.close()

    # -- 输出：plot_series（行预算约束的绘图产物，流式写出）----------------
    # 预算守恒已在落盘前判定（上方计数遍）；本遍重计 written 数（与计数遍
    # 同一生成器工厂，两次遍历行数恒等）作为 summary 的落盘实数。
    dense_rows_total = 0
    stream, close = open_output(args.output, "slo_hbm_plot_series.csv",
                                args.run_dir)
    try:
        if dense_refused:
            write_csv(stream, SERIES_COLUMNS, iter(()))
        else:
            series_rows = _plot_series_rows(cplog, repo_variant, span_start,
                                            span_end, effective_bucket_ns)
            def counted_series():
                nonlocal dense_rows_total
                for row in series_rows:
                    dense_rows_total += 1
                    yield row
            write_csv(stream, SERIES_COLUMNS, counted_series())
    finally:
        if close:
            stream.close()

    # -- 违规口径（tier 决定正式性）----------------------------------------
    if journal_replay is not None:
        violation_events = journal_replay.physical_violation_events
        violation_instances = journal_replay.violation_by_instance
        max_exceed_bytes = journal_replay.max_exceed_bytes
        violation_kind = ("per_rank_physical_gt_capacity（正式判决）"
                          if trust_tier == TRUST_TIER_CERTIFIED else
                          f"per_rank_physical_gt_capacity（{trust_tier} 层"
                          f"仅报告，不作正式判决）")
    else:
        violation_events = replay.violation_events
        violation_instances = replay.violation_by_instance
        max_exceed_bytes = replay.max_exceed_bytes
        violation_kind = ("aggregate_occupancy_gt_aggregate_capacity"
                          "（upper_bound_only 层诊断：不构成物理违规认证"
                          "、退出码不为 3）")

    # -- 输出：instances CSV ------------------------------------------------
    upper_bound_peaks: dict[int, int] = {}
    if journal_replay is not None:
        for instance, unit in replay.cplog.units.items():
            if unit.has_event:
                upper_bound_peaks[instance] = unit.peak
    instances_csv_coverage = ("journal_exact" if journal_replay is not None
                              else coverage)
    estream, eclose = open_output(
        args.instances_csv, "slo_hbm_watermark_instances.csv", args.run_dir)
    try:
        def instance_rows():
            for instance in units:
                stat = cplog.units[instance].stats()
                yield (repo_variant, instance, trust_tier,
                       instances_csv_coverage,
                       "true" if occupancy_valid else "false",
                       capacity if capacity is not None else NA,
                       capacity_source, bucket_ns,
                       "true" if bucket_provisional else "false",
                       effective_bucket_ns,
                       "true" if resolution_adjusted else "false",
                       stat["first_event_ns"], stat["last_event_ns"],
                       stat["duration_ns"], stat["peak_occupancy_bytes"],
                       f"{stat['mean_occupancy_bytes']:.3f}",
                       stat["residual_occupancy_bytes"],
                       stat["evict_events"],
                       # P1-⑤：full_reconciled 也输出真实逐出 bytes（JSON
                       # 本就有，NA 展示分支只保留给 count_only 档）。
                       # journal 路径激活时逐出列与 summary total 同源
                       # （journal 重放，hbm-P2 单一来源），不再受
                       # decision-log coverage 门控。
                       stat["evict_bytes"] if (
                           journal_replay is not None
                           or coverage in ("full", "full_reconciled"))
                       else NA,
                       violation_instances.get(instance, 0),
                       upper_bound_peaks.get(instance, NA))
        write_csv(estream, INSTANCE_COLUMNS, instance_rows())
    finally:
        if eclose:
            estream.close()

    if coverage == "full":
        occupancy_note = "full：逐出条目带 bytes+victim，重放闭合"
    elif coverage == "full_reconciled":
        occupancy_note = (
            "full_reconciled：逐出条目带 bytes+victim；另有静默释放在恢"
            "复点按账本断言对账（silent_evictions_reconciled="
            f"{replay.report.actions['silent_evictions_reconciled']}、"
            f"restore_prefix_reconciled="
            f"{replay.report.actions['restore_prefix_reconciled']}）；对账"
            f"窗口（静默释放真实时刻不可见）内占用为上界")
    else:
        occupancy_note = (
            "S2 decision log 只有 *_eviction_count（无 victim/bytes）：他"
            f"人逐出不可归因，occupancy 为上界（未归因逐出 "
            f"{replay.report.actions['unattributed_evictions']} 例不扣减）")

    summary = {
        "command": "hbm_watermark",
        "repo_variant": repo_variant,
        "algorithm": ("kv_delta_journal 权威重放（tier>=resident_kv_exact）"
                      "或 decision-log KV action replay（upper_bound_"
                      "only；evict -> restore/move -> grow, per-record "
                      "order）-> RLE 变点区间 + 事件流 stats + 行预算绘"
                      "图 series（WP8 offline primary source）"),
        "trust_tier": trust_tier,
        "trust_tier_semantics": {
            TRUST_TIER_CERTIFIED:
                "journal+checksum 在场且 sha256/行级链/终态/守恒四项全过"
                "——逐 rank physical 认证，违规=exit 3（正式容量判决）",
            TRUST_TIER_RESIDENT:
                "journal 在场、链自洽，checksum 缺失——逐 rank 时序精确、"
                "无守恒证书，容量检查仅报告",
            TRUST_TIER_LIFECYCLE:
                "journal+checksum 在场、链与终态一致，守恒 checks 未全过"
                "——生命周期精确、无守恒证书，容量检查仅报告",
            TRUST_TIER_UPPER_BOUND:
                "journal 缺失——decision-log 重放为上界，occupancy_valid"
                "=false，无物理违规认证（退出码不为 3）",
        },
        "occupancy_source": occupancy_source,
        "bucket_ns": bucket_ns,
        "bucket_ns_provisional": bucket_provisional,
        "bucket_ns_source": "slo_params_manifest.watermark_sample_period_ns"
                            + ("（临时锚点 5,000,000 ns）"
                               if bucket_provisional else ""),
        "plot_series": {
            "requested_bucket_ns": bucket_ns,
            "effective_bucket_ns": effective_bucket_ns,
            "resolution_adjusted": resolution_adjusted,
            "adjustment_reason": adjustment_reason,
            "row_budget": row_budget,
            "span_ns": span_ns,
            "bucket_origin_ns": span_start,
            "b_eff_formula": "max(B_requested, ceil(S·N/(R−N)))；"
                             "R≤N 拒绝稠密输出只给 RLE",
            "eventful_units": n_units,
            "dense_rows_written": dense_rows_total,
            "dense_output_refused": dense_refused,
            "worst_dense_rows_estimate_requested": worst_rows_requested,
            "worst_dense_rows_estimate_effective": worst_rows_effective,
        },
        "span_start_ns": span_start,
        "span_end_ns": span_end,
        "n_buckets": (span_end - span_start) // effective_bucket_ns + 1,
        "n_instances": n_units,
        "interval_rows": interval_rows_total,
        "coef_bytes_per_token": coef,
        "coef_formula": "2*layers*hidden_size*bytes_per_elem",
        "trace_config_source": trace["path"],
        "token_manifest_source": tokens["path"],
        "capacity_bytes_per_instance": capacity,
        "capacity_source": capacity_source,
        "capacity_field_evidence": mapping["capacity_field_evidence"],
        "capacity_calibers": calibers if calibers is not None else NA,
        "capacity_calibers_note": calibers_note,
        "eviction_coverage": coverage,
        "occupancy_valid": occupancy_valid,
        "occupancy_note": occupancy_note,
        "violation_check": violation_check,
        "violation_kind": violation_kind,
        "violation_events": violation_events,
        "violation_instances": {str(k): v for k, v in
                                sorted(violation_instances.items())},
        "max_exceed_bytes": max_exceed_bytes,
        # 逐出总量单一来源（hbm-P2）：journal 路径激活时取 journal 重放
        # 计数（与 instances CSV 逐出列同源），decision-log 路径取重放
        # 报告——不再三源混用。
        "total_evict_events": (journal_replay.evict_events_total
                               if journal_replay is not None
                               else replay.report.actions["evictions"]),
        "total_evict_bytes": (journal_replay.evict_bytes_total
                              if journal_replay is not None
                              else replay.report.actions["evict_bytes"]),
        "partial_evictions": replay.report.actions["partial_evictions"],
        "actions": replay.report.actions,
        "anomalies": replay.report.anomalies,
        "restore_bytes_ratio_histogram": replay.report.restore_ratio_hist,
        "instances": {
            str(instance): {
                "peak_occupancy_bytes": stat["peak_occupancy_bytes"],
                "mean_occupancy_bytes": stat["mean_occupancy_bytes"],
                "duration_ns": stat["duration_ns"],
                "residual_occupancy_bytes":
                    stat["residual_occupancy_bytes"],
                "evict_events": stat["evict_events"],
                "evict_bytes": stat["evict_bytes"],
                # 与顶层 interval_rows 同源（merged 变点流行数，含逐出
                # 叠加行）——事件变点数会漏掉独立逐出 tick 的行，Σ 对不
                # 上顶层（hbm-P2）。
                "interval_rows": sum(
                    1 for _ in cplog.units[instance].merged_change_points()),
                "violation_events": violation_instances.get(instance, 0),
            }
            for instance, stat in ((u, cplog.units[u].stats())
                                   for u in units)
        },
    }
    if journal_replay is not None:
        summary["journal_replay"] = {
            "journal_path": str(prep["journal_path"]),
            "checksum_path": (str(prep["checksum_path"])
                              if prep["checksum_present"] else None),
            "sha256": journal_result["sha256_hex"],
            "rows": journal_replay.rows,
            "sequence_first": journal_replay.sequence_first,
            "sequence_last": journal_replay.sequence_last,
            "transaction_max": journal_replay.transaction_max,
            "causes": journal_replay.causes,
            "checks": journal_result["checks"],
            "per_rank_final": journal_replay.final_rank_state(),
            "per_rank_physical_violations": {
                str(rank): count for rank, count in
                sorted(journal_replay.violation_by_rank.items())},
            "resident_hard_limit_exceed_events":
                journal_replay.hard_limit_exceed_events,
            "resident_hard_limit_exceed_ranks": {
                str(rank): count for rank, count in
                sorted(journal_replay.hard_limit_exceed_ranks.items())},
        }
        # 对照列：decision-log 重放（上界）per instance 峰值/残差——与
        # journal 权威值的差 = 账本缺口可视化（17.24TB 幻影的对照面）。
        # "-1" 哨兵（无归因逐出标记的挂靠桶）不是实例，滤除不进对照表。
        comparison = {}
        for instance in sorted({u for u in units}
                               | set(replay.cplog.units)):
            if instance < 0:
                continue
            unit = replay.cplog.units.get(instance)
            entry = {
                "upper_bound_peak_occupancy_bytes":
                    unit.peak if unit is not None and unit.has_event else NA,
                "upper_bound_residual_occupancy_bytes":
                    unit.occupancy if unit is not None and unit.has_event
                    else NA,
            }
            auth = cplog.units.get(instance)
            if auth is not None and auth.has_event:
                entry["authoritative_peak_occupancy_bytes"] = auth.peak
                entry["authoritative_residual_occupancy_bytes"] = \
                    auth.occupancy
                if isinstance(entry["upper_bound_peak_occupancy_bytes"],
                              int):
                    entry["peak_gap_bytes"] = (
                        entry["upper_bound_peak_occupancy_bytes"]
                        - auth.peak)
            comparison[str(instance)] = entry
        summary["decision_log_upper_bound_comparison"] = comparison

    print("", file=sys.stderr)
    emit_json(sys.stderr, summary)
    if args.json:
        jstream, jclose = open_output(args.json,
                                      "slo_hbm_watermark.json", args.run_dir)
        try:
            emit_json(jstream, summary)
        finally:
            if jclose:
                jstream.close()

    if violation_events and trust_tier == TRUST_TIER_CERTIFIED:
        # 正式容量判决（仅 certified 层）：fail-loud。
        print("", file=sys.stderr)
        print("!! [hbm-watermark] 容量违规（正式判决，per_rank_total_hbm_"
              f"certified）：逐 rank physical > capacity_bytes 共 "
              f"{violation_events} 个事件点（rank 分布 "
              f"{summary['journal_replay']['per_rank_physical_violations']}，"
              f"最大超出 {max_exceed_bytes} B）——manager 逐 rank 不变量"
              f"被破坏；宁可报错不可静默，退出码 {EXIT_VIOLATION}",
              file=sys.stderr)
        return EXIT_VIOLATION
    if violation_events:
        # 非 certified 层：诊断报告（tier 已标注），不构成物理违规认证。
        print("", file=sys.stderr)
        print("[hbm-watermark] 超限诊断（非正式判决；trust_tier="
              f"{trust_tier}）：{violation_kind} 共 {violation_events} 个"
              f"事件点（实例分布 {summary['violation_instances']}，最大"
              f"超出 {max_exceed_bytes} B）——tier 语义见 summary；"
              f"退出码保持 0", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hbm_watermark.py",
        description="WP8 补充主数据源：离线重放重建每实例 HBM KV 水位线"
                    "（journal 权威重放或 decision-log 上界重放，四层可"
                    "信度自动判定）→ RLE 权威区间 + 行预算绘图 series + "
                    "事件流 stats + 逐 rank 三口径容量认证。fail-closed："
                    "缺文件/缺列/重放不一致/journal 损坏即退出码 2；容量"
                    "违规退出码 3 仅 per_rank_total_hbm_certified 层。")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 results/online_decision_log."
                             "jsonl、cpp.log；token manifest 与 "
                             "trace_config 可自动定位或显式指定；含 "
                             "results/kv_delta_journal.jsonl 时走 journal "
                             "权威路径）")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="slo_params_manifest.json（默认：脚本同目录；"
                             "watermark_sample_period_ns null → 5,000,000 ns"
                             " 临时锚点并标 provisional；"
                             "watermark_series_row_budget 为绘图 series "
                             "全局行预算）")
    parser.add_argument("--token-manifest", type=Path, default=None,
                        help="token manifest（manifest.json：requests[]."
                             "prefill_context_tokens/final_context_tokens/"
                             "history_tokens_before；默认：run_dir/manifest."
                             "json > cpp.log init manifest_path 同目录）")
    parser.add_argument("--request-manifest", type=Path, default=None,
                        help="request manifest（metrics_manifest.json："
                             "npus_per_instance 取 requests[].prefill_ranks "
                             "长度）")
    parser.add_argument("--trace-config", type=Path, default=None,
                        help="trace_config.csv（config 行 layers/hidden_size/"
                             "bytes_per_elem/local_hbm_capacity_profile 及三"
                             "口径剖面所需模型行）")
    parser.add_argument("--hardware-config", type=Path, default=None,
                        help="hardware json（local-hbm.capacity-profiles；"
                             "默认搜脚本所在仓 sh_test_mesh/hardware/）")
    parser.add_argument("--intervals-csv", default="",
                        help="权威 RLE 变点区间 CSV（'-'=stdout；缺省写 "
                             "run_dir/slo_hbm_intervals.csv）")
    parser.add_argument("-o", "--output", default="",
                        help="绘图 series CSV（行预算约束；'-'=stdout；"
                             "缺省写 run_dir/slo_hbm_plot_series.csv；旧产"
                             "物名 slo_hbm_watermark_series.csv 已退役）")
    parser.add_argument("--instances-csv", default="",
                        help="逐实例汇总 CSV（缺省写 run_dir/"
                             "slo_hbm_watermark_instances.csv）")
    parser.add_argument("--json", default="",
                        help="汇总 JSON 输出路径（可选）")
    parser.add_argument("--repo-variant", default=None,
                        help="显式指定 repo_variant（默认读 cpp.log init 行）")
    args = parser.parse_args()
    return int(cmd_hbm_watermark(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
