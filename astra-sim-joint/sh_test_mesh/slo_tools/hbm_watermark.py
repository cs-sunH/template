#!/usr/bin/env python3
"""WP8 补充主数据源：离线 HBM KV 水位线重建（五仓逐字节相同，标准库实现）。

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
    一致，但证书四项 checks 有 false（run 末守恒未通过，如残差非零）：
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

M17（2026-09-29，如实披露）：per-rank 权威 journal（results/
kv_delta_journal.jsonl + kv_delta_journal_checksum.json）在本仓**没有
任何生产者**——注意 online_service.dump_joint_kv_ledgers 落盘的第四键
"kv_delta_journal" 是 C14 逐请求结算事实 sidecar（JSON 键内嵌列表、
行结构 = seq/session_id/home_before/...），与本工具消费的 per-rank
delta 权威账本（JSONL、schema_version=1、rank/delta/before/after 字段
面）同名不同物，全仓对后者零写入方。故本仓自产 run 恒落
upper_bound_only 层，certified 容量判决路径（exit 3）在自产 run 上
不可达；上层文档描述的该判决路径需外部 journal 生产者落盘后才真实
存在。四层判定逻辑本身不变（tier 由 run_dir 内容自动判定，外部
journal 在场即按上表升级）；summary 以 ``certification_layer`` 字段
显式披露当前层，静默降级就此变为显式披露。

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
    load_request_manifest, load_slo_manifest, manifest_requests,
    open_output, read_init_record, require_param_int, run_main, write_csv,
)

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
JOURNAL_TIERS = (TRUST_TIER_CERTIFIED, TRUST_TIER_RESIDENT,
                 TRUST_TIER_LIFECYCLE)

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
    """FACE/W 逐出条目：shard_bytes 列表 → 实例合计。

    bool 排除（2026-09-29，对齐姊妹文件 hopbytes.py D2 的 _byte_count
    口径）：isinstance(True, int) 为真，bool shard 曾按 1 字节计入合计；
    现 bool 一律 None → 调用方 _evict_action 走既定"缺可解析 bytes"
    fail-closed 出口，不把布尔值当字节。
    """
    shards = entry.get("shard_bytes")
    if not isinstance(shards, list) or not shards:
        return None
    total = 0
    for value in shards:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        total += value
    return total


def _scalar_bytes(entry: dict) -> Optional[int]:
    # bool 排除（2026-09-29，对齐 hopbytes.py D2 口径）：
    # isinstance(True, int) 为真，total_bytes=true 曾按 1 字节放行；
    # bool → None → 调用方 _evict_action fail-closed，不静默当 1 字节。
    value = entry.get("total_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _strict_present_int(value, what: str, where: str) -> int:
    """标量 bytes 臂的在场值严格校验（R8-F1，2026-09-29 face 移植）。

    "在场但非 int"（float/str/bool/list…）一律 fail-closed：静默按 0 处理
    会把"在场非法"与"缺席=0"混为一谈、零 anomaly 直通幽灵占用出口；
    bool 经 isinstance int 混入会按 1 计账。与 restore_list_field 严格臂
    同强度（该臂缺/非法 total_bytes 即 fail），臂间校验强度对称。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{where}: {what} 在场但不是整数（实得 {value!r}）——"
             f"float 截断/str 裸崩/bool 按 1 计均禁止，拒绝静默按 0 处理")
    return value


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
        # 幻影占用零异常）——fail-closed（批次 D 镜像，2026-09-30）。
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
    # R17-4：层区间与新序列化键（resident_prefix_layers_before/after，
    # 仿真侧 R17-1d 新增；旧日志无此键 → None）透传给 apply_evict 的
    # 驻留前缀校验；非 joint 仓不消费这些键。
    return {"type": "evict", "tick": eff_tick, "session": victim_session,
            "bytes": nbytes, "instance": victim_instance,
            "kind": entry.get("kind"), "reason": entry.get("reason"),
            "layer_start": entry.get("layer_start"),
            "layer_end": entry.get("layer_end"),
            "resident_prefix_layers_before":
                entry.get("resident_prefix_layers_before"),
            "resident_prefix_layers_after":
                entry.get("resident_prefix_layers_after")}


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
            holders = [row for row in rows if isinstance(row, dict)]
        else:
            fail(f"{where}: {restore_list_field} 必须是数组")
        actions = []
        for row in holders:
            row_kind = row.get("kind")
            row_bytes = row.get("total_bytes")
            if isinstance(row_bytes, bool) or not isinstance(row_bytes, int) \
                    or row_bytes < 0:
                fail(f"{where}: {restore_list_field} 成员缺非负整数 "
                     f"total_bytes（实得 {row_bytes!r}）")
            if row_kind in ("local_hit", "LOCAL_HIT"):
                row_bytes = 0
            row_source = row.get("source_instance_index")
            if row_source is not None and not isinstance(row_source, int):
                row_source = None
            actions.append({
                "type": "restore", "tick": record.get("tick", 0),
                "session": record.get("session_hint"), "bytes": row_bytes,
                "kind": row_kind, "logged_source": row_source,
                "state_before": None,
                "target": decision.get("prefill_instance_index")})
        return actions

    for path_element in spec.get("bytes_path", []):
        if isinstance(holder, dict):
            holder = holder.get(path_element)
        else:
            holder = None
    if isinstance(holder, dict):
        if "total_bytes" in holder:
            # R8-F1（2026-09-29 face 移植）：在场值严格校验
            # （缺席=0 与"在场非法"不得混同）。
            nbytes = _strict_present_int(holder.get("total_bytes"),
                                         "history 恢复 total_bytes", where)
        kind = holder.get("kind")
        source = holder.get("source_instance_index")
        # F2_PY 修复（2026-09-29，RV5-F7 等价移植）：source 在场非整数
        # （含 bool、浮点、字符串）此前静默降 None，下游
        # restore_source_mismatch 诊断计数随之被静默抑制（异常漏报）、
        # bool 冒充实例 1——与 bytes 臂的 _strict_present_int 同强度，
        # 在场即严格；键缺席/None 保留 None（跟踪态归位）语义。
        if source is not None and (isinstance(source, bool)
                                   or not isinstance(source, int)):
            fail(f"{where}: history 恢复 source_instance_index 非整数"
                 f"（{source!r}）——类型谎报不静默降级")
        if isinstance(source, int):
            logged_source = source
    elif holder is not None:
        nbytes = _strict_present_int(holder, "history 恢复 bytes", where)
    if spec.get("kind_field"):
        value = decision.get(spec["kind_field"])
        if isinstance(value, str):
            kind = value
    if spec.get("source_field"):
        value = decision.get(spec["source_field"])
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, int)):
            fail(f"{where}: history 恢复 {spec['source_field']} 非整数"
                 f"（{value!r}）——类型谎报不静默降级")
        if isinstance(value, int):
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
        if "total_bytes" in holder:
            # R8-F1（2026-09-29 face 移植）：在场值严格校验
            # （同 restore 标量臂，臂间对称）。
            nbytes = _strict_present_int(holder.get("total_bytes"),
                                         "decode 迁移 total_bytes", where)
        source = holder.get("source_instance_index")
        # F2_PY 修复（2026-09-29，RV5-F7 等价移植）：同 restore 标量臂——
        # source 在场非整数（含 bool、浮点、字符串）不再静默降 None/按 1
        # 计，fail-closed；键缺席/None 保留 None（跟踪态归位）语义。
        if source is not None and (isinstance(source, bool)
                                   or not isinstance(source, int)):
            fail(f"{where}: decode 迁移 source_instance_index 非整数"
                 f"（{source!r}）——类型谎报不静默降级")
        if isinstance(source, int):
            logged_source = source
    elif holder is not None:
        nbytes = _strict_present_int(holder, "decode 迁移 bytes", where)
    if spec.get("source_field"):
        value = decision.get(spec["source_field"])
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, int)):
            fail(f"{where}: decode 迁移 {spec['source_field']} 非整数"
                 f"（{value!r}）——类型谎报不静默降级")
        if isinstance(value, int):
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
            ("prefill", "admission_evictions", _sum_shard_bytes),
            ("prefill", "decode_target_evictions", _sum_shard_bytes),
            ("completion", "completion_evictions", _sum_shard_bytes),
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
    # index（0.5 比值=半层恢复）；decode 恒 P==D（红线 #4）无迁移记录；
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
    "astra-sim-joint": {
        "eviction_lists": (
            ("completion", "completion_evictions", _scalar_bytes),
            ("prefill", "history_evictions", _scalar_bytes),
            # R17-2a（2026-09-16）：prefill_evictions 不读取——joint 下该
            # 字段结构性恒空（准入 R1' 预约覆盖 prefill 全动作足迹、drain
            # expand gap≡0，2026-09-16 三方裁决）；旧 joint 日志该字段恒空，
            # 移除零行为差（sh_1.0 的同名字段不受影响）。
            ("decode", "decode_evictions", _scalar_bytes),
        ),
        "restore": {"bytes_path": ["history_transfer_bytes"],
                    "source_field": "history_source_instance_index"},
        # joint 行不走通用 _decode_move_actions：P→D 迁移由 _joint_decode
        # 按决策行 prefill_decode_transfer 披露重放（源扣/目标加，W1
        # 2026-09-27）；本键置 None 仅表示通用路径不消费。
        "decode_move": None,
        "completion_relocation_field": None,
        "eviction_count_fields": (),
        "eviction_coverage": "full_reconciled",
        "capacity_field_evidence":
            "trace_config: local_hbm_capacity_profile → hardware json 同 FACE",
        # R12（2026-09-14）：joint 专属事件入重放词表——跨实例工作副本
        # 双驻留记账（base/working）、merge 增量回传（noc→home）与池写、
        # home_merge_base_degrade 自降级、recompute@home 缺失后缀物化
        # （R13）。决策行的 joint 专属审计行
        # （joint_admission/joint_admission_failed/joint_admission_wait/
        # joint_decode_stall/joint_decode_wake）非账本行，跳过不进重放。
        # merge v2（2026-09-17《部分层逐出kv管理改造分析方案》需求①②）：
        # completion 行 merge_direction=forward/reverse/in_place 走 v2
        # 重放（_joint_completion_v2——零池写断言/方向交叉校验/home 迁移
        # 计数 joint_home_migration；copy PARTIAL 基的准入池恢复后缀经
        # history_transfers 腿入账）；无该字段的旧日志走 legacy。
        # W2（2026-09-27，C4/D12 三位置形态）：decode 行
        # prefill_decode_transfer 的无迁移边界披露（local_hit@新执行端、
        # 字节 0、reason=boundary_rebuild_elsewhere）把 i_P 旧工作副本
        # 转来源侧记录
        # （session 第三槽），completion 行按 winner/loser 位置集合做三位置
        # 结算（_joint_completion_three_position——逐败者整份腿源扣/胜者加、
        # 重建形态零载荷静默释放、胜者终态守恒 == f(final)、reverse ⟺
        # 胜者=执行端的方向交叉校验）；两位置 forward/reverse/in_place
        # 行为零改动（无来源槽时不进新分支）。
        #
        # C19（2026-09-22，C15 §25.7 登记的跨车道观察项裁置）：C8/C11/
        # C14 起决策日志新增的**非账本披露行**并入跳过词表——
        # readplan_reconcile/readplan_settle（C8 #readplan 预登记对账/
        # 残差核销披露）、quota_oneshot_overflow（C11 配额 oneshot 超
        # 额披露）、merge_done（C14 合并结算闭合披露行——KV 账本走
        # completion 行 merge_direction v2 重放，本行仅审计）、
        # link_telemetry_coverage / joint_decision_metrics（C8/C11 的
        # run 级收尾披露行，request_id 恒空）。这些行不载运逐出/恢复/
        # 增长等任何 KV 账本事件，跳过 = 零重放语义差；缺 request_id
        # 的 run 级行在原 kind 门下会误触 fail-closed（C15 stress 对拍
        # 实证）。修法约束履行：仅本工具侧 kind 门扩容（C16 domain_
        # metrics 的非账本行静默跳过同款模式），C5 冻结 schema 与 SH
        # 零改动；词表仅挂 astra-sim-joint 映射，非 joint 仓零影响。
        # M4（2026-09-26）：kv_eviction_done（逐出尾标记完成披露行，
        # 与 kv_eviction 行按 decision.watch_id 配对得实际写回时长）同
        # 款入表——账本侧逐出事件已在 kv_eviction 行重放，本行仅时间
        # 线测量披露，跳过 = 零重放语义差。
        # W3（2026-09-27，C6 纯度门拒收披露）：joint_sample_contended
        # （ServiceFactors 争用样本拒收 + 观测窗口端口占用摘要）入表
        # ——纯采样纯度披露，不载运任何 KV 账本事件；request_id 可为空
        # （own_request_ids 空时发射面落 ""），跳过位于 request_id 门之
        # 前故不需豁免。readplan_reconcile/readplan_settle 属历史 run 的
        # 纯审计行词表（同款零占用语义，逐字节对拍钉测随远读退役批整删）。
        # W4（2026-09-27，D 边界就地并基折叠）：decode 行 prefill_decode_
        # transfer 的迁移腿落点 == base 槽实例 = workload 折叠形态
        # （complete_copy_base_fold / increment_copy_home_u_only——
        # base_* 转正/释放后清账，下一轮 stash 是合法重建）——按迁移腿
        # 字节与 f(P) 的恒等式二分清账（完整型：重复基础副本释放；
        # 增量型：基础前缀就地转正并入、增长基切完整上下文族），见
        # _apply_boundary_base_fold；折叠轮
        # 的 completion 行 merge_direction=stay 与 base 槽在册互斥
        # （未知形态 fail-closed）。钉测 test_hbm_watermark_base_fold。
        "joint": True,
        "joint_audit_kinds": (
            "joint_admission", "joint_admission_failed",
            "joint_admission_wait", "joint_decode_stall",
            "joint_decode_wake",
            "readplan_reconcile", "readplan_settle",
            "quota_oneshot_overflow", "merge_done",
            "link_telemetry_coverage", "joint_decision_metrics",
            "kv_eviction_done",
            "joint_sample_contended",
            # G1（2026-09-27，sh30 D 边界决策行消费侧补词表）：
            # boundary_duplicate_ignored / boundary_commit_fallback /
            # boundary_terminal 是 sh30 生产端无条件落决策行的纯披露行
            # （来源锚：sh30_online_scheduler._on_prefill_drain 落前两
            # 行、_settle_terminal_at_boundary 落第三行）——duplicate 行
            # 注释自证幂等忽略零账本副作用；terminal/fallback 的结算走
            # 主账行（release/completion 等账本行重放），本行仅披露。
            # 三行不载运任何 KV 账本事件，跳过 = 零重放语义差；缺词表时
            # 消费端唯一顶层未知-kind 门会 fail-closed 误红。
            "boundary_duplicate_ignored", "boundary_commit_fallback",
            "boundary_terminal",
            # W5（2026-09-29，§7.4 统一结算）：settle_done 是统一结算闭
            # 合披露行（SETTLE_COMMITTED -> SETTLE_DONE；echo service_
            # done_ns/merge_done_ns/载荷字节/策略）——纯审计行，KV 账
            # 本走 completion 行 merge_direction v2 重放（与 merge_done
            # 行同款），跳过 = 零重放语义差。
            "settle_done",
            # W9（2026-09-29，§7.4/§10）：distributed_directory_stats 是
            # W9 目录开销 run 级披露行（条目峰值/逐会话逐轮来源面/目录
            # 操作计数与累计墙时）——纯审计行，不载运 KV 账本事件，跳过
            # = 零重放语义差。
            "distributed_directory_stats",
            # cfg1 补词表（2026-10-01 审查轮）：forecast_status 是预测引
            # 擎收尾披露行（sh30_online_scheduler 的
            # _joint_forecast_status_decision 在 run 末落行、request_id 恒
            # 为空串）——背景预测引擎 disclosure 与 P/D 完成预测成熟账
            # 的纯披露行，不载运 KV 账本事件，跳过 = 零重放语义差。缺
            # 词表时消费端顶层"决策记录缺 request_id"门会 fail-closed
            # 误红（任何正常 joint run 的水印重放必崩 exit 2）。
            "forecast_status",
        ),
    },
}


# ---------------------------------------------------------------------------
# manager 原函数（逐字拷贝）+ 容量三口径（P1-③）
# ---------------------------------------------------------------------------
# 以下六个函数逐字拷贝自仓内 workload/llama2_7b_inference/
# session_kv_manager.py（_require_nonnegative_int / partition_values_exact /
# attention_heads_by_tp_rank / estimate_model_weight_bytes /
# model_weight_shard_bytes_by_tp_rank(:185) / kv_cache_shard_bytes_for_
# tokens(:220)）——manager 是逐 rank HBM 记账的权威实现，本工具不得另立
# 口径。**同步义务**：session_kv_manager.py 上述函数任何改动必须同步拷贝
# 到本节（五仓同改，md5 对齐）；函数是五仓共性，模型参数由 run_dir 自带
# 的 trace_config/hardware 配置实例化（五仓 hardware 配置可不同）。

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
    """Match the existing LLaMA-family model-size estimate exactly."""

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


def kv_cache_shard_bytes_for_tokens(
    model: Any,
    tokens: int,
    tp_degree: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV shard bytes for a complete model cache."""

    _require_nonnegative_int(tokens, "tokens")
    if model.hidden_size % model.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    head_dim = model.hidden_size // model.num_heads
    bytes_per_head = 2 * model.layers * tokens * head_dim * model.bytes_per_elem
    shards = tuple(
        bytes_per_head * head_count
        for head_count in attention_heads_by_tp_rank(model.num_heads, tp_degree)
    )
    expected = 2 * model.layers * tokens * model.hidden_size * model.bytes_per_elem
    if sum(shards) != expected:
        raise RuntimeError("whole-head KV shards do not preserve total KV bytes")
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
    # 负容量档门（2026-09-29，报告 4.3）：capacity < 任一 rank 的 weight
    # shard 时 ⌊(capacity−weight)/kv⌋ 为负 → hard_tokens/limit 全为负值，
    # journal 硬上限诊断随之全量误报。硬件容量装不下权重分片属配置矛盾，
    # 走本函数既有的 ValueError 通道（调用方 warning + capacity_calibers=
    # NA，绝不静默 clamp 出负上限）；journal 正式认证用行内 capacity_bytes，
    # 不受本门影响。
    if capacity_bytes_per_npu < max(weight_shards):
        raise ValueError(
            f"capacity_bytes_per_npu ({capacity_bytes_per_npu}) < 逐 rank "
            f"weight shard 最大值 ({max(weight_shards)})——容量档装不下模型"
            f"权重，拒绝产出负 hard_tokens/limit")
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
    coef = sum(kv_shards_per_token)
    hard_tokens_by_weight = {}
    ambiguous_weights = set()
    for weight, kv_per_token in zip(weight_shards, kv_shards_per_token):
        if kv_per_token <= 0:
            continue  # 0 头 shard 无 KV 上限语义，不参与绑定
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
            "limit_bytes_per_instance": coef * hard_tokens,
            "limit_resident_bytes_by_relative_rank": [
                kv_per_token * hard_tokens
                for kv_per_token in kv_shards_per_token],
            "limit_by_weight_bytes": hard_tokens_by_weight,
            "ambiguous_weight_binding": sorted(ambiguous_weights),
            "semantics": "reservation=0 时逐 rank ⌊(capacity−weight_r)/"
                         "kv_r⌋ 最小 token 数 × Σkv_r——任意时刻成立的 "
                         "resident-KV 硬上限（诊断口径）",
        },
        "watermark_reserve_target": {
            "reserve_context_tokens": reserve_context_tokens,
            "reserve_shard_bytes_by_relative_rank": list(reserve_shards),
            "target_tokens": target_tokens,
            "target_bytes_per_instance": coef * target_tokens,
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
# 的 _transfer_entry_rows）落地后的 full 口径升级。判定字段 =
# history_transfers（partial 两段式恢复逐段对象）：新产物每个 prefill 决策
# 恒在场（空列表亦在场），旧产物（B3 前基线）与跨仓通用 fixture 均无此
# 字段——比检测 *_evictions 键名更严格（后者在通用 fixture 的默认
# decision 里也会出现，会把 count_only 旧语义误升级）。
S2_FULL_TRIGGER_FIELD = "history_transfers"


def upgrade_s2_mapping_if_fields_present(mapping: dict,
                                         run_dir: Path) -> dict:
    """字段在场 → full 口径；旧产物（B3 前基线）→ 原 count_only 映射。

    full 口径语义（与 S1 对齐）：逐出条目可归因（victim session + bytes +
    source 实例）；restore 按 history_transfers 逐段对账（partial 前缀
    noc_migrate 搬移 + suffix remote_load 只增，标量聚合会把跨实例 partial
    迁移错记成整体搬移而击穿占用下界）；completion 的自身释放已含在
    completion_evictions（enforce_reserve 含自身逐出）→ 关闭
    kv_location_after_completion 归因避免双重扣减；*_eviction_count 不再
    计入 unattributed（逐出已逐条归因）。"""
    log_path = run_dir / DECISION_LOG_RELPATH
    present = False
    for record in iter_jsonl(log_path):
        decision = record.get("decision")
        if record.get("kind") == "prefill" and isinstance(decision, dict) \
                and isinstance(decision.get(S2_FULL_TRIGGER_FIELD), list):
            present = True
            break
    if not present:
        return mapping
    upgraded = dict(mapping)
    upgraded["eviction_lists"] = (
        ("prefill", "history_evictions", _scalar_bytes),
        ("prefill", "prefill_evictions", _scalar_bytes),
        ("decode", "decode_evictions", _scalar_bytes),
        ("completion", "completion_evictions", _scalar_bytes),
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
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
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
                # bool 冒充门（2026-09-29，报告 3.4）：isinstance(True, int)
                # 为真，true 曾通过下方的 int 门按 1 token 放行——token 事实
                # 被布尔值谎报。bool 显式拒绝，错误信息说明收到布尔值。
                if isinstance(value, bool):
                    fail(f"token manifest {request_id} {field} 收到布尔值"
                         f"（{value!r}）——bool 是 int 子类但不是合法 "
                         f"token 数，不放行（{candidate}）")
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
            config[row[1]] = row[2]
    return config


def load_trace_config(run_dir: Path, explicit: Optional[Path]) -> dict:
    """trace_config.csv 的 config 行（layers/hidden_size/bytes_per_elem/
    local_hbm_capacity_profile）。

    优先级：--trace-config > run_dir/trace_config.csv >
    run_dir/trace_config.csv.snapshot（B0 归档名）> 脚本所在仓的
    sh_test_mesh/workload/llama2_7b_inference/trace_config.csv。
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        candidates.append(run_dir / "trace_config.csv")
        candidates.append(run_dir / "trace_config.csv.snapshot")
        candidates.append(
            Path(__file__).resolve().parent.parent
            / "workload" / "llama2_7b_inference" / "trace_config.csv")
    for candidate in candidates:
        if not candidate.is_file():
            continue
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
    fail(f"找不到 trace_config.csv（尝试：{[str(c) for c in candidates]}）"
         f"——用 --trace-config 指定")


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
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
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
        # bool 冒充门（2026-09-29，报告 3.4）：isinstance(True, int) 为真，
        # capacity_bytes=true 曾按 1 字节容量放行 → journal 权威重放逐事件
        # 判违规 → 正式物理违规判决（退出码 3）被布尔值错误触发。bool 显式
        # 拒绝，错误信息说明收到布尔值。
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

    __slots__ = ("tick", "start", "end", "peak", "delta")

    def __init__(self, tick: int, start: int) -> None:
        self.tick = tick
        self.start = start  # 本 tick 首事件前占用
        self.end = start
        self.peak = start  # 本 tick 内事件后占用的最大值（含进入值）
        self.delta = 0


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
        self.open.delta += delta
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
        """journal 逐出行记账（M-31 会话口径镜像，2026-09-30）：事件计数
        仅在该行的会话是本事务内首次出现时 +1（同会话一次逐出跨 rank 落
        多行，其余行只累加字节）；decision-log 路径的 add_evict（一次调用
        =一个事件）不变。"""
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
            # L18（2026-09-29）：单实例全部事件落在同一 tick（duration=0）
            # 时面积分为 0，退化口径 mean := peak（该实例在唯一事件 tick
            # 的占用即为峰；行为保留，仅补注记说明口径）。
            "mean_occupancy_bytes": (self.area / duration) if duration > 0
            else float(self.peak),
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

    add(tick, instance, delta) 要求 tick 对同一实例单调不减（decision-log
    重放的 tick 回退检查与 journal 的 planner_time_ns 单调检查保证）；
    逐出标记走独立 per-unit 表。内存 O(变点数+实例数)。
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
    桶 j，末桶 bucket_end 钳到 span_end+1（含恰落在 span_end 的事件）；
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
    """会话跟踪态：当前实例 + 当前本地 bytes（分层仓可为部分层）。

    R12（2026-09-14，joint）：跨实例工作副本期间双驻留记账——
    base_instance/base_bytes 描述 home 侧权威基础驻留（primary 的
    instance/bytes 此时描述执行端工作副本），消除 joint 专属事件
    （merge 回传/池写回/工作副本释放）不在词表时的"home 低估、执行端
    高估"系统性偏差。

    R17-4（2026-09-16，joint）：(primary, base) 二元组驻留前缀跟踪——
    prefix_layers/base_prefix_layers = 驻留层前缀 [0, n) 的层数（None=
    未知）；prefill 行成功处理后建立，逐出条目逐条校验推进（连锁不变
    量，前缀未知而逐出到场 = fail-closed，宁爆不漏）。

    W2（2026-09-27，joint C4/D12 三位置形态）：source_instance/source_
    bytes = 边界第二来源槽（i_P 侧旧工作副本——decode 行
    boundary_rebuild_elsewhere 无迁移披露建立，三位置合并结算时清账）；
    boundary_kind = 该边界的执行面动作族（"rebuild"=重建副本持完整
    并集——三动作下唯一形态），决定 decode 行增长口径与 completion
    结算形态，仅在来源槽建立至结算之间在册。

    W4（2026-09-27，joint 边界折叠）：base 槽除合并结算（v2/three-
    position）外另有折叠清账路径——D 边界迁移腿落 base 驻留实例的
    就地并基（完整型释放重复副本 / 增量型前缀转正并入），清账后下一
    轮跨实例 prefill 重新 stash 是合法重建（见 _apply_boundary_base_fold）。
    """

    __slots__ = ("instance", "bytes", "tokens", "base_instance",
                 "base_bytes", "prefix_layers", "base_prefix_layers",
                 "source_instance", "source_bytes", "boundary_kind")

    def __init__(self) -> None:
        self.instance: Optional[int] = None
        self.bytes: int = 0
        self.tokens: int = 0
        self.base_instance: Optional[int] = None
        self.base_bytes: int = 0
        self.prefix_layers: Optional[int] = None
        self.base_prefix_layers: Optional[int] = None
        self.source_instance: Optional[int] = None
        self.source_bytes: int = 0
        self.boundary_kind: Optional[str] = None


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
            # R6-U4/R7-F4（2026-09-29 face 移植）：restore 改指目标实例时
            # 占用随迁计数（LOCAL_HIT/0 字节臂）。
            "restore_occupancy_relocated": 0,
            "restore_occupancy_relocated_bytes": 0,
            # R17-2d 哨兵：kv_eviction 决策行逐条计数（reason 分桶键
            # evict_reason_kv_eviction:<reason> 动态写入）。
            "evict_entries_kv_eviction": 0,
        }
        self.anomalies = {
            "restore_bytes_mismatch": 0, "restore_source_mismatch": 0,
            "restore_kind_mismatch": 0, "move_bytes_mismatch": 0,
            "move_source_mismatch": 0, "local_hit_location_mismatch": 0,
            "evict_location_mismatch": 0,
            "evict_untracked_session": 0, "evict_exceeds_tracked": 0,
            # R8-F3（2026-09-29 face 移植）：own_relocation 去向字段
            # 缺席/枚举外值留痕计数。
            "own_relocation_missing_location": 0,
            "own_relocation_unknown_location": 0,
        }
        self.restore_ratio_hist: dict[str, int] = {}


class WatermarkReplay:
    """按文件顺序重放决策记录，产出逐实例 delta 事件流与违规计数。"""

    def __init__(self, repo_variant: str, mapping: dict, tokens: dict,
                 coef_bytes_per_token: int,
                 capacity_per_instance: Optional[int]) -> None:
        self.repo_variant = repo_variant
        self.mapping = mapping
        self.tokens = tokens
        self.coef = coef_bytes_per_token
        self.capacity = capacity_per_instance
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

    def stash_base(self, session_id: str) -> None:
        """R12（joint）：跨实例工作副本建立时把 home 侧基础驻留转 base
        记账——primary 的 instance/bytes 此后描述执行端工作副本。

        R17-4：驻留前缀随字节一并转入 base_prefix_layers；primary 前缀
        置 None（未知），由工作副本行的 history_transfers 区间/增长重建。"""
        session = self._session(session_id)
        if session.base_instance is not None:
            fail(f"joint base stash on already-stashed session "
                 f"{session_id!r}——重复建立工作副本，账本与重放不一致")
        if session.source_instance is not None:
            # W2：来源槽在册（D 边界已把 i_P 旧副本转来源侧）而新 prefill
            # 又要 stash base——两次边界之间无合并结算，合同违规（镜像
            # face_scheduler 边界的第二来源守卫）。
            fail(f"joint base stash on session {session_id!r} with a "
                 f"pending boundary source ({session.source_instance})——"
                 f"边界转记录后必须先经三位置合并结算，账本与重放不一致")
        session.base_instance = session.instance
        session.base_bytes = session.bytes
        session.base_prefix_layers = session.prefix_layers
        session.instance = None
        session.bytes = 0
        session.tokens = 0
        session.prefix_layers = None

    def _expected_bytes(self, tokens: int) -> int:
        return self.coef * tokens

    # -- R17-4：(primary, base) 驻留前缀连锁不变量（joint） ----------------

    def _advance_resident_prefix(self, session_id: str, session: SessionState,
                                 entry: dict, where: str,
                                 base: bool = False) -> None:
        """逐出条目驱动的驻留前缀推进（joint 连锁不变量，fail-closed）。

        条目带 resident_prefix_layers_before/after（仿真侧 R17-1d 新增
        序列化键）→ 断言 before == 当前跟踪前缀后推进到 after；旧日志无
        该键 → 回退 layer_end 规则（before=上一前缀、after=layer_start，
        不另立断言）。跟踪前缀未知（None）而逐出条目到场 = fail-closed
        （真阳性，宁爆不漏）。base=True 时作用于 base 前缀
        （home_merge_base_degrade 自降级——降级离开的是 home 侧基础驻
        留，不是 primary 工作副本）。非 joint 仓不适用（直接返回）。
        """
        if not self.mapping.get("joint"):
            return
        before = entry.get("resident_prefix_layers_before")
        after = entry.get("resident_prefix_layers_after")
        tracked = session.base_prefix_layers if base else session.prefix_layers
        label = "base_prefix_layers" if base else "prefix_layers"
        if before is not None or after is not None:
            if not isinstance(before, int) or isinstance(before, bool) \
                    or not isinstance(after, int) or isinstance(after, bool) \
                    or before < 0 or after < 0 or after > before:
                fail(f"{where}: 会话 {session_id!r} 逐出条目 "
                     f"resident_prefix_layers_before/after 结构非法"
                     f"（before={before!r}, after={after!r}, "
                     f"kind={entry.get('kind')!r}）——fail-closed")
            if tracked is None:
                fail(f"{where}: 会话 {session_id!r} 的 {label} 未知（None）"
                     f"而逐出条目已到场（kind={entry.get('kind')!r}, "
                     f"reason={entry.get('reason')!r}）——驻留前缀连锁"
                     f"断裂，宁爆不漏")
            if tracked != before:
                fail(f"{where}: 会话 {session_id!r} 逐出前缀不符（{label} "
                     f"跟踪 {tracked} != 条目 before={before}, kind="
                     f"{entry.get('kind')!r}）——账本与重放不一致")
            if base:
                session.base_prefix_layers = after
            else:
                session.prefix_layers = after
            return
        layer_start = entry.get("layer_start")
        if layer_start is None:
            # bytes-only 条目（无层区间）：前缀不变，由调用方在 bytes 归零
            # 时清零。
            return
        if not isinstance(layer_start, int) or isinstance(layer_start, bool) \
                or layer_start < 0:
            fail(f"{where}: 会话 {session_id!r} 逐出条目 layer_start 非"
                 f"非负整数（{layer_start!r}）——fail-closed")
        if tracked is None:
            fail(f"{where}: 会话 {session_id!r} 的 {label} 未知（None）而"
                 f"逐出条目已到场（kind={entry.get('kind')!r}）——驻留"
                 f"前缀连锁断裂，宁爆不漏")
        if base:
            session.base_prefix_layers = layer_start
        else:
            session.prefix_layers = layer_start

    def _establish_prefill_prefix(self, session: SessionState,
                                  decision: dict) -> None:
        """本会话 prefill 行成功处理后建立 primary 驻留前缀。

        行内 history_transfers 带层区间 → 按区间建立（最大 layer_end——
        copy@PARTIAL 的 noc 前缀段 + remote 后缀段合并即全覆盖）；否则
        := 全层数（model_layers，watermark_prepare 注入 mapping）。"""
        prefix: Optional[int] = None
        for transfer in decision.get("history_transfers") or ():
            end = transfer.get("layer_end")
            if isinstance(end, int) and not isinstance(end, bool) and end >= 0:
                if prefix is None or end > prefix:
                    prefix = end
        if prefix is None:
            prefix = self.mapping.get("model_layers")
        session.prefix_layers = prefix

    # -- 动作 ------------------------------------------------------------

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
            self.report.actions["partial_evictions"] += 1  # 分层部分逐出（S3 等）
        # R17-4：驻留前缀连锁校验（joint；条目带 R17-1d 新键则断言 before，
        # 旧日志回退 layer_start 规则；前缀未知即 fail-closed）。
        self._advance_resident_prefix(
            action["session"], session, action,
            f"tick={action['tick']} trigger={trigger}")
        self._apply(action["tick"], session.instance, -action["bytes"],
                    evict_bytes=action["bytes"])
        session.bytes -= action["bytes"]
        if session.bytes == 0:
            session.instance = None
            session.tokens = 0
            if self.mapping.get("joint"):
                session.prefix_layers = 0
        self.report.actions["evictions"] += 1
        self.report.actions["evict_bytes"] += action["bytes"]

    def _relocate_session_occupancy(self, tick: int, source: Optional[int],
                                    target: int,
                                    session: SessionState) -> bool:
        """会话归属切实例时占用随迁（R6-U4/R7-F4，2026-09-29 face 移植）。

        只改 session.instance 而不迁占用会让占用滞留 source、后续
        grow/evict 记到 target——instances 占用与 instance 指向分裂
        （source 侧幽灵占用 + target 侧账目错位）。无占用可迁时为空操作。
        三态适配（joint）：随迁对象 = primary 槽跟踪 bytes（可为部分层
        驻留），base/source 槽各有独立记账不受影响；prefix_layers 是会话
        属性随 instance 指向自然跟随，不参与迁移。
        """
        if source is None or session.bytes <= 0:
            return False
        self._apply(tick, source, -session.bytes)
        self._apply(tick, target, session.bytes)
        self.report.actions["restore_occupancy_relocated"] += 1
        self.report.actions["restore_occupancy_relocated_bytes"] += \
            session.bytes
        return True

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
            ratio = nbytes / expected if expected else None
            bucket = (f"{ratio:.3f}" if ratio is not None else "no-tokens")
            self.report.restore_ratio_hist[bucket] = \
                self.report.restore_ratio_hist.get(bucket, 0) + 1
            if ratio is None or not (math.isclose(ratio, 1.0, rel_tol=1e-9)
                                     or math.isclose(ratio, 0.5,
                                                     rel_tol=1e-9)):
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
            session.tokens = 0
            source = None

        # 显式 kind 优先（S1 的 local_hit.total_bytes 是"本地复用 KV 大小"
        # 信息量而非搬移量——基线实测 174 例 local_hit 携带 f(h) bytes，
        # 误当搬移会把占用翻倍）。
        if kind in ("LOCAL_HIT", "local_hit"):
            if nbytes and nbytes != session.bytes:
                self.report.anomalies["restore_bytes_mismatch"] += 1
            if source != target:
                self.report.anomalies["local_hit_location_mismatch"] += 1
                # R6-U4（2026-09-29 face 移植）：LOCAL_HIT 改指目标实例时
                # 占用必须同步随迁，否则跟踪态分裂（占用滞留 source、
                # 增长/逐出记到 target）。
                self._relocate_session_occupancy(action["tick"], source,
                                                 target, session)
            self.report.actions["restore_none"] += 1
            session.instance = target  # 会话归属目标实例（增长在目标发生）
            return
        if kind in ("NO_HISTORY", "RECOMPUTE", "recompute", "no_history") \
                and nbytes:
            self.report.anomalies["restore_kind_mismatch"] += 1
        # S3 半层恢复（bytes = suffix）：恢复前本地应恰持有 f(h) − bytes
        # （prefix 驻留）；跟踪态超出该值的部分即未落盘的静默 suffix 释放，
        # 在此对账扣减（字节守恒；时刻同样为上界窗口）。
        if nbytes and source == target and session.bytes > expected - nbytes:
            excess = session.bytes - (expected - nbytes)
            self._apply(action["tick"], target, -excess)
            session.bytes -= excess
            self.report.actions["restore_prefix_reconciled"] += 1
            self.report.actions["restore_prefix_reconcile_bytes"] += excess

        if nbytes == 0:
            if source != target:
                # R7-F4（2026-09-29 face 移植）：0 字节 restore 与 LOCAL_HIT
                # 同型——改指目标时占用必须随迁，否则幽灵占用滞留 source。
                self._relocate_session_occupancy(action["tick"], source,
                                                 target, session)
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
        # mismatch 检查先于 stay 分支（R6-U3，2026-09-29 face 移植）：
        # 0 字节跨实例（P≠D）交割此前在 stay 分支短路 return——占用滞留
        # source + anomaly 零留痕（幽灵占用不可见）。检查前置后该情形留下
        # move_bytes_mismatch 计数；stay 分支本身占用不变（同实例交割
        # S1/S2 P==D 仍记 f(c)）。
        if nbytes != self._expected_bytes(prefill_tokens):
            self.report.anomalies["move_bytes_mismatch"] += 1
        # D2（批次 D 镜像，2026-09-30）：跨实例零字节交割不再静默转移归属
        # ——账本宣称没有迁移、但会话跟踪 bytes 仍挂在源实例且归属要切到
        # 目标实例，源实例占用从此永不回收（R6-U3 的 anomaly 留痕不改变
        # exit 0 的静默性，此处升为 fail-closed）；会话本无驻留
        # （bytes==0）时仅实例归属切换，仍按 stay 计。
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
        session.tokens = tokens
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
            session.tokens = 0
            self.report.actions["own_relocation_remote"] += 1
        elif location == "partial_hbm_remote":
            # 分层部分去向：留下/离开比例账本未落——保留 bytes（上界），
            # 显式计数，不臆造比例。
            # R17-4：驻留前缀已知且层数可得时，按前缀字节比例反推本地
            # 留存层数并做整性断言（非整层数 = 账本与重放不一致）。
            total_layers = self.mapping.get("model_layers")
            if session.prefix_layers is not None and total_layers \
                    and session.tokens > 0 and session.bytes > 0:
                full = self._expected_bytes(session.tokens)
                per_layer, rem = divmod(full, total_layers)
                if rem or per_layer <= 0:
                    fail(f"partial_hbm_remote 层数反推失败（会话 "
                         f"{session_id!r}: f(tokens)={full} 不能被 "
                         f"layers={total_layers} 整除，tick={tick}）——"
                         f"fail-closed")
                layers_local, rem2 = divmod(session.bytes, per_layer)
                if rem2:
                    fail(f"partial_hbm_remote 留存 bytes 非整层数（会话 "
                         f"{session_id!r}: {session.bytes} 对每层 "
                         f"{per_layer} 余 {rem2}，tick={tick}）——fail-closed")
                session.prefix_layers = layers_local
            self.report.actions["own_relocation_partial_unknown"] += 1
        elif location == "local_hbm":
            pass  # 驻留本地：保留，无事。
        elif location is None:
            # 字段缺席（生产方未序列化）：不得静默按驻留本地处理——补
            # 计数留痕（与 R7-F8 修复面对齐：缺字段≠local_hbm）。
            # R8-F3（2026-09-29 face 移植）。
            self.report.anomalies["own_relocation_missing_location"] += 1
        else:
            # 枚举（local_hbm/partial_hbm_remote/remote_memory）外的值
            # 拒绝静默按 local_hbm no-op——anomaly 留痕
            # （R8-F3，2026-09-29 face 移植）。
            self.report.anomalies["own_relocation_unknown_location"] += 1

    # -- W3（2026-09-27）：KVStagingLease 执行期持有重放（joint） ----------

# ---------------------------------------------------------------------------
# 决策记录 → 动作流
# ---------------------------------------------------------------------------

class WatermarkScan:
    """A4 driver 复用面：单条决策记录一次 consume，扫完 finish。

    与独立 CLI 的 replay_decision_log 循环体逐语句等价（含 fail 消息与
    记录内「逐出→恢复/迁移→增长」固定次序、session_hint 回写）。注意
    consume 会向 record 注入 session_hint 键——driver 的 sink 次序中
    watermark 必须最后（kv/load/hop 不读该键，注入对其不可见）。
    """

    def __init__(self, run_dir: Path, repo_variant: str, mapping: dict,
                 tokens: dict, coef: int,
                 capacity: Optional[int]) -> None:
        self.replay = WatermarkReplay(repo_variant, mapping, tokens, coef,
                                      capacity)
        self.mapping = mapping
        self.tokens = tokens
        self.seen_kinds: dict[tuple[str, str], int] = {}
        self.log_path = run_dir / DECISION_LOG_RELPATH
        self.last_tick = -1

    def consume(self, record: dict) -> None:
        replay = self.replay
        mapping = self.mapping
        tokens = self.tokens
        where = f"{self.log_path}:seq={record.get('seq', '?')}"
        request_id = record.get("request_id")
        kind = record.get("kind")
        if kind in mapping.get("joint_audit_kinds", ()):
            # joint 审计行（全候选成本表/准入失败折叠/stall-wake 事件）——
            # 非账本行，跳过不进重放。
            return
        if not isinstance(request_id, str) or not request_id:
            fail(f"{where}: 决策记录缺 request_id")
        if kind == "kv_eviction" and mapping.get("joint"):
            # R17-2b（2026-09-16）：decode 增长逐出/准入失败逐出的
            # kv_eviction 决策行——账本行（非审计行），必须在 kind 门之前
            # 分流（否则按未知 kind fail-closed）。同请求允许多根（不经
            # seen_kinds 去重）；触发请求在飞，不查 token manifest；
            # request_id 校验复用主循环（kv_eviction 行必带触发请求 id）；
            # 分支自行承担 tick 单调检查（复用 last_tick 推进与回退判罚
            # 语义，不绕过次序纪律）。
            decision = record.get("decision") or {}
            entries = decision.get("evictions")
            if entries is None:
                entries = []
            if not isinstance(entries, list):
                fail(f"{where}: evictions 必须是数组")
            tick = record.get("tick")
            if not isinstance(tick, int) or tick < 0:
                fail(f"{where}: 缺非负整数 tick")
            if tick < self.last_tick:
                fail(f"{where}: tick 回退（{tick} < {self.last_tick}，"
                     f"kind=kv_eviction）——文件顺序与时间顺序不一致，"
                     f"无法安全重放")
            self.last_tick = tick
            for entry in entries:
                if not isinstance(entry, dict):
                    fail(f"{where}: evictions 成员必须是对象")
                action = _evict_action(tick, entry, _scalar_bytes, where)
                replay.apply_evict(action, request_id)
                # R17-2d 哨兵：逐条计数 + 按 reason 分桶（随 summary.actions
                # 落盘）。
                replay.report.actions["evict_entries_kv_eviction"] = (
                    replay.report.actions.get("evict_entries_kv_eviction", 0)
                    + 1)
                bucket = f"evict_reason_kv_eviction:{action.get('reason')}"
                replay.report.actions[bucket] = (
                    replay.report.actions.get(bucket, 0) + 1)
            return
        if kind not in ("prefill", "decode", "completion"):
            fail(f"{where}: 未知 kind={kind!r}")
        key = (request_id, kind)
        if key in self.seen_kinds:
            fail(f"{where}: 请求 {request_id} 的 {kind} 决策出现两次"
                 f"（先于 seq={self.seen_kinds[key]}）——账本次序异常")
        self.seen_kinds[key] = record.get("seq", -1)
        tick = record.get("tick")
        if not isinstance(tick, int) or tick < 0:
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
        for action in _collect_eviction_lists(record, mapping, where):
            replay.apply_evict(action, request_id)

        decision = record.get("decision") or {}
        joint_action = (
            decision.get("joint_action")
            if mapping.get("joint") else None)

        # 2) 按记录类型主动作。
        if joint_action is not None:
            # R12：joint 专属事件重放（双驻留记账）。
            if kind == "prefill":
                self._joint_prefill(
                    replay, decision, tick, session_id, token_row)
            elif kind == "decode":
                self._joint_decode(
                    replay, decision, tick, session_id, token_row, joint_action)
            else:
                self._joint_completion(
                    replay, decision, tick, session_id, token_row, joint_action)
        elif kind == "prefill":
            target_instance = decision.get("prefill_instance_index")
            if not isinstance(target_instance, int):
                fail(f"{where}: prefill 决策缺整数 prefill_instance_index")
            for action in _prefill_restore_actions(record, mapping, where):
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
            else:
                session = replay.sessions.get(session_id)
                if session is not None:
                    session.tokens = token_row["final_context_tokens"]

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

    # ------------------------------------------------------ joint 重放 --
    def _joint_prefill(self, replay, decision, tick, session_id,
                       token_row) -> None:
        """R12：joint prefill 行重放——工作副本语义按动作分流。

        copy/recompute 跨实例：home 侧基础驻留转 base 记账（源不扣减、
        不低估）；工作副本 = 迁移字节（copy，PARTIAL 基时另含池恢复
        后缀的 remote_load 腿）/0（recompute）起步，再增长到工作上下文
        （prefill_context）。recompute@home 的缺失后缀物化（R13，无传
        输记录）由"增长到完整上下文"自然覆盖。
        stay：走通用恢复/增长（suffix 池恢复 + 全上下文增长）。"""
        action = decision["joint_action"]
        target = decision.get("prefill_instance_index")
        # 终审-中2：joint 行与通用路径同级容错——缺整数实例索引即账本
        # 破损（此前产出 {None: ...} 假实例桶）。
        if not isinstance(target, int):
            fail(f"joint prefill 决策缺整数 prefill_instance_index"
                 f"（session={session_id!r}, got {target!r}）")
        prefill_context = token_row["prefill_context_tokens"]
        # R17-3（2026-09-16）：fail-early——丢弃历史词元的未演练词表显式
        # 拒收（重放无此语义，静默吞掉会系统性漏记被丢弃的驻留）。
        discarded = decision.get("history_tokens_discarded")
        if isinstance(discarded, int) and discarded > 0:
            fail(f"joint prefill 行 history_tokens_discarded={discarded} > 0"
                 f"（session={session_id!r}, tick={tick}）——丢弃历史词元"
                 f"属未演练词表，显式拒收（fail-closed）")
        session = replay._session(session_id)
        if action == "stay":
            session.instance = target
            replay.report.actions["joint_stay_prefill"] = (
                replay.report.actions.get("joint_stay_prefill", 0) + 1)
            replay.apply_grow(tick, session_id, target, prefill_context,
                              "prefill_grow")
            # R17-4：stay 单驻留——前缀按行内区间/全层数重建。
            replay._establish_prefill_prefix(session, decision)
            return
        # 跨实例工作动作：基础驻留转 base（home 低估缺口修复的核心）。
        if session.instance is not None and session.instance != target:
            replay.stash_base(session_id)
        session.instance = target
        for transfer in decision.get("history_transfers") or ():
            kind = transfer.get("kind")
            # 四审-低7：同 merge 段——缺失即 raise，不静默归零。
            nbytes = transfer.get("total_bytes")
            if not isinstance(nbytes, int) or nbytes < 0:
                fail(f"joint history transfer total_bytes 必须为非负整数"
                     f"（session={session_id!r}, kind={kind!r}, "
                     f"got {nbytes!r}）")
            if kind == "local_hit" or not nbytes:
                continue
            if kind == "remote_load":
                replay._apply(tick, target, nbytes)
                session.bytes += nbytes
                replay.report.actions["restore_remote_add"] += 1
            elif kind == "noc_migrate":
                # copy 前缀：源端保留（base 记账），目标端增副本。
                replay._apply(tick, target, nbytes)
                session.bytes += nbytes
                replay.report.actions["joint_working_copy_add"] = (
                    replay.report.actions.get("joint_working_copy_add", 0)
                    + 1)
            else:
                # 终审-中2：未知 kind 且有字节 → fail-closed（不得静默
                # 丢弃账本字节）。
                fail(f"joint history transfer 出现未知 kind={kind!r}"
                     f"（session={session_id!r}, bytes={nbytes}）——"
                     f"joint 重放词表未覆盖，拒绝静默丢弃")
        grow_tokens = prefill_context
        desired = replay._expected_bytes(grow_tokens)
        # R17-3（2026-09-16）：对称校验——增长语义只增不减（与 decode_grow
        # 的 apply_grow 负增长 fail 同构）；缩小时静默保留旧值会把逐出缺口
        # 的幻影 bytes 结转下去，必须在产生点 fail-early。
        if desired < session.bytes:
            fail(f"prefill_shrink：joint prefill 增长为负（会话 "
                 f"{session_id!r}: desired={desired} < session.bytes="
                 f"{session.bytes}, tick={tick}）——上下文不允许收缩，"
                 f"重放口径与账本不一致")
        if desired > session.bytes:
            replay._apply(tick, target, desired - session.bytes)
            session.bytes = desired
        session.tokens = prefill_context
        # R17-4：跨实例工作副本行——primary 前缀按 history_transfers 区间
        # （noc 前缀段 + remote 后缀段的并）/全层数重建。
        replay._establish_prefill_prefix(session, decision)
        replay.report.actions["prefill_grow"] += 1

    def _joint_decode(self, replay, decision, tick, session_id, token_row,
                      joint_action) -> None:
        """joint decode 行——先 P→D 迁移、后 D 增长（manager 次序：先
        move_prefill_to_decode 再 expand_decode）。

        P→D 跨实例迁移：决策行 ``prefill_decode_transfer`` 披露搬移时
        （noc_migrate 携 source_instance_index+total_bytes），把 P 阶段
        工作副本从源实例扣、目标实例加——见
        ``_apply_prefill_decode_move``。缺此步会把工作副本留在源实例成
        幻影，完成结算在目标实例多扣 → 负占用 fail-closed。

        增长口径：工作副本增长基 = final_context（三动作统一；远读增量
        增长基随远读退役删除，2026-10-14）。

        W2（2026-09-27，C4 三位置边界后的增长基，按 D18 边界动作族）：
        来源槽在册（本行边界刚建立）时按 boundary_kind 覆盖——重建
        （rebuild）j 侧副本自 0 物化至完整终上下文（三动作下唯一边界
        动作族）。无来源槽的行为与 W1 逐位一致。

        W4（2026-09-27，边界折叠后的增长基）：迁移腿落 base 驻留实例的
        就地并基折叠把工作副本口径并回完整上下文族——增量型折叠
        （工作副本 = S+U）并基后 session.bytes = f(prefill_context)，增长
        基随之切 final_context；完整型折叠工作副本本就是 f(P)，增长基
        不变。"""
        target = decision.get("decode_instance_index")
        if not isinstance(target, int):
            fail(f"joint decode 决策缺整数 decode_instance_index"
                 f"（session={session_id!r}, got={target!r}）")
        session = replay._session(session_id)
        self._apply_prefill_decode_move(
            replay, decision, tick, session_id, target, session)
        fold_kind = self._apply_boundary_base_fold(
            replay, decision, tick, session_id, token_row, session)
        if fold_kind == "increment":
            # 折叠并基后工作副本 = f(P)（完整上下文族），增长基 = final。
            grow_tokens = token_row["final_context_tokens"]
        elif session.boundary_kind == "rebuild":
            grow_tokens = token_row["final_context_tokens"]
        else:
            grow_tokens = token_row["final_context_tokens"]
        replay.apply_grow(tick, session_id, target, grow_tokens,
                          "decode_grow")

    @staticmethod
    def _apply_prefill_decode_move(replay, decision, tick, session_id,
                                   target, session) -> None:
        """P→D 迁移重放（对齐 move_prefill_to_decode 的占用语义）。

        * 跨实例（noc_migrate/prefill_decode_instance_migrate）：manager
          搬走 session 全量本地 KV（字节按完整驻留上下文派生），故披露
          bytes 必须与重放跟踪的工作副本逐字节相等——少搬即源实例幻影、
          多搬即凭空字节，均 fail-closed；源端还须与跟踪实例一致。
        * 同实例（local_hit/prefill_decode_local_reuse）：交割字节为信息
          量（S1 同款口径），占用不变；跟踪实例与披露实例矛盾即账本
          破损。
        * 无披露：仅跟踪态无驻留或已在目标实例才允许无搬移；跨实例而
          无披露 = 迁移字节未落盘，拒绝静默漏记。"""
        transfer = decision.get("prefill_decode_transfer")
        if transfer is None:
            if session.instance is not None and session.instance != target:
                fail(f"joint decode 跨实例（{session.instance}→{target}）"
                     f"但决策行无 prefill_decode_transfer 披露"
                     f"（session={session_id!r}, tick={tick}）——P→D 迁移"
                     f"字节必须落盘，拒绝静默漏记")
            return
        if not isinstance(transfer, dict):
            fail(f"joint decode prefill_decode_transfer 结构非法"
                 f"（session={session_id!r}, "
                 f"got {type(transfer).__name__}）")
        source = transfer.get("source_instance_index")
        kind = transfer.get("kind")
        if source == target:
            if kind != "local_hit":
                fail(f"joint decode 同实例交割（instance={source!r}）携带 "
                     f"kind={kind!r}（session={session_id!r}, "
                     f"tick={tick}）——与交割语义矛盾")
            if session.instance is not None and session.instance != source:
                # W2（2026-09-27，C4/D19 三位置边界）：local_hit 披露的
                # source 是**新执行端 j**（face_scheduler 边界事务先零化
                # 会话再产 local_hit，字节恒 0），而跟踪主副本仍在 i_P——
                # 无迁移边界形态：旧工作副本不搬、留 i_P 转来源侧记录，
                # j 侧新副本零化起步（rebuild 由增长钩子物化）。
                # reason 词表 = face_scheduler 边界事务的 j≠i_P 无迁移
                # kind（三动作下唯一值）；其他 reason 保持原 fail（宁爆
                # 不漏）。
                reason = transfer.get("reason")
                nbytes0 = transfer.get("total_bytes")
                if (reason == "boundary_rebuild_elsewhere"
                        and isinstance(nbytes0, int) and nbytes0 == 0):
                    if session.source_instance is not None:
                        fail(f"joint decode 边界重复建立来源侧（session="
                             f"{session_id!r}，已有来源 {session.source_instance}"
                             f"，tick={tick}）——两次边界之间无合并结算，"
                             f"合同违规")
                    if decision.get("joint_action") == "copy":
                        # R9（修复轮三，2026-09-27）镜像：P=copy 交接轮的
                        # D 边界无迁移——face_scheduler 在边界转记录点先
                        # 闭合 C13 journal（home 侧物理留存已随逐块交接在
                        # 轮内释放完，journal 闭合断言 H_home=0）再按
                        # pool-only 基同科目把 base_* **重录到 i_P**：旧工
                        # 作副本转 base 记账、不建来源槽，轮末 merge 退化
                        # 常规两位置（base@i_P 对 exec 增量/重建）。W2 的
                        # 来源侧三位置模型对本形态不成立（实测 combo_none
                        # session_2：stale base@3 ＋ 误建 source@0 使
                        # forward 胜者误裁到 stale base，败者 0 无 noc 腿
                        # fail-closed）。stale base 槽释放 = journal home
                        # 侧逐块释放的重放镜像（上界口径滞留至闭合点统一
                        # 释放；去重释放非容量逐出，不记 evict 标记，审计
                        # 量走 actions 计数器——W4 折叠点同科目）。
                        stale_base = session.base_instance
                        stale_bytes = session.base_bytes
                        if stale_base is not None and stale_bytes:
                            replay._apply(tick, stale_base, -stale_bytes)
                            replay.report.actions[
                                "joint_copy_handoff_base_release"] = (
                                    replay.report.actions.get(
                                        "joint_copy_handoff_base_release",
                                        0) + 1)
                            replay.report.actions[
                                "joint_copy_handoff_base_release_bytes"] = (
                                    replay.report.actions.get(
                                        "joint_copy_handoff_base_release_"
                                        "bytes", 0) + stale_bytes)
                        session.base_instance = session.instance
                        session.base_bytes = session.bytes
                        session.base_prefix_layers = session.prefix_layers
                        session.instance = None
                        session.bytes = 0
                        session.tokens = 0
                        # 边界即设全层驻留前缀（同 W2：manager 零化同点置
                        # resident_prefix_layers = model.layers）。
                        session.prefix_layers = (
                            replay.mapping.get("model_layers"))
                        session.boundary_kind = "rebuild"
                        replay.report.actions[
                            "joint_boundary_base_rerecord"] = (
                                replay.report.actions.get(
                                    "joint_boundary_base_rerecord", 0) + 1)
                        return
                    session.source_instance = session.instance
                    session.source_bytes = session.bytes
                    replay.report.actions["joint_boundary_source_stash"] = (
                        replay.report.actions.get(
                            "joint_boundary_source_stash", 0) + 1)
                    replay.report.actions[
                        "joint_boundary_source_stash_bytes"] = (
                            replay.report.actions.get(
                                "joint_boundary_source_stash_bytes", 0)
                            + session.bytes)
                    session.instance = None
                    session.bytes = 0
                    session.tokens = 0
                    # 边界即设全层驻留前缀（manager 在零化同点置
                    # resident_prefix_layers = model.layers；重建/增量
                    # 均为全层跨度增长）。
                    session.prefix_layers = (
                        replay.mapping.get("model_layers"))
                    session.boundary_kind = "rebuild"
                    return
                fail(f"joint decode 同实例交割披露 instance={source} != 跟踪"
                     f"实例 {session.instance}（session={session_id!r}, "
                     f"tick={tick}）——账本与重放不一致")
            return
        if not isinstance(source, int):
            fail(f"joint decode 跨实例迁移缺整数 source_instance_index"
                 f"（session={session_id!r}, got {source!r}）")
        if kind != "noc_migrate":
            fail(f"joint decode 跨实例迁移（{source}→{target}）携带 "
                 f"kind={kind!r}（session={session_id!r}, tick={tick}）"
                 f"——与搬移语义矛盾")
        nbytes = transfer.get("total_bytes")
        if not isinstance(nbytes, int) or nbytes < 0:
            fail(f"joint decode 迁移 total_bytes 必须为非负整数"
                 f"（session={session_id!r}, got {nbytes!r}）")
        if session.instance is None:
            fail(f"joint decode 迁移时会话 {session_id!r} 无跟踪实例"
                 f"（tick={tick}）——重放次序或账本缺失")
        if session.instance != source:
            fail(f"joint decode 迁移披露源端 {source} != 跟踪工作副本实例 "
                 f"{session.instance}（session={session_id!r}, "
                 f"tick={tick}）——账本与重放不一致")
        if nbytes != session.bytes:
            fail(f"joint decode 迁移 bytes {nbytes} != 会话跟踪工作副本 "
                 f"{session.bytes}（session={session_id!r}, tick={tick}）"
                 f"——move_prefill_to_decode 整份搬移全量本地 KV，"
                 f"少搬/多搬均属账本与重放不一致")
        replay._apply(tick, source, -nbytes)
        replay._apply(tick, target, nbytes)
        session.instance = target
        replay.report.actions["joint_decode_move"] = (
            replay.report.actions.get("joint_decode_move", 0) + 1)
        replay.report.actions["joint_decode_move_bytes"] = (
            replay.report.actions.get("joint_decode_move_bytes", 0) + nbytes)

    def _apply_boundary_base_fold(self, replay, decision, tick, session_id,
                                  token_row, session):
        """W4（2026-09-27，D 边界就地并基折叠的词表补齐）：迁移腿落点
        == base 驻留实例 → workload 侧的折叠（complete_copy_base_fold /
        increment_copy_home_u_only）把 base_* 记录清账（基础转正或重复
        副本释放），下一轮跨实例 prefill 重新 stash base 是合法重建——
        重放必须在折叠点同步清 base 槽。

        占用语义按迁移腿字节 W 与 f(prefill_context) 的恒等式二分（镜像
        face_scheduler 折叠分支的字节守恒；两式均不成立 = 未演练词表，
        fail-closed）：
          * W == f(P)（完整型工作副本并基）：home 侧基础副本已释放
            （逐块交接轮内释放 / 折叠点就地释放），重放在折叠点扣减
            base 槽字节；工作副本口径不变（本就 = f(P)）。
          * W + base_bytes == f(P)（增量型工作副本并基）：基础前缀就地
            转正并入工作副本（字节已在位，占用不动），session.bytes 并
            至 f(P)；返回 "increment" 供调用方把 decode 增长基切到完整
            上下文族（仍按增量域增长会把折叠后的全上下文当收缩）。

        无迁移腿（local_hit/无披露）或落点 != base 槽实例：非折叠形态，
        返回 None 零行为差。"""
        transfer = decision.get("prefill_decode_transfer")
        if (session.base_instance is None
                or not isinstance(transfer, dict)
                or transfer.get("kind") != "noc_migrate"):
            return None
        if decision.get("decode_instance_index") != session.base_instance:
            return None
        target = session.base_instance
        moved = transfer.get("total_bytes")
        if not isinstance(moved, int) or moved < 0:
            # 迁移腿字节结构校验已由 _apply_prefill_decode_move 完成；
            # 此处防御性复核（折叠恒等式不得建立在破损披露上）。
            fail(f"joint 边界折叠迁移腿 total_bytes 必须为非负整数"
                 f"（session={session_id!r}, got={moved!r}）")
        full_context = replay._expected_bytes(
            token_row["prefill_context_tokens"])
        base_instance = session.base_instance
        base_bytes = session.base_bytes
        if moved == full_context:
            # 完整型并基：重复基础副本释放——折叠点扣减、清槽（去重释
            # 放非容量逐出，不记 evict 标记；审计量走 actions 计数器）。
            if base_bytes:
                replay._apply(tick, base_instance, -base_bytes)
            replay.report.actions["joint_boundary_base_fold"] = (
                replay.report.actions.get("joint_boundary_base_fold", 0) + 1)
            replay.report.actions["joint_boundary_base_fold_release_bytes"] = (
                replay.report.actions.get(
                    "joint_boundary_base_fold_release_bytes", 0) + base_bytes)
            self._clear_base_slot(session, replay)
            return "complete"
        if moved + base_bytes == full_context:
            # 增量型并基：基础前缀就地转正——字节并入、占用不动、清槽。
            session.bytes += base_bytes
            replay.report.actions["joint_boundary_base_fold"] = (
                replay.report.actions.get("joint_boundary_base_fold", 0) + 1)
            replay.report.actions["joint_boundary_base_fold_absorb_bytes"] = (
                replay.report.actions.get(
                    "joint_boundary_base_fold_absorb_bytes", 0) + base_bytes)
            self._clear_base_slot(session, replay)
            return "increment"
        fail(f"joint 边界迁移腿落 base 驻留实例但字节恒等式不成立"
             f"（session={session_id!r}, moved={moved}, base={base_bytes},"
             f" f(P)={full_context}）——未演练折叠词表，拒绝静默抹账")

    @staticmethod
    def _clear_base_slot(session, replay) -> None:
        """折叠清 base 槽：base 三字段归零/None，primary 前缀转全层
        （折叠产物恒为本地完整上下文，镜像 manager 折叠点的
        resident_prefix_layers = model.layers）。"""
        session.base_instance = None
        session.base_bytes = 0
        session.base_prefix_layers = None
        session.prefix_layers = replay.mapping.get("model_layers")

    def _joint_completion(self, replay, decision, tick, session_id,
                          token_row, joint_action) -> None:
        """R12：joint completion 行——merge 流 + 工作副本释放 + base 恢复。

        merge v2（2026-09-17《部分层逐出kv管理改造分析方案》需求②）：
        携带 ``merge_direction`` 的新行走 v2 重放（forward/reverse/
        in_place——见 ``_joint_completion_v2``）；"stay" 本地提交；无该
        字段的旧日志走本方法尾部 legacy 分支（增量拆分归并回 home 的
        旧语义，金样对照用）。

        merge_transfers：noc 增量回传（+home）、池写回（无本地变化）、
        home_merge_base_degrade 自降级（home 侧 base 扣减）。工作副本
        判定（K3）：优先读完成行显式披露的 ``joint_working_copy``（merge
        前 working_kind 真值；REMOTE 基/跨实例 recompute 无 stash 但确有
        执行端副本须释放，stay / recompute@home / 新会话本地提交是单驻
        留不得释放）；字段缺省（旧日志）回退 origin_home 启发式。home
        侧结算（K2）：基础字节从未离开 home（stash_base 不扣占用账），
        完成时只**并入增量**——不得再加 base，否则逐轮复利双计。"""
        direction = decision.get("merge_direction")
        if direction in ("forward", "reverse", "in_place"):
            self._joint_completion_v2(
                replay, decision, tick, session_id, token_row, direction)
            return
        if direction == "distributed":
            # W9（§7.4，2026-09-29）：目录提交型结算——零物理归并，全部
            # 位置字节原样保留（base/来源/旧 primary 转跨轮留存，下一轮
            # 可换址新起）。**水印重放的会话状态机是单 primary + base +
            # source 三槽模型**：distributed 跨轮留存的多位置布局（>3 位
            # 置、换址新起 primary）超出该模型——本重放器对 distributed
            # run 显式 fail-closed（不出 certified 水印；仿真正确性由
            # KVCacheManager 侧目录守恒/rank 台账对账裁决），多位置重放
            # 状态机登记为 W10/W11 遗留。拒收信息携带精确诊断（不静默
            # 忽略分布式事件——§10.2"未知不得默认忽略后仍出 certified
            # 结论"）。
            fail(
                f"joint completion merge_direction=distributed（session="
                f"{session_id!r}）：水印重放器的三槽会话模型不支持目录"
                "提交型结算的跨轮多位置布局（retained 位置 + 换址新起"
                " primary）——distributed run 不出 certified 水印（W9 已"
                "知边界，W10/W11 多位置重放状态机遗留）；仿真正确性由 "
                "KVCacheManager 目录守恒与 rank 台账对账裁决")
        if direction == "stay":
            session = replay._session(session_id)
            if session.source_instance is not None:
                # W2：三位置来源槽只能由 merge v2 结算清账——stay 本地提交
                # 与来源侧在册互斥（边界动作族必产工作副本）。
                fail(f"joint completion merge_direction=stay 但来源侧在册"
                     f"（session={session_id!r}, 来源 {session.source_instance}"
                     f"）——三位置结算不得走 stay 本地提交，账本与重放不一致")
            if session.base_instance is not None:
                # W4：stay 本地提交 = 无工作副本的单驻留形态——base 槽在册
                # 说明跨实例工作副本的折叠清账没有在 D 边界词表内表达，
                # 未知形态显式拒收（下一轮 stash 会在册守卫处再爆，宁在此
                # 早爆，账本定位更准）。
                fail(f"joint completion merge_direction=stay 但 base 槽在册"
                     f"（session={session_id!r}, base {session.base_instance}"
                     f"）——折叠清账必须经 D 边界迁移腿词表表达，"
                     f"账本与重放不一致")
            session.tokens = token_row["final_context_tokens"]
            replay.report.actions["joint_merge_stay"] = (
                replay.report.actions.get("joint_merge_stay", 0) + 1)
            return
        session = replay._session(session_id)
        if session.source_instance is not None:
            # W2：来源槽在册（D 边界无迁移披露）而完成行无 merge_direction
            # ——三位置结算必须有 v2 方向披露，legacy 增量归并口径不覆盖
            # 三位置形态，显式拒收（拒绝 legacy 抹账）。
            fail(f"joint completion 缺 merge_direction 但来源侧在册"
                 f"（session={session_id!r}, 来源 {session.source_instance}"
                 f"）——三位置合并必须走 merge v2 披露重放")
        merge_home = (
            session.base_instance
            if session.base_instance is not None
            else decision.get("origin_home_instance"))
        # 终审-中1/中2：merge 流预分类（stay 早退前执行——校验不得被
        # 无工作副本行绕过）。判别：传输摘要恒含 session_id——等于本行
        # 会话 = 自身流（增量 noc 回 home / 自降级 / 自身增量池写回，均
        # 无第三方账效果）；不等于 = home 侧空间准备的第三方 victim 逐出
        # （home_merge_capacity，victim=其它会话，经 apply_evict 入账）。
        home_add = 0
        degrade_bytes = 0
        degrade_entries = []
        third_party_evictions = []
        for transfer in decision.get("merge_transfers") or ():
            kind = transfer.get("kind")
            # 四审-低7：缺失字段不得 or 0 静默归零（与"非非负整数即
            # raise"的声称一致——None 同样 raise）。
            nbytes = transfer.get("total_bytes")
            if not isinstance(nbytes, int) or nbytes < 0:
                fail(f"joint merge transfer total_bytes 必须为非负整数"
                     f"（session={session_id!r}, kind={kind!r}, "
                     f"got {nbytes!r}）")
            if kind == "noc_migrate":
                home_add += nbytes
            elif kind == "remote_store":
                if transfer.get("reason") == "home_merge_base_degrade":
                    degrade_bytes += nbytes
                    degrade_entries.append(transfer)
                elif transfer.get("session_id") == session_id:
                    pass  # 自身增量池写回（REMOTE 结算）：无本地变化
                else:
                    victim = transfer.get("session_id")
                    if victim is None or merge_home is None:
                        fail(f"joint merge 逐出传输无法归类（session="
                             f"{session_id!r}, transfer={transfer!r}）——"
                             f"缺 victim session_id 或 merge home")
                    third_party_evictions.append(transfer)
            elif kind in ("local_hit", "remote_load"):
                continue
            else:
                fail(f"joint merge transfer 出现未知 kind={kind!r}"
                     f"（session={session_id!r}, bytes={nbytes}）——"
                     f"joint 重放词表未覆盖，拒绝静默丢弃")
        for transfer in third_party_evictions:
            nbytes = transfer["total_bytes"]
            if nbytes:
                # R17-4：层区间/R17-1d 新键透传（home_merge_capacity 的
                # victim 逐出同受驻留前缀连锁校验约束）。
                replay.apply_evict(
                    {"session": transfer.get("session_id"), "tick": tick,
                     "bytes": nbytes, "instance": merge_home,
                     "kind": transfer.get("kind"),
                     "reason": transfer.get("reason"),
                     "layer_start": transfer.get("layer_start"),
                     "layer_end": transfer.get("layer_end"),
                     "resident_prefix_layers_before":
                         transfer.get("resident_prefix_layers_before"),
                     "resident_prefix_layers_after":
                         transfer.get("resident_prefix_layers_after")},
                    f"joint_merge:{session_id}")
        explicit_working_copy = decision.get("joint_working_copy")
        if explicit_working_copy is not None:
            had_working_copy = bool(explicit_working_copy)
        else:
            origin_home = decision.get("origin_home_instance")
            had_working_copy = (
                session.base_instance is not None
                or (session.instance is not None
                    and origin_home is not None
                    and origin_home != session.instance))
        if not had_working_copy:
            # stay / recompute@home / 新会话本地提交：单驻留不动。
            session.tokens = token_row["final_context_tokens"]
            return
        # 执行端工作副本释放（primary 全量；REMOTE 基/跨实例 recompute
        # 无 stash 但同样释放）。
        if session.instance is not None and session.bytes:
            replay._apply(tick, session.instance, -session.bytes)
            replay.report.actions["joint_working_release"] = (
                replay.report.actions.get("joint_working_release", 0) + 1)
            replay.report.actions["joint_working_release_bytes"] = (
                replay.report.actions.get(
                    "joint_working_release_bytes", 0) + session.bytes)
        session.bytes = 0
        session.instance = None
        # R17-4：primary 前缀随工作副本释放清零——kv_location_after_
        # completion == "remote_memory" 时 primary 前缀必为零（清零实现，
        # 与 bytes 清零同构，不与既有断言冲突）。
        session.prefix_layers = 0
        if session.base_instance is None:
            # K3：REMOTE 基会话——基础历史在池 backing、无 home 驻留可
            # 恢复；增量整份写池（merge_transfers 只有池写回，无
            # noc_migrate，home_add == 0），会话回到无驻留态。
            session.tokens = token_row["final_context_tokens"]
            replay.report.actions["joint_merge_settle"] = (
                replay.report.actions.get("joint_merge_settle", 0) + 1)
            return
        # home 侧 base：自降级扣减（真实离开 home）+ 增量并入（base 从未
        # 离开，不得重复加回——K2）。
        base_instance = session.base_instance
        # R17-4：home_merge_base_degrade 条目作用于 base 前缀（不是
        # primary——降级离开的是 home 侧基础驻留）；前缀未知而降级到场
        # 同样 fail-closed（宁爆不漏）。
        for transfer in degrade_entries:
            replay._advance_resident_prefix(
                session_id, session, transfer,
                f"tick={tick} joint_merge_base_degrade", base=True)
        base_bytes = max(0, session.base_bytes - degrade_bytes)
        if degrade_bytes:
            replay._apply(tick, base_instance, -degrade_bytes,
                          evict_bytes=degrade_bytes)
            replay.report.actions["joint_merge_base_degrade"] = (
                replay.report.actions.get("joint_merge_base_degrade", 0) + 1)
        if home_add:
            replay._apply(tick, base_instance, home_add)
        if base_bytes > 0 or home_add > 0:
            session.instance = base_instance
            session.bytes = base_bytes + home_add
        # R17-4：base 归位 primary——前缀随字节并回（可能经自降级缩减）。
        session.prefix_layers = session.base_prefix_layers
        session.base_instance = None
        session.base_bytes = 0
        session.base_prefix_layers = None
        session.tokens = token_row["final_context_tokens"]
        replay.report.actions["joint_merge_settle"] = (
            replay.report.actions.get("joint_merge_settle", 0) + 1)

    def _joint_completion_v2(self, replay, decision, tick, session_id,
                             token_row, direction) -> None:
        """merge v2 少并多重放（需求②，2026-09-17）。

        占用语义（与物化 face_scheduler.merge_back v2 同构）：
        * forward：exec 工作副本整份搬 home——exec 释放（−工作副本字节）、
          home 接收（+noc 字节，恒等于释放量）；base 保留并归位 primary。
        * reverse：home 侧 base 释放（−base 字节；含传输=noc 搬至 exec /
          零字节翻转=直接删除——copy/recompute 执行端恒持并集）；exec
          工作副本**保留**并接收 noc 字节（零字节翻转=0）。
        * in_place：REMOTE 基无主就地保留——零传输零池写，工作副本直接
          转正（occupancy 不动），home := exec（重放侧 home 由下一轮
          准入行 origin_home 重建，无需显式迁移）。
        I6 断言：merge 事务不得出现本会话 remote_store（合并零池写）；
        noc 方向按 source_instance_index 与胜者/败者实例交叉校验。
        终态恒全层 LOCAL@胜者（prefix = model_layers）。

        W2（2026-09-27，C4/D12）：来源槽在册（D 边界无迁移披露）时本方
        法分流到 ``_joint_completion_three_position``——三位置确定性合并
        （base/source/exec 少并多）有独立的方向裁决与守恒断言；两位置
        forward/reverse/in_place 路径零改动。
        """
        session = replay._session(session_id)
        if session.source_instance is not None:
            self._joint_completion_three_position(
                replay, decision, tick, session_id, token_row, direction)
            return
        exec_instance = session.instance
        base_instance = session.base_instance
        if direction == "forward" and base_instance is None:
            fail(f"merge forward 无 base 记账（session={session_id!r}）"
                 f"——账本与重放不一致")
        if direction == "in_place" and base_instance is not None:
            fail(f"merge in_place 到达但 base 记账存在"
                 f"（session={session_id!r}）——无主语义与账本矛盾")
        if direction == "in_place" and exec_instance is None:
            fail(f"merge in_place 无执行端驻留（session={session_id!r}）")
        winner = (base_instance if direction == "forward" else exec_instance)
        # ---- 传输预分类（方向校验＋第三方 victim 逐出＋I6 池写断言）。----
        noc_bytes = 0
        third_party_evictions = []
        for transfer in decision.get("merge_transfers") or ():
            kind = transfer.get("kind")
            nbytes = transfer.get("total_bytes")
            if not isinstance(nbytes, int) or nbytes < 0:
                fail(f"joint merge transfer total_bytes 必须为非负整数"
                     f"（session={session_id!r}, kind={kind!r}, "
                     f"got={nbytes!r}）")
            if kind == "noc_migrate":
                source = transfer.get("source_instance_index")
                if direction == "forward" and source != exec_instance:
                    fail(f"merge forward 的 noc 源端 {source!r} != 执行端 "
                         f"{exec_instance!r}（session={session_id!r}）——"
                         f"方向披露与传输矛盾")
                if direction == "reverse" and source != base_instance:
                    fail(f"merge reverse 的 noc 源端 {source!r} != home 端 "
                         f"{base_instance!r}（session={session_id!r}）——"
                         f"方向披露与传输矛盾")
                if direction == "in_place":
                    fail(f"merge in_place 不得携带 noc 传输"
                         f"（session={session_id!r}, bytes={nbytes}）")
                noc_bytes += nbytes
            elif kind == "remote_store":
                if transfer.get("session_id") == session_id:
                    fail(f"merge v2 产生本会话池写（session={session_id!r}, "
                         f"bytes={nbytes}）——合并零池写（I6）被违反")
                if transfer.get("session_id") is None:
                    fail(f"joint merge 逐出传输缺 victim session_id"
                         f"（session={session_id!r}）")
                third_party_evictions.append(transfer)
            elif kind in ("local_hit", "remote_load"):
                continue
            else:
                fail(f"joint merge transfer 出现未知 kind={kind!r}"
                     f"（session={session_id!r}, bytes={nbytes}）——"
                     f"joint 重放词表未覆盖，拒绝静默丢弃")
        for transfer in third_party_evictions:
            if transfer["total_bytes"]:
                # 胜者侧 _ensure_capacity 的统一 T+E victim 逐出（R17-4
                # 层区间/前缀连锁键透传，同 legacy 口径）。
                replay.apply_evict(
                    {"session": transfer.get("session_id"), "tick": tick,
                     "bytes": transfer["total_bytes"], "instance": winner,
                     "kind": transfer.get("kind"),
                     "reason": transfer.get("reason"),
                     "layer_start": transfer.get("layer_start"),
                     "layer_end": transfer.get("layer_end"),
                     "resident_prefix_layers_before":
                         transfer.get("resident_prefix_layers_before"),
                     "resident_prefix_layers_after":
                         transfer.get("resident_prefix_layers_after")},
                    f"joint_merge:{session_id}")
        full_layers = replay.mapping.get("model_layers")
        if direction == "in_place":
            session.tokens = token_row["final_context_tokens"]
            session.prefix_layers = full_layers
            replay.report.actions["joint_merge_in_place"] = (
                replay.report.actions.get("joint_merge_in_place", 0) + 1)
            if decision.get("home_flipped_to") is not None:
                replay.report.actions["joint_home_migration"] = (
                    replay.report.actions.get(
                        "joint_home_migration", 0) + 1)
            return
        if direction == "forward":
            if exec_instance is None or not session.bytes:
                fail(f"merge forward 缺执行端工作副本记账"
                     f"（session={session_id!r}, bytes={session.bytes}）")
            if noc_bytes != session.bytes:
                fail(f"merge forward noc 字节 {noc_bytes} != 释放的工作"
                     f"副本 {session.bytes}（session={session_id!r}）——"
                     f"两侧账本不一致")
            replay._apply(tick, exec_instance, -session.bytes)
            replay.report.actions["joint_working_release"] = (
                replay.report.actions.get("joint_working_release", 0) + 1)
            replay.report.actions["joint_working_release_bytes"] = (
                replay.report.actions.get(
                    "joint_working_release_bytes", 0) + session.bytes)
            base_bytes = session.base_bytes
            if noc_bytes:
                replay._apply(tick, base_instance, noc_bytes)
            session.instance = base_instance
            session.bytes = base_bytes + noc_bytes
            session.prefix_layers = full_layers
        else:  # reverse（含零字节翻转）
            if base_instance is None:
                fail(f"merge reverse 无 base 记账（session={session_id!r}）")
            base_bytes = session.base_bytes
            if noc_bytes and noc_bytes != base_bytes:
                fail(f"merge reverse noc 字节 {noc_bytes} != 释放的 base "
                     f"{base_bytes}（session={session_id!r}）——两侧账本"
                     f"不一致")
            replay._apply(tick, base_instance, -base_bytes)
            if noc_bytes:
                replay._apply(tick, exec_instance, noc_bytes)
                session.bytes += noc_bytes
            session.prefix_layers = full_layers
            replay.report.actions["joint_merge_reverse"] = (
                replay.report.actions.get("joint_merge_reverse", 0) + 1)
            replay.report.actions["joint_merge_reverse_bytes"] = (
                replay.report.actions.get(
                    "joint_merge_reverse_bytes", 0) + noc_bytes)
            if not noc_bytes:
                replay.report.actions["joint_merge_zero_byte_flip"] = (
                    replay.report.actions.get(
                        "joint_merge_zero_byte_flip", 0) + 1)
        session.base_instance = None
        session.base_bytes = 0
        session.base_prefix_layers = None
        # R9 镜像（2026-09-27）：P=copy 交接轮的 D 边界 base 重录把
        # boundary_kind 设为增长基信号而不建来源槽（常规 v2 结算清账）
        # ——此处同步清账，维持"boundary_kind 在每次合并结算点归零"的
        # 生命周期不变量（残留会污染下一轮 decode 增长基裁决）。
        session.boundary_kind = None
        session.tokens = token_row["final_context_tokens"]
        if direction == "forward":
            replay.report.actions["joint_merge_forward"] = (
                replay.report.actions.get("joint_merge_forward", 0) + 1)
            replay.report.actions["joint_merge_forward_bytes"] = (
                replay.report.actions.get(
                    "joint_merge_forward_bytes", 0) + noc_bytes)
        if decision.get("home_flipped_to") is not None:
            replay.report.actions["joint_home_migration"] = (
                replay.report.actions.get("joint_home_migration", 0) + 1)
        replay.report.actions["joint_merge_settle"] = (
            replay.report.actions.get("joint_merge_settle", 0) + 1)

    def _joint_completion_three_position(
            self, replay, decision, tick, session_id, token_row,
            direction) -> None:
        """W2（2026-09-27，C4/D12 三位置确定性合并重放）。

        跟踪面 = {(base), source, exec} 三位置：source = i_P 侧旧工作副本
        （decode 行 boundary_rebuild_elsewhere 无迁移披露建立）；base =
        跨实例工作副本的 home 侧 stash（LOCAL/PARTIAL 基；REMOTE 基不
        成位置，manager 侧不会产生来源槽）。占用语义与
        face_scheduler._merge_back_three_position 同构：

        * 多位置形态：逐败者一腿 noc 整份搬入胜者（腿字节 = 该位置跟踪
          全量——镜像 W1 整份搬移口径），胜者 = 腿字节合计的接收端；
          方向 reverse ⟺ 胜者 = exec，否则 forward。
        * 重建形态（rebuild，exec 持完整并集）：零传输（合并账
          zero_byte_flip），败者重复副本直接释放；胜者恒 = exec（reverse）。
          重合持有者唯一合法形态 = 重建×base 与 exec 同实例（recompute@
          基础驻留端：重建副本持并集、覆盖 base 的 token 域，同实例
          base 记录是重复副本，零载荷就地释放——镜像 manager 的
          coincident-holders 豁免）；其余任何重合仍 fail-closed。

        胜者裁决（决策行无显式 winner_instance 披露，home_flipped_to 仅在
        home 迁移时落胜者）：reverse → exec；forward → home_flipped_to，
        缺省 = base 记账位（= home；home 不迁且胜者非 exec 的唯一形态）。
        守恒：胜者终态字节 == coef(final_context_tokens)（manager 结算守恒
        断言的镜像：base_history + source_history + context == final）。
        I6 同 v2：本会话 remote_store 即 fail；他人 victim 逐出归胜者侧。
        """
        session = replay._session(session_id)
        exec_instance = session.instance
        source_instance = session.source_instance
        if exec_instance is None:
            fail(f"three-position merge 无执行端工作副本记账"
                 f"（session={session_id!r}）——账本与重放不一致")
        if source_instance is None:  # 分派守卫的镜像（不可达，防御）
            fail(f"three-position merge 无来源侧记账"
                 f"（session={session_id!r}）——分派与跟踪态不一致")
        if direction == "in_place":
            fail(f"three-position merge 方向披露 in_place"
                 f"（session={session_id!r}）——三位置结算必有胜者，与"
                 f"无主就地语义矛盾")
        rebuild = session.boundary_kind == "rebuild"
        # 位置登记（base 可缺省——REMOTE 基形态；重合守卫镜像 manager 的
        # coincident-holders 断言，唯一放行重合 = 重建×base 与 exec 同实
        # 例：重建副本持完整并集、已覆盖 base 的 token 域，同实例 base
        # 记录是重复副本——不占位置，结算时零载荷就地释放；其余任何
        # 重合（含非重建的 base==exec、base==source、source==exec）仍
        # fail-closed）。
        positions: dict[int, int] = {}
        duplicate_base_bytes = 0
        if session.base_instance is not None:
            if rebuild and session.base_instance == exec_instance:
                duplicate_base_bytes = session.base_bytes
            else:
                positions[session.base_instance] = session.base_bytes
        for side, nbytes in ((source_instance, session.source_bytes),
                             (exec_instance, session.bytes)):
            if side in positions:
                fail(f"three-position merge 位置重合（instance={side}，"
                     f"session={session_id!r}）——边界迁移合同违规，"
                     f"账本与重放不一致")
            positions[side] = nbytes
        # ---- 胜者裁决 + 方向交叉校验。----
        home_flipped_to = decision.get("home_flipped_to")
        if home_flipped_to is not None \
                and not isinstance(home_flipped_to, int):
            fail(f"three-position merge home_flipped_to 非整数"
                 f"（session={session_id!r}, got={home_flipped_to!r}）")
        if direction == "reverse":
            winner = exec_instance
            if home_flipped_to is not None and home_flipped_to != winner:
                fail(f"three-position merge reverse 的 home_flipped_to="
                     f"{home_flipped_to} != 胜者（执行端）{winner}"
                     f"（session={session_id!r}）——方向披露与胜者矛盾")
        else:  # forward
            if home_flipped_to is not None:
                winner = home_flipped_to
            elif session.base_instance is not None:
                winner = session.base_instance
            elif source_instance is not None:
                # W3（2026-09-27，combo_none 实测形态）：REMOTE 基三位置
                # forward——base 不成位置（池 backing 无本地驻留），跟踪面
                # = {source, exec}；forward ⟺ 胜者≠执行端 ⟹ 胜者唯一 =
                # source 位（manager 式(21) 保留量降序选出的整份持有者，
                # home 未迁移即 home==source）。锚定 face_scheduler._
                # merge_back_three_position：base_location==REMOTE 时
                # base_shards=() 不进 positions，target 落 source/exec
                # 之一；direction=forward 即 target!=exec。
                winner = source_instance
            else:
                fail(f"three-position merge forward 无 home_flipped_to 且无"
                     f" base/source 记账（session={session_id!r}）——胜者"
                     f"不可裁决，账本与重放不一致")
            if winner == exec_instance:
                fail(f"three-position merge forward 的胜者 {winner} == 执行"
                     f"端 {exec_instance}（session={session_id!r}）——"
                     f"forward ⟺ 胜者≠执行端，方向披露矛盾")
        if winner not in positions:
            fail(f"three-position merge 胜者 {winner} 不在跟踪位置 "
                 f"{sorted(positions)}（session={session_id!r}）——账本与"
                 f"重放不一致")
        if rebuild and direction != "reverse":
            fail(f"three-position 重建合并方向披露 {direction!r}"
                 f"（session={session_id!r}）——重建副本持完整并集，胜者"
                 f"恒为执行端（reverse），方向披露矛盾")
        # ---- 传输预分类（方向校验＋整份搬移对账＋I6 池写断言）。----
        noc_total = 0
        leg_sources: dict[int, int] = {}
        third_party_evictions = []
        for transfer in decision.get("merge_transfers") or ():
            kind = transfer.get("kind")
            nbytes = transfer.get("total_bytes")
            if not isinstance(nbytes, int) or nbytes < 0:
                fail(f"joint merge transfer total_bytes 必须为非负整数"
                     f"（session={session_id!r}, kind={kind!r}, "
                     f"got={nbytes!r}）")
            if kind == "noc_migrate":
                source = transfer.get("source_instance_index")
                if rebuild:
                    fail(f"three-position 重建合并不得携带 noc 传输"
                         f"（session={session_id!r}, bytes={nbytes}）——"
                         f"重复副本直接释放（零载荷），方向披露与传输矛盾")
                if not isinstance(source, int):
                    fail(f"three-position merge noc 腿缺整数 "
                         f"source_instance_index（session={session_id!r}, "
                         f"got={source!r}）")
                if source == winner:
                    fail(f"three-position merge noc 腿源端 {source} == 胜者 "
                         f"{winner}（session={session_id!r}）——方向披露与"
                         f"传输矛盾")
                if source not in positions:
                    fail(f"three-position merge noc 腿源端 {source} 不在跟"
                         f"踪位置 {sorted(positions)}（session={session_id!r}）"
                         f"——账本与重放不一致")
                if source in leg_sources:
                    fail(f"three-position merge 位置 {source} 出现第二根 "
                         f"noc 腿（session={session_id!r}）——每败者恰一腿"
                         f"（整份搬移），账本与重放不一致")
                if nbytes != positions[source]:
                    fail(f"three-position merge 腿字节 {nbytes} != 位置 "
                         f"{source} 跟踪 {positions[source]}"
                         f"（session={session_id!r}）——败者整份搬移，"
                         f"少搬/多搬均属账本与重放不一致")
                leg_sources[source] = nbytes
                noc_total += nbytes
            elif kind == "remote_store":
                if transfer.get("session_id") == session_id:
                    fail(f"merge v2 产生本会话池写（session={session_id!r}, "
                         f"bytes={nbytes}）——合并零池写（I6）被违反")
                if transfer.get("session_id") is None:
                    fail(f"joint merge 逐出传输缺 victim session_id"
                         f"（session={session_id!r}）")
                third_party_evictions.append(transfer)
            elif kind in ("local_hit", "remote_load"):
                continue
            else:
                fail(f"joint merge transfer 出现未知 kind={kind!r}"
                     f"（session={session_id!r}, bytes={nbytes}）——"
                     f"joint 重放词表未覆盖，拒绝静默丢弃")
        disclosed = decision.get("merge_transferred_bytes")
        if isinstance(disclosed, int) and disclosed != noc_total:
            fail(f"three-position merge 披露 transferred_bytes {disclosed} !="
                 f" 腿合计 {noc_total}（session={session_id!r}）——两侧账本"
                 f"不一致")
        # ---- 胜者侧空间准备（统一 T+E victim 逐出，先于搬移）。----
        for transfer in third_party_evictions:
            if transfer["total_bytes"]:
                replay.apply_evict(
                    {"session": transfer.get("session_id"), "tick": tick,
                     "bytes": transfer["total_bytes"], "instance": winner,
                     "kind": transfer.get("kind"),
                     "reason": transfer.get("reason"),
                     "layer_start": transfer.get("layer_start"),
                     "layer_end": transfer.get("layer_end"),
                     "resident_prefix_layers_before":
                         transfer.get("resident_prefix_layers_before"),
                     "resident_prefix_layers_after":
                         transfer.get("resident_prefix_layers_after")},
                    f"joint_merge:{session_id}")
        # ---- 结算：败者源扣 / 胜者加（多源逐腿；重建静默释放）。----
        released_bytes = 0
        for side in sorted(positions):
            if side == winner:
                continue
            tracked = positions[side]
            if side in leg_sources:
                replay._apply(tick, side, -leg_sources[side])
                replay._apply(tick, winner, leg_sources[side])
            elif tracked:
                if not rebuild:
                    fail(f"three-position merge 败者 {side}（跟踪 {tracked} B）"
                         f"无 noc 腿（session={session_id!r}）——多源形态每"
                         f"败者恰一腿，拒绝静默漏记")
                replay._apply(tick, side, -tracked)
                released_bytes += tracked
        if duplicate_base_bytes:
            # 豁免重合（重建×base==exec）的重复 base 副本零载荷就地释放
            # ——重建方向恒 reverse，胜者 = exec = base 驻留实例，其工作
            # 副本已持该 token 域的权威内容，终态不含这份字节（去重释
            # 放非容量逐出，审计量并入 zero_payload 计数器族）。
            replay._apply(tick, winner, -duplicate_base_bytes)
            released_bytes += duplicate_base_bytes
        winner_bytes = positions[winner] + noc_total
        final_expected = replay._expected_bytes(
            token_row["final_context_tokens"])
        if winner_bytes != final_expected:
            fail(f"three-position merge 结算守恒失败：胜者 {winner} 终态 "
                 f"{winner_bytes} != f(final)={final_expected}"
                 f"（session={session_id!r}，positions={positions}，腿="
                 f"{leg_sources}）——账本与重放不一致")
        # ---- 会话终态：全层 LOCAL@胜者，三槽清账。----
        session.instance = winner
        session.bytes = winner_bytes
        session.prefix_layers = replay.mapping.get("model_layers")
        session.base_instance = None
        session.base_bytes = 0
        session.base_prefix_layers = None
        session.source_instance = None
        session.source_bytes = 0
        session.boundary_kind = None
        session.tokens = token_row["final_context_tokens"]
        replay.report.actions["joint_merge_three_position"] = (
            replay.report.actions.get("joint_merge_three_position", 0) + 1)
        replay.report.actions["joint_merge_three_position_bytes"] = (
            replay.report.actions.get(
                "joint_merge_three_position_bytes", 0) + noc_total)
        if rebuild:
            replay.report.actions[
                "joint_merge_three_position_zero_payload"] = (
                    replay.report.actions.get(
                        "joint_merge_three_position_zero_payload", 0) + 1)
            replay.report.actions[
                "joint_merge_three_position_release_bytes"] = (
                    replay.report.actions.get(
                        "joint_merge_three_position_release_bytes", 0)
                    + released_bytes)
        if decision.get("home_flipped_to") is not None:
            replay.report.actions["joint_home_migration"] = (
                replay.report.actions.get("joint_home_migration", 0) + 1)
        replay.report.actions["joint_merge_settle"] = (
            replay.report.actions.get("joint_merge_settle", 0) + 1)

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


def replay_decision_log(run_dir: Path, repo_variant: str, mapping: dict,
                        tokens: dict, coef: int,
                        capacity: Optional[int]) -> WatermarkReplay:
    scan = WatermarkScan(run_dir, repo_variant, mapping, tokens, coef,
                         capacity)
    for record in iter_jsonl(scan.log_path):
        scan.consume(record)
    return scan.finish()


# ---------------------------------------------------------------------------
# journal 权威重放（P1-①）与 tier 判定
# ---------------------------------------------------------------------------

JOURNAL_REQUIRED_FIELDS = (
    "schema_version", "sequence", "transaction_id", "planner_time_ns",
    "rank", "instance_index", "capacity_bytes", "weight_delta_bytes",
    "resident_kv_delta_bytes", "reserved_kv_delta_bytes", "before_bytes",
    "after_bytes", "cause",
)


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
        self.state: dict[int, dict] = {}  # rank -> {w, r, s}（当前）
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
        self.sha256 = hashlib.sha256()
        self._prev_sequence: Optional[int] = None
        self._prev_time: Optional[int] = None
        # M-31 会话去重态（批次 D 镜像，2026-09-30）：journal 行是逐 rank
        # 提交点，同一会话的一次逐出按 TP 数落多行——逐行计数 ×TP 虚增。
        # writer 契约保证同一 mutation 的行连续且 transaction_id 单调——
        # 事务切换即重置会话集；bytes 逐行求和（每 rank 只报自己分片的
        # 字节，总和本就不虚增）。
        self._evict_tx_current: Optional[int] = None
        self._evict_sessions_tx: set[str] = set()
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
        self._require(row.get("schema_version") == 1,
                      f"schema_version 必须为 1（实得 "
                      f"{row.get('schema_version')!r}）")
        sequence = row["sequence"]
        tick = row["planner_time_ns"]
        self._require(isinstance(sequence, int) and not isinstance(
            sequence, bool) and sequence >= 0, "sequence 非非负整数")
        self._require(isinstance(tick, int) and not isinstance(tick, bool)
                      and tick >= 0, "planner_time_ns 非非负整数")
        self._require(self._prev_sequence is None or sequence >
                      self._prev_sequence,
                      f"sequence 回退（{sequence} <= {self._prev_sequence}）")
        self._require(self._prev_time is None or tick >= self._prev_time,
                      f"planner_time_ns 回退（{tick} < {self._prev_time}）")
        self._prev_sequence = sequence
        self._prev_time = tick
        if self.sequence_first is None:
            self.sequence_first = sequence
        self.sequence_last = sequence
        transaction_id = row["transaction_id"]
        if isinstance(transaction_id, int) and not isinstance(
                transaction_id, bool) and transaction_id > self.transaction_max:
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
        deltas = {}
        for field, key in (("weight_delta_bytes", "w"),
                           ("resident_kv_delta_bytes", "r"),
                           ("reserved_kv_delta_bytes", "s")):
            value = row[field]
            self._require(isinstance(value, int) and not isinstance(
                value, bool), f"{field} 非整数")
            deltas[key] = value
        before = row["before_bytes"]
        after = row["after_bytes"]
        self._require(isinstance(before, dict) and isinstance(after, dict),
                      "before_bytes/after_bytes 必须是对象")
        parsed_before = {}
        parsed_after = {}
        for source, target in ((before, parsed_before), (after, parsed_after)):
            for field, key in (("weight", "w"), ("resident", "r"),
                               ("reserved", "s")):
                value = source.get(field)
                self._require(isinstance(value, int) and not isinstance(
                    value, bool) and value >= 0,
                    f"{field if source is after else 'before.' + field}"
                    f" 非非负整数")
                target[key] = value
        current = self.state.get(rank)
        expected_before = current if current is not None \
            else {"w": 0, "r": 0, "s": 0}
        for key, label in (("w", "weight"), ("r", "resident"),
                           ("s", "reserved")):
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
        # R6-U9（2026-09-29 face 移植）：weight==0 的 rank 同样查绑定表
        # ——退化切分可能为 weight 0 绑定硬上限，条件跳过会让该 rank 漏检
        # （诊断口径必须覆盖全部 rank；无绑定时 .get 恒 None，与原行为
        # 一致）。
        hard_limit = self.hard_limit_by_weight.get(parsed_after["w"])
        if hard_limit is not None and parsed_after["r"] > hard_limit:
            self.hard_limit_exceed_events += 1
            self.hard_limit_exceed_ranks[rank] = \
                self.hard_limit_exceed_ranks.get(rank, 0) + 1
        cause = row.get("cause")
        cause_label = cause if isinstance(cause, str) else repr(cause)
        self.causes[cause_label] = self.causes.get(cause_label, 0) + 1
        if deltas["r"]:
            self.cplog.add(tick, instance, deltas["r"])
        if "evict" in cause_label.lower() and deltas["r"] < 0:
            # M-31（批次 D 镜像，2026-09-30）：evict 计数以会话数为口径
            # ——同一会话跨 rank 多行只计一次事件；bytes 逐行求和不虚增。
            transaction = row["transaction_id"]
            if transaction != self._evict_tx_current:
                self._evict_tx_current = transaction
                self._evict_sessions_tx = set()
            # 会话键：session_id 可缺席（合成行）——repr 兜底保证可哈希。
            session_key = repr(row.get("session_id"))
            new_event = session_key not in self._evict_sessions_tx
            if new_event:
                self._evict_sessions_tx.add(session_key)
            self.cplog.add_evict_row(tick, instance, -deltas["r"], new_event)
        self.rows += 1

    def final_rank_state(self) -> dict:
        return {rank: {
            "instance_index": self.rank_instance[rank],
            "capacity_bytes": self.rank_capacity[rank],
            "weight": state["w"],
            "resident": state["r"],
            "reserved": state["s"],
            "physical": state["w"] + state["r"] + state["s"],
        } for rank, state in sorted(self.state.items())}


def load_journal_checksum(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"kv_delta_journal_checksum.json 非法 JSON：{path}: {exc}")
    if not isinstance(data, dict):
        fail(f"kv_delta_journal_checksum.json 必须是对象：{path}")
    return data


def replay_journal(journal_path: Path,
                   checksum_path: Optional[Path],
                   hard_limit_by_weight: Optional[dict[int, int]]) -> dict:
    """journal 单遍流式重放 + tier 判定（P1-① 的权威路径）。

    返回 {tier, replay, checksum, sha256_hex, checks}。fail-closed 条件
    （退出码 2，不降级）：sha256 不匹配 / 行级链断裂 / 行数与证书不符 /
    重放终态与证书终态矛盾。checks 有 false → lifecycle_replay_exact
    （run 末守恒未过：journal 行级仍精确，但不作正式判决）。
    """
    replay = JournalReplay(journal_path, hard_limit_by_weight)
    with journal_path.open("rb") as handle:
        for lineno, raw in enumerate(handle, start=1):
            replay.sha256.update(raw)
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
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
    if checksum_path is not None:
        checksum = load_journal_checksum(checksum_path)
        expected_sha = checksum.get("sha256")
        if expected_sha is not None and expected_sha != sha256_hex:
            fail(f"{journal_path}: journal sha256 与证书不符（journal="
                 f"{sha256_hex}，checksum={expected_sha}）——账本完整性"
                 f"破坏，fail-closed 不降级")
        expected_lines = checksum.get("line_count")
        if isinstance(expected_lines, int) and expected_lines != replay.rows:
            fail(f"{journal_path}: journal 行数与证书不符（journal="
                 f"{replay.rows}，checksum={expected_lines}）")
        ranks_block = checksum.get("ranks")
        if isinstance(ranks_block, dict) and ranks_block:
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
                # M-32（批次 D 镜像，2026-09-30）：终态分量全字段严格
                # 类型校验。旧的 `isinstance(value, int) and ...` 形状对
                # 缺失/None/str/float 静默空过、bool 冒充整数混过——证书
                # 残缺也能 certified，第三道门形同虚设。分量字段缺失/类型
                # 不符 = 证书无法证明终态，与终态不符同罪，fail-closed。
                for field, key in (("weight", "w"), ("resident", "r"),
                                   ("reserved", "s")):
                    value = entry.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) \
                            or value < 0:
                        fail(
                            f"{checksum_path}: rank {rank} 终态 {field} 缺失"
                            f"或非非负整数（实得 {value!r}）——证书残缺，"
                            f"fail-closed 不降级")
                    if value != state[key]:
                        fail(
                            f"{checksum_path}: rank {rank} 终态 {field} 与"
                            f" journal 重放不符（证书 {value}，重放 "
                            f"{state[key]}）——账本与证书矛盾")
                # M-32：physical 字段纳入校验（此前完全不查）。口径 =
                # weight + resident（schema 的 physical 两分量口径）。
                expected_physical = entry.get("physical")
                if isinstance(expected_physical, bool) or \
                        not isinstance(expected_physical, int):
                    fail(
                        f"{checksum_path}: rank {rank} 终态 physical 缺失"
                        f"或非整数（实得 {expected_physical!r}）——证书"
                        f"残缺，fail-closed 不降级")
                replay_physical = state["w"] + state["r"]
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
            missing = sorted(set(replay.state) - {
                int(key) for key in ranks_block if str(key).lstrip("-").isdigit()})
            if missing:
                fail(f"{checksum_path}: journal 含证书未覆盖的 rank"
                     f"（{missing[:5]}，共 {len(missing)}）——账本与证书"
                     f"矛盾")
        checks = checksum.get("checks") if isinstance(
            checksum.get("checks"), dict) else {}
        if checks and all(checks.get(name) is True for name in (
                "manager_state_match", "physical_equals_weight",
                "residual_reserved_zero", "residual_resident_zero")):
            tier = TRUST_TIER_CERTIFIED
        else:
            tier = TRUST_TIER_LIFECYCLE
    return {
        "tier": tier,
        "replay": replay,
        "checksum": checksum,
        "sha256_hex": sha256_hex,
        "checks": checks,
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

    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    bucket_ns, bucket_provisional = load_bucket_ns(manifest)

    tokens = load_token_manifest(args.run_dir, args.token_manifest)
    trace = load_trace_config(args.run_dir, args.trace_config)
    config = trace["values"]
    coef = 2 * config["layers"] * config["hidden_size"] * \
        config["bytes_per_elem"]
    # R17-4：模型层数随 mapping 下发（joint 驻留前缀跟踪的"全层数"基准；
    # REPO_VARIANTS 是模块级登记表，注入进副本、不动全局。A4 driver 经
    # prep["mapping"] 复用，无需改签名）。
    mapping = dict(mapping)
    mapping["model_layers"] = config["layers"]

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
            npus_for_calibers = load_npus_per_instance(
                args.run_dir, args.request_manifest, request_manifest_loader) \
                if not npus else npus
            try:
                calibers = compute_capacity_calibers(
                    model, npus_for_calibers, hardware["bytes"],
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
        # M-33（批次 D 镜像，2026-09-30）：孤儿证书（journal 缺失、
        # checksum 在场）诚实标注，不与"阶段2 前旧 run"混同。
        violation_check = "orphan_checksum_upper_bound_only"
    else:
        violation_check = "diagnostic_upper_bound_only"
    if capacity is None and not journal_present:
        # M-32（批次 D 镜像，2026-09-30）：degraded 覆盖只适用于
        # decision-log 上界层——journal 路径的逐 rank 认证用行内
        # capacity_bytes（不依赖配置容量）。旧的无条件覆盖让 certified 层
        # exit 3 正式判决与 "degraded_peak_recorded" 标注同框自相矛盾。
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
    scan = WatermarkScan(args.run_dir, repo_variant, prep["mapping"],
                         prep["tokens"], prep["coef"], prep["capacity"])
    comparison_failure = None
    if prep["journal_present"]:
        # R6-U5（2026-09-29 face 移植）：journal 在场时 decision-log 重放
        # 仅为对照列——重放失败隔离为对照列 FAIL，权威（journal）产物照常
        # 产出，不连坐；journal 缺席时重放即权威，失败原样 fail-closed
        # 不降级。
        try:
            for record in iter_jsonl(scan.log_path):
                scan.consume(record)
        except SloToolError as exc:
            comparison_failure = str(exc)
            print(f"[hbm-watermark] 警告：decision-log 对照重放失败——"
                  f"对照列标记 FAIL，journal 权威产物照常产出（{exc}）",
                  file=sys.stderr)
    else:
        for record in iter_jsonl(scan.log_path):
            scan.consume(record)
    try:
        replay = scan.finish()
    except SloToolError:
        if comparison_failure is None:
            raise
        # 对照重放已失败：finish 的恰一次/非空断言对部分消费态不再适用，
        # 以部分重放态支撑对照列（权威侧不受影响）。
        replay = scan.replay
    return watermark_emit(args, repo_variant, prep, replay,
                          comparison_failure=comparison_failure)


def _intervals_rows(cplog: ChangePointLog, repo_variant: str):
    """intervals.csv 的行流：per instance 变点区间（含逐出列）。"""
    for instance in cplog.eventful_units():
        unit = cplog.units[instance]
        next_tick: Optional[int] = None
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
                   replay: WatermarkReplay,
                   comparison_failure: Optional[str] = None) -> int:
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
            # M-33（批次 D 镜像，2026-09-30）：孤儿证书诚实诊断——journal
            # 丢失（归档/拷贝不完整）不是"阶段2 前旧 run"；判决面不变
            # （无权威重放即无正式判决，退出码 3 废除），只修误导文案。
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
    if span_start is None or span_end is None or span_end <= span_start:
        fail("KV 动作时间跨度为 0/空事件流，无法分桶（decision-log 与 "
             "journal 两侧均无可重放 KV 动作）")
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
    worst_rows_requested = max(
        (cplog.units[u].last_tick - span_start) // bucket_ns for u in units) \
        + 1
    worst_rows_effective = max(
        (cplog.units[u].last_tick - span_start) // effective_bucket_ns
        for u in units) + 1

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
    if dense_rows_total > row_budget:
        fail(f"绘图 series 行数 {dense_rows_total} 超全局行预算 "
             f"{row_budget}（B_eff 公式失效）——fail-closed（不变量破坏，"
             f"拒绝静默超预算交付）")

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
                       stat["evict_bytes"] if coverage in ("full",
                                                           "full_reconciled")
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
        # M17（2026-09-29）：认证层显式披露。本仓自产 run 无 per-rank
        # kv_delta_journal 生产者（results/kv_delta_journal.jsonl 全仓零
        # 写入方——online_service 的 C14 kv_delta_journal 是逐请求结算
        # sidecar 键，行结构/文件均非本工具消费的 per-rank delta 账本，
        # 见模块 docstring M17 注记）——certified 判决路径在自产 run 上
        # 不可达，恒落 upper_bound_only；该字段使静默降级变为显式披露
        # （certified 时取 "certified"，journal 在场但无守恒证书的
        # resident/lifecycle 层透传 tier 名）。
        "certification_layer": (
            "certified" if trust_tier == TRUST_TIER_CERTIFIED
            else "upper_bound_only"
            if trust_tier == TRUST_TIER_UPPER_BOUND else trust_tier),
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
        # L18（2026-09-29）：顶层逐出计数恒为 decision-log 重放（对照列）
        # 数字——journal 权威路径（tier>=resident_kv_exact）下亦然。加
        # _upper_bound 后缀 + 来源标注字段，消除与 journal 认证值的混用；
        # ``actions`` 仍为 decision-log 重放完整动作计数（对照列）。
        "total_evict_events_upper_bound":
            replay.report.actions["evictions"],
        "total_evict_bytes_upper_bound":
            replay.report.actions["evict_bytes"],
        "partial_evictions_upper_bound":
            replay.report.actions["partial_evictions"],
        "eviction_counters_source": (
            "decision-log KV action replay (upper bound; comparison column "
            "when the kv_delta_journal authority path is active — not a "
            "journal-certified value)"),
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
                "interval_rows": len(cplog.units[instance].points),
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
        comparison = {}
        for instance in sorted({**{u: None for u in units},
                                **{u: None for u in replay.cplog.units}}):
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
        # R6-U5（2026-09-29 face 移植）：对照重放失败时对照列标记 FAIL
        # （部分消费态 + 失败消息），权威产物照常产出——不因对照面问题
        # 连坐丢 intervals/series/instances/summary。
        summary["decision_log_comparison_status"] = \
            "ok" if comparison_failure is None else "FAIL"
        if comparison_failure is not None:
            summary["decision_log_comparison_failure"] = comparison_failure
            print(f"[hbm-watermark] 警告：decision-log 对照列 FAIL"
                  f"（对照重放失败，上界对照值仅部分消费态）："
                  f"{comparison_failure}", file=sys.stderr)

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
