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

算法（重放，非策略复刻）：
  按 (文件顺序=seq) 重放每条决策记录，记录内固定次序「逐出 → 恢复/迁移
  → 增长」。会话状态（当前所在实例、当前本地 bytes）由本脚本自行跟踪，
  恢复/迁移的方向由跟踪态判定（本地跨实例=搬移、远端/同实例=只增），
  已落盘 bytes 与 f(tokens)=2·layers·hidden_size·bytes_per_elem·tokens
  逐条对账（f 已在五仓基线数据上核对：restore 比值全 1.0，S3 另有 0.5
  分层半恢复）。增长按 manager 语义"长到 f(目标 tokens)"取增量。

  → 每 instance 占用时间序列（桶长 = manifest watermark_sample_period_ns，
    null → 5,000,000 ns 临时锚点，全输出标 provisional）；
  → 汇总 JSON：每 instance peak/mean/时长、逐出次数/bytes、容量违规计
    数（occupancy > capacity 必须为 0；>0 → 退出码 3 + 标红报错，宁可
    报错不可静默——既可能是重建口径错误，也可能是真实问题）。

容量链（逐仓登记，不编造）：
  trace_config.csv config 行 local_hbm_capacity_profile → 仓内
  sh_test_mesh/hardware/*.json 的 local-hbm.capacity-profiles[profile].bytes
  （每 NPU 字节）× npus_per_instance（取 request manifest requests[].
  prefill_ranks 长度）。任一环节缺失 → capacity=NA，违规检查降级为
  "峰值记录"并注明，绝不代拟容量值。

覆盖度（fail-closed 语义的对偶面，显式降级、绝不静默）：
  * FACE/W/S1/S3：逐出条目带 bytes+victim → eviction_coverage=full，
    违规检查生效；
  * S2：decision log 只落 *_eviction_count（无 victim/bytes），他人逐出
    无法归因 → occupancy 为上界（未归因逐出不扣减），违规检查降级为
    峰值记录（occupancy_valid=false）。

退出码：0 正常；2 fail-closed（缺文件/缺列/结构错/重放不可续）；
3 容量违规（occupancy > capacity，coverage=full 且 capacity 已知时）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, NA, SloToolError, default_manifest_path,
    detect_repo_variant, emit_json, fail, iter_jsonl,
    load_request_manifest, load_slo_manifest, manifest_requests,
    open_output, read_init_record, run_main, write_csv,
)

EXIT_VIOLATION = 3  # 容量违规（与 fail-closed 的 2 区分）

# watermark_sample_period_ns 未推导（null）时的临时桶长锚点（ns）。
# 全输出（JSON/CSV/stderr）都会带 bucket_ns_provisional=true 标注。
PROVISIONAL_BUCKET_NS = 5_000_000

TOKEN_MANIFEST_FILENAME = "manifest.json"

SERIES_COLUMNS = (
    "repo_variant", "instance_index", "bucket_index", "bucket_start_ns",
    "bucket_end_ns", "occupancy_end_bytes", "occupancy_peak_in_bucket_bytes",
    "evict_events", "evict_bytes",
)
INSTANCE_COLUMNS = (
    "repo_variant", "instance_index", "eviction_coverage", "occupancy_valid",
    "capacity_bytes", "capacity_source", "bucket_ns",
    "bucket_ns_provisional", "first_event_ns", "last_event_ns",
    "duration_ns", "peak_occupancy_bytes", "mean_occupancy_bytes",
    "residual_occupancy_bytes", "evict_events", "evict_bytes",
    "violation_events",
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
        if not isinstance(value, int) or value < 0:
            return None
        total += value
    return total


def _scalar_bytes(entry: dict) -> Optional[int]:
    value = entry.get("total_bytes")
    if isinstance(value, int) and value >= 0:
        return value
    return None


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
    if victim_instance is not None and not isinstance(victim_instance, int):
        victim_instance = None
    time_ns = entry.get("time_ns")
    eff_tick = time_ns if isinstance(time_ns, int) and time_ns >= 0 else tick
    return {"type": "evict", "tick": eff_tick, "session": victim_session,
            "bytes": nbytes, "instance": victim_instance,
            "kind": entry.get("kind"), "reason": entry.get("reason")}


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
    for path_element in spec.get("bytes_path", []):
        if isinstance(holder, dict):
            holder = holder.get(path_element)
        else:
            holder = None
    if isinstance(holder, dict):
        value = holder.get("total_bytes")
        if isinstance(value, int):
            nbytes = value
        kind = holder.get("kind")
        source = holder.get("source_instance_index")
        if isinstance(source, int):
            logged_source = source
    elif isinstance(holder, int):
        nbytes = holder
    if spec.get("kind_field"):
        value = decision.get(spec["kind_field"])
        if isinstance(value, str):
            kind = value
    if spec.get("source_field"):
        value = decision.get(spec["source_field"])
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
        value = holder.get("total_bytes")
        if isinstance(value, int):
            nbytes = value
        source = holder.get("source_instance_index")
        if isinstance(source, int):
            logged_source = source
    elif isinstance(holder, int):
        nbytes = holder
    if spec.get("source_field"):
        value = decision.get(spec["source_field"])
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
    # decode 记录不含逐出列表）不落盘——802 例 RECOMPUTE(state_before=
    # EVICTED) 隐含静默逐出；重放按账本断言在恢复点对账扣减
    # （silent_evictions_reconciled），逐出真实时刻 ∈ 上次可见事件与恢复
    # tick 之间，占用在该窗口为上界。
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
}


# ---------------------------------------------------------------------------
# 输入装载（fail-closed）
# ---------------------------------------------------------------------------

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
        if not isinstance(value, int) or value <= 0:
            fail(f"hardware 容量档 {profile!r}.bytes 非正整数：{candidate}")
        return {"path": str(candidate), "profile": profile, "bytes": value}
    fail(f"hardware 配置中找不到容量档 {profile!r}"
         f"（尝试：{[str(c) for c in candidates]}）——用 --hardware-config "
         f"指定含 local-hbm.capacity-profiles 的 json；无法解析时本命令"
         f"拒绝执行（不编造容量）")


def load_npus_per_instance(run_dir: Path, explicit: Optional[Path]) -> int:
    """每实例 NPU 数：request manifest requests[].prefill_ranks 长度。

    优先级：--request-manifest > run_dir/metrics_manifest.json >
    cpp.log init 行 manifest_path。
    """
    try:
        manifest = load_request_manifest(run_dir, explicit)
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

class SessionState:
    """会话跟踪态：当前实例 + 当前本地 bytes（分层仓可为部分层）。"""

    __slots__ = ("instance", "bytes", "tokens")

    def __init__(self) -> None:
        self.instance: Optional[int] = None
        self.bytes: int = 0
        self.tokens: int = 0


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

    def __init__(self, repo_variant: str, mapping: dict, tokens: dict,
                 coef_bytes_per_token: int,
                 capacity_per_instance: Optional[int]) -> None:
        self.repo_variant = repo_variant
        self.mapping = mapping
        self.tokens = tokens
        self.coef = coef_bytes_per_token
        self.capacity = capacity_per_instance
        self.sessions: dict[str, SessionState] = {}
        self.events: list[tuple[int, int, int]] = []  # (tick, instance, delta)
        self.evict_marks: list[tuple[int, int, int]] = []  # (tick,inst,bytes)
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
        self.events.append((tick, instance, delta))
        if evict_bytes is not None:
            self.evict_marks.append((tick, instance, evict_bytes))
        if self.capacity is not None and current > self.capacity:
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
        self._apply(action["tick"], session.instance, -action["bytes"],
                    evict_bytes=action["bytes"])
        session.bytes -= action["bytes"]
        if session.bytes == 0:
            session.instance = None
            session.tokens = 0
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
            self.report.actions["own_relocation_partial_unknown"] += 1
        # local_hbm：保留，无事。


# ---------------------------------------------------------------------------
# 决策记录 → 动作流
# ---------------------------------------------------------------------------

def replay_decision_log(run_dir: Path, repo_variant: str, mapping: dict,
                        tokens: dict, coef: int,
                        capacity: Optional[int]) -> WatermarkReplay:
    replay = WatermarkReplay(repo_variant, mapping, tokens, coef, capacity)
    seen_kinds: dict[tuple[str, str], int] = {}
    log_path = run_dir / DECISION_LOG_RELPATH
    last_tick = -1
    for record in iter_jsonl(log_path):
        where = f"{log_path}:seq={record.get('seq', '?')}"
        request_id = record.get("request_id")
        kind = record.get("kind")
        if not isinstance(request_id, str) or not request_id:
            fail(f"{where}: 决策记录缺 request_id")
        if kind not in ("prefill", "decode", "completion"):
            fail(f"{where}: 未知 kind={kind!r}")
        key = (request_id, kind)
        if key in seen_kinds:
            fail(f"{where}: 请求 {request_id} 的 {kind} 决策出现两次"
                 f"（先于 seq={seen_kinds[key]}）——账本次序异常")
        seen_kinds[key] = record.get("seq", -1)
        tick = record.get("tick")
        if not isinstance(tick, int) or tick < 0:
            fail(f"{where}: 缺非负整数 tick")
        if tick < last_tick:
            fail(f"{where}: tick 回退（{tick} < {last_tick}）——文件顺序"
                 f"与时间顺序不一致，无法安全重放")
        last_tick = tick
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

        # 2) 按记录类型主动作。
        if kind == "prefill":
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
                    replay.evict_marks.append(
                        (tick, instance if instance is not None else -1, 0))

    missing = []
    for request_id in tokens["requests"]:
        for kind in ("prefill", "decode", "completion"):
            if (request_id, kind) not in seen_kinds:
                missing.append(f"{request_id}:{kind}")
    if missing:
        fail(f"{log_path}: token manifest 中的请求缺决策记录（前 5 例："
             f"{missing[:5]}，共 {len(missing)}）——两源请求集不一致")
    if not replay.events:
        fail(f"{log_path}: 没有任何可重放的 KV 动作")
    return replay


# ---------------------------------------------------------------------------
# 分桶与汇总
# ---------------------------------------------------------------------------

def bucketize(replay: WatermarkReplay, bucket_ns: int) -> dict:
    """逐实例桶时序（桶末占用 + 桶内峰值 + 逐出叠加）。

    桶 j 覆盖 [span_start + j·B, +B)；事件 tick 落在 [start, end) 计入
    桶 j；末桶补记 span_end 处事件。active 桶 = 有事件或占用非零。
    """
    events = sorted(replay.events, key=lambda item: (item[0],))
    span_start = events[0][0]
    span_end = events[-1][0]
    if span_end <= span_start:
        fail("KV 动作时间跨度为 0，无法分桶")
    # 末桶必须能容纳恰落在 span_end 的事件（其桶号 = span//B）——
    # 用 ceil 会把末事件静默丢出网格。
    n_buckets = (span_end - span_start) // bucket_ns + 1

    per_instance_events: dict[int, list[tuple[int, int]]] = {}
    for tick, instance, delta in events:
        per_instance_events.setdefault(instance, []).append((tick, delta))
    per_instance_evicts: dict[int, list[tuple[int, int]]] = {}
    for tick, instance, nbytes in replay.evict_marks:
        per_instance_evicts.setdefault(instance, []).append((tick, nbytes))

    series: dict[int, list[dict]] = {}
    stats: dict[int, dict] = {}
    for instance, ievents in per_instance_events.items():
        ievents.sort(key=lambda item: item[0])
        evicts = per_instance_evicts.get(instance, [])
        evict_by_bucket: dict[int, list[int]] = {}
        for tick, nbytes in evicts:
            index = min((tick - span_start) // bucket_ns, n_buckets - 1)
            pair = evict_by_bucket.setdefault(index, [0, 0])
            pair[0] += 1
            pair[1] += nbytes
        rows: list[dict] = []
        occupancy = 0
        cursor = 0
        first_tick = ievents[0][0]
        last_tick = ievents[-1][0]
        peak = 0
        area = 0  # Σ(occupancy × dt)（先整数后除，展示层才转浮点）
        prev_tick = first_tick
        # 逐桶推进：每桶先吸收事件，再记录桶末占用。
        for index in range(n_buckets):
            bucket_start = span_start + index * bucket_ns
            bucket_end = min(bucket_start + bucket_ns,
                             span_end + 1)  # 末桶含 span_end 事件
            bucket_peak = occupancy
            while cursor < len(ievents) and ievents[cursor][0] < bucket_end:
                tick, delta = ievents[cursor]
                area += occupancy * (tick - prev_tick)  # 先记旧占用×时长
                occupancy += delta
                bucket_peak = max(bucket_peak, occupancy)
                peak = max(peak, occupancy)
                prev_tick = tick
                cursor += 1
            evict_pair = evict_by_bucket.get(index)
            if occupancy > 0 or bucket_peak > 0 or evict_pair:
                rows.append({
                    "bucket_index": index,
                    "bucket_start_ns": bucket_start,
                    "bucket_end_ns": bucket_start + bucket_ns,
                    "occupancy_end_bytes": occupancy,
                    "occupancy_peak_in_bucket_bytes": bucket_peak,
                    "evict_events": evict_pair[0] if evict_pair else 0,
                    "evict_bytes": evict_pair[1] if evict_pair else 0,
                })
        # 事件间隔用 tick 步进（线性插值不做：只累计事件间驻留面积）。
        duration = last_tick - first_tick
        stats[instance] = {
            "first_event_ns": first_tick,
            "last_event_ns": last_tick,
            "duration_ns": duration,
            "peak_occupancy_bytes": peak,
            "mean_occupancy_bytes": (area / duration) if duration > 0
            else float(peak),
            "residual_occupancy_bytes": occupancy,
            "evict_events": sum(pair[0] for pair in
                                evict_by_bucket.values()),
            "evict_bytes": sum(pair[1] for pair in evict_by_bucket.values()),
        }
        series[instance] = rows
    return {
        "span_start_ns": span_start,
        "span_end_ns": span_end,
        "bucket_ns": bucket_ns,
        "n_buckets": n_buckets,
        "series": series,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def cmd_hbm_watermark(args: argparse.Namespace) -> int:
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    mapping = REPO_VARIANTS.get(repo_variant)
    if mapping is None:
        fail(f"未登记的 repo_variant：{repo_variant}"
             f"（REPO_VARIANTS 需扩表并附基线核对证据）")

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
                npus = load_npus_per_instance(args.run_dir,
                                              args.request_manifest)
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

    coverage = mapping["eviction_coverage"]
    occupancy_valid = coverage in ("full", "full_reconciled")
    violation_check = "occupancy_gt_capacity" if (
        capacity is not None and occupancy_valid) else (
        "degraded_peak_recorded" if capacity is None else
        "degraded_upper_bound_no_certification")

    replay = replay_decision_log(args.run_dir, repo_variant, mapping,
                                 tokens, coef, capacity
                                 if violation_check ==
                                 "occupancy_gt_capacity" else None)
    result = bucketize(replay, bucket_ns)

    # -- 输出 ------------------------------------------------------------
    stream, close = open_output(args.output, "slo_hbm_watermark_series.csv",
                                args.run_dir)
    try:
        def series_rows():
            for instance in sorted(result["series"]):
                for row in result["series"][instance]:
                    yield (repo_variant, instance, row["bucket_index"],
                           row["bucket_start_ns"], row["bucket_end_ns"],
                           row["occupancy_end_bytes"],
                           row["occupancy_peak_in_bucket_bytes"],
                           row["evict_events"], row["evict_bytes"])
        write_csv(stream, SERIES_COLUMNS, series_rows())
    finally:
        if close:
            stream.close()

    istream, iclose = open_output(
        args.instances_csv, "slo_hbm_watermark_instances.csv", args.run_dir)
    try:
        def instance_rows():
            for instance in sorted(result["stats"]):
                stat = result["stats"][instance]
                yield (repo_variant, instance, coverage,
                       "true" if occupancy_valid else "false",
                       capacity if capacity is not None else NA,
                       capacity_source, bucket_ns,
                       "true" if bucket_provisional else "false",
                       stat["first_event_ns"], stat["last_event_ns"],
                       stat["duration_ns"], stat["peak_occupancy_bytes"],
                       f"{stat['mean_occupancy_bytes']:.3f}",
                       stat["residual_occupancy_bytes"],
                       stat["evict_events"],
                       stat["evict_bytes"] if coverage == "full" else NA,
                       replay.violation_by_instance.get(instance, 0))
        write_csv(istream, INSTANCE_COLUMNS, instance_rows())
    finally:
        if iclose:
            istream.close()

    total_violations = replay.violation_events
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
        "algorithm": "decision-log KV action replay (evict -> restore/"
                     "move -> grow, per-record order) -> per-instance "
                     "occupancy series (WP8 offline primary source)",
        "bucket_ns": bucket_ns,
        "bucket_ns_provisional": bucket_provisional,
        "bucket_ns_source": "slo_params_manifest.watermark_sample_period_ns"
                            + ("（临时锚点 5,000,000 ns）"
                               if bucket_provisional else ""),
        "span_start_ns": result["span_start_ns"],
        "span_end_ns": result["span_end_ns"],
        "n_buckets": result["n_buckets"],
        "n_instances": len(result["stats"]),
        "coef_bytes_per_token": coef,
        "coef_formula": "2*layers*hidden_size*bytes_per_elem",
        "trace_config_source": trace["path"],
        "token_manifest_source": tokens["path"],
        "capacity_bytes_per_instance": capacity,
        "capacity_source": capacity_source,
        "capacity_field_evidence": mapping["capacity_field_evidence"],
        "eviction_coverage": coverage,
        "occupancy_valid": occupancy_valid,
        "occupancy_note": occupancy_note,
        "violation_check": violation_check,
        "violation_events": total_violations,
        "violation_instances": {str(k): v for k, v in
                                sorted(replay.violation_by_instance.items())},
        "max_exceed_bytes": replay.max_exceed_bytes,
        "total_evict_events": replay.report.actions["evictions"],
        "total_evict_bytes": replay.report.actions["evict_bytes"],
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
                "active_buckets": len(result["series"][instance]),
                "violation_events":
                    replay.violation_by_instance.get(instance, 0),
            }
            for instance, stat in sorted(result["stats"].items())
        },
    }
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

    if total_violations:
        print("", file=sys.stderr)
        print("!! [hbm-watermark] 容量违规：occupancy > capacity 共 "
              f"{total_violations} 个事件点（实例分布 "
              f"{summary['violation_instances']}，最大超出 "
              f"{replay.max_exceed_bytes} B）——可能是重建口径错误，也可"
              f"能是真实超卖；宁可报错不可静默，退出码 {EXIT_VIOLATION}",
              file=sys.stderr)
        return EXIT_VIOLATION
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hbm_watermark.py",
        description="WP8 补充主数据源：离线重放 python 侧 ledger（decision "
                    "log KV 动作 + plan manifest token 事实）重建每实例 "
                    "HBM KV 占用水位线（时序 CSV + 峰值/均值/逐出/容量违"
                    "规汇总）。fail-closed：缺文件/缺列/重放不一致即非零"
                    "退出；容量违规退出码 3。")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 results/online_decision_log."
                             "jsonl、cpp.log；token manifest 与 "
                             "trace_config 可自动定位或显式指定）")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="slo_params_manifest.json（默认：脚本同目录；"
                             "watermark_sample_period_ns null → 5,000,000 ns"
                             " 临时锚点并标 provisional）")
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
                             "bytes_per_elem/local_hbm_capacity_profile）")
    parser.add_argument("--hardware-config", type=Path, default=None,
                        help="hardware json（local-hbm.capacity-profiles；"
                             "默认搜脚本所在仓 sh_test_mesh/hardware/）")
    parser.add_argument("-o", "--output", default="",
                        help="逐实例桶时序 CSV（'-'=stdout；缺省写 run_dir/"
                             "slo_hbm_watermark_series.csv）")
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
