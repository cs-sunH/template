#!/usr/bin/env python3
"""Hop-Bytes = Σ_actions bytes × noc_hops（主规格 §1.5-A；纯后处理）。

从运行产物中"含 noc hops 的传输记录"聚合，分请求/总量两档输出。各仓
真实字段名已在基线产物中核对并登记于 REPO_HOP_SOURCES（先定位字段、
逐仓登记，不猜测）：

  astra-sim-sh_1.0  shard 级：decision log 的 history_transfer /
                    prefill_decode_transfer / 各类逐出传输对象的
                    shards[].noc_path（hops = len(noc_path)-1；生成侧
                    sh_1.0 仓 generate_face_trace.py 亦落 noc_hops 字段——
                    读取时优先用显式 noc_hops，缺省由 noc_path 推导）。
  astra-sim-wscllm  实例级：decode 决策的 static_route.hop_count ×
                    prefill_decode_transfer.total_bytes（粒度为实例间
                    hop，非 rank 级）。2026-08-26 B2wp9py 起 prefill 决策
                    对 history NOC_MIGRATE 迁移附加 noc_hops 字段
                    （实例图最短路，同粒度）→ history 迁移纳入覆盖
                    （字段为 A/B 对拍剥离清单条目）。
                    2026-09-05 问题 4b 起 decode 行 prefill_decode_
                    transfer.shards[] 逐 shard bytes×noc_hops（mesh 跳
                    口径；旧产物回退聚合口径，数值不变）。
                    2026-09-24（M29）起逐出/回迁契约行（history_
                    evictions/prefill_evictions/decode_evictions/
                    completion_evictions；旧产物 admission_evictions+
                    decode_target_evictions）计入 bytes_without_hops
                    ——契约行无 shard 级路由，coverage 如实低于 1
                    （此前逐出字节整类不进账、notes 误称满覆盖）。
                    2026-09-26（G7-15）起顶层 noc_hops 缺席的 prefill 行
                    （REMOTE_RESTORE 远端回迁）逐对象消费 transfer_hop_
                    bytes 行的 shard 级 hops 补全 hop 分子（行字节合计与
                    history_transfer_bytes 相符才启用；旧产物回退标量
                    bytes_without_hops 口径）。
  astra-sim-face    per-TP-shard 级（B2wp9py 起决策日志附带只读
                    hops 列表：prefill=history_noc_hops、decode=
                    kv_noc_hops，与 shards 一一对齐；hops 全等时
                    聚合 bytes×hops[0] 与逐 shard 求和严格相等）。
                    旧产物无字段 → bytes_without_hops。
  astra-sim-sh_2.0  WP9-线5（2026-08-26）起 decision 逐传输对象序列化
                    transfer_hop_bytes[].{total_bytes, noc_hop_bytes,
                    shard_count}（KVTransfer shards 按 deterministic_xy_
                    route 生成、在线复用）→ hop_bytes 可聚合;local_hit
                    无物理搬运不计入。旧产物（字段缺失）回退 coverage=0。
  astra-sim-sh_3.0  B4 起（2026-08-26）消费 shards[].noc_hops/noc_path
                    （现状 completion_evictions 落盘；count/fallback 语义
                    同 S1）；无路由的聚合 bytes 字段不计入本账。

覆盖率为 0 的仓输出 hop_bytes=0 并显式标注 coverage，不臆造 hop 数。

口径警示（2026-09-30 审查登记）：``hop_bytes_total`` 是单一总和列，
astra-sim-wscllm 变体内混合两种 hop 单位——decode 行 shards[]/
transfer_hop_bytes 为 mesh 跳（deterministic_xy_route），prefill 行
history 迁移 noc_hops 与 static_route.hop_count 回退为实例图跳（实例
邻接最短路）；granularity 列已登记该混合粒度，但同一总和内跨动作类
（decode 传输 / prefill 迁移 / 恢复）比较或二次加工 hop 分子不可比。
不拆列（下游消费面大、列结构冻结），以本警示 + REPO_HOP_SOURCES
notes + slo_tools/README.md 对应行为权威口径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, NA, detect_repo_variant, emit_json, fail,
    fmt_ratio, iter_jsonl, open_output, run_main, write_csv,
)

TOTAL_COLUMNS = ("repo_variant", "granularity", "actions_with_hops",
                 "hop_bytes_total", "bytes_with_hops", "bytes_without_hops",
                 "coverage_bytes_ratio", "notes")
PER_REQUEST_COLUMNS = ("request_id", "actions_with_hops", "hop_bytes",
                       "bytes_with_hops", "bytes_without_hops")


def _shard_hops(shard: dict) -> Optional[int]:
    """优先显式 noc_hops 字段；缺省由 noc_path 推导（len-1）。"""
    hops = shard.get("noc_hops")
    # bool 显式排除（True 是 int 子类，旧实现当 1 跳计费——与本文件
    # G7-15 处的排除纪律对齐）。
    if isinstance(hops, int) and not isinstance(hops, bool) and hops >= 0:
        return hops
    path = shard.get("noc_path")
    if isinstance(path, list) and path:
        return max(0, len(path) - 1)
    return None


def _accumulate_shard_transfers(holder: dict, request_id: str, acc: dict,
                                per_request: dict) -> None:
    if not isinstance(holder, dict):
        return
    shards = holder.get("shards")
    if not isinstance(shards, list):
        return
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        nbytes = shard.get("bytes")
        # bool 显式排除(True 是 int 子类,旧实现当 1 字节计——与本文件
        # _shard_hops 的 hops 门排除纪律对齐)。
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) \
                or nbytes <= 0:
            continue
        hops = _shard_hops(shard)
        slot = per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})
        if hops is None:
            slot["bytes_without"] += nbytes
            acc["bytes_without_hops"] += nbytes
        else:
            slot["actions"] += 1
            slot["hop_bytes"] += nbytes * hops
            slot["bytes_with"] += nbytes
            acc["actions_with_hops"] += 1
            acc["hop_bytes_total"] += nbytes * hops
            acc["bytes_with_hops"] += nbytes


def _collect_shard_route_record(record: dict, acc: dict,
                                per_request: dict) -> None:
    """shard 级 noc 路由消费（S1/S3 共用实现，逐语句同体；两仓 holder 集
    与 count/fallback 语义完全一致——显式 noc_hops 优先、noc_path 推导
    兜底、无路由 bytes 计 bytes_without_hops）。"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    if kind == "prefill":
        _accumulate_shard_transfers(decision.get("history_transfer"),
                                    request_id, acc, per_request)
        for field in ("history_evictions", "prefill_evictions"):
            for entry in decision.get(field) or []:
                _accumulate_shard_transfers(entry, request_id, acc,
                                            per_request)
    elif kind == "decode":
        _accumulate_shard_transfers(decision.get("prefill_decode_transfer"),
                                    request_id, acc, per_request)
        for entry in decision.get("decode_evictions") or []:
            _accumulate_shard_transfers(entry, request_id, acc, per_request)
    elif kind == "completion":
        for entry in decision.get("completion_evictions") or []:
            _accumulate_shard_transfers(entry, request_id, acc, per_request)


def collect_sh10(record: dict, acc: dict, per_request: dict) -> None:
    """S1 shard 级消费（_collect_shard_route_record 薄包装，公开 API）。"""
    _collect_shard_route_record(record, acc, per_request)


def collect_face(record: dict, acc: dict, per_request: dict) -> None:
    """face 变体（2026-08-27 B3 补齐）：B2wp9py 起决策日志附带只读
    noc 观测字段——prefill 决策 history_noc_hops、decode 决策
    kv_noc_hops（均为 per-TP-shard hops 列表，与 shards 一一对齐，
    无物理传输时空列表）。基线事实核对（60s 窗）：decode 1244 条
    shards/hops 长度零错位、每条 hops 全等；prefill 1079 条 hops 全等。
    聚合口径：
      decode  = Σ shards[i].bytes × hops[i]（逐 shard 精确；shards/hops
      长度不齐或字段缺席 → total_bytes/Σshard bytes 计 bytes_without_
      hops 兜底，不整条丢弃）；
      prefill = history_transfer_bytes × hops[0]（per-shard hops 全等
      ⇒ 与逐 shard bytes×hops 求和严格相等，不依赖 shard bytes 分布）。
    旧产物无字段 → bytes_without_hops（向后兼容，coverage 如实为低）。
    """
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")

    def _slot():
        return per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})

    if kind == "prefill":
        nbytes = decision.get("history_transfer_bytes")
        hops = decision.get("history_noc_hops")
        if (not isinstance(nbytes, bool)
                and isinstance(nbytes, int) and nbytes > 0):
            if isinstance(hops, list) and hops and all(
                    isinstance(h, int) and not isinstance(h, bool)
                    and h >= 0 for h in hops):
                if any(h != hops[0] for h in hops):
                    fail(
                        f"hopbytes: prefill history_noc_hops 非全等"
                        f"（request={request_id!r}, hops={hops}）——聚合"
                        f"口径 bytes×hops[0] 与逐 shard bytes×hops 求和"
                        f"严格相等的前提破坏（基线核对：per-shard hops"
                        f" 全等），拒绝臆造聚合值")
                hop_bytes = nbytes * hops[0]
                slot = _slot()
                slot["actions"] += 1
                slot["hop_bytes"] += hop_bytes
                slot["bytes_with"] += nbytes
                acc["actions_with_hops"] += 1
                acc["hop_bytes_total"] += hop_bytes
                acc["bytes_with_hops"] += nbytes
            else:
                slot = _slot()
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes
    elif kind == "decode":
        transfer = decision.get("prefill_decode_transfer")
        hops = decision.get("kv_noc_hops")
        shards = (transfer or {}).get("shards") if isinstance(
            transfer, dict) else None
        if isinstance(shards, list) and isinstance(hops, list) \
                and len(shards) == len(hops):
            for shard, hop in zip(shards, hops):
                if not isinstance(shard, dict):
                    # 非 dict 真值成员跳过（与 _accumulate_shard_transfers
                    # 同纪律），不裸 AttributeError。
                    continue
                nbytes = shard.get("bytes")
                if (isinstance(nbytes, bool)
                        or not isinstance(nbytes, int)
                        or nbytes <= 0):
                    continue
                if isinstance(hop, int) and hop >= 0:
                    slot = _slot()
                    slot["actions"] += 1
                    slot["hop_bytes"] += nbytes * hop
                    slot["bytes_with"] += nbytes
                    acc["actions_with_hops"] += 1
                    acc["hop_bytes_total"] += nbytes * hop
                    acc["bytes_with_hops"] += nbytes
                else:
                    slot = _slot()
                    slot["bytes_without"] += nbytes
                    acc["bytes_without_hops"] += nbytes
        else:
            # 兜底（修 2026-09-24）：shards/hops 长度不齐、hops 缺席或
            # shards 缺席时旧实现整条丢弃——该 decode 迁移字节从覆盖
            # 分母消失。按无路由口径计入 bytes_without_hops（宁缺勿造，
            # 与本分支 prefill 侧的旧产物向后兼容语义一致）；字节取
            # total_bytes，缺席时回退 Σ shards[].bytes。
            nbytes = None
            if isinstance(transfer, dict):
                # local_hit 守卫（R3-1，2026-09-30）：名义整份 KV 非物理
                # 搬运，与 collect_wscllm decode 回退臂同口径。
                if transfer.get("kind") != "local_hit":
                    total = transfer.get("total_bytes")
                    if isinstance(total, int) and total > 0:
                        nbytes = total
            if nbytes is None and isinstance(shards, list):
                shard_total = 0
                for shard in shards:
                    if isinstance(shard, dict) and isinstance(
                            shard.get("bytes"), int) and shard["bytes"] > 0:
                        shard_total += shard["bytes"]
                if shard_total > 0:
                    nbytes = shard_total
            if (not isinstance(nbytes, bool)
                    and isinstance(nbytes, int) and nbytes > 0):
                slot = _slot()
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes


# （2026-10-01 合并）_route_hops 与 _shard_hops 此前是 AST 级逐字重复
# 的两份拷贝——现为单一实现 + 别名。
_route_hops = _shard_hops


def _consume_wscllm_eviction_rows(rows, request_id: str, acc: dict,
                                  per_request: dict) -> None:
    """wscllm 逐出/回迁契约行（_kv_transfer_rows 形状：kind/reason/
    total_bytes/...，无 shard 级路由）→ 覆盖分母 bytes_without_hops
    （宁缺勿造，不臆造 hop 数）；total_bytes 非正 = 无物理搬移不计。"""
    for entry in rows or ():
        if not isinstance(entry, dict):
            continue
        nbytes = entry.get("total_bytes")
        # bool 显式排除(True 是 int 子类,旧实现当 1 字节计——同上排 bool
        # 纪律)。
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) \
                or nbytes <= 0:
            continue
        slot = per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})
        slot["bytes_without"] += nbytes
        acc["bytes_without_hops"] += nbytes


# F1 同款跨行去重态（R5-batch3，2026-10-01；机制照 kv_cache_adapter 2026-09-30
# 的 F1 移植）：B2 窗口产物的同一笔 #2 预占逐出在 prefill 行（准入快照）与
# decode 行（全集）各序列化一次，legacy 回退臂两臂同读会双计入
# bytes_without_hops。去重键 = (session_id, reason, total_bytes)（与
# kv_cache_adapter._legacy_eviction_key 同键同语义：_kv_transfer_rows 行
# 结构不携带 time_ns，同一 KVTransfer 的两次序列化三元组恒等；同一会话在
# 一个请求持有期内不可能以同 reason 同字节被逐出两次）。去重态随 acc 生命
# 周期（hopbytes_prepare 每 run 新建；CLI/driver/直连调用共用同一 acc 即
# 共用同一去重态），以 "_" 前缀私有键承载、hopbytes_emit 的 summary 过滤
# 私有键——不新增 collector 形参，契约路径（分段字段/decode_evictions
# 在场）不触碰该态，现行产物字节不变。
LEGACY_DEDUP_ACC_KEY = "_wscllm_legacy_eviction_keys"


def _legacy_eviction_key(entry: dict) -> tuple:
    """legacy 逐出条目的跨行去重键（kv_cache_adapter._legacy_eviction_key
    同款；调用方保证 entry 为 dict）。"""
    return (str(entry.get("session_id")), str(entry.get("reason")),
            entry.get("total_bytes"))


def _consume_wscllm_evictions(decision: dict, kind, request_id: str,
                              acc: dict, per_request: dict) -> None:
    """逐出传输字段消费（M29，2026-09-24）：此前 collect_wscllm 整类
    不消费，逐出/回迁 NoC 字节既不进 hop_bytes 也不进
    bytes_without_hops，coverage 分母缺整类。

    新旧产物单键取用（同 kv_cache_adapter/hbm_watermark 的防双计纪律
    ——admission_evictions 是 history_evictions+prefill_evictions 的
    union；decode_evictions 与 decode_target_evictions 同内容；B3 起
    union 与分段字段双序列化，同批消费会双计）：
      契约分段字段在场（B3+）：prefill 行读 history_evictions+
      prefill_evictions，decode 行读 decode_evictions，completion 行读
      completion_evictions；
      否则（旧产物）：prefill 行读 admission_evictions+decode_target_
      evictions（准入快照，decode 行不另读），completion 行读
      completion_evictions。

    F1 跨行去重（2026-10-01，见 LEGACY_DEDUP_ACC_KEY）：B2 窗口产物的
    同一笔 #2 预占逐出在 prefill 行（准入快照）与 decode 行（全集）各
    序列化一次——prefill legacy 臂消费后把去重键记入 acc 私有去重态，
    decode legacy 臂跳过已记键（与 kv_cache_adapter.extract_events_
    face_wscllm 的 F1 同键同两侧；契约路径不触碰去重态）。
    """
    legacy_seen = acc.setdefault(LEGACY_DEDUP_ACC_KEY, {})
    if kind == "prefill":
        if ("history_evictions" in decision
                or "prefill_evictions" in decision):
            for field in ("history_evictions", "prefill_evictions"):
                _consume_wscllm_eviction_rows(decision.get(field),
                                              request_id, acc, per_request)
        else:
            # F1 登记侧：本行 legacy 逐出条目（含无 bytes/零 bytes 的
            # 条目，与 kv_cache_adapter 登记口径一致）按三元组记键。
            seen = legacy_seen.setdefault(str(request_id), set())
            for field in ("admission_evictions", "decode_target_evictions"):
                rows = decision.get(field) or []
                for entry in rows:
                    if isinstance(entry, dict):
                        seen.add(_legacy_eviction_key(entry))
                _consume_wscllm_eviction_rows(rows, request_id, acc,
                                              per_request)
    elif kind == "decode":
        # 单键取用（与 prefill 行同式）：契约字段 decode_evictions 在场
        # （B3+）优先；缺席（B2 窗口/旧产物）回退 legacy decode_target_
        # evictions——decode 侧逐出（#2 预占/#4 move/#5 grow_decode 累积）
        # 自 B2 起随 decode 行落盘，不回退则 B2 move/grow 逐出字节两桶
        # 皆不入。F1 去重：跳过 prefill 行准入快照已消费的 #2 条目（同
        # 键=同一次物理逐出的双序列化；非 dict 成员本就不计数，不过滤）。
        if "decode_evictions" in decision:
            _consume_wscllm_eviction_rows(decision.get("decode_evictions"),
                                          request_id, acc, per_request)
        else:
            seen = legacy_seen.get(str(request_id))
            rows = decision.get("decode_target_evictions") or []
            if seen:
                rows = [entry for entry in rows
                        if not isinstance(entry, dict)
                        or _legacy_eviction_key(entry) not in seen]
            _consume_wscllm_eviction_rows(rows, request_id, acc, per_request)
    elif kind == "completion":
        _consume_wscllm_eviction_rows(decision.get("completion_evictions"),
                                      request_id, acc, per_request)


def collect_wscllm(record: dict, acc: dict, per_request: dict) -> None:
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    _consume_wscllm_evictions(decision, kind, request_id, acc, per_request)
    if kind != "decode":
        # 历史迁移（prefill 决策）：聚合口径 history_transfer_bytes×
        # noc_hops（实例图最短路，2026-08-26 B2wp9py 起 prefill 决策对
        # history NOC_MIGRATE 迁移附加 noc_hops 输出字段）；字段缺席
        # （REMOTE 恢复——远端池无实例语义，或旧产物）按
        # bytes_without_hops 处理（向后兼容）。曾有 prefill 行
        # history_transfer_shards[] 的 shard 级分支（2026-09-05 问题 4b
        # 规划）：写侧从未序列化该字段（生成侧为 history_transfers
        # 行 + transfer_hop_bytes），对本仓产物恒死，已删除
        # （2026-09-24）。
        if kind == "prefill":
            # local_hit 守卫（2026-10-01 补齐，与 decode 回退臂/face/sh 系
            # 同款 R3-1 口径）：LOCAL_HIT 的 history_transfer_bytes 是名义
            # 整份 KV（信息量，非物理搬运），携带 noc_hops 字段的旧产物
            # 若不排除会把非物理字节推高 bytes_with_hops/coverage 分子。
            if decision.get("history_action") == "LOCAL_HIT":
                nbytes = 0
            else:
                nbytes = decision.get("history_transfer_bytes")
            if (not isinstance(nbytes, bool)
                    and isinstance(nbytes, int) and nbytes > 0):
                hops = decision.get("noc_hops")
                slot = per_request.setdefault(
                    request_id, {"actions": 0, "hop_bytes": 0,
                                 "bytes_with": 0, "bytes_without": 0})
                # bool 显式排除（同 G7-15 行级门）。
                if isinstance(hops, int) and not isinstance(hops, bool) \
                        and hops >= 0:
                    slot["actions"] += 1
                    slot["hop_bytes"] += nbytes * hops
                    slot["bytes_with"] += nbytes
                    acc["actions_with_hops"] += 1
                    acc["hop_bytes_total"] += nbytes * hops
                    acc["bytes_with_hops"] += nbytes
                else:
                    # G7-15（2026-09-26）：顶层 noc_hops 缺席（REMOTE_RESTORE
                    # 远端回迁——B3 起远端池无实例语义、不发射实例间 hop 诊断
                    # 字段）时，旧实现把整笔回迁字节落 bytes_without_hops，
                    # hop 分子缺整类；写侧 transfer_hop_bytes 行逐对象携带
                    # shard 级 hops（max shard noc_path len−1），此处逐对象
                    # 消费补全远程迁移度量。行集合缺席/形状异常/行字节合计
                    # 与标量不符（旧产物）→ 回退旧标量口径，不臆造。
                    # G6 口径披露（2026-09-30 审查）：写侧 hops 是"max
                    # shard hops"——逐分片回迁各 rank 曼哈顿跳数不同时，
                    # total_bytes×max_hops 是 Hop-Bytes 的**上界口径**（严格
                    # 定义为 Σ bytes_i×hops_i，恒高估）；同文件 collect_face
                    # 对非全等 hops 是 fail-closed 拒绝，本处用 max 聚合是
                    # 写侧序列化粒度（单标量/行）所限的已知近似。根治需
                    # 写侧逐 shard 序列化 bytes×hops 对，另立批次。
                    rows = decision.get("transfer_hop_bytes")
                    row_total = 0
                    usable = isinstance(rows, list) and bool(rows)
                    if usable:
                        for row in rows:
                            if not isinstance(row, dict):
                                usable = False
                                break
                            value = row.get("total_bytes")
                            if not isinstance(value, int) or isinstance(
                                    value, bool) or value < 0:
                                usable = False
                                break
                            row_total += value
                        usable = usable and row_total == nbytes
                    if usable:
                        for row in rows:
                            value = row["total_bytes"]
                            if value <= 0:
                                continue  # 零长段：无物理搬运
                            hop = row.get("hops")
                            # 零跳口径统一：hops=0 = 已知零跳路由 = 计入
                            # 覆盖（与 decode 分支/_shard_hops 的 >=0 口径
                            # 一致；零跳只是 hop 分子为 0，不是"无路由"）。
                            if isinstance(hop, int) and not isinstance(
                                    hop, bool) and hop >= 0:
                                slot["actions"] += 1
                                slot["hop_bytes"] += value * hop
                                slot["bytes_with"] += value
                                acc["actions_with_hops"] += 1
                                acc["hop_bytes_total"] += value * hop
                                acc["bytes_with_hops"] += value
                            else:
                                slot["bytes_without"] += value
                                acc["bytes_without_hops"] += value
                    else:
                        slot["bytes_without"] += nbytes
                        acc["bytes_without_hops"] += nbytes
        return
    # decode：问题 4b（2026-09-05）shard 级优先——prefill_decode_transfer
    # 含非空 shards[] 且任一 shard 携带 noc_hops/noc_path 时逐 shard
    # bytes×hops（生成侧 _kv_transfer_dict(hardware=...) 落的路由字段；
    # 无路由 shard 计 bytes_without_hops，不臆造）；否则（旧产物/旧行为，
    # shards 无路由信息或字段缺席）回退聚合口径
    # total_bytes×static_route.hop_count（实例间 hop 粒度）。
    transfer = decision.get("prefill_decode_transfer")
    shards = (transfer or {}).get("shards") if isinstance(
        transfer, dict) else None
    if isinstance(shards, list) and shards and any(
            _route_hops(shard) is not None
            for shard in shards if isinstance(shard, dict)):
        _accumulate_shard_transfers(transfer, request_id, acc, per_request)
        return
    static_route = decision.get("static_route")
    nbytes = (transfer or {}).get("total_bytes") if isinstance(
        transfer, dict) else None
    # local_hit 守卫（R2-3/R3-1，2026-09-30）：名义整份 KV 非物理搬运
    # （写侧 shards=()，本文件头注 sh_2.0 行同口径"local_hit 无物理
    # 搬运不计入"）——聚合回退臂漏守卫会把 stay 边界的名义字节计入
    # hop 账；collect_face/collect_aggregate_only 的同型臂已同口径补齐。
    if isinstance(transfer, dict) and transfer.get("kind") == "local_hit":
        nbytes = None
    hops = (static_route or {}).get("hop_count") if isinstance(
        static_route, dict) else None
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
        return
    slot = per_request.setdefault(
        request_id, {"actions": 0, "hop_bytes": 0,
                     "bytes_with": 0, "bytes_without": 0})
    # bool 显式排除（static_route.hop_count 谎报 true=1 跳）。
    if isinstance(hops, int) and not isinstance(hops, bool) and hops >= 0:
        slot["actions"] += 1
        slot["hop_bytes"] += nbytes * hops
        slot["bytes_with"] += nbytes
        acc["actions_with_hops"] += 1
        acc["hop_bytes_total"] += nbytes * hops
        acc["bytes_with_hops"] += nbytes
    else:
        slot["bytes_without"] += nbytes
        acc["bytes_without_hops"] += nbytes


def collect_aggregate_only(record: dict, acc: dict, per_request: dict) -> None:
    """S2 旧产物（transfer_hop_bytes 缺席，collect_sh20 回退）：
    bytes 全部计入无覆盖侧。"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    nbytes = None
    if kind == "prefill":
        nbytes = decision.get("history_transfer_bytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int):
            transfer = decision.get("history_transfer")
            if isinstance(transfer, dict):
                total = transfer.get("total_bytes")
                if isinstance(total, int):
                    nbytes = total
    elif kind == "decode":
        nbytes = decision.get("prefill_decode_transfer_bytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int):
            transfer = decision.get("prefill_decode_transfer")
            if isinstance(transfer, dict):
                # local_hit 守卫（R3-1，2026-09-30）：同上，跨变体补齐。
                if transfer.get("kind") == "local_hit":
                    nbytes = None
                else:
                    total = transfer.get("total_bytes")
                    if isinstance(total, int):
                        nbytes = total
    if isinstance(nbytes, int) and nbytes > 0:
        slot = per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})
        slot["bytes_without"] += nbytes
        acc["bytes_without_hops"] += nbytes


def collect_sh20(record: dict, acc: dict, per_request: dict) -> None:
    """S2：WP9-线5（2026-08-26）起 decision 附加 transfer_hop_bytes
    （逐传输对象 {total_bytes, noc_hop_bytes, shard_count}；shards 按
    deterministic_xy_route 生成，在线路径复用）。shard_count>0 的传输
    计入覆盖侧（hop_bytes 用决策时点序列化的 noc_hop_bytes，不再回读
    shards）；local_hit（shard_count=0，无物理搬运）不计入任何桶。
    旧产物（字段缺失）回退 collect_aggregate_only 语义（bytes 全部
    计入无覆盖侧），新旧产物可区分。"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    relevant_kinds = ("prefill", "decode", "completion")
    hop_rows = decision.get("transfer_hop_bytes")
    if kind in relevant_kinds and isinstance(hop_rows, list):
        for row in hop_rows:
            if not isinstance(row, dict):
                continue
            nbytes = row.get("total_bytes")
            hop_bytes = row.get("noc_hop_bytes")
            shard_count = row.get("shard_count")
            if (not isinstance(nbytes, int) or isinstance(nbytes, bool)
                    or nbytes <= 0
                    or not isinstance(shard_count, int)
                    or isinstance(shard_count, bool) or shard_count <= 0):
                continue  # local_hit / 名义规模：无物理搬运
            slot = per_request.setdefault(
                request_id, {"actions": 0, "hop_bytes": 0,
                             "bytes_with": 0, "bytes_without": 0})
            # M-36：与 collect_wscllm 的 G7-15 口径对齐——shard_count>0 而
            # noc_hop_bytes 缺席/类型谎报（str/float/bool）的行不再计入
            # 覆盖侧（hop 分子无贡献却进 bytes_with_hops 分母 → coverage
            # 虚增），落 bytes_without_hops（宁缺勿造，不臆造 hop 数）；
            # 显式 0 = 已知零跳路由，仍计入覆盖侧（与本文件零跳口径
            # 一致：零跳只是 hop 分子为 0，不是"无路由"）。
            if isinstance(hop_bytes, int) \
                    and not isinstance(hop_bytes, bool) and hop_bytes >= 0:
                slot["actions"] += 1
                slot["bytes_with"] += nbytes
                acc["actions_with_hops"] += 1
                acc["bytes_with_hops"] += nbytes
                slot["hop_bytes"] += hop_bytes
                acc["hop_bytes_total"] += hop_bytes
            else:
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes
        return
    collect_aggregate_only(record, acc, per_request)


def collect_sh30(record: dict, acc: dict, per_request: dict) -> None:
    """S3（B4 升级，2026-08-26）：消费 decision log 的 shards[].noc_hops/
    noc_path 新字段（B3 60s 证据：completion_evictions 96 行 / 960 shards
    全带 noc_hops；ad-hoc 复核 hop_bytes=2383397232640）。

    count/fallback 语义与 S1 分支一致（_accumulate_shard_transfers）；
    holder 集照 S1（history_transfer / prefill_decode_transfer / 各类
    逐出列表）——函数体与 collect_sh10 逐语句相同（2026-09-29 合并为
    _collect_shard_route_record 单一实现，本名保留为薄包装，公开 API
    不变）。S3 现仅 completion_evictions 落 shards，其余 holder 留作
    字段同步后的前向兼容。B3 之前的 aggregate-only 停用口径
    （history_transfer_bytes / prefill_decode_transfer_bytes 无路由可聚）
    不再计入本账——那些记录无 noc 路由信息，不属于 Hop-Bytes 可判定
    宇宙；其量级（60s 参考跑 1045595684864 B）由门 JSON 另行登记。
    """
    _collect_shard_route_record(record, acc, per_request)


REPO_HOP_SOURCES: dict[str, dict] = {
    "astra-sim-sh_1.0": {
        "collector": collect_sh10,
        "granularity": "shard(noc_path)",
        "notes": "shards[].noc_path/noc_hops 已落盘 decision log",
    },
    "astra-sim-wscllm": {
        "collector": collect_wscllm,
        "granularity": "shard(noc_path:decode)"
                       "+instance-aggregate(prefill noc_hops/"
                       "static_route.hop_count)+evictions(no-route)",
        "notes": "decode 行 prefill_decode_transfer.shards[] 逐 shard "
                 "bytes×noc_hops（2026-09-05 问题 4b，mesh 跳口径；"
                 "count/fallback 语义同 S1），旧产物回退聚合口径 "
                 "（total_bytes×static_route.hop_count，实例级粒度）；"
                 "prefill 行 history_transfer_bytes×noc_hops（NOC_"
                 "MIGRATE 实例图最短路），REMOTE 恢复无 noc_hops 计 "
                 "bytes_without_hops。2026-09-24（M29）起逐出/回迁契约行"
                 "（history_evictions/prefill_evictions/decode_evictions/"
                 "completion_evictions；旧产物 admission_evictions+"
                 "decode_target_evictions，单键取用防 B3 双序列化双计）"
                 "计入 bytes_without_hops——契约行无 shard 级路由，"
                 "coverage 如实低于 1（不臆造 hop 数；此前逐出字节整类"
                 "不进账）。2026-09-26（G7-15）起顶层 noc_hops 缺席的 "
                 "prefill 行（REMOTE_RESTORE 远端回迁）逐对象消费 "
                 "transfer_hop_bytes 行的 shard 级 hops 补全 hop 分子"
                 "（行字节合计与标量相符才启用，旧产物回退标量口径）。"
                 "2026-09-05 规划的 prefill 行 history_transfer_"
                 "shards[] shard 级分支因写侧零生产（生成侧为 history_"
                 "transfers 行 + transfer_hop_bytes）删除。"
                 "口径警示：hop_bytes_total 混合 mesh 跳（decode shard 级）"
                 "与实例图跳（prefill 迁移 noc_hops/static_route 回退）"
                 "两种单位，跨动作类比较不可比（模块 docstring 口径警示）。",
    },
    "astra-sim-face": {
        "collector": collect_face,
        "granularity": "per-TP-shard(history_noc_hops/kv_noc_hops)",
        "notes": "B2wp9py 起决策日志附带只读 noc hops 列表"
                 "（prefill=history_noc_hops，decode=kv_noc_hops，"
                 "与 shards 一一对齐）；旧产物无字段 → "
                 "bytes_without_hops（coverage 如实降低，不臆造）",
    },
    "astra-sim-sh_2.0": {
        "collector": collect_sh20,
        "granularity": "shard-route(noc_hop_bytes)",
        "notes": "WP9-线5（2026-08-26）起 decision 逐传输对象序列化 "
                 "noc_hop_bytes（shards 按 deterministic_xy_route 生成，"
                 "决策时点即知）；local_hit 无物理搬运不计入；旧产物"
                 "（字段缺失）回退 coverage=0 口径",
    },
    "astra-sim-sh_3.0": {
        "collector": collect_sh30,
        "granularity": "shard(noc_path)",
        "notes": "B4 起消费 shards[].noc_hops/noc_path（现状仅 "
                 "completion_evictions 落盘；其余 holder 前向兼容）；"
                 "无路由的聚合 bytes 字段不计入本账（count/fallback "
                 "语义同 S1）",
    },
}


def hopbytes_prepare(repo_variant: str) -> tuple[dict, dict, dict]:
    """A4 driver 用：source 登记 + 空累积态。"""
    source = REPO_HOP_SOURCES.get(repo_variant)
    if source is None:
        fail(f"未登记的 repo_variant：{repo_variant}（REPO_HOP_SOURCES 需扩表）")
    # hop_bytes_total 混合单位警示：wscllm 变体下该列同时累计 mesh 跳
    # （decode shard 级）与实例图跳（prefill 迁移/static_route 回退）
    # 两种分子，跨动作类不可比（见模块 docstring 口径警示）。
    acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
           "bytes_with_hops": 0, "bytes_without_hops": 0,
           # F1 跨行去重态（私有键，emit 过滤；仅 wscllm legacy 回退臂
           # 读写）：request_id → {(session_id, reason, total_bytes), ...}。
           LEGACY_DEDUP_ACC_KEY: {}}
    return source, acc, {}


def cmd_hopbytes(args: argparse.Namespace) -> int:
    # CLI 入口（独立运行行为不变）。A4 driver 经 hopbytes_prepare /
    # collector / hopbytes_emit 组合复用同一逻辑（单遍 decision log）。
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    source, acc, per_request = hopbytes_prepare(repo_variant)
    for record in iter_jsonl(args.run_dir / DECISION_LOG_RELPATH):
        source["collector"](record, acc, per_request)
    return hopbytes_emit(args, repo_variant, source, acc, per_request)


def hopbytes_emit(args: argparse.Namespace, repo_variant: str, source: dict,
                  acc: dict, per_request: dict[str, dict]) -> int:
    total_bytes = acc["bytes_with_hops"] + acc["bytes_without_hops"]
    coverage = (acc["bytes_with_hops"] / total_bytes) if total_bytes else 0.0
    total_row = (repo_variant, source["granularity"],
                 acc["actions_with_hops"], acc["hop_bytes_total"],
                 acc["bytes_with_hops"], acc["bytes_without_hops"],
                 fmt_ratio(coverage), source["notes"])
    stream, close = open_output(args.output, "slo_hopbytes_total.csv",
                                args.run_dir)
    try:
        write_csv(stream, TOTAL_COLUMNS, [total_row])
    finally:
        if close:
            stream.close()
    pr_stream, pr_close = open_output(args.per_request,
                                      "slo_hopbytes_per_request.csv",
                                      args.run_dir)
    try:
        write_csv(
            pr_stream, PER_REQUEST_COLUMNS,
            ((rid, slot["actions"], slot["hop_bytes"], slot["bytes_with"],
              slot["bytes_without"])
             for rid, slot in sorted(per_request.items())))
    finally:
        if pr_close:
            pr_stream.close()
    summary = {
        "command": "hopbytes",
        "repo_variant": repo_variant,
        "formula": "Hop-Bytes = sum(bytes * noc_hops) over transfer "
                   "records carrying hop fields (spec 1.5-A)",
        # 私有键（LEGACY_DEDUP_ACC_KEY 去重态）不进 summary——acc 的
        # "_" 前缀键是运行态，不是指标。
        **{key: value for key, value in acc.items()
           if not str(key).startswith("_")},
        "coverage_bytes_ratio": coverage,
        "notes": source["notes"],
        "granularity": source["granularity"],
    }
    emit_json(sys.stderr, summary)
    if coverage == 0.0:
        print(f"[hopbytes] 警告：{repo_variant} 产物无可用的 noc hops 字段"
              f"（coverage=0）——Hop-Bytes=0 仅表示无覆盖，见 notes",
              file=sys.stderr)
    if args.json:
        jstream, jclose = open_output(args.json, "slo_hopbytes.json",
                                      args.run_dir)
        try:
            emit_json(jstream, summary)
        finally:
            if jclose:
                jstream.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hopbytes.py",
        description="Hop-Bytes = Σ bytes×noc_hops（分请求/总量两档；"
                    "仓内产物无 hops 字段时 coverage=0 并显式标注）")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 results/online_decision_log.jsonl）")
    parser.add_argument("-o", "--output", default="",
                        help="总量 CSV 输出（'-'=stdout；缺省写 "
                             "run_dir/slo_hopbytes_total.csv）")
    parser.add_argument("--per-request", default="",
                        help="分请求 CSV 输出（同上；缺省写 "
                             "run_dir/slo_hopbytes_per_request.csv）")
    parser.add_argument("--json", default="",
                        help="汇总 JSON 输出路径（可选）")
    parser.add_argument("--repo-variant", default=None,
                        help="显式指定 repo_variant（默认读 cpp.log init 行）")
    args = parser.parse_args()
    return int(cmd_hopbytes(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
