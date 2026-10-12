#!/usr/bin/env python3
"""Hop-Bytes = Σ_actions bytes × noc_hops（主规格 §1.5-A；纯后处理）。

从运行产物中"含 noc hops 的传输记录"聚合，分请求/总量两档输出。各仓
真实字段名已在基线产物中核对并登记于 REPO_HOP_SOURCES（先定位字段、
逐仓登记，不猜测）：

  astra-sim-sh_1.0  shard 级：decision log 的 history_transfer /
                    prefill_decode_transfer / 各类逐出传输对象的
                    shards[].noc_path（hops = len(noc_path)-1；生成侧
                    generate_face_trace.py 的 shard 路由/传输发射亦落
                    noc_hops 字段——读取时优先用显式 noc_hops，缺省由
                    noc_path 推导）。
  astra-sim-wscllm  实例级：decode 决策的 static_route.hop_count ×
                    prefill_decode_transfer.total_bytes（粒度为实例间
                    hop，非 rank 级）。2026-08-26 B2wp9py 起 prefill 决策
                    对 history NOC_MIGRATE 迁移附加 noc_hops 字段
                    （实例图最短路，同粒度）→ history 迁移纳入覆盖
                    （60s 基线覆盖 70.4% → 满覆盖；字段为 A/B 对拍
                    剥离清单条目）。
  astra-sim-face    per-TP-shard 级（B2wp9py 起决策日志附带只读
                    hops 列表：prefill=history_noc_hops、decode=
                    kv_noc_hops，与 shards 一一对齐；hops 全等时
                    聚合 bytes×hops[0] 与逐 shard 求和严格相等）。
                    旧产物无字段 → bytes_without_hops。
  astra-sim-face-LRU
                    本仓：同 face（产物恒附带只读 hops 列表；collect_face
                    保留的 legacy dict 兜底分支仅为 face 旧产物路径——
                    R1 py-slo-C1，2026-09-26 注释如实化）。
                    B3 契约逐出行（history/prefill/decode_evictions、
                    completion_eviction_transfers）与 decode PD 迁移的
                    hops 权威源 = 同决策的 transfer_hop_bytes 行
                    （noc_hop_bytes；py-slo-E1/C1，2026-09-30）；行缺失/
                    不可解析的旧产物回退 bytes_without_hops（分母完备）。
                    非法 shard（非 dict/错型 bytes/负值）计数留痕，
                    可估字节走 dropped 通道入分母（R7-F5）。
  astra-sim-sh_2.0  WP9-线5（2026-08-26）起 decision 逐传输对象序列化
                    transfer_hop_bytes[].{total_bytes, noc_hop_bytes,
                    shard_count}（KVTransfer shards 按 deterministic_xy_
                    route 生成、在线复用）→ hop_bytes 可聚合;local_hit
                    无物理搬运不计入。旧产物（字段缺失）回退 coverage=0。
  astra-sim-sh_3.0  B4 起（2026-08-26）消费 shards[].noc_hops/noc_path
                    （现状 completion_evictions 落盘；count/fallback 语义
                    同 S1）；B3 前 aggregate-only 标量字段（无路由可聚）
                    不计入本账，无 shards 键 holder 的 total_bytes 计
                    bytes_without_hops（2026-09-26 E5 兜底，同 S1）。

覆盖率为 0 的仓输出 hop_bytes=0 并显式标注 coverage，不臆造 hop 数。
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
    # (2026-10-01, ported from the face repo) bool 经 isinstance int 混入
    # 会按 1 hop 计账——显式拒绝。
    if isinstance(hops, int) and not isinstance(hops, bool) and hops >= 0:
        return hops
    path = shard.get("noc_path")
    if isinstance(path, list) and path:
        return max(0, len(path) - 1)
    return None


def _drop_invalid_shard_bytes(nbytes) -> int:
    """(2026-09-30, ported from the face repo) R7-F5：非法 shard bytes 的
    可估字节数（积分值 float）；不可估返 0（不臆造字节数，由
    dropped_shards 计数留痕）。"""
    if isinstance(nbytes, float) and nbytes.is_integer() and nbytes > 0:
        return int(nbytes)
    return 0


def _accumulate_shard_transfers(holder: dict, request_id: str, acc: dict,
                                per_request: dict) -> None:
    if not isinstance(holder, dict):
        return
    shards = holder.get("shards")
    if not isinstance(shards, list):
        # (2026-09-30, ported from the face repo) 2026-09-26 A5-3 处置：
        # 缺 shards 的 holder 不再静默丢弃——holder 本体可估的
        # total_bytes 计入 dropped_bytes（coverage 分母含之，见
        # hopbytes_emit），holder 计数 dropped_holders；否则字节从分母
        # 消失会抬高 coverage。（本仓 2026-09-26 E5 曾把这些字节计入
        # bytes_without_hops；A5-3 与其同守恒目标，改走 dropped 通道后
        # 无覆盖侧不再混入结构损坏字节，summary 的 dropped_* 计数可诊断。）
        acc["dropped_holders"] = acc.get("dropped_holders", 0) + 1
        nbytes = holder.get("total_bytes")
        if isinstance(nbytes, int) and nbytes > 0:
            acc["dropped_bytes"] = acc.get("dropped_bytes", 0) + nbytes
        return
    counted_bytes = 0  # 实际计入分母桶（覆盖侧/无覆盖侧）的 shard 字节
    dropped_shards = 0  # 非法 bytes/结构的 shard 计数（R7-F5 留痕）
    dropped_estimated = 0  # 非法 shard 中可估的字节（float 积分值）
    for shard in shards:
        if not isinstance(shard, dict):
            # (2026-09-30, ported from the face repo) R7-F5：非 dict shard
            # 不再静默跳过——计数留痕。
            dropped_shards += 1
            continue
        nbytes = shard.get("bytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int):
            # (2026-09-30, ported from the face repo) R7-F5：混合 holder 内
            # 单个非法 bytes 不再静默蒸发——可估值暂记，混合场景计入
            # dropped_bytes（分母守恒）；全无效场景由 holder 级
            # total_bytes 代表（同源不双计，B4① 口径）。
            dropped_shards += 1
            dropped_estimated += _drop_invalid_shard_bytes(nbytes)
            continue
        if nbytes < 0:
            # (2026-09-30, ported from the face repo) 负字节同为账本垃圾：
            # 计数留痕，不臆造守恒。
            dropped_shards += 1
            continue
        if nbytes == 0:
            continue  # 零字节 shard 无可守恒字节（合法，不计数）
        hops = _shard_hops(shard)
        slot = per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})
        counted_bytes += nbytes
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
    if dropped_shards:
        acc["dropped_shards"] = acc.get("dropped_shards", 0) + dropped_shards
    if counted_bytes and dropped_estimated:
        # (2026-09-30, ported from the face repo) 混合 holder：holder 级
        # 回退不触发（避免与有效 shard 双计），非法 shard 的可估字节在
        # 此入账留痕。
        acc["dropped_bytes"] = acc.get("dropped_bytes", 0) + dropped_estimated
    if counted_bytes == 0:
        # (2026-09-30, ported from the face repo) 2026-09-28 B4① 处置
        # （A5-3 同型）：shards 为空列表或全无效时 holder 不再静默消失
        # ——holder 计入 dropped_holders，可估的 total_bytes 计
        # dropped_bytes（coverage 分母含之），否则字节从分母消失会抬高
        # coverage。
        acc["dropped_holders"] = acc.get("dropped_holders", 0) + 1
        nbytes = holder.get("total_bytes")
        if isinstance(nbytes, int) and nbytes > 0:
            acc["dropped_bytes"] = acc.get("dropped_bytes", 0) + nbytes


def collect_sh10(record: dict, acc: dict, per_request: dict) -> None:
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


def collect_face(record: dict, acc: dict, per_request: dict) -> None:
    """face 变体（2026-08-27 B3 补齐）：B2wp9py 起决策日志附带只读
    noc 观测字段——prefill 决策 history_noc_hops、decode 决策
    kv_noc_hops（均为 per-TP-shard hops 列表，与 shards 一一对齐，
    无物理传输时空列表）。基线事实核对（60s 窗）：decode 1244 条
    shards/hops 长度零错位、每条 hops 全等；prefill 1079 条 hops 全等。
    聚合口径：
      decode  = transfer_hop_bytes 行为 PD 迁移的 hops 权威源
      （noc_hop_bytes = 逐 shard bytes×hops 的精确和；py-slo-E1/C1，
      2026-09-30——此前 hops[0]×Σbytes 依赖 M18 全等前提）；旧 dict
      产物仍按 shards[i].bytes × hops[i] 逐 shard 精确。
      prefill = history_transfer_bytes × hops[0]（per-shard hops 全等
      ⇒ 与逐 shard bytes×hops 求和严格相等，不依赖 shard bytes 分布）；
      恢复族（REMOTE_RESTORE remote_load，hops 恒空）改消费 transfer_
      hop_bytes 行的精确 noc_hop_bytes（T3，2026-09-29——restore 物理
      上仍经池边缘端口→目标 NPU 的 NoC 路由，此前整体漏出覆盖分子；
      行源 = history_transfers，与逐出账本无交集，不双计）。
      逐出    = transfer_hop_bytes 行为逐出传输的 hops 权威源
      （py-slo-E1/C1，2026-09-30：生产者 face_online_scheduler 的
      _transfer_hop_rows/_eviction_hop_rows 行恰好覆盖 history/
      prefill/decode_evictions 与 completion_eviction_transfers 的
      全部传输对象，含逐传输 noc_hop_bytes——此前整类计入 bytes_
      without_hops 使 coverage 系统性偏低。按 kind 区分行源：
      remote_load 行属 T3 恢复域、local_hit 无物理搬移不计入；
      防双计：行级 total_bytes 不再同时经 _eviction_rows_without_
      hops 计入 bytes_without。行缺失/不可解析的旧产物回退
      bytes_without_hops（向后兼容，coverage 如实为低）。
    旧产物无字段 → bytes_without_hops（向后兼容，coverage 如实为低）。
    """
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")

    def _slot():
        return per_request.setdefault(
            request_id, {"actions": 0, "hop_bytes": 0,
                         "bytes_with": 0, "bytes_without": 0})

    def _eviction_rows_without_hops(rows) -> None:
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            if row.get("kind") in ("local_hit", "LOCAL_HIT"):
                continue
            nbytes = row.get("total_bytes")
            if not isinstance(nbytes, int) or nbytes <= 0:
                continue
            slot = _slot()
            slot["bytes_without"] += nbytes
            acc["bytes_without_hops"] += nbytes

    def _consume_transfer_hop_rows(rows, restore_skip_kinds) -> None:
        """(2026-09-30) py-slo-E1/C1：逐出/PD 传输经 transfer_hop_bytes
        行消费（与 T3 remote_load 消费同法）。按 kind 区分行源：
        restore_skip_kinds 由调用分支给出——prefill 分支传
        ("remote_load", "noc_migrate")：两者同属恢复域，恰由该分支的
        标量路径（history_transfer_bytes × history_noc_hops，M18 全等
        门）覆盖，行消费须跳过防双计（第二轮审计 XC-1：noc_migrate
        漏列曾导致跨实例历史迁移翻倍计入）；decode 分支传 ()：
        prefill_decode_transfer 的 noc_migrate 迁移腿只有行消费这一条
        计数路径（该分支行消费后即返回，无标量路径），不得跳过。
        local_hit 无物理搬移不计入（两分支共通）。行可解析
        （total_bytes/noc_hop_bytes/shard_count 型正确）时以
        noc_hop_bytes 为 hops 权威计入覆盖侧；行缺失 hop 字段或型
        不对时 total_bytes 保守落 bytes_without（不臆造 hop 数）。"""
        skip = set(restore_skip_kinds)
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            if row.get("kind") in skip:
                continue
            if row.get("kind") in ("local_hit", "LOCAL_HIT"):
                continue
            nbytes = row.get("total_bytes")
            if not isinstance(nbytes, int) or nbytes <= 0:
                continue  # 名义规模/无搬运（同 sh20 行口径）
            hop_bytes = row.get("noc_hop_bytes")
            shard_count = row.get("shard_count")
            slot = _slot()
            # (2026-10-01, ported from the face repo) bool 经 isinstance
            # int 混入会按 1 hop 计账——显式拒绝。
            if (isinstance(hop_bytes, int)
                    and not isinstance(hop_bytes, bool) and hop_bytes >= 0
                    and isinstance(shard_count, int)
                    and not isinstance(shard_count, bool)
                    and shard_count > 0):
                slot["actions"] += 1
                slot["hop_bytes"] += hop_bytes
                slot["bytes_with"] += nbytes
                acc["actions_with_hops"] += 1
                acc["hop_bytes_total"] += hop_bytes
                acc["bytes_with_hops"] += nbytes
            else:
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes

    def _require_uniform_hops(hops: list, field: str) -> None:
        """M18（2026-09-29）：hops[0]×Σbytes 聚合前提 = per-shard hops
        全等（face 实例同构 rank 块）。非全等（非同构实例块形状）时旧
        实现静默取 hops[0] 失真——fail-closed 并列各 shard 值，不猜。"""
        if len(set(hops)) != 1:
            fail(f"hopbytes: request={request_id} kind={kind} "
                 f"{field} 非全等（{hops}）——hops[0]×Σbytes 聚合前提"
                 f"不成立（非同构实例块形状？），拒绝静默失真")

    if kind == "prefill":
        hop_rows = decision.get("transfer_hop_bytes")
        if isinstance(hop_rows, list):
            # (2026-09-30) py-slo-E1/C1：逐出行（history/prefill_evictions
            # 的传输对象）以 transfer_hop_bytes 行为 hops 权威源；
            # remote_load/noc_migrate 恢复行由下方标量/T3 路径按字节口径
            # 消费（kind 区分，不双计——XC-1 修复：noc_migrate 与
            # remote_load 同属恢复域）。旧产物（无 transfer_hop_bytes
            # 字段）回退 _eviction_rows_without_hops。
            _consume_transfer_hop_rows(
                hop_rows, ("remote_load", "noc_migrate"))
        else:
            _eviction_rows_without_hops(decision.get("history_evictions"))
            _eviction_rows_without_hops(decision.get("prefill_evictions"))
        nbytes = decision.get("history_transfer_bytes")
        hops = decision.get("history_noc_hops")
        if isinstance(nbytes, int) and nbytes > 0:
            if isinstance(hops, list) and hops and all(
                    isinstance(h, int) and not isinstance(h, bool)
                    and h >= 0 for h in hops):
                _require_uniform_hops(hops, "history_noc_hops")
                hop_bytes = nbytes * hops[0]
                slot = _slot()
                slot["actions"] += 1
                slot["hop_bytes"] += hop_bytes
                slot["bytes_with"] += nbytes
                acc["actions_with_hops"] += 1
                acc["hop_bytes_total"] += hop_bytes
                acc["bytes_with_hops"] += nbytes
            elif hops == [] or hops is None:
                # T3(face-LRU 审计,2026-09-29):hops 空 = 恢复族
                # (REMOTE_RESTORE remote_load;history_noc_hops 以池为源,
                # 源实例索引 None 恒返 [])。恢复传输物理上仍经 NoC
                # (池边缘端口→目标 NPU 的 XY 路由),本仓标志性"逐出→回迁"
                # 流量此前被整体排除出覆盖分子。改消费同记录
                # transfer_hop_bytes 行中 kind=remote_load 的精确
                # noc_hop_bytes(行源 = history_transfers,无其他消费路径,
                # 不构成双计);旧行缺失/不可解析的字节仍保守落无覆盖侧。
                restored = 0
                for row in decision.get("transfer_hop_bytes") or ():
                    if not isinstance(row, dict) or row.get(
                            "kind") != "remote_load":
                        continue
                    row_bytes = row.get("total_bytes")
                    row_hops = row.get("noc_hop_bytes")
                    if not (isinstance(row_bytes, int) and row_bytes > 0
                            and isinstance(row_hops, int)
                            and not isinstance(row_hops, bool)
                            and row_hops >= 0):
                        continue
                    slot = _slot()
                    slot["actions"] += 1
                    slot["hop_bytes"] += row_hops
                    slot["bytes_with"] += row_bytes
                    acc["actions_with_hops"] += 1
                    acc["hop_bytes_total"] += row_hops
                    acc["bytes_with_hops"] += row_bytes
                    restored += row_bytes
                remainder = nbytes - restored
                if remainder > 0:
                    slot = _slot()
                    slot["bytes_without"] += remainder
                    acc["bytes_without_hops"] += remainder
            else:
                slot = _slot()
                slot["bytes_without"] += nbytes
                acc["bytes_without_hops"] += nbytes
    elif kind == "decode":
        transfer = decision.get("prefill_decode_transfer")
        hops = decision.get("kv_noc_hops")
        hop_rows = decision.get("transfer_hop_bytes")
        if isinstance(hop_rows, list):
            # (2026-09-30) py-slo-E1/C1：decode 行源 = decode_evictions
            # 传输 + prefill_decode_transfer（_transfer_hop_rows 恰好覆盖
            # 两者的全部传输对象）——整表经 transfer_hop_bytes 消费：
            # PD 迁移的 hops 权威 = 行内 noc_hop_bytes（逐 shard
            # bytes×hops 精确和，不再依赖 kv_noc_hops 全等前提/M18），
            # 逐出行同法。防双计：decode_evictions 行级 total_bytes 与
            # PD 迁移字节均不再走旧路径（_eviction_rows_without_hops +
            # hops[0]×Σbytes）。restore_skip_kinds 为空：本分支 PD 的
            # noc_migrate 迁移腿只有行消费这一条计数路径，不得豁免。
            _consume_transfer_hop_rows(hop_rows, ())
            return
        _eviction_rows_without_hops(decision.get("decode_evictions"))
        if isinstance(transfer, list):
            # 旧产物回退（无 transfer_hop_bytes 行而 PD 传输为契约行
            # 列表的中间形态；face_online_scheduler._transfer_rows，行
            # 结构无 shards 键）——local_hit 段无物理搬移不计数（同
            # hbm_watermark 口径）；迁移行 total_bytes × hops[0]（face
            # 实例为同构 rank 块，_kv_noc_hops 逐 shard 跳数恒一致 ⇒
            # 与逐 shard bytes×hops 求和严格相等，同 prefill 分支论证）。
            for row in transfer:
                if not isinstance(row, dict) or row.get(
                        "kind") in ("local_hit", "LOCAL_HIT"):
                    continue
                nbytes = row.get("total_bytes")
                if not isinstance(nbytes, int) or nbytes <= 0:
                    continue
                if isinstance(hops, list) and hops and all(
                        isinstance(h, int) and not isinstance(h, bool)
                        and h >= 0 for h in hops):
                    _require_uniform_hops(hops, "kv_noc_hops")
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
        else:
            shards = (transfer or {}).get("shards") if isinstance(
                transfer, dict) else None
            if isinstance(shards, list) and isinstance(hops, list) \
                    and len(shards) == len(hops):
                for shard, hop in zip(shards, hops):
                    nbytes = (shard or {}).get("bytes")
                    if not isinstance(nbytes, int) or nbytes <= 0:
                        continue
                    if (isinstance(hop, int) and not isinstance(hop, bool)
                            and hop >= 0):
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
                # 旧产物（kv_noc_hops 缺席或 shards/hops 错位）的 dict 型
                # prefill_decode_transfer：与 prefill 分支同款兜底——bytes
                # 计入无覆盖侧（docstring 向后兼容口径，不得静默漏计）。
                nbytes = (transfer or {}).get("total_bytes") if isinstance(
                    transfer, dict) else None
                if isinstance(nbytes, int) and nbytes > 0:
                    slot = _slot()
                    slot["bytes_without"] += nbytes
                    acc["bytes_without_hops"] += nbytes
    elif kind == "completion":
        # face 完成段逐出契约行（completion_evictions 为 8 字段行、无
        # total_bytes；transfer 契约行在 completion_eviction_transfers，
        # 见 face_online_scheduler B3 完成决策行序列化）。
        # (2026-09-30) py-slo-E1/C1：transfer_hop_bytes 行源 =
        # completion_evictions（_eviction_hop_rows），逐出传输的 hops
        # 权威源；防双计——completion_eviction_transfers 行级
        # total_bytes 不再同时计入 bytes_without。旧产物回退旧行口径。
        completion_hop_rows = decision.get("transfer_hop_bytes")
        if isinstance(completion_hop_rows, list):
            _consume_transfer_hop_rows(completion_hop_rows, ())
        else:
            _eviction_rows_without_hops(
                decision.get("completion_eviction_transfers"))


def collect_wscllm(record: dict, acc: dict, per_request: dict) -> None:
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    if record.get("kind") != "decode":
        # 历史迁移（prefill 决策）：2026-08-26 B2wp9py 起附带 noc_hops
        # 字段（实例图最短路）→ 纳入覆盖；旧产物无该字段时按
        # bytes_without_hops 处理（向后兼容）。
        if record.get("kind") == "prefill":
            nbytes = decision.get("history_transfer_bytes")
            if isinstance(nbytes, int) and nbytes > 0:
                hops = decision.get("noc_hops")
                slot = per_request.setdefault(
                    request_id, {"actions": 0, "hop_bytes": 0,
                                 "bytes_with": 0, "bytes_without": 0})
                if isinstance(hops, int) and not isinstance(hops, bool) \
                        and hops >= 0:
                    slot["actions"] += 1
                    slot["hop_bytes"] += nbytes * hops
                    slot["bytes_with"] += nbytes
                    acc["actions_with_hops"] += 1
                    acc["hop_bytes_total"] += nbytes * hops
                    acc["bytes_with_hops"] += nbytes
                else:
                    slot["bytes_without"] += nbytes
                    acc["bytes_without_hops"] += nbytes
        return
    transfer = decision.get("prefill_decode_transfer")
    static_route = decision.get("static_route")
    nbytes = (transfer or {}).get("total_bytes") if isinstance(
        transfer, dict) else None
    hops = (static_route or {}).get("hop_count") if isinstance(
        static_route, dict) else None
    if not isinstance(nbytes, int) or nbytes <= 0:
        return
    slot = per_request.setdefault(
        request_id, {"actions": 0, "hop_bytes": 0,
                     "bytes_with": 0, "bytes_without": 0})
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
        if not isinstance(nbytes, int):
            transfer = decision.get("history_transfer")
            if isinstance(transfer, dict):
                total = transfer.get("total_bytes")
                if isinstance(total, int):
                    nbytes = total
    elif kind == "decode":
        nbytes = decision.get("prefill_decode_transfer_bytes")
        if not isinstance(nbytes, int):
            transfer = decision.get("prefill_decode_transfer")
            if isinstance(transfer, dict):
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
    计入无覆盖侧），新旧产物可区分。2026-09-26 A5-4 处置：字段在但行
    全部结构无效（无一行可判 local_hit/有效搬运）时同样回退 aggregate
    口径，不再静默丢字节（aggregate 字节计入无覆盖侧，coverage 如实
    降低而非虚高）。2026-09-28 B4② 处置：混合无效行可估的 total_bytes
    计入 dropped_bytes；空列表与 completion 记录同样不再绕过回退
    （completion 无 aggregate 字段，可估字节计 dropped_bytes）。
    R2-F3（2026-09-29）：noc_hop_bytes 缺失/非法的行不计入 coverage
    分子（零贡献虚高且零留痕），可估字节转 dropped_bytes 留痕。
    （2026-10-01 自 face 仓对齐移植。）"""
    decision = record.get("decision") or {}
    request_id = record.get("request_id") or NA
    kind = record.get("kind")
    relevant_kinds = ("prefill", "decode", "completion")
    hop_rows = decision.get("transfer_hop_bytes")
    if kind in relevant_kinds and isinstance(hop_rows, list):
        wellformed = 0  # 结构可判的行（含 local_hit / 零字节名义行）
        dropped_row_bytes = 0  # 结构无效行中可估的 total_bytes（B4② 留痕）
        for row in hop_rows:
            if not isinstance(row, dict):
                continue
            nbytes = row.get("total_bytes")
            shard_count = row.get("shard_count")
            if (not isinstance(nbytes, int) or isinstance(nbytes, bool)
                    or not isinstance(shard_count, int)
                    or isinstance(shard_count, bool)):
                # 2026-09-28 B4② 处置：结构无效行不再静默蒸发——可估的
                # total_bytes 暂记，混合场景（本记录尚有有效行）计入
                # dropped_bytes 留痕并入分母；行全无效时由下方回退分支
                # 代表（aggregate 字段与行字节同源，不双计）。
                if isinstance(nbytes, int) and nbytes > 0:
                    dropped_row_bytes += nbytes
                continue
            wellformed += 1
            if nbytes <= 0 or shard_count <= 0:
                continue  # local_hit / 名义规模：无物理搬运
            hop_bytes = row.get("noc_hop_bytes")
            if isinstance(hop_bytes, bool) or not isinstance(hop_bytes, int) \
                    or hop_bytes < 0:
                # R2-F3：noc_hop_bytes 缺失/非法的行不得计入 coverage 分子
                # （零贡献虚高 coverage 且零留痕）——可估字节转入
                # dropped_bytes 留痕（分母守恒，B4② 同通道）；wellformed
                # 计数不变（行仍是真实搬运，不触发 aggregate 回退以免
                # aggregate 字段同源双计）。
                dropped_row_bytes += nbytes
                continue
            slot = per_request.setdefault(
                request_id, {"actions": 0, "hop_bytes": 0,
                             "bytes_with": 0, "bytes_without": 0})
            slot["actions"] += 1
            slot["bytes_with"] += nbytes
            acc["actions_with_hops"] += 1
            acc["bytes_with_hops"] += nbytes
            slot["hop_bytes"] += hop_bytes
            acc["hop_bytes_total"] += hop_bytes
        if wellformed == 0:
            # A5-4 回退口径（2026-09-28 B4② 补盲区）：hop_rows 为空列表
            # 同样回退（旧守卫 `hop_rows and` 把空列表漏在回退之外）；
            # completion 无 aggregate 字段可回退，其无效行可估字节直接
            # 计 dropped_bytes（分母守恒），不再蒸发。
            if kind == "completion":
                if dropped_row_bytes:
                    acc["dropped_bytes"] = (
                        acc.get("dropped_bytes", 0) + dropped_row_bytes)
            else:
                collect_aggregate_only(record, acc, per_request)
        elif dropped_row_bytes:
            acc["dropped_bytes"] = (
                acc.get("dropped_bytes", 0) + dropped_row_bytes)
        return
    collect_aggregate_only(record, acc, per_request)


def collect_sh30(record: dict, acc: dict, per_request: dict) -> None:
    """S3（B4 升级，2026-08-26）：消费 decision log 的 shards[].noc_hops/
    noc_path 新字段（B3 60s 证据：completion_evictions 96 行 / 960 shards
    全带 noc_hops；ad-hoc 复核 hop_bytes=2383397232640）。

    count/fallback 语义与 S1 分支一致（_accumulate_shard_transfers：显式
    noc_hops 优先、noc_path 推导兜底、无路由 bytes 计 bytes_without_hops）。
    holder 集照 S1（history_transfer / prefill_decode_transfer / 各类
    逐出列表）；S3 现仅 completion_evictions 落 shards，其余 holder 留作
    字段同步后的前向兼容。B3 之前的 aggregate-only 停用口径
    （history_transfer_bytes / prefill_decode_transfer_bytes 无路由可聚）
    不再计入本账——那些记录无 noc 路由信息，不属于 Hop-Bytes 可判定
    宇宙；其量级（60s 参考跑 1045595684864 B）由门 JSON 另行登记。
    """
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


REPO_HOP_SOURCES: dict[str, dict] = {
    "astra-sim-sh_1.0": {
        "collector": collect_sh10,
        "granularity": "shard(noc_path)",
        "notes": "shards[].noc_path/noc_hops 已落盘 decision log",
    },
    "astra-sim-wscllm": {
        "collector": collect_wscllm,
        "granularity": "instance(static_route.hop_count)",
        "notes": "decode=static_route.hop_count（实例级粒度）；B2wp9py 起 "
                 "prefill 决策附 noc_hops（实例图最短路），history 迁移"
                 "纳入覆盖；字段缺席 → bytes_without_hops",
    },
    "astra-sim-face": {
        "collector": collect_face,
        "granularity": "per-TP-shard(history_noc_hops/kv_noc_hops)",
        "notes": "B2wp9py 起决策日志附带只读 noc hops 列表"
                 "（prefill=history_noc_hops，decode=kv_noc_hops，"
                 "与 shards 一一对齐）；旧产物无字段 → "
                 "bytes_without_hops（coverage 如实降低，不臆造）；"
                 "B3 契约逐出行 bytes 同入 bytes_without_hops（分母完备）",
    },
    "astra-sim-face-LRU": {
        # 本仓（face-LRU）：同 face——决策日志恒附带只读 noc hops 列表
        # （映射复制 astra-sim-face 条目；collect_face 的 legacy dict 兜底
        # 分支仅为 face 旧产物保留，同 collect_face docstring 向后兼容
        # 口径——R1 py-slo-C1，2026-09-26 注释如实化）。
        # (2026-09-30) py-slo-E1/C1：逐出/PD 传输的 hops 权威源改为
        # transfer_hop_bytes 行（noc_hop_bytes），行缺失回退 bytes_without_
        # hops；非法 shard 走 dropped 计数留痕（R7-F5）。
        "collector": collect_face,
        "granularity": "per-TP-shard(history_noc_hops/kv_noc_hops)",
        "notes": "同 face（恒附带只读 hops 列表：prefill=history_noc_hops、"
                 "decode=kv_noc_hops，与 shards 一一对齐）；逐出/PD 传输"
                 "（history/prefill/decode_evictions、completion_eviction_"
                 "transfers、prefill_decode_transfer）hops 权威源 = "
                 "transfer_hop_bytes 行的 noc_hop_bytes（remote_load 行由"
                 "恢复域按字节口径消费）；行缺失/不可解析 → "
                 "bytes_without_hops（分母完备）",
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
                 "B3 前 aggregate-only 标量字段（无路由可聚）不计入"
                 "本账；无 shards 键 holder 的 total_bytes 计 "
                 "bytes_without_hops（E5 兜底，count/fallback 语义"
                 "同 S1）",
    },
}


def hopbytes_prepare(repo_variant: str) -> tuple[dict, dict, dict]:
    """A4 driver 用：source 登记 + 空累积态。"""
    source = REPO_HOP_SOURCES.get(repo_variant)
    if source is None:
        fail(f"未登记的 repo_variant：{repo_variant}（REPO_HOP_SOURCES 需扩表）")
    # (2026-09-30, ported from the face repo) dropped_* 计数（R7-F5/A5-3/
    # B4①）：结构损坏的 shard/holder 可估字节走 dropped 通道留痕。
    acc = {"actions_with_hops": 0, "hop_bytes_total": 0,
           "bytes_with_hops": 0, "bytes_without_hops": 0,
           "dropped_holders": 0, "dropped_bytes": 0,
           "dropped_shards": 0}
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
    # (2026-09-30, ported from the face repo) 2026-09-26 A5-3 处置：
    # dropped_bytes（缺 shards holder 可估的字节）计入分母——dropped 不
    # 再静默抬高 coverage；dropped_* 计数随 summary JSON 一并输出（**acc）。
    total_bytes = (acc["bytes_with_hops"] + acc["bytes_without_hops"]
                   + acc["dropped_bytes"])
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
        **acc,
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
