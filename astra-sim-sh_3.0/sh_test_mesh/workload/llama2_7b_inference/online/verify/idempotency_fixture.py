#!/usr/bin/env python3
"""idempotency_fixture.py -- 阶段 4 §7.2 幂等 fixture(重放同一 delivery)。

把一次真实运行(C++ 侧)产出的全部 request_<seq>.json(delta 序列)逐一喂给
一个真实的 Sh30OnlineScheduler,每个 delta 喂两次:

  1. 第一次 = 正常应用(delivery_sequence 单调 +1);
  2. 第二次(同一 delta 原样重放)= 幂等重放:必须返回与第一次逐字段相等的
     batch(digest),且调度器状态零变更(不重新产生任何 assignment/KV
     action/图节点;KV 账本逐会话状态与逐 rank HBM 占用、图节点计数、
     in-flight、provisional/committed 账本、日志行、digest 计数全部不动)。

fail-closed 路径同样验证(不满足即非 0 退出):
  - 跳号(seq != last+1)抛 ValueError;
  - 同 seq 不同内容(篡改 delta)抛 ValueError;
  - ack 领先(seq > last_applied_sequence)、schema 版本错、batch_id 发散
    抛 ValueError;
  - 重复 ack 幂等忽略(ack_count 不重复计)。

结束:verify_run_end() 必须通过(ack_count == delivery_count ==
last_applied_sequence;reply cache 覆盖最后一笔交付)。

用法(与 online_service.py 相同的参数子集):
    python3 online/verify/idempotency_fixture.py \
        --bridge-dir <真实运行 bridge 目录> --plan-dir <离线 plan 目录> \
        [--config <trace_config.csv>] [--limit N]

默认 strategy 模式(真实策略产图;§7.2 是基类级合同)。--limit 只喂前 N 个
delta(快速冒烟),缺省全量。

本文件自 astra-sim-face 同名夹具整文件分发(R2 修复 07;含 response 消费
即删的 has_any_response 判定),按本仓适配三处(其余与 face 版逐字一致):

  1. 调度器装配段:Sh30OnlineScheduler(与本仓 online_service.py 相同的
     关键字装配);
  2. _snapshot 的 KV 口径:face 版读 kv_manager.events 事件流计数(本仓
     KVCacheManager 无事件流),改用等价状态投影——逐会话 session_snapshots()
     (含本仓三态账本的 resident_prefix_layers)+ 逐 rank hbm_snapshots()
     的 kv_cache_bytes(形态对齐批 F1 sh_1.0 同名夹具);
  3. --config 传参:与本仓 online_service.py 相同,args.config(str)统一
     转 Path 再进 load_face_trace_config(裸传 str 会在 .exists() 处崩溃)。
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

_ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_face_trace import load_face_trace_config  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import Sh30OnlineScheduler  # noqa: E402


class _CountingDigestSink:
    """只计数的 digest sink(不写文件;断言幂等重放不重复产出 digest)。"""

    def __init__(self):
        self.count = 0

    def __call__(self, row: dict) -> None:
        self.count += 1


def _snapshot(scheduler, sink):
    """调度器可观测状态快照(幂等断言用;重复 delivery 必须逐项不变)。"""
    return {
        "delivery_count": scheduler.delivery_count,
        "last_applied_sequence": scheduler.last_applied_sequence,
        "ack_count": scheduler.ack_count,
        "in_flight": dict(sorted(scheduler.in_flight.items())),
        "completed_request_ids": sorted(scheduler.completed_request_ids),
        "online_log_rows": scheduler.online_log_count,
        "emitted_by_delivery": len(scheduler._emitted_by_delivery),
        "seen_acks": len(scheduler._seen_ack_delivery_seqs),
        # sh_3.0 适配(批 F2a 2026-08-21,形态对齐批 F1 sh_1.0):face 版此处读
        # kv 事件水位(kv_manager.events 长度 + _kv_events_emitted);本仓
        # KVCacheManager 无事件流水,KV 动作不经 kv_actions 批次通道,等价
        # 幂等观测源 = 逐会话 KV 状态(位置/实例/上下文/字节/驻留前缀层数/
        # 完成时刻/active)+ 逐 rank HBM KV 占用——任何重放误重入账都会
        # 改变其中至少一项(reserve/prepare/expand/move/enforce/mark_complete
        # 均落到会话或 rank 字节;重放误重跑策略会先撞基类 fail-closed)。
        "kv_sessions": [
            (snapshot.session_id, snapshot.location, snapshot.instance_index,
             snapshot.context_tokens, snapshot.total_bytes,
             snapshot.resident_prefix_layers, snapshot.last_completion_ns,
             snapshot.active)
            for snapshot in scheduler.kv_manager.session_snapshots()
        ],
        "kv_hbm_bytes": [
            (snapshot.rank, snapshot.kv_cache_bytes)
            for snapshot in scheduler.kv_manager.hbm_snapshots()
        ],
        # 阶段 5 §8.2:provisional KV 账本也纳入幂等快照(重放不重复入账,
        # fail-closed 尝试不污染;ack 流把暂存条目逐笔转入 committed 层)。
        "provisional_kv_actions": sorted(
            scheduler._provisional_kv_actions),
        "committed_kv_actions_count": len(
            scheduler._committed_kv_actions),
        "digest_count": sink.count,
        "reply_cache_seq": (
            -1 if scheduler._delivery_reply_cache is None
            else scheduler._delivery_reply_cache["seq"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="sh_3.0 phase-4 §7.2 idempotency fixture")
    parser.add_argument("--bridge-dir", required=True,
                        help="真实运行的 bridge 目录(含 request_<seq>.json)")
    parser.add_argument("--plan-dir", required=True,
                        help="离线 plan/manifest 目录(含 manifest.json)")
    parser.add_argument("--config", default=None,
                        help="trace_config.csv 路径(缺省用 workload 默认)")
    parser.add_argument("--limit", type=int, default=0,
                        help="只喂前 N 个 delta(0 = 全量)")
    args = parser.parse_args(argv)

    manifest_path = os.path.join(args.plan_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    if args.config:
        # args.config 是 str;loader 签名是 Path(默认参数 CONFIG_CSV_PATH 亦
        # 为 Path),裸传 str 会在 .exists() 处崩溃——与本仓 online_service.py
        # 相同,统一转 Path。
        config = load_face_trace_config(Path(args.config))
    else:
        config = load_face_trace_config()

    # 真实 delta 序列:重读 C++ 运行产出的 request_<seq>.json,按 seq 升序。
    # 只取已获 response 的交付(运行尾 C++ 可能写出最后一笔 request 后即
    # 结束,该笔无 response/ack,Python 侧从未应用)。
    deltas = []
    has_any_response = any(
        name.startswith("response_") and name.endswith(".json")
        for name in os.listdir(args.bridge_dir))
    for name in sorted(os.listdir(args.bridge_dir)):
        if name.startswith("request_") and name.endswith(".json"):
            seq = int(name[len("request_"):-len(".json")])
            if has_any_response and not os.path.exists(os.path.join(
                    args.bridge_dir, "response_{}.json".format(seq))):
                continue  # 运行尾未响应交付,Python 从未应用,不重放
            # (阶段 7 §10.3 response 消费即删:成功运行的 bridge 无任何
            #  response_*.json,此时全部 request_*.json 均为已应用交付。)
            with open(os.path.join(args.bridge_dir, name),
                      "r", encoding="utf-8") as source:
                deltas.append((seq, json.load(source)))
    deltas.sort(key=lambda item: item[0])
    if not deltas:
        raise RuntimeError("no request_<seq>.json found in {}".format(
            args.bridge_dir))
    if args.limit > 0:
        deltas = deltas[:args.limit]

    sink = _CountingDigestSink()
    graph = GraphBatchBuilder(config)
    # B3(2026-08-23):本夹具逐 delta 双喂,是重放路径的唯一常规触发者——
    # 显式开启防御性深拷缓存(生产 online_service 走默认引用缓存)。
    scheduler = Sh30OnlineScheduler(
        manifest=manifest,
        config=config,
        graph=graph,
        digest_sink=sink,
        mode="strategy",
        defensive_reply_cache=True,
    )

    # ------------------------------------------------------------- 主循环 --
    for index, (seq, delta) in enumerate(deltas):
        if delta.get("delivery_sequence") != seq:
            raise RuntimeError(
                "delta file seq {} != payload delivery_sequence {}"
                .format(seq, delta.get("delivery_sequence")))
        if seq != index:  # C++ 自 0 起(合同 §3.2 修正记录)
            raise RuntimeError(
                "delta sequence not contiguous: got {} at index {}"
                .format(seq, index))

        batch1 = scheduler.on_decision_batch(delta)
        before = _snapshot(scheduler, sink)

        # 同一 delivery 原样重放 -> 幂等 digest,状态零变更。
        batch2 = scheduler.on_decision_batch(delta)
        if batch2 != batch1:
            raise RuntimeError(
                "delivery {} idempotent replay returned a different batch "
                "(digest mismatch)".format(seq))
        after = _snapshot(scheduler, sink)
        if after != before:
            raise RuntimeError(
                "delivery {} replay mutated scheduler state:\n  before={}\n  "
                "after ={}".format(seq, before, after))
        if after["delivery_count"] != index + 1 or after[
                "last_applied_sequence"] != seq:
            raise RuntimeError(
                "delivery {}: delivery_count={} last_applied_sequence={} "
                "(expect {} and {})".format(
                    seq, after["delivery_count"],
                    after["last_applied_sequence"], index + 1, seq))

        if index == 0:
            # -------------------------------------------------- fail-closed --
            # 恰在第一笔交付之后:last_applied == 0,篡改命中"同 seq 不同
            # 内容"的幂等 byte-match 路径;跳号命中 gap 路径。两个尝试都
            # 必须在任何状态变更前抛 ValueError 且不留污染。
            state_before_fail_closed = _snapshot(scheduler, sink)
            jumped = copy.deepcopy(delta)
            jumped["delivery_sequence"] = seq + 2  # 0 -> 2:跳号
            try:
                scheduler.on_decision_batch(jumped)
            except ValueError:
                pass
            else:
                raise RuntimeError(
                    "delivery sequence gap ({} -> {}) was not rejected"
                    .format(seq, seq + 2))
            tampered = copy.deepcopy(delta)
            tampered["reasons"] = (list(tampered.get("reasons", []))
                                   + ["TAMPERED"])
            try:
                scheduler.on_decision_batch(tampered)
            except ValueError:
                pass
            else:
                raise RuntimeError(
                    "tampered duplicate of delivery {} was not rejected"
                    .format(seq))
            if _snapshot(scheduler, sink) != state_before_fail_closed:
                raise RuntimeError(
                    "fail-closed attempt mutated scheduler state")

    # ------------------------------------------------------------- ack 流 --
    for seq, _ in deltas:
        ack = {
            "schema_version": 1,
            "delivery_sequence": seq,
            "batch_id": seq,
            "success": True,
        }
        scheduler.on_commit_ack(ack)
        scheduler.on_commit_ack(ack)  # 重复 ack 幂等忽略
    if scheduler.ack_count != len(deltas):
        raise RuntimeError(
            "ack_count={} != deliveries={} after ack stream"
            .format(scheduler.ack_count, len(deltas)))
    # ack fail-closed:版本错 / batch_id 发散 / 领先。
    for label, ack in (
        ("schema version", {"schema_version": 0, "delivery_sequence": 1,
                            "batch_id": 1, "success": True}),
        ("batch_id divergence", {"schema_version": 1, "delivery_sequence": 1,
                                 "batch_id": 2, "success": True}),
        ("ahead of last applied", {"schema_version": 1,
                                   "delivery_sequence": len(deltas) + 1,
                                   "batch_id": len(deltas) + 1,
                                   "success": True}),
    ):
        try:
            scheduler.on_commit_ack(ack)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "ack fail-closed violation not rejected: {}".format(label))
    if scheduler.ack_count != len(deltas):
        raise RuntimeError(
            "rejected acks still counted: ack_count={}".format(
                scheduler.ack_count))

    # ------------------------------------------------------------- 收尾 --
    if args.limit:
        # 冒烟模式只喂前缀交付:结束审计(in-flight 清空等)依赖全量输入,
        # 只验证交付级幂等断言;全量模式(limit=0)才做 verify_run_end。
        print(
            "idempotency fixture PASS (smoke, first {} deliveries): {} "
            "replays, {} acks, {} digest rows (all applied once)"
            .format(args.limit, len(deltas), scheduler.ack_count, sink.count),
            flush=True)
        return 0
    scheduler.verify_run_end()
    print(
        "idempotency fixture PASS: {} deliveries, {} deliveries replayed, "
        "{} acks, {} digest rows (all applied once), verify_run_end ok"
        .format(len(deltas), len(deltas), scheduler.ack_count, sink.count),
        flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("idempotency_fixture: fatal: {}: {}".format(
            type(exc).__name__, exc),
            file=sys.stderr)
        sys.exit(1)
