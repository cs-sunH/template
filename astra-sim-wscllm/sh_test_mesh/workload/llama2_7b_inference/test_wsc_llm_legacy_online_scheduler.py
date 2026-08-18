#!/usr/bin/env python3
"""test_wsc_llm_legacy_online_scheduler.py -- 阶段 7 §10.6 legacy 第二变体
在线调度器单元测试(WscLlmLegacyOnlineScheduler)。

合成 delta 流直接驱动 on_decision_batch(不经 C++/bridge),验证 §0.4 红线
与 legacy 语义:

  - FCFS 队头阻塞:head try_allocate 失败 -> 整队停(队内后续请求不发射),
    后续调度事件重查容量,释放后阻塞解除重查;
  - WSC Relevant(P,D) 静态域分配:decode 实例优先、选中 prefill 次之、
    同域 sibling 最后,永不扩大到其他 Decode 域;分配扣减/释放恢复;
  - 静态 P->D 路由:decode 目标 = static_mapping.route_for_prefill,
    path 首尾与角色校验;
  - turn>0 前序 KV 释放 + history_source_instance_index / history_
    transfer_bytes 记录;决策日志 history_action 按离线口径 NO_HISTORY
    (构图字段 NOC_MIGRATE);
  - terminal 释放、run-end 断言(completed / 实例空闲 / session KV 全释放 /
    heap/frontier 清空)、kv_actions 恒空 + kv_event_payload_legacy 终值。

请求规模动态推导:legacy Relevant(P,D) 域 = decode 实例 + 选中路径 +
同 decode 域 sibling(空域容量由 probe allocator 实测),避免硬编码 token
数越界(空域不足 = try_allocate raise,非队头阻塞)或填不满域(无法构造
阻塞场景)。

运行: 在 llama2_7b_inference 目录
    python3 -m pytest test_wsc_llm_legacy_online_scheduler.py -q
(与 test_wsc_llm_scheduler.py 同一运行方式)
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import tempfile

from pathlib import Path

import pytest

from generate_wsc_llm_trace import load_wsc_llm_trace_config
from online.graph_batch_builder import GraphBatchBuilder
from online.wsc_llm_legacy_online_scheduler import (
    P_CHUNK,
    WscLlmLegacyOnlineScheduler,
)
from wsc_llm_scheduler import (
    DECODE_ROLE,
    PREFILL_ROLE,
    kv_cache_bytes_for_tokens,
)

# 模型: llama2_7b(32 layers, hidden 4096, bytes_per_elem 2)
# kv 字节/token = 2 * layers * hidden * bytes_per_elem = 524288 B = 0.5 MiB
# request-neutral 合成 fixture(方案 §0.3 / §3 步骤 0-1):checked-in legacy
# 配置的 request_queue_csv 为占位路径(正式入口缺失输入 fail-closed);测试
# 需要队列槽位时用本文件内手写的合成队列,绝不引用任何真实 trace 数据。
# __file__ 锚定:相对路径会随 pytest 调用根(cwd)不同而失效(workload 目录
# 单跑可过,sh_test_mesh 根带 tests 一起跑 FileNotFound)。

_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="wscllm_legacy_fixture_")
_FIXTURE_HEADER = (
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns,description"
)
# 8 个 turn-0 槽位 + 2 个 turn>0 槽位:_pick_slots 需要 >=7 个 turn-0
# (session_arrival_time_ns)与 >=1 个 turn>0(inter_request_interval_ns)。
_FIXTURE_ROWS = (
    ("fixture_s0", "0", "fixture_s0_r0", "100", "5", "0", "", "synthetic fixture"),
    ("fixture_s1", "0", "fixture_s1_r0", "200", "6", "0", "", "synthetic fixture"),
    ("fixture_s2", "0", "fixture_s2_r0", "300", "7", "0", "", "synthetic fixture"),
    ("fixture_s3", "0", "fixture_s3_r0", "400", "8", "0", "", "synthetic fixture"),
    ("fixture_s4", "0", "fixture_s4_r0", "500", "9", "0", "", "synthetic fixture"),
    ("fixture_s5", "0", "fixture_s5_r0", "600", "10", "0", "", "synthetic fixture"),
    ("fixture_s6", "0", "fixture_s6_r0", "700", "11", "0", "", "synthetic fixture"),
    ("fixture_s7", "0", "fixture_s7_r0", "800", "12", "0", "", "synthetic fixture"),
    ("fixture_s0", "1", "fixture_s0_r1", "50", "4", "", "1000", "synthetic fixture"),
    ("fixture_s1", "1", "fixture_s1_r1", "60", "5", "", "2000", "synthetic fixture"),
)


def _load_fixture_config():
    """request-neutral fixture:legacy checked-in 配置 + 手写合成队列。

    trace_config_legacy.csv 的相对路径(hardware/、system/)仍按 SH_TEST_DIR
    解析;仅 request_queue_csv 指向合成队列。调用方(调用方)物化真实输入后,
    测试即按物化规则(方案 §3 步骤 0-1)验证同一代码路径。
    """
    config_source = Path(__file__).resolve().parent / "trace_config_legacy.csv"
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    if not queue_path.exists():
        queue_path.write_text(
            _FIXTURE_HEADER
            + "\n"
            + "\n".join(",".join(row) for row in _FIXTURE_ROWS)
            + "\n",
            encoding="utf-8",
        )
    config_path = Path(_FIXTURE_DIR.name) / "trace_config_legacy.csv"
    if not config_path.exists():
        lines = []
        for raw_line in config_source.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("config,request_queue_csv,"):
                lines.append(
                    "config,request_queue_csv,"
                    + str(queue_path)
                    + ",,,,,request-neutral synthetic fixture queue (unit test)"
                )
            else:
                lines.append(raw_line)
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_wsc_llm_trace_config(config_path)

# 小请求: 任意域空置余量下必然可分配(≈5 MB)。
SMALL_TOKENS = 10_000
# 阻塞场景的"塞不进去"请求: 需求 = 域空置余量 + 0.5 MiB(见
# test_fcfs_head_of_line_blocking 的动态推导,此处仅作为空域必可容纳的
# 下界——try_allocate 对空域不足会 raise,不能用超大常数)。
BLOCKED_TOKENS = 1_000_000


def _request_record(queue_index, session_id, turn_index, request_id,
                    history_tokens_before, prefill_context_tokens,
                    final_context_tokens, prefill_length=100,
                    decode_length=5):
    return {
        "queue_index": queue_index,
        "session_id": session_id,
        "turn_index": turn_index,
        "request_id": request_id,
        "prefill_length": prefill_length,
        "decode_length": decode_length,
        "history_tokens_before": history_tokens_before,
        "prefill_context_tokens": prefill_context_tokens,
        "final_context_tokens": final_context_tokens,
    }


def _delta(seq, tick, arrivals=(), completed=(), reasons=None):
    """合成 StateDelta v1(request_*.json 同构;schema 校验按基类)。"""
    return {
        "schema_version": 1,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": tick,
        "deferred_from_tick": 0,
        "reasons": list(reasons or ["TICK_END"]),
        "arrivals": list(arrivals),
        "completed_groups": list(completed),
        "completed_nodes": [],
        "retry_items": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick, "kind": ""},
        "ledger_summary": {"injected_unfinished": []},
    }


def _arrival(queue_index, request_id, session_id, turn_index,
             prefill_length=100, decode_length=5,
             arrival_world_ns=0, interval_ns=0):
    return {
        "arrival_world_ns": arrival_world_ns,
        "decode_length": decode_length,
        "ingress_seq": queue_index,
        "inter_request_interval_ns": interval_ns,
        "prefill_length": prefill_length,
        "queue_index": queue_index,
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": turn_index,
    }


def _complete(request_id, stage):
    return {"request_id": request_id, "stage": stage}


def _probe(config):
    """probe 调度器:拓扑/静态映射/allocator 只读实例(用于域容量推导)。
    manifest 为空不触发任何运行期逻辑。"""
    manifest = {"requests": [], "selected_request_count": 0,
                "selected_session_count": 0}
    return WscLlmLegacyOnlineScheduler(
        manifest=manifest, config=config,
        graph=GraphBatchBuilder(config))


def _build_scheduler(records, config=None):
    if config is None:
        config = _load_fixture_config()
    assert config.kv_cache_policy == "legacy"
    manifest = {
        "requests": records,
        "selected_request_count": len(records),
        "selected_session_count": len(
            {record["session_id"] for record in records}),
    }
    graph = GraphBatchBuilder(config)
    scheduler = WscLlmLegacyOnlineScheduler(
        manifest=manifest, config=config, graph=graph)
    return config, scheduler


def _pick_slots(config, turn0_count, later_count):
    """合成 fixture config 的 queue_index 槽位池:合成 manifest 的
    queue_index 会被 graph_batch_builder._request_spec 用于取 gate 事实,
    必须指向真实槽位——turn-0 记录需要 session_arrival_time_ns(合成队列
    8 个),turn>0 记录需要 inter_request_interval_ns(2 个)。返回升序槽位。"""
    turn0 = [i for i, spec in enumerate(config.request_queue)
             if spec.session_arrival_time_ns is not None]
    later = [i for i, spec in enumerate(config.request_queue)
             if spec.inter_request_interval_ns is not None]
    return turn0[:turn0_count], later[:later_count]


def _ack(scheduler, seq):
    scheduler.on_commit_ack({
        "schema_version": 1,
        "batch_id": seq,
        "delivery_sequence": seq,
    })


def _apply(scheduler, seq, tick, arrivals=(), completed=(), reasons=None):
    batch = scheduler.on_decision_batch(
        _delta(seq, tick, arrivals=arrivals, completed=completed,
               reasons=reasons))
    _ack(scheduler, seq)
    return batch


def _watched_requests(batch):
    return {watch["request_id"] for watch in batch["watches"]}


def _assigned(batch):
    return {assignment["request_id"] for assignment in batch["assignments"]}


# --------------------------------------------------------------------- 红线 --


def test_import_and_p_chunk_calibration():
    """p_chunk 标定常数 = 4954(用户裁决 2026-08-15;推导 = ceil(mean(
    prefill_length)) = ceil(5830711/1177),登记于 online/wsc_llm_legacy_
    online_scheduler.py 模块注释与方案文档 §3 步骤 0-1 物化规则/附录 C;
    离线 legacy 运行目录名 pc4954 实测同值)。"""
    assert P_CHUNK == 4954


def test_policy_dispatch_fail_closed():
    """分发 fail-closed:mode / kv_cache_policy / sensing 校验。"""
    config = _load_fixture_config()
    manifest = {"requests": [], "selected_request_count": 0,
                "selected_session_count": 0}
    graph = GraphBatchBuilder(config)
    # mode 白名单断言——构造器对非 strategy 模式 fail-closed
    # (词法层由 online_service --mode choices 拒绝,机制层由本断言拒绝)。
    with pytest.raises(ValueError):
        WscLlmLegacyOnlineScheduler(
            manifest=manifest, config=config, graph=graph, mode="replay")
    with pytest.raises(ValueError):
        WscLlmLegacyOnlineScheduler(
            manifest=manifest, config=config, graph=graph, sensing=True)


def test_kv_actions_always_empty_and_terminal_payload():
    """legacy 无逐事件 KV 日志:每批 kv_actions 恒空;run-end 终值报告与
    metrics_integration.kv_event_payload_legacy 同构(9 实例);terminal
    释放后 allocator 剩余容量 = 初始容量。"""
    records = [_request_record(
        0, "fx_s0", 0, "fx_s0_r0", 0, SMALL_TOKENS, SMALL_TOKENS)]
    config, scheduler = _build_scheduler(records)
    initial = list(scheduler.allocator.remaining_capacity)
    batch0 = _apply(scheduler, 0, 1000,
                    arrivals=[_arrival(0, "fx_s0_r0", "fx_s0", 0)])
    assert batch0["kv_actions"] == []
    _apply(scheduler, 1, 2000, completed=[_complete("fx_s0_r0", "prefill")])
    batch2 = _apply(scheduler, 2, 3000, completed=[
        _complete("fx_s0_r0", "decode"), _complete("fx_s0_r0", "")])
    assert batch2["kv_actions"] == []
    scheduler.verify_run_end()
    assert scheduler.runtime_by_request_id[
        "fx_s0_r0"].terminal_kv_release_at_completion
    assert list(scheduler.allocator.remaining_capacity) == initial
    payload = scheduler.kv_event_payload_legacy()
    assert payload["policy"] == "wsc_relevant_pd_static_decode_domain"
    assert len(payload["final_remaining_capacity_bytes"]) == 9
    assert all(byte > 0 for byte in payload["final_remaining_capacity_bytes"])


def test_static_pd_route_and_domain_allocation():
    """静态 P->D 路由 + Relevant(P,D) 域分配:decode 目标 = route_for_prefill;
    pieces 序 = decode 优先 -> selected_prefill -> sibling;总字节与
    kv_cache_bytes_for_tokens(final_context) 一致;永不出域(不含其他
    Decode 域实例);terminal 释放后容量完全恢复。"""
    config = _load_fixture_config()
    probe = _probe(config)
    first_prefill = min(probe.topology.indices_for_role(PREFILL_ROLE))
    route0 = probe.static_mapping.route_for_prefill(first_prefill)
    bytes_per_token = kv_cache_bytes_for_tokens(config.model, 1)
    # 需求 = 空域 - 1M token -> 必然跨多个实例(decode 单实例容量
    # ≈ 1.94M token,单 piece 装不下),触发 decode + selected_prefill +
    # sibling 三段式。
    tokens = (
        probe.allocator.empty_domain_capacity_bytes(route0) // bytes_per_token
        - BLOCKED_TOKENS)
    records = [_request_record(0, "fx_s0", 0, "fx_s0_r0", 0, tokens, tokens)]
    config2, scheduler = _build_scheduler(records)
    batch = _apply(scheduler, 0, 1000,
                   arrivals=[_arrival(0, "fx_s0_r0", "fx_s0", 0)])
    runtime = scheduler.runtime_by_request_id["fx_s0_r0"]
    assert runtime.kv_allocation is not None
    prefill = runtime.prefill_instance_index
    route = scheduler.static_mapping.route_for_prefill(prefill)
    assert runtime.decode_instance_index == route.decode_instance_index
    assert scheduler.topology.instance(prefill).phase_role == PREFILL_ROLE
    assert scheduler.topology.instance(
        runtime.decode_instance_index).phase_role == DECODE_ROLE
    assert route.path[0] == route.prefill_instance_index
    assert route.path[-1] == route.decode_instance_index
    allocation = runtime.kv_allocation
    assert allocation.total_bytes == kv_cache_bytes_for_tokens(
        config2.model, tokens)
    assert sum(piece.bytes for piece in allocation.pieces) == \
        allocation.total_bytes
    assert allocation.pieces[0].location_priority == "decode"
    assert allocation.pieces[0].instance_index == \
        runtime.decode_instance_index
    priorities = [piece.location_priority for piece in allocation.pieces]
    assert "selected_prefill" in priorities
    # 域内:pieces 全部落在本 decode 域(本域全部路径实例);域外:不含任何
    # 其他 Decode 域的路径实例(永不扩大到其他 Decode 域)。
    decode = route.decode_instance_index
    this_domain = {decode} | {
        instance_index
        for sibling in scheduler.static_mapping.routes_for_decode(decode)
        for instance_index in sibling.path
    }
    other_domains = {
        instance_index
        for sibling in scheduler.static_mapping.routes
        if sibling.decode_instance_index != decode
        for instance_index in sibling.path
    }
    assert all(piece.instance_index in this_domain
               for piece in allocation.pieces)
    assert not any(piece.instance_index in other_domains
                   for piece in allocation.pieces)
    # 完成 + terminal 释放 -> 容量完全恢复。
    _apply(scheduler, 1, 2000, completed=[_complete("fx_s0_r0", "prefill")])
    _apply(scheduler, 2, 3000, completed=[
        _complete("fx_s0_r0", "decode"), _complete("fx_s0_r0", "")])
    scheduler.verify_run_end()
    assert list(scheduler.allocator.remaining_capacity) == \
        list(scheduler.allocator.initial_capacity)


def test_fcfs_head_of_line_blocking_and_recheck():
    """核心红线:FCFS 队头阻塞 + 后续调度事件重查 + 阻塞解除重查。

    A(需求 = 空域 - BLOCKED)准入后其 Relevant(P,D) 域仅剩
    < BLOCKED 字节;D1..D5 占住其余 5 个 prefill 实例使 B 与 A 同实例
    排队;A 完成前 B(head)try_allocate 失败 -> 整队停(B 不发射);
    后续空批(调度事件)重查仍阻塞;A terminal 释放后重查成功发射。
    """
    config = _load_fixture_config()
    probe = _probe(config)
    prefill_indices = sorted(probe.topology.indices_for_role(PREFILL_ROLE))
    assert len(prefill_indices) == 6  # 6P3D face case
    first_prefill = prefill_indices[0]
    route_a = probe.static_mapping.route_for_prefill(first_prefill)
    bytes_per_token = kv_cache_bytes_for_tokens(config.model, 1)
    huge_tokens = (
        probe.allocator.empty_domain_capacity_bytes(route_a) // bytes_per_token
        - BLOCKED_TOKENS)
    assert huge_tokens > BLOCKED_TOKENS  # 域至少容下两个阻塞级请求
    slots, _ = _pick_slots(config, 7, 0)
    records = [
        _request_record(slots[0], "fx_sA", 0, "fx_sA_r0", 0, huge_tokens,
                        huge_tokens),
        _request_record(slots[1], "fx_sD1", 0, "fx_sD1_r0", 0,
                        SMALL_TOKENS, SMALL_TOKENS),
        _request_record(slots[2], "fx_sD2", 0, "fx_sD2_r0", 0,
                        SMALL_TOKENS, SMALL_TOKENS),
        _request_record(slots[3], "fx_sD3", 0, "fx_sD3_r0", 0,
                        SMALL_TOKENS, SMALL_TOKENS),
        _request_record(slots[4], "fx_sD4", 0, "fx_sD4_r0", 0,
                        SMALL_TOKENS, SMALL_TOKENS),
        _request_record(slots[5], "fx_sD5", 0, "fx_sD5_r0", 0,
                        SMALL_TOKENS, SMALL_TOKENS),
        _request_record(slots[6], "fx_sB", 0, "fx_sB_r0", 0,
                        BLOCKED_TOKENS, BLOCKED_TOKENS),
    ]
    _, scheduler = _build_scheduler(records, config)
    by_id = scheduler.runtime_by_request_id

    # seq0: A + D1..D5 同批到达(queue_index 冻结序升序;各占一个 prefill
    # 实例,6 实例 count 全 1)。A 准入成功(D 们域空,全部准入)。
    batch0 = _apply(scheduler, 0, 1000, arrivals=[
        _arrival(slots[0], "fx_sA_r0", "fx_sA", 0),
        _arrival(slots[1], "fx_sD1_r0", "fx_sD1", 0),
        _arrival(slots[2], "fx_sD2_r0", "fx_sD2", 0),
        _arrival(slots[3], "fx_sD3_r0", "fx_sD3", 0),
        _arrival(slots[4], "fx_sD4_r0", "fx_sD4", 0),
        _arrival(slots[5], "fx_sD5_r0", "fx_sD5", 0),
    ])
    a = by_id["fx_sA_r0"]
    assert a.prefill_instance_index == first_prefill
    assert a.kv_allocation is not None  # A 准入成功
    assert "fx_sA_r0" in _watched_requests(batch0)
    assert scheduler.instances[a.prefill_instance_index].busy

    # seq1: B 到达 -> 6 个 prefill 实例全 count=1 -> min(ordering_key)
    # 平局取最小 index = A 所在实例(队尾)。A 忙,实例不进 frontier。
    batch1 = _apply(scheduler, 1, 1100,
                    arrivals=[_arrival(slots[6], "fx_sB_r0", "fx_sB", 0)])
    b = by_id["fx_sB_r0"]
    assert b.prefill_instance_index == a.prefill_instance_index  # 同实例
    assert b.kv_allocation is None
    assert "fx_sB_r0" not in _watched_requests(batch1)

    # seq2: A prefill 完成 -> B 成为队头 -> try_allocate 失败
    # (FCFS 队头阻塞:head 失败 -> 整队停,同队后续一概不尝试)。
    batch2 = _apply(scheduler, 2, 2000,
                    completed=[_complete("fx_sA_r0", "prefill")])
    assert b.kv_allocation is None  # 队头阻塞:未准入
    assert "fx_sB_r0" not in _watched_requests(batch2)
    assert "fx_sB_r0" not in _assigned(batch2)
    assert b in scheduler.instances[b.prefill_instance_index].qp  # 仍在队中

    # seq3: 空批(后续调度事件重查容量)——仍阻塞,零副作用。
    batch3 = _apply(scheduler, 3, 3000)
    assert b.kv_allocation is None
    assert batch3["watches"] == [] and batch3["assignments"] == []

    # seq4: A terminal decode 完成 -> 释放 Relevant(P,D) 域 -> B 重查
    # 成功(A 释放后域 = 空域 >= B 需求)。
    batch4 = _apply(scheduler, 4, 4000,
                    completed=[_complete("fx_sA_r0", "decode"),
                               _complete("fx_sA_r0", "")])
    assert b.kv_allocation is not None  # 阻塞解除
    assert "fx_sB_r0" in _watched_requests(batch4)
    assert a.terminal_kv_release_at_completion

    # seq5-6: B 完成(terminal 释放)。
    _apply(scheduler, 5, 5000, completed=[_complete("fx_sB_r0", "prefill")])
    _apply(scheduler, 6, 6000, completed=[_complete("fx_sB_r0", "decode"),
                                          _complete("fx_sB_r0", "")])
    # D1..D5 完成(各自域空,无阻塞)。
    seq = 7
    for d in ("fx_sD1_r0", "fx_sD2_r0", "fx_sD3_r0", "fx_sD4_r0",
              "fx_sD5_r0"):
        _apply(scheduler, seq, 7000 + seq, completed=[
            _complete(d, "prefill"), _complete(d, "decode"),
            _complete(d, "")])
        seq += 1
    scheduler.verify_run_end()
    assert scheduler.completed_requests == 7
    assert scheduler.session_allocations == {}
    assert list(scheduler.allocator.remaining_capacity) == \
        list(scheduler.allocator.initial_capacity)


def test_turn1_history_release_and_decision_log_legacy_scope():
    """turn>0 前序 KV 释放 + history 记录;决策日志 history_action 按离线
    legacy 口径为 NO_HISTORY(构图字段 NOC_MIGRATE);释放与再准入的
    容量净效应 = 初始 - turn-1 需求。"""
    config = _load_fixture_config()
    turn0_slots, later_slots = _pick_slots(config, 1, 1)
    records = [
        _request_record(turn0_slots[0], "fx_s1", 0, "fx_s1_r0", 0,
                        BLOCKED_TOKENS, BLOCKED_TOKENS),
        _request_record(later_slots[0], "fx_s1", 1, "fx_s1_r1",
                        BLOCKED_TOKENS,
                        BLOCKED_TOKENS + SMALL_TOKENS,
                        BLOCKED_TOKENS + 2 * SMALL_TOKENS),
    ]
    _, scheduler = _build_scheduler(records, config)
    by_id = scheduler.runtime_by_request_id
    bytes_per_token = kv_cache_bytes_for_tokens(config.model, 1)
    r0_bytes = BLOCKED_TOKENS * bytes_per_token
    r1_final_bytes = (BLOCKED_TOKENS + 2 * SMALL_TOKENS) * bytes_per_token
    initial_total = sum(scheduler.allocator.remaining_capacity)

    _apply(scheduler, 0, 1000, arrivals=[
        _arrival(turn0_slots[0], "fx_s1_r0", "fx_s1", 0)])
    assert sum(scheduler.allocator.remaining_capacity) == \
        initial_total - r0_bytes
    _apply(scheduler, 1, 2000, completed=[_complete("fx_s1_r0", "prefill")])
    batch2 = _apply(scheduler, 2, 3000,
                    completed=[_complete("fx_s1_r0", "decode"),
                               _complete("fx_s1_r0", "")])
    # turn-0 非 terminal:decode 完成后 KV 保留(等待 turn-1 arrival)。
    assert scheduler.session_allocations["fx_s1"].request_id == "fx_s1_r0"
    assert sum(scheduler.allocator.remaining_capacity) == \
        initial_total - r0_bytes  # 保留期间容量未恢复
    # turn-1 future alarm:interval 来自冻结 config(离线同源口径)。
    interval = config.request_queue[later_slots[0]].inter_request_interval_ns
    assert batch2["future_alarms"] == [{
        "arrival_world_ns": 3000 + interval,
        "envelope": {
            "request_id": "fx_s1_r1",
            "session_id": "fx_s1",
            "turn_index": 1,
            "prefill_length": 100,
            "decode_length": 5,
            "inter_request_interval_ns": interval,
        },
    }]

    batch3 = _apply(scheduler, 3, 3000 + interval, arrivals=[
        _arrival(later_slots[0], "fx_s1_r1", "fx_s1", 1,
                 interval_ns=interval)])
    r1 = by_id["fx_s1_r1"]
    # 前序释放 + turn-1 再准入:净容量 = 初始 - turn-1 需求(释放已生效)。
    assert sum(scheduler.allocator.remaining_capacity) == \
        initial_total - r1_final_bytes
    assert r1.history_source_instance_index == \
        by_id["fx_s1_r0"].decode_instance_index
    assert r1.history_transfer_bytes == r0_bytes
    # 决策日志:legacy 口径 history_action = NO_HISTORY(离线 decision_log
    # 同字段);构图字段为 NOC_MIGRATE(_plan_dict)。
    prefill_rows = [row for row in scheduler.online_log_rows
                    if row["kind"] == "prefill"
                    and row["request_id"] == "fx_s1_r1"]
    assert len(prefill_rows) == 1
    assert prefill_rows[0]["decision"]["history_action"] == "NO_HISTORY"
    assert prefill_rows[0]["decision"]["history_source_instance_index"] == \
        by_id["fx_s1_r0"].decode_instance_index
    assert prefill_rows[0]["decision"]["history_transfer_bytes"] == r0_bytes
    # 构图:turn-1 用 NOC_MIGRATE(单 history transfer,共享构图器分支)。
    plan = scheduler._plan_dict(r1)
    assert plan["history_action"] == "NOC_MIGRATE"
    assert plan["history_transfer_bytes"] == r1.history_transfer_bytes
    assert plan["history_recompute_tokens"] == 0

    _apply(scheduler, 4, 4000 + interval,
           completed=[_complete("fx_s1_r1", "prefill")])
    _apply(scheduler, 5, 5000 + interval, completed=[
        _complete("fx_s1_r1", "decode"), _complete("fx_s1_r1", "")])
    scheduler.verify_run_end()
    assert by_id["fx_s1_r1"].terminal_kv_release_at_completion
    assert list(scheduler.allocator.remaining_capacity) == \
        list(scheduler.allocator.initial_capacity)


def test_full_run_verify_and_decision_log_shape():
    """小规模全量运行(2 session,一单 turn 一大一小的混合):完整 delta 流
    + ack 流,verify_run_end 通过;决策日志每 request 3 条(kind 序
    prefill -> decode -> completion,seq 单调 1 起)。

    注意 decode 侧 FCFS:fx_x_r0 与 fx_y_r0 的前两个到达落在 prefill
    1/4,二者静态路由同归 decode 0——decode 队首整段发射使 fx_y_r0 的
    decode 延后一个决策批(completion_gates 由 emit_decode_batch 写入,
    与真实 C++ 完成流一致,测试流按此排程)。"""
    config = _load_fixture_config()
    turn0_slots, later_slots = _pick_slots(config, 2, 1)
    records = [
        _request_record(turn0_slots[0], "fx_x", 0, "fx_x_r0", 0,
                        SMALL_TOKENS, SMALL_TOKENS),
        _request_record(turn0_slots[1], "fx_y", 0, "fx_y_r0", 0,
                        BLOCKED_TOKENS, BLOCKED_TOKENS),
        _request_record(later_slots[0], "fx_y", 1, "fx_y_r1",
                        BLOCKED_TOKENS,
                        BLOCKED_TOKENS + SMALL_TOKENS,
                        BLOCKED_TOKENS + 2 * SMALL_TOKENS),
    ]
    _, scheduler = _build_scheduler(records, config)
    by_id = scheduler.runtime_by_request_id

    # seq0: 两 session 首 turn 同批到达,全部准入发射。
    batch0 = _apply(scheduler, 0, 1000, arrivals=[
        _arrival(turn0_slots[0], "fx_x_r0", "fx_x", 0),
        _arrival(turn0_slots[1], "fx_y_r0", "fx_y", 0),
    ])
    assert set(_watched_requests(batch0)) == {"fx_x_r0", "fx_y_r0"}

    # seq1: 两 prefill 完成;decode 0 队首整段发射 = 只发射 fx_x_r0 的
    # decode(fx_y_r0 的 decode 排队等待)。
    batch1 = _apply(scheduler, 1, 2000, completed=[
        _complete("fx_x_r0", "prefill"), _complete("fx_y_r0", "prefill")])
    assert _watched_requests(batch1) == {"fx_x_r0"}

    # seq2: fx_x_r0 decode 完成(terminal 释放)-> decode 0 队头让位,
    # fx_y_r0 的 decode 本批发射(completion_gates["fx_y"] 由此写入)。
    interval = config.request_queue[later_slots[0]].inter_request_interval_ns
    batch2 = _apply(scheduler, 2, 3000, completed=[
        _complete("fx_x_r0", "decode"), _complete("fx_x_r0", "")])
    assert _watched_requests(batch2) == {"fx_y_r0"}
    assert batch2["future_alarms"] == []

    # seq3: fx_y_r0 decode 完成(非 terminal,保留 KV)+ 排程 turn-1。
    batch3 = _apply(scheduler, 3, 4000, completed=[
        _complete("fx_y_r0", "decode"), _complete("fx_y_r0", "")])
    assert batch3["future_alarms"] == [{
        "arrival_world_ns": 4000 + interval,
        "envelope": {
            "request_id": "fx_y_r1",
            "session_id": "fx_y",
            "turn_index": 1,
            "prefill_length": 100,
            "decode_length": 5,
            "inter_request_interval_ns": interval,
        },
    }]
    assert scheduler.session_allocations["fx_y"].request_id == "fx_y_r0"

    # seq4: fx_y 的 turn-1 到达(前序释放 + 再准入)。
    batch4 = _apply(scheduler, 4, 4000 + interval, arrivals=[
        _arrival(later_slots[0], "fx_y_r1", "fx_y", 1,
                 interval_ns=interval)])
    assert "fx_y_r1" in _watched_requests(batch4)
    assert by_id["fx_y_r1"].history_source_instance_index == \
        by_id["fx_y_r0"].decode_instance_index

    # seq5-6: turn-1 完成(terminal)。
    _apply(scheduler, 5, 5000 + interval,
           completed=[_complete("fx_y_r1", "prefill")])
    _apply(scheduler, 6, 6000 + interval, completed=[
        _complete("fx_y_r1", "decode"), _complete("fx_y_r1", "")])

    scheduler.verify_run_end()
    assert scheduler.completed_requests == 3
    assert scheduler.ack_count == scheduler.delivery_count == 7
    assert scheduler.session_allocations == {}
    assert list(scheduler.allocator.remaining_capacity) == \
        list(scheduler.allocator.initial_capacity)
    # 决策日志:每 request 3 条,kind 序 prefill -> decode -> completion,
    # seq 单调 1 起(在线 seq 约定;跨 request 交错,按 request 分组核对)。
    rows = scheduler.online_log_rows
    assert len(rows) == 9
    for request_id in ("fx_x_r0", "fx_y_r0", "fx_y_r1"):
        kinds = [row["kind"] for row in rows
                 if row["request_id"] == request_id]
        assert kinds == ["prefill", "decode", "completion"]
    seqs = [row["seq"] for row in rows]
    assert seqs == list(range(1, 10))
