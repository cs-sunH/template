#!/usr/bin/env python3
"""relevant_kv_emission.py -- relevant_distributed 变体的 KV 通信发射薄包装
(总文档《wscllm要补充的选择分析方案总文档》§2.4/§3.2/§3.3,裁决 #9-13/#35;
执行文档 §4.2/§5.2,2026-09-02)。

三个发射函数操作 GraphBatchBuilder/OnlineTraceBuilder(经传入的 graph
对象 duck-typing,不 import online/ 模块),复用 _paired_transfer 的
整头 shard 逐 rank 字节语义与命名/tag 惯例,但**不改其本体**:

  - emit_history_pulls(1000 族):多轮历史拉回,旧 pieces 的每个非 P
    实际源实例一条聚合边 源→新 P;源=P 零边(LOCAL_HIT 语义);两端
    hbm_charge=true;总字节 = kv(history_tokens_before)。
  - emit_piece_scatter(3100):drain 散布写边 P→owner(owner∈D 剩余
    容量上的 piece 与中间 die piece);P-piece 零边;send 侧插显式锚 =
    PREFILL_DRAIN watch members(每 rank 末个真实 prefill 节点 id,
    R4 的 fallback 语义由此统一实现);recv 正常链入 owner 的 D/die
    frontier(与旧 3000 同口径);两端 hbm_charge=true;函数内 fail-closed
    守恒断言 Σ3100 + P-piece = kv_cache_bytes_for_tokens(prefill_ctx)。
  - emit_remote_reads(3300):per (列车,成员,源) 聚合读边,send/recv
    **双端侧插**(saved=previous_id → previous_id=None → arm_timer_gate
    (锚) → comm_send/recv → previous_id=saved);send 依赖 = 源 piece
    数据就绪锚两档(P-piece→prefill 块末节点 id / 中间 die piece→该
    piece 的 3100 recv 节点 id),不进源实例主链;recv 依赖 = 列车
    join 标记,不进 D 主链,recv 节点 id 注入**该成员自己的 exit 标记**
    (成员 exit = max(体尾, 其全部读边));中途成员(本列车不退出)的
    recv 不注入任何标记;字节 = p_m × R_{m,s} 精确式逐 rank 整头 shard;
    send hbm_charge=true、recv hbm_charge=false(全仓首个显式 false)。

解耦规则(执行文档 §4.2):本模块**不 import wsc_relevant_memory_scheduler**
(B1 产物);pieces 序列按字段名 duck-typing(instance_index/token_start/
token_end,tier 仅元数据不消费),placement→发射参数的粘合由 B3 调度器
完成。kv_remote_read=ideal_masked 时调用方整体跳过 3300 发射与 KV 分量
裁剪(现状行为)——发射函数自身不读配置,天然可整体跳过;开关判定用
remote_reads_enabled。

反环约束(总文档 §3.2,写进断言):侧插节点依赖集 ⊆ {源实例上更早的
数据就绪锚(prefill 块末 / 3100 recv)} ∪ {D 的 join 标记};每个侧插
节点恰有一条来自锚的父边,且锚节点 id 严格小于侧插节点 id(因果更早)。

批次折叠加注:B2 三函数自带 _mark/_collect 对(与 emit_* 方法同款)——
builder 的节点/边缓存在每次 emit_* 的 _collect 后即清空,若在两次
emit_* 之间裸发射而不折扫,节点会被下一轮 _mark/_collect 的前缀切掉
而丢失;调用方无须(也不应)自己包 mark/collect。
"""

from __future__ import annotations

import os
import sys
from functools import partial
from typing import Callable, Iterable, Mapping, Optional

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/,共享发射原语在
# 本目录。路径只做 import 用途(红线:generate_wsc_llm_trace.py /
# wsc_llm_scheduler.py / session_kv_manager.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from generate_trace import sanitize_node_prefix  # noqa: E402
from generate_wsc_llm_trace import (  # noqa: E402
    _xy_route,
    kv_cache_bytes_for_tokens,
)
from session_kv_manager import kv_cache_shard_bytes_for_tokens  # noqa: E402


def remote_reads_enabled(kv_remote_read: str) -> bool:
    """kv_remote_read A/B 开关判定(总文档 §3.2,裁决 #11)。

    physical(缺省)= 发射 3300 读边 + 裁远程 KV 分量;ideal_masked =
    调用方**整体跳过** 3300 发射与 KV 分量裁剪(现状字节口径)。非法值
    fail-closed(与 generate_wsc_llm_trace 的键校验同款值域)。
    """

    if kv_remote_read not in {"physical", "ideal_masked"}:
        raise ValueError(
            "kv_remote_read must be physical or ideal_masked, "
            f"got {kv_remote_read!r}")
    return kv_remote_read == "physical"


def local_kv_override_value(model, local_tokens: int, tp_degree: int,
                            relative_rank: int) -> int:
    """local_kv_bytes 覆盖值(总文档 §3.3 KV 归因覆盖;M2 契约钉死)。

    transformer_pass_aggregated 的 k/v_cache 字段是**逐层**张量(kv_length
    × attention_hidden_per_rank),聚合发射时 ×layers;故覆盖值的口径 =
    该 rank 该 span 的**单层单份 K(=V)本地字节** = kv_cache_shard_bytes_
    for_tokens(model, local_tokens, tp)[rank] ÷ (2 × layers)。该值在体内
    K/V 各计一次再 ×layers 后,恰等于 D 本地 piece 的完整 KV 字节(K+V、
    全层)——远程分量改由 3300 读边在源端计费,防双计(总文档 §5.6)。
    B3 用本函数从 KVPlacement 的 D 本地 token 区间逐 rank 求值。
    """

    shards = kv_cache_shard_bytes_for_tokens(model, local_tokens, tp_degree)
    divisor = 2 * int(model.layers)
    value, remainder = divmod(int(shards[relative_rank]), divisor)
    if remainder:
        raise RuntimeError(
            "local KV override value is not integral: shard "
            f"{shards[relative_rank]} not divisible by {divisor}")
    return value


def assert_pieces_on_path(pieces: Iterable, static_route) -> None:
    """pieces ⊆ path 实例集合(总文档 §8.2 KVPlacement 不变量)。

    B3 在准入/placement 冻结后自查;违反即 RuntimeError(fail-closed,
    与构图校验同款)。static_route = 到 D 的静态路径实例号序列。
    """

    allowed = {int(index) for index in static_route}
    for piece in pieces:
        owner = int(piece.instance_index)
        if owner not in allowed:
            raise RuntimeError(
                "KV piece owner instance "
                f"{owner} is not on the static route {sorted(allowed)}")


def remote_read_sources(decode_instance_index: int, pieces: Iterable) -> list:
    """读边源集合 = pieces 的非 D 实例集合(总文档 §8.2 不变量)。

    返回按 instance_index 升序的 [(source_instance_index, piece_tokens)]
    ——decode 段 piece 与 D 上的 prefill 段 piece 都是 D 本地读(其字节
    由列车体 local_kv_bytes 覆盖参数计入),不产生 3300 读边。B3 用本
    函数从 KVPlacement 构造 member_reads 的 sources(升序 = 源序号的
    规范序,跨列车保持同序以保证 R3-② 的同 tag 重复语义)。
    """

    totals: dict[int, int] = {}
    for piece in pieces:
        owner = int(piece.instance_index)
        if owner == int(decode_instance_index):
            continue
        start, end = int(piece.token_start), int(piece.token_end)
        if start < 0 or end < start:
            raise ValueError(
                f"KV piece token range [{start}, {end}) is not well formed")
        totals[owner] = totals.get(owner, 0) + (end - start)
    return sorted(totals.items())


# --------------------------------------------------------------------------
# 内部助手
# --------------------------------------------------------------------------

def _stage_tag(queue_index: int, category: int, relative_rank: int, *,
               source_ordinal: Optional[int] = None,
               tp_degree: Optional[int] = None) -> int:
    """tag 规则(总文档 §2.3):queue_index*10000 + category + relative_rank;
    同请求多源时源序号编入第三段 源序号×TP度数 + relative_rank,断言扩展
    段整体 < 100(源数 ≤ |D′|−1 ≤ 8 → ≤ 8×6+5 = 53)。与既有 _stage_tag
    语义一致,仅增加 fail-closed 的扩展段上界断言。"""

    if source_ordinal is None:
        extension = relative_rank
    else:
        extension = source_ordinal * int(tp_degree) + relative_rank
    if not 0 <= extension < 100:
        raise ValueError(
            "stage tag extension must stay within [0, 100) "
            f"(got {extension}; source ordinal {source_ordinal}, "
            f"tp {tp_degree}, relative rank {relative_rank})")
    return queue_index * 10000 + category + extension


def _require_group(graph, instance_index: int):
    """实例号 → 推理组;未知实例号 fail-closed(pieces ⊆ path 实例集合
    的发射侧守卫:owner/源不在拓扑内直接报错,而非静默错边)。"""

    try:
        return graph.group_by_index[int(instance_index)]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "unknown instance index "
            f"{instance_index!r}: KV pieces must stay on the mapped "
            "instances") from error


def _kv_shard_bytes(model, tokens: int, tp_degree: int) -> tuple:
    """整头 shard 逐 rank KV 字节(与 _paired_transfer 的整头所有权口径
    一致;session_kv_manager.kv_cache_shard_bytes_for_tokens 只读 import)。"""

    if (isinstance(tokens, bool) or not isinstance(tokens, int)
            or tokens < 0):
        raise ValueError("KV token count must be a non-negative integer")
    return kv_cache_shard_bytes_for_tokens(model, tokens, tp_degree)


def _side_insert(builder, anchor_id: int, emit: Callable[[], None],
                 where: str) -> int:
    """双端侧插模式(总文档 §3.2/§5.5):saved=previous_id →
    previous_id=None → arm_timer_gate(锚) → comm_send/recv →
    previous_id=saved。侧插节点不进该 rank 主链(frontier 不动),依赖集
    恰为 {锚};反环断言:唯一父边 = 锚→节点,且锚 id < 节点 id
    (锚必须是源实例/D 上因果更早的数据就绪锚或 join 标记)。"""

    if anchor_id is None:
        raise RuntimeError(f"{where}: side-inserted node needs an anchor id")
    if builder.pending_extra_dependencies:
        raise RuntimeError(
            f"{where}: side-insert requires empty pending dependencies "
            "(arm_timer_gate leftovers would be swallowed)")
    saved = builder.previous_id
    builder.previous_id = None
    builder.arm_timer_gate(int(anchor_id))
    edge_mark = len(builder.edges)
    emit()
    node_id = builder.previous_id
    builder.previous_id = saved
    if node_id is None or node_id == saved:
        raise RuntimeError(f"{where}: side-inserted node was not emitted")
    expected = {"rank": builder.rank, "from": int(anchor_id),
                "to": node_id, "kind": "data"}
    if builder.edges[edge_mark:] != [expected]:
        raise RuntimeError(
            f"{where}: anti-cycle invariant violated (总文档 §3.2): a "
            f"side-inserted node must depend on exactly its anchor "
            f"{int(anchor_id)}, got edges {builder.edges[edge_mark:]!r}")
    if not int(anchor_id) < node_id:
        raise RuntimeError(
            f"{where}: side-insert anchor {int(anchor_id)} must precede "
            f"the node {node_id} on rank {builder.rank}")
    return node_id


def _resolve_join_anchors(graph, train_id: str, decode_ranks,
                          join_anchors: Optional[Mapping]) -> dict:
    """列车 join 锚(3300 recv 的依赖,总文档 §3.2):每 decode rank 的
    列车头末节点 = 该 rank 最后一个 join 标记(因果覆盖全部 join 标记,
    保守锚;【物】取舍见模块 docstring)。

    获取方式两档:显式 join_anchors 参数优先(无 joiner 的列车——例如
    全员继续成员——由调用方供头前 frontier 节点);否则按名字锚恢复
    ({train_id}_join_* 节点,_emit_train_head 命名),要求本函数与列车
    发射同批次调用(节点已在 graph.batch["nodes"] 中)。两档皆缺 →
    fail-closed。"""

    resolved: dict[int, Optional[int]] = {rank: None for rank in decode_ranks}
    explicit = dict(join_anchors or {})
    prefix = f"{train_id}_join_"
    batch_nodes = graph.batch["nodes"] if graph.batch else ()
    for node in batch_nodes:
        rank = node.get("rank")
        if rank not in resolved:
            continue
        if not str(node.get("name", "")).startswith(prefix):
            continue
        node_id = int(node["id"])
        if resolved[rank] is None or node_id > resolved[rank]:
            resolved[rank] = node_id
    for rank in decode_ranks:
        anchor = explicit.get(rank)
        if anchor is None:
            anchor = resolved[rank]
        if anchor is None:
            raise RuntimeError(
                f"{train_id}: no join anchor for decode rank {rank} "
                "(trains without join markers need explicit head anchors)")
        resolved[rank] = int(anchor)
    return {rank: int(anchor) for rank, anchor in resolved.items()}


# --------------------------------------------------------------------------
# 1000 族:多轮历史拉回(总文档 §2.1/§3.4,裁决 #15)
# --------------------------------------------------------------------------

def emit_history_pulls(
    graph,
    *,
    queue_index: int,
    prefix: str,
    request_id: str,
    old_pieces: Iterable,
    new_prefill_instance_index: int,
    history_tokens_before: int,
    source_timer_gates: Optional[Mapping] = None,
    stage: str = "prefill",
    generation: int = 0,
) -> dict:
    """turn>0 历史拉回:旧 KVPlacement.pieces 的每个非 P 实际源实例一条
    聚合边 源→新 P(1000 族;源=P 零边 = LOCAL_HIT 语义)。与既有
    _paired_transfer(legacy 单源 history_kv)同款挂法:send 链源实例
    frontier、可带 timer gate 锚(interval gate;None = 不锚),recv 链
    新 P frontier——历史拉回是到达边界的串行前置,无需侧插。

    字节 = 逐源聚合 token 的整头 shard(kv_cache_bytes_for_tokens 语义,
    实例级总量守恒);tag 第三段编入源序号(源按 instance_index 升序取
    规范序 0..k-1,同请求内稳定)。两端 hbm_charge=true。函数内
    fail-closed 守恒:Σ各源字节(+P 本地零边部分)= kv(history_tokens_
    before)。

    返回 {"routes": [...], "sources": [...], "local_hit_tokens": int,
    "total_bytes": int}(决策日志/hopbytes 元数据,noc_path 复用 _xy_route)。
    """

    model = graph.config.model
    prefill_group = _require_group(graph, new_prefill_instance_index)
    tp = len(prefill_group.ranks)
    if (isinstance(history_tokens_before, bool)
            or not isinstance(history_tokens_before, int)
            or history_tokens_before < 0):
        raise ValueError(
            "history_tokens_before must be a non-negative integer")

    # 逐源聚合旧 pieces(裁剪到 [0, history_tokens_before);owner == 新 P
    # 的部分零边 = LOCAL_HIT,只进守恒账)。
    tokens_by_source: dict[int, int] = {}
    local_hit_tokens = 0
    for piece in old_pieces:
        start, end = int(piece.token_start), int(piece.token_end)
        if start < 0 or end < start:
            raise ValueError(
                f"old KV piece token range [{start}, {end}) is not well "
                "formed")
        low, high = max(start, 0), min(end, history_tokens_before)
        if high <= low:
            continue
        tokens = high - low
        if int(piece.instance_index) == int(new_prefill_instance_index):
            local_hit_tokens += tokens
        else:
            owner = int(piece.instance_index)
            tokens_by_source[owner] = tokens_by_source.get(owner, 0) + tokens

    # 守恒断言(总文档 §3.4):拉回 + 本地命中 = kv(history_tokens_before)。
    scatter_total = sum(tokens_by_source.values())
    expected_bytes = kv_cache_bytes_for_tokens(model, history_tokens_before)
    emitted_bytes = kv_cache_bytes_for_tokens(model, scatter_total)
    local_bytes = kv_cache_bytes_for_tokens(model, local_hit_tokens)
    if emitted_bytes + local_bytes != expected_bytes:
        raise RuntimeError(
            "history pull conservation violated (总文档 §3.4): "
            f"{emitted_bytes} + {local_bytes} != {expected_bytes} for "
            f"{history_tokens_before} history tokens")

    timer_gates = dict(source_timer_gates or {})
    marker = graph._mark()
    routes: list[dict] = []
    sources_meta: list[dict] = []
    for source_ordinal, source_index in enumerate(sorted(tokens_by_source)):
        tokens = tokens_by_source[source_index]
        if tokens == 0:
            continue
        source_group = _require_group(graph, source_index)
        if len(source_group.ranks) != tp:
            raise RuntimeError(
                "WSC-LLM direct KV shard pairing requires equal TP degree")
        bytes_by_rank = _kv_shard_bytes(model, tokens, tp)
        name = f"{prefix}_kv_history_pull_i{source_index}"
        for relative_rank, (source, target) in enumerate(
                zip(source_group.ranks, prefill_group.ranks)):
            tag = _stage_tag(
                queue_index, 1000, relative_rank,
                source_ordinal=source_ordinal, tp_degree=tp)
            graph.builders[source].set_context(request_id, stage, generation)
            graph.builders[target].set_context(request_id, stage, generation)
            # send:legacy history_kv 同款——timer gate 锚(有则 arm)+ 正常
            # 链源 frontier;两端 hbm_charge=true(源读出 + P 落盘)。
            gate = timer_gates.get(source)
            if gate is not None:
                graph.builders[source].arm_timer_gate(int(gate))
            graph.builders[source].comm_send(
                f"{name}_send_rank{source}_to_rank{target}",
                src=source, dst=target,
                comm_size=bytes_by_rank[relative_rank],
                comm_tag=tag, hbm_charge=True,
            )
            graph.builders[target].comm_recv(
                f"{name}_recv_rank{source}_to_rank{target}",
                src=source, dst=target,
                comm_size=bytes_by_rank[relative_rank],
                comm_tag=tag, hbm_charge=True,
            )
            path = _xy_route(graph.config.hardware, source, target)
            routes.append({
                "category": 1000,
                "source_instance_index": source_index,
                "target_instance_index": int(new_prefill_instance_index),
                "relative_shard": relative_rank,
                "source_rank": source,
                "target_rank": target,
                "bytes": bytes_by_rank[relative_rank],
                "noc_path": path,
                "noc_hops": len(path) - 1,
            })
        sources_meta.append({
            "source_instance_index": source_index,
            "source_ordinal": source_ordinal,
            "tokens": tokens,
            "bytes": kv_cache_bytes_for_tokens(model, tokens),
        })
    graph._collect(marker)
    return {
        "routes": routes,
        "sources": sources_meta,
        "local_hit_tokens": local_hit_tokens,
        "total_bytes": emitted_bytes,
    }


# --------------------------------------------------------------------------
# 3100:drain 散布写边(总文档 §2.1/§3.2,裁决 #9/#13)
# --------------------------------------------------------------------------

def emit_piece_scatter(
    graph,
    *,
    queue_index: int,
    prefix: str,
    request_id: str,
    prefill_instance_index: int,
    pieces: Iterable,
    prefill_context_tokens: int,
    prefill_end_members: Mapping,
    stage: str = "decode",
    generation: int = 1,
) -> dict:
    """PREFILL_DRAIN 边界的散布写边:P → owner(owner = D 剩余容量上的
    piece 与中间 die piece);P-piece(prefill_stay)零边。

    挂法(总文档 §3.2):
      - send **侧插**、显式锚 = prefill_end_members(PREFILL_DRAIN watch
        members,每 rank 末个真实 prefill 节点 id)——不依赖 R4 的
        frontier 等价假设(R4 的 fallback 语义由此统一实现,§9 R4);
      - recv 正常链入 owner 的 D/die frontier(与旧 3000 同口径;成员
        drain 后才成为列车候选,列车体天然因果序于 recv);
      - 两端 hbm_charge=true(P 读出 + owner 落盘,write-once 完整计费)。

    字节 = piece ∩ [0, prefill_context_tokens) 的整头 shard 逐 rank;
    同 owner 多 piece 聚合为一条边(贪心放置下每 owner 恰一 piece)。
    函数内 fail-closed 守恒断言(总文档 §2.1/裁决 #9):
    Σ3100 + P-piece 字节 = kv_cache_bytes_for_tokens(prefill_context)。

    返回 {"routes": [...], "recv_ids": {owner_instance: {rank: 节点 id}},
    "scatter_bytes": int, "prefill_stay_bytes": int, ...};recv_ids 是
    3300 读边 send 锚的第二档(中间 die piece → 该 piece 的 3100 recv
    节点 id)的数据就绪锚来源。
    """

    model = graph.config.model
    prefill_group = _require_group(graph, prefill_instance_index)
    tp = len(prefill_group.ranks)
    if (isinstance(prefill_context_tokens, bool)
            or not isinstance(prefill_context_tokens, int)
            or prefill_context_tokens < 0):
        raise ValueError(
            "prefill_context_tokens must be a non-negative integer")

    # 逐 owner 聚合散布 token(裁剪到 prefill 段 [0, prefill_context));
    # decode 段 piece(>= prefill_ctx)不散布;owner == P 零边只进守恒账。
    tokens_by_owner: dict[int, int] = {}
    stay_tokens = 0
    for piece in pieces:
        start, end = int(piece.token_start), int(piece.token_end)
        if start < 0 or end < start:
            raise ValueError(
                f"KV piece token range [{start}, {end}) is not well formed")
        low, high = max(start, 0), min(end, prefill_context_tokens)
        if high <= low:
            continue
        tokens = high - low
        owner = int(piece.instance_index)
        if owner == int(prefill_instance_index):
            stay_tokens += tokens
        else:
            tokens_by_owner[owner] = tokens_by_owner.get(owner, 0) + tokens

    # 守恒断言(总文档 §2.1/裁决 #9,做 golden):Σ3100 + P-piece 字节 =
    # kv_cache_bytes_for_tokens(prefill_context)(= 旧 3000 总量)。
    scatter_tokens = sum(tokens_by_owner.values())
    expected_bytes = kv_cache_bytes_for_tokens(model, prefill_context_tokens)
    scatter_bytes = kv_cache_bytes_for_tokens(model, scatter_tokens)
    stay_bytes = kv_cache_bytes_for_tokens(model, stay_tokens)
    if scatter_bytes + stay_bytes != expected_bytes:
        raise RuntimeError(
            "piece scatter conservation violated (总文档 §2.1): "
            f"{scatter_bytes} + {stay_bytes} != {expected_bytes} for "
            f"prefill_context_tokens={prefill_context_tokens}")

    # send 显式锚:本请求 prefill 块末,须覆盖 P 全部 rank。
    for rank in prefill_group.ranks:
        if rank not in prefill_end_members:
            raise RuntimeError(
                f"prefill end members miss P rank {rank}: the 3100 send "
                "anchor is the explicit PREFILL_DRAIN watch member")

    marker = graph._mark()
    routes: list[dict] = []
    recv_ids: dict[int, dict] = {}
    for owner_index in sorted(tokens_by_owner):
        tokens = tokens_by_owner[owner_index]
        if tokens == 0:
            continue
        owner_group = _require_group(graph, owner_index)
        if len(owner_group.ranks) != tp:
            raise RuntimeError(
                "WSC-LLM direct KV shard pairing requires equal TP degree")
        bytes_by_rank = _kv_shard_bytes(model, tokens, tp)
        name = f"{prefix}_kv_scatter_i{owner_index}"
        owner_recv_ids: dict[int, int] = {}
        for relative_rank, (source, target) in enumerate(
                zip(prefill_group.ranks, owner_group.ranks)):
            tag = _stage_tag(queue_index, 3100, relative_rank)
            graph.builders[source].set_context(request_id, stage, generation)
            graph.builders[target].set_context(request_id, stage, generation)
            # send:侧插 + 显式 prefill 块末锚(不进 P 主链,P 可立即服务
            # 下一请求的 prefill;散布与其后继物理重叠)。
            _side_insert(
                graph.builders[source], int(prefill_end_members[source]),
                partial(
                    graph.builders[source].comm_send,
                    f"{name}_send_rank{source}_to_rank{target}",
                    src=source, dst=target,
                    comm_size=bytes_by_rank[relative_rank],
                    comm_tag=tag, hbm_charge=True,
                ),
                where=f"{name}_send_rank{source}_to_rank{target}",
            )
            # recv:正常链入 owner 的 D/die frontier(旧 3000 同口径)。
            graph.builders[target].comm_recv(
                f"{name}_recv_rank{source}_to_rank{target}",
                src=source, dst=target,
                comm_size=bytes_by_rank[relative_rank],
                comm_tag=tag, hbm_charge=True,
            )
            owner_recv_ids[target] = graph.builders[target].previous_id
            path = _xy_route(graph.config.hardware, source, target)
            routes.append({
                "category": 3100,
                "source_instance_index": int(prefill_instance_index),
                "target_instance_index": owner_index,
                "relative_shard": relative_rank,
                "source_rank": source,
                "target_rank": target,
                "bytes": bytes_by_rank[relative_rank],
                "noc_path": path,
                "noc_hops": len(path) - 1,
            })
        recv_ids[owner_index] = owner_recv_ids
    graph._collect(marker)
    return {
        "routes": routes,
        "recv_ids": recv_ids,
        "scatter_bytes": scatter_bytes,
        "prefill_stay_bytes": stay_bytes,
        "scatter_tokens_by_owner": dict(sorted(tokens_by_owner.items())),
        "prefill_stay_tokens": stay_tokens,
    }


# --------------------------------------------------------------------------
# 3300:列车远程读边(总文档 §2.1/§3.2/§5.4/§5.5,裁决 #10/#11)
# --------------------------------------------------------------------------

def emit_remote_reads(
    graph,
    *,
    train_id: str,
    decode_instance_index: int,
    member_reads: Iterable,
    scatter_recv_ids: Optional[Mapping] = None,
    join_anchors: Optional[Mapping] = None,
) -> dict:
    """per (列车, 成员, 源) 的聚合读边(列车发射后、同批次调用——join
    锚按名字恢复需要列车节点仍在当前批中)。

    member_reads 每元素(成员粒度,总文档 §5.4:成员级 exit 汇合语义):
      request_id / queue_index / participation(p_m,本列车迭代数,≥1)
      prefill_instance_index(该成员的 P,源两档判别)
      sources: [{"source_instance_index", "piece_tokens"(R_{m,s} 的
        token 数), "source_ordinal"(可选,缺省 = 列表位置)}]——跨列车
        必须保持同序(R3-② 的同 tag 重复语义)
      prefill_end_members: {rank: id}(该成员 PREFILL_DRAIN watch members,
        P-piece 源锚第一档)
      exit_anchors: {rank: exit 标记节点 id} 或 None——本列车退出的成员
        传 emit_iteration_train 返回的 exit_members[request_id];中途成员
        (本列车不退出)传 None,其 recv 不注入任何标记(读完成语义由其
        退出列车的 recv 承载)。

    send 锚两档(总文档 §3.2;源实例主链不被读边污染):
      源 = P(prefill_stay piece,零 3100 边——默认拓扑唯一情形)→ 锚 =
        该成员 prefill 块末节点 id(member["prefill_end_members"]);
      源 = 中间 die(scatter 目标,仅合成拓扑)→ 锚 = 该 piece 的 3100
        recv 节点 id(scatter_recv_ids[源实例],emit_piece_scatter 返回)。
    recv 依赖 = 列车 join 标记(_resolve_join_anchors:显式参数或按
    {train_id}_join_* 名字恢复的每 rank 头末节点),不进 D 主链。

    字节 = p_m × R_{m,s} 精确式(decode 段恒钉 D ⇒ 每迭代远程字节为
    常量;逐 rank 整头 shard × 迭代数)。send hbm_charge=true(源端
    COMM_READ,与其本机负载真实争用)、recv hbm_charge=false(读边不落
    D 的 HBM,全仓首个显式 false)。3300 非 collective,send/recv 仅靠
    (src,dst,tag) 会合。

    返回 {"recv_ids": {request_id: {rank: [recv 节点 id]}}, "routes":
    [...], "join_anchors": {...}}。本函数无跨调用状态:继续成员跨列车
    重复同 tag(R3-②,T_max 截断常态)不做任何唯一性假设——后端按
    注册序(= 每 rank 发射序)FIFO 匹配。
    """

    model = graph.config.model
    decode_group = _require_group(graph, decode_instance_index)
    decode_ranks = tuple(decode_group.ranks)
    tp = len(decode_ranks)
    scatter_lookup = {
        int(index): dict(ranks) for index, ranks in (scatter_recv_ids or {}).items()
    }
    join_anchor_by_rank = _resolve_join_anchors(
        graph, train_id, decode_ranks, join_anchors)

    marker = graph._mark()
    routes: list[dict] = []
    recv_ids: dict[str, dict] = {}
    for member in member_reads:
        request_id = member["request_id"]
        queue_index = member["queue_index"]
        participation = member["participation"]
        if (isinstance(participation, bool)
                or not isinstance(participation, int) or participation <= 0):
            raise ValueError(
                f"member {request_id!r}: participation must be a positive "
                "integer (p_m = iterations of this train)")
        prefill_group = _require_group(
            graph, member["prefill_instance_index"])
        prefill_end_members = dict(member["prefill_end_members"])
        exit_anchors = member.get("exit_anchors")
        member_recv_ids: dict[int, list] = {rank: [] for rank in decode_ranks}
        for position, source in enumerate(member.get("sources", ())):
            source_index = int(source["source_instance_index"])
            piece_tokens = int(source["piece_tokens"])
            source_ordinal = int(source.get("source_ordinal", position))
            if piece_tokens <= 0:
                continue  # 该源零字节(全本地成员):无读边
            if source_index == int(decode_instance_index):
                raise RuntimeError(
                    f"member {request_id!r}: decode instance cannot be a "
                    "remote-read source (D 本地读由 local_kv_bytes 计入, "
                    "3300 源集合 = pieces 非 D 实例集合)")
            source_group = _require_group(graph, source_index)
            if len(source_group.ranks) != tp:
                raise RuntimeError(
                    "WSC-LLM direct KV shard pairing requires equal TP "
                    "degree")
            # send 锚两档解析(见 docstring;非 P 源必须有 3100 recv 锚)。
            if source_index == int(member["prefill_instance_index"]):
                anchors = prefill_end_members
            else:
                anchors = scatter_lookup.get(source_index)
                if anchors is None:
                    raise RuntimeError(
                        f"member {request_id!r}: source instance "
                        f"{source_index} has no 3100 recv anchor "
                        "(scatter_recv_ids misses it)")
            # 字节 = p_m × R_{m,s} 逐 rank 整头 shard(精确式)。
            bytes_by_rank = [
                shard_bytes * participation
                for shard_bytes in _kv_shard_bytes(model, piece_tokens, tp)
            ]
            for relative_rank, (source_rank, decode_rank) in enumerate(
                    zip(source_group.ranks, decode_ranks)):
                anchor = anchors.get(source_rank)
                if anchor is None:
                    raise RuntimeError(
                        f"member {request_id!r}: source instance "
                        f"{source_index} has no data-ready anchor on rank "
                        f"{source_rank}")
                tag = _stage_tag(
                    queue_index, 3300, relative_rank,
                    source_ordinal=source_ordinal, tp_degree=tp)
                name = (
                    f"{train_id}_kv_remote_read_"
                    f"{sanitize_node_prefix(request_id)}_src{source_ordinal}")
                # send:双端侧插之源端——不进源实例主链(否则把该实例
                # 流水线钉在数十 ms 量级的 HBM 读上,总文档 §3.2)。
                graph.builders[source_rank].set_context(
                    request_id, "decode", 1)
                _side_insert(
                    graph.builders[source_rank], int(anchor),
                    partial(
                        graph.builders[source_rank].comm_send,
                        f"{name}_send_rank{source_rank}_to_rank{decode_rank}",
                        src=source_rank, dst=decode_rank,
                        comm_size=bytes_by_rank[relative_rank],
                        comm_tag=tag, hbm_charge=True,
                    ),
                    where=(f"{name}_send_rank{source_rank}_to_rank"
                           f"{decode_rank}"),
                )
                # recv:双端侧插之 D 端——依赖列车 join 标记,不进 D 主链;
                # hbm_charge=False(读边不落 D 的 HBM)。
                graph.builders[decode_rank].set_context(
                    train_id, "decode", 1)
                recv_id = _side_insert(
                    graph.builders[decode_rank],
                    join_anchor_by_rank[decode_rank],
                    partial(
                        graph.builders[decode_rank].comm_recv,
                        f"{name}_recv_rank{source_rank}_to_rank{decode_rank}",
                        src=source_rank, dst=decode_rank,
                        comm_size=bytes_by_rank[relative_rank],
                        comm_tag=tag, hbm_charge=False,
                    ),
                    where=(f"{name}_recv_rank{source_rank}_to_rank"
                           f"{decode_rank}"),
                )
                member_recv_ids[decode_rank].append(recv_id)
                path = _xy_route(
                    graph.config.hardware, source_rank, decode_rank)
                routes.append({
                    "category": 3300,
                    "train_id": train_id,
                    "request_id": request_id,
                    "source_instance_index": source_index,
                    "decode_instance_index": int(decode_instance_index),
                    "source_ordinal": source_ordinal,
                    "relative_shard": relative_rank,
                    "source_rank": source_rank,
                    "decode_rank": decode_rank,
                    "bytes": bytes_by_rank[relative_rank],
                    "participation": participation,
                    "noc_path": path,
                    "noc_hops": len(path) - 1,
                })
        # recv 节点 id 注入该成员自己的 exit 标记(成员 exit = max(体尾,
        # 其全部读边),总文档 §5.4/§5.5;per-member 粒度——按源跨成员
        # 聚合会使短成员被长成员拖尾)。中途成员不注入任何标记。
        # 注:exit 标记已随列车发射(id 更小),此处补一条 recv→exit 的
        # parent edge——与 timer_gate 的 after_node_id 跨节点依赖同构
        # (builder.edges 的既有 dict 结构),不新增 builder 方法。
        if exit_anchors is not None:
            for rank in decode_ranks:
                exit_id = exit_anchors.get(rank)
                if exit_id is None:
                    raise RuntimeError(
                        f"member {request_id!r}: exit anchors miss decode "
                        f"rank {rank}")
                for recv_id in member_recv_ids[rank]:
                    graph.builders[rank].edges.append({
                        "rank": rank,
                        "from": recv_id,
                        "to": int(exit_id),
                        "kind": "data",
                    })
        recv_ids[request_id] = member_recv_ids
    graph._collect(marker)
    return {
        "recv_ids": recv_ids,
        "routes": routes,
        "join_anchors": join_anchor_by_rank,
    }
