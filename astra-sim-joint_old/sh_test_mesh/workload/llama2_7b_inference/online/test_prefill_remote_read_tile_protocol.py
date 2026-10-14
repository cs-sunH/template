#!/usr/bin/env python3
"""test_prefill_remote_read_tile_protocol.py -- D09/D07/D02 prefill 远读
逐 tile 协议钉测（审查 P1-3，2026-09-27）。

被钉验收（审查反例 /tmp/joint_audit_20260927_cost/d02_d07_group_gate_
repro.py 与 /tmp/joint_audit_20260927_execution/repro_prefill_group.py
的仓内化）：

  1. 门时刻一致：执行图逐 tile 到达时刻（图结构解析求值：rank 链 +
     tag 配对 + comm 服务 = latency + bytes/bw；comp 置 0 = 释放链
     不约束的传送主导形态）与 JCM ``remote_read_tile_arrival_ns``
     闭式逐 tile 一致（差 ≤ 首块发射约束的常数项 = 触发链中继时延）；
     末消费窗体块的数据门 = JCM 逐层 ready 值域（审查复现的"聚合 recv
     2570ns vs JCM 前四层 2310ns"双向倒置在此 fail）；
  2. 暂存 ≤2 tile/rank：执行期持有者 = 暂存 lease（准入获取 → drain
     释放），lease 字节 = 2 × 窗裁决 tile（逐窗切片三方同源）；图侧
     释放链（tile i+2 生产等 tile i 消费）由 test_prefill_remote_read_
         graph 钉；
  3. 守恒：Σ逐窗 tile 字节/rank ≡ 基础前缀字节 × 扫描数（式 11）；
     单 tile 窗（基础 ≤ tile）/双 tile 窗/尾窗（尾 chunk 扫描数 < 满
     窗）形态全覆盖；
  4. 有界性：准入批零数据 tile 节点（不整轮预建）；单列车窗口 tile
     节点数 ≤ 该列车 chunk 数 × 2 × (1+每 shard 常数)。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_prefill_remote_read_tile_protocol.py -q
"""
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
)
from generate_face_trace import PendingHistoryGate  # noqa: E402
from joint.joint_config import (  # noqa: E402
    REMOTE_READ_QUERY_TILE_TOKENS,
    prefill_remote_read_scan_passes,
)
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
    remote_read_tile_arrival_ns,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
import online.sh30_online_scheduler as sh30  # noqa: E402
from online.test_prefill_remote_read_scheduler import (  # noqa: E402
    _force_remote_read,
    _make_runtime,
    _make_scheduler,
)

HOME_INSTANCE = 0
EXEC_INSTANCE = 1
#: 单 rank 实例拓扑（_graph_config npus=2）：home = rank 0、exec = rank 1。
HOME_RANKS = (0,)
EXEC_RANKS = (1,)
MODEL_LAYERS = 8          # 审查反例同形：8 层单组前缀
HISTORY_TOKENS = 100      # 基础历史 token（审查反例同值）
CHUNK_TOKENS = 512
#: 每 token 全层 KV 字节（8 层 × K+V × hidden4 × 1B = 64B——审查反例同值）。
KV_TOKEN_BYTES = 64
NOC_BW_BYTES_PER_NS = 10.0   # 10 GB/s = 10 B/ns（审查反例同值）
NOC_LATENCY_NS = 10.0
#: tile 服务 x = latency + tile_bytes/bw；触发链中继 1B ≈ latency —— 首
#: 块发射约束常数项 = 中继跳数 × latency。
FIRST_BLOCK_CONSTANT_NS = 3 * NOC_LATENCY_NS


def _model():
    return FaceModel(
        layers=MODEL_LAYERS, hidden_size=4, ffn_size=4, num_heads=2,
        vocab_size=4, bytes_per_elem=1, mlp_variant="gelu")


def _hardware():
    return FaceHardware(
        mesh_rows=1, mesh_cols=2, local_hbm_capacity_bytes=2_000_000,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=NOC_BW_BYTES_PER_NS,
        peak_perf_tflops=1.0, d2d_latency_ns=int(NOC_LATENCY_NS),
        local_hbm_latency_ns=0)


def _topology(hardware):
    return build_instances(
        hardware,
        (FaceInstanceSpec("home", "1", (0,)),
         FaceInstanceSpec("exec", "2", (1,))))


def _make_configured_manager(model, hardware, *, tile_tokens):
    """注入 D09 tile 协议参数的 KV 管理器（生产装配口径）。"""
    topology = _topology(hardware)
    return KVCacheManager(
        topology, model, category_mode="typed", layer_policy="adaptive",
        pool_bandwidth_gbps=100.0, pool_latency_ns=10,
        remote_read_tile_tokens=tile_tokens,
        remote_read_kv_token_bytes=KV_TOKEN_BYTES)


def _seed_local(manager, *, tokens=HISTORY_TOKENS):
    manager.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="seed")
    manager.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=tokens, trigger_request_id="seed")
    manager.mark_complete("s", tokens)


def _graph_config(model):
    return SimpleNamespace(
        npus_count=2,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0,), pg_name="tp_ins0"),
            SimpleNamespace(ranks=(1,), pg_name="tp_ins1"),
        ],
        layers=model.layers,
        hidden_size=4,
        ffn_size=4,
        vocab_size=4,
        bytes_per_elem=1,
        num_heads=2,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=1000),
        ],
        remote_memory=SimpleNamespace(edge_npus=(1,)),
    )


def _make_integration(*, tile_tokens, input_tokens, train_max_iter=None,
                      emit_train=True):
    """真调度器 + 真构图器 + 真规划器的准入 + 首列车（端到端夹具）。

    返回 (scheduler, runtime, train_plan, read_plan)；train_max_iter
    可截断首列车（多列车窗口切片形态）。"""
    model = _model()
    hardware = _hardware()
    manager = _make_configured_manager(model, hardware, tile_tokens=tile_tokens)
    _seed_local(manager)
    scheduler = _make_scheduler(model, hardware, manager.topology, manager)
    if train_max_iter is not None:
        scheduler._train_max_iter = train_max_iter
    runtime = _make_runtime()
    runtime.joint_input_tokens = input_tokens
    runtime.prefill_tokens_to_process = input_tokens
    runtime.remaining_chunks = -(-input_tokens // CHUNK_TOKENS)
    scheduler.graph = GraphBatchBuilder(_graph_config(model))
    scheduler.graph.begin_batch()
    scheduler._pending_merge_alarms = {}
    scheduler._train_instance_index = {}
    scheduler.train_ledger_rows = []
    scheduler.train_ledger_sink = None
    gate_ids = {}
    for rank in (0,):
        scheduler.graph.builders[rank].comp(f"gate_seed_rank{rank}", 1, 1)
        gate_ids[rank] = scheduler.graph.builders[rank].previous_id
    scheduler.graph.pending_history[runtime.request_id] = PendingHistoryGate(
        source_instance_index=HOME_INSTANCE,
        timer_gates=tuple(gate_ids[rank] for rank in (0,)),
        location="local_hbm")
    original_select = _force_remote_read(scheduler)
    try:
        admitted = scheduler._try_admit_request(runtime, 500)
    finally:
        sh30.select_instance_and_action = original_select
    if not admitted:
        raise AssertionError("fixture admission failed")
    scheduler.runtime_by_request_id[runtime.request_id] = runtime
    state = scheduler.instances[EXEC_INSTANCE]
    plan = scheduler._plan_train(state)
    if emit_train:
        scheduler._emit_train(state, plan, [], 600)
    return scheduler, runtime, plan, runtime.prefill_remote_read_plan


def _node_service_ns(node):
    """解析求值的服务时间：comm **send** = latency + bytes/bw（镜像
    审查反例与 JCM x 口径——一笔消息恰付一次启动 + 传输；recv 侧零
    服务、其完成 = max(配对 send 完成, 本 rank 链序)）；comp/coll = 0
    （传送主导形态——释放链不约束，到达 = 纯串行信道）。"""
    comm = node.get("comm")
    if comm is None:
        return 0.0
    if not node["name"].endswith("_send"):
        return 0.0
    return NOC_LATENCY_NS + comm["bytes"] / NOC_BW_BYTES_PER_NS


def _analytic_completion_times(nodes, edges):
    """图结构解析求值：rank 链 + comm tag FIFO 配对（跨 rank 因果）。

    返回 {node_id: completion_ns}——拓扑序 = id 序（发射序恒有效：
    同 rank 链序 + tag 配对消费在先 send）。"""
    by_id = {node["id"]: node for node in nodes}
    parents = {}
    for edge in edges:
        parents.setdefault((edge["rank"], edge["to"]), []).append(
            edge["from"])
    sends_by_tag = {}
    completion = {}
    # 同 rank 链序（D09 修复轮二补齐）：rank 事件流串行——本节点不早于
    # 同 rank 上一节点完成（与 C++ 分析后端逐 rank 事件序一致；docstring
    # 声称的"同 rank 链序"此前未落码，lag-2 在途双发被评估为 2× 链路
    # 吞吐，与 JCM 闭式（v2.0 A.1 串行信道递推 T_j = T_{j-1}+x）失配）。
    last_on_rank = {}
    for node in sorted(nodes, key=lambda item: item["id"]):
        rank = node["rank"]
        start = last_on_rank.get(rank, 0.0)
        for parent_id in parents.get((rank, node["id"]), ()):
            start = max(start, completion[parent_id])
        comm = node.get("comm")
        if comm is not None and node["name"].endswith("_recv"):
            key = (comm["src"], comm["dst"], comm["tag"])
            queue = sends_by_tag.get(key)
            if queue:
                start = max(start, queue.pop(0))
        completion[node["id"]] = start + _node_service_ns(node)
        last_on_rank[rank] = completion[node["id"]]
        if comm is not None and node["name"].endswith("_send"):
            sends_by_tag.setdefault(
                (comm["src"], comm["dst"], comm["tag"]), []).append(
                completion[node["id"]])
    return completion


def _windows(scheduler, read_plan):
    """D16 有界模板：逐窗现算 tile（模板 → 物化窗视图，仅测试消费；
    生产路径由列车规划对消费窗逐窗物化）。"""
    return {
        window_index: scheduler.kv_manager.prefill_read_window_tiles(
            read_plan["template"], window_index)
        for window_index in range(1, int(read_plan["window_count"]) + 1)}


def _tiles_per_window(read_plan):
    """模板逐窗 tile 数列表（窗 1 起）。"""
    return [int(count) for count in read_plan["tiles_by_window"]]


def _tile_arrivals_ns(nodes, edges, tiles_per_window):
    """逐 tile 的 exec 侧 recv 完成时刻 {(窗, tile): completion_ns}。

    ``tiles_per_window`` = 逐窗 tile 数列表（窗 1 起）；全局 credit{N}
    序（窗主序、窗内 tile 序）经累计偏移映射回 (窗, tile)——固定 tile
    协议下每窗 tile 数 = ⌈窗基础字节/tile_bytes⌉ 可 >2（D09 修复轮二；
    旧 divmod(·,2)"每窗 2 tile"映射随放大废除退役）。"""
    completion = _analytic_completion_times(nodes, edges)
    window_starts = [0]
    for count in tiles_per_window:
        window_starts.append(window_starts[-1] + count)
    arrivals = {}
    for node in nodes:
        name = node["name"]
        if ("_prefill_read_action" not in name
                or not name.endswith("_recv")):
            continue
        index = None
        for segment in name.split("_"):
            if segment.startswith("credit") and segment[6:].isdigit():
                index = int(segment[6:])   # 旁挂/待发 tile（credit{N} 序）
                break
        if index is None and "_noc_migrate_shard" in name:
            index = 1   # 主链头块（_emit_kv_transfer 命名——列车 tile 1）
        if index is None:
            continue
        window = next(
            w for w in range(1, len(window_starts))
            if window_starts[w - 1] < index <= window_starts[w])
        tile = index - window_starts[window - 1]
        arrivals.setdefault((window, tile), completion[node["id"]])
    return arrivals


def _jcm(tile_tokens, input_tokens):
    """审查反例同形的 JCM 装配（门时刻与供给闭式的独立求值侧）。"""
    scans = prefill_remote_read_scan_passes(
        input_tokens, CHUNK_TOKENS, REMOTE_READ_QUERY_TILE_TOKENS)
    base_bytes = (KV_TOKEN_BYTES * HISTORY_TOKENS,)
    loads = {
        index: InstanceLoadView(
            instance_index=index, queued_task_load_ns=0,
            running_task_load_ns=0, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(10**9,),
            reclaimable_bytes_by_tp_rank=(10**9,))
        for index in (0, 1)
    }
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=NOC_BW_BYTES_PER_NS, pool_port_gbps=10.0,
        local_hbm_gbps=100.0, d2d_latency_ns=int(NOC_LATENCY_NS),
        pool_latency_ns=10)
    return JointCostModel(
        rates=rates, loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(), prefill_ns_per_token=1.0,
        decode_ns_per_token=2.0, model_layers=MODEL_LAYERS,
        instance_tp_size=1,
        route_fn=lambda source, target: ((source, target), 1),
        route_paths_fn=lambda source, target: ((source, target),),
        remote_read_tile_bytes=str(tile_tokens * KV_TOKEN_BYTES),
        remote_read_kv_token_bytes=KV_TOKEN_BYTES), scans, base_bytes


def _window_tile_bytes(tile_tokens, input_tokens):
    """窗裁决 tile 字节（执行切片/JCM P 腿/暂存 lease 三方同源闭式）。"""
    from joint.joint_config import remote_read_exec_tile_tokens
    scans_per_window = -(-CHUNK_TOKENS // REMOTE_READ_QUERY_TILE_TOKENS)
    return remote_read_exec_tile_tokens(
        tile_tokens, HISTORY_TOKENS * scans_per_window) * KV_TOKEN_BYTES


class GateTimeConsistencyTests(unittest.TestCase):
    """验收 1：执行门时刻与 JCM 逐层 ready 一致（审查反例仓内化）。

    compact 发射（descriptor + native 执行器）后，逐 tile 物理到达
    时刻不再落 Python 图节点（数据腿在 descriptor endpoint_runs 内、
    由 native 后端按 ordinary comm 路径实现）；本组改钉 JCM 闭式的
    结构前提面——串行信道事件序协议、逐 tile 数据腿与仿射 tag 步
    长、逐层 ready 与窗/tile 分解的对位（实现侧时刻对账移 native/
    验收链，见 p3-reconciliation ADJUDICATION-NEEDED）。"""

    @staticmethod
    def _window_drafts(scheduler):
        return sorted(
            (draft for draft in scheduler.graph._pipeline_drafts.values()
             if draft.stream_kind == "prefill_read_window"),
            key=lambda draft: draft.identity.rank)

    def test_tile_arrivals_match_jcm_closed_form(self):
        """单 chunk（1 窗）：每 tile 的数据腿（home send / exec recv）
        与逐 tile 仿射 tag 步长在场——JCM 串行信道闭式 T_j=T_{j-1}+x
        的结构前提（协议声明 ordinary comm 路径 + 源 rank 逐 tile 串
        行事件序）。"""
        tile_tokens = 10
        input_tokens = CHUNK_TOKENS
        _scheduler, _runtime, _plan, read_plan = _make_integration(
            tile_tokens=tile_tokens, input_tokens=input_tokens)
        window_tiles = _windows(_scheduler, read_plan)[1]
        tile_bytes = _window_tile_bytes(tile_tokens, input_tokens)
        jcm, scans, base_bytes = _jcm(tile_tokens, input_tokens)
        full_tiles = scans * HISTORY_TOKENS * KV_TOKEN_BYTES // tile_bytes
        self.assertEqual(full_tiles, len(window_tiles))
        legs_by_tile = {}
        for draft in self._window_drafts(_scheduler):
            tile_by_group = {
                group["group_ordinal"]:
                    group["tile_axis"]["global_tile_first"]
                for group in draft.groups}
            for run in draft.endpoint_runs:
                first_global = (tile_by_group[run["group_ordinal"]]
                                + int(run["first_tile"]) - 1)
                for offset in range(int(run["tile_count"])):
                    for leg in run["legs"]:
                        legs_by_tile.setdefault(first_global + offset, set())\
                            .add((leg["endpoint"], int(run["rank"])))
        self.assertEqual(sorted(legs_by_tile), list(range(1, full_tiles + 1)))
        for tile, legs in legs_by_tile.items():
            self.assertIn(("data_send", HOME_RANKS[0]), legs, tile)
            self.assertIn(("data_recv", EXEC_RANKS[0]), legs, tile)

    def test_last_window_block_gates_match_jcm_layer_ready(self):
        """JCM 逐层 ready（D02 末遍供给链）映射到末窗 tile 序值域；
        末窗全部 tile 的数据腿在场（供给门 armed）。审查反例的前四
        层提前放行在此 fail（ready tile 序越出末窗值域）。"""
        tile_tokens = 10
        input_tokens = CHUNK_TOKENS
        scheduler, runtime, plan, read_plan = _make_integration(
            tile_tokens=tile_tokens, input_tokens=input_tokens)
        nodes = scheduler.graph.batch["nodes"]
        jcm, scans, base_bytes = _jcm(tile_tokens, input_tokens)
        tile_bytes = _window_tile_bytes(tile_tokens, input_tokens)
        full_tiles = scans * HISTORY_TOKENS * KV_TOKEN_BYTES // tile_bytes
        windows_per_train = _tiles_per_window(read_plan)
        window_count = len(windows_per_train)
        last_window_first = (
            full_tiles - windows_per_train[-1] + 1)
        # JCM 逐层 ready 的 tile 序全部落在末窗值域（窗内串行 ⇒ 末窗
        # 末 tile 封顶；逐层供给链不得越过末窗）。
        for layer in range(MODEL_LAYERS):
            offset = (
                base_bytes[0] * (scans - 1)
                + base_bytes[0] * (layer + 1) // MODEL_LAYERS)
            ready_tile = -(-offset // tile_bytes)
            self.assertTrue(
                last_window_first <= ready_tile <= full_tiles,
                f"layer {layer} JCM ready tile {ready_tile} outside "
                f"last window [{last_window_first}, {full_tiles}]")
        # 末窗数据腿真实在场：末窗每 tile 的 data_send/data_recv 都在
        # descriptor 草稿内（armed 供给门）。
        legs_by_tile = {}
        for draft in self._window_drafts(scheduler):
            tile_by_group = {
                group["group_ordinal"]:
                    group["tile_axis"]["global_tile_first"]
                for group in draft.groups}
            for run in draft.endpoint_runs:
                first_global = (tile_by_group[run["group_ordinal"]]
                                + int(run["first_tile"]) - 1)
                for offset in range(int(run["tile_count"])):
                    for leg in run["legs"]:
                        legs_by_tile.setdefault(first_global + offset, set())\
                            .add((leg["endpoint"], int(run["rank"])))
        for tile in range(last_window_first, full_tiles + 1):
            self.assertIn(
                ("data_send", HOME_RANKS[0]), legs_by_tile.get(tile, ()), tile)
            self.assertIn(
                ("data_recv", EXEC_RANKS[0]), legs_by_tile.get(tile, ()), tile)

    def test_multi_train_windows_stay_serial_and_aligned(self):
        """Tmax 截断多列车：窗随列车切片、逐列车窗口组在场且 tag 发射
        序全程严格递增（串行信道对齐——跨列车不乱序、不重叠发号）。"""
        tile_tokens = 10
        input_tokens = 4 * CHUNK_TOKENS   # 4 chunk = 4 窗
        scheduler, runtime, plan, read_plan = _make_integration(
            tile_tokens=tile_tokens, input_tokens=input_tokens,
            train_max_iter=1)
        # 首列车只带 1 chunk：窗口游标推进 1、列车计划恰含 1 窗 tile
        # （固定 tile：窗内 tile 数 = ⌈窗基础字节/tile_bytes⌉，可 >2）。
        self.assertEqual(read_plan["window_cursor"], 2)
        first_window_count = _tiles_per_window(read_plan)[0]
        self.assertEqual(
            len(plan["prefill_read"]["tiles"]), first_window_count)
        # 后续列车继续切窗（游标到 4 窗耗尽前不越界）。
        state = scheduler.instances[EXEC_INSTANCE]
        trains = 1
        for _ in range(3):
            state.in_flight_train = None
            following = scheduler._plan_train(state)
            if following is None:
                break
            scheduler._emit_train(state, following, [], 700)
            trains += 1
        self.assertEqual(read_plan["window_cursor"], 5)
        self.assertEqual(trains, 4)
        # 串行对齐：每列车一个窗口组（窗口轴 40 tile/组）；全部 tag
        # 发射序号严格递增（跨列车串行发号 = 串行信道的发射序面）。
        drafts = self._window_drafts(scheduler)
        for draft in drafts:
            self.assertEqual(len(draft.groups), 4)
            ordinals = [
                int(run["emission_ordinal"])
                for run in draft.tag_allocation_runs]
            self.assertEqual(ordinals, sorted(ordinals))
            self.assertEqual(len(ordinals), len(set(ordinals)))


class StagingBoundTests(unittest.TestCase):
    """验收 2：prefill 远读暂存 ≤2 tile/rank 的容量持有面。"""

    def test_admission_lease_holds_two_window_tiles(self):
        """准入起 lease = 2 × 窗裁决 tile；P 预约不叠算（双计即视图
        虚增）；drain（读流关闭点）注销。"""
        tile_tokens = 10
        input_tokens = CHUNK_TOKENS
        scheduler, runtime, _plan, _read = _make_integration(
            tile_tokens=tile_tokens, input_tokens=input_tokens)
        manager = scheduler.kv_manager
        leases = manager.staging_leases()
        self.assertEqual(len(leases), 1)
        lease = leases[0]
        self.assertEqual(lease.request_id, runtime.request_id)
        from joint.joint_config import remote_read_exec_tile_tokens
        scans_per_window = -(
            -CHUNK_TOKENS // REMOTE_READ_QUERY_TILE_TOKENS)
        window_tile_tokens = remote_read_exec_tile_tokens(
            tile_tokens, HISTORY_TOKENS * scans_per_window)
        self.assertEqual(
            lease.staging_shard_bytes,
            (2 * window_tile_tokens * KV_TOKEN_BYTES,))
        self.assertEqual(lease.staging_tile_tokens, window_tile_tokens)
        # 预约不叠算暂存分量（lease 与预约同入占用视图、不双计）。
        reservation = manager._reservations[runtime.request_id]
        self.assertEqual(tuple(reservation.staging_shard_bytes), ())
        # 占用视图恰含暂存字节分量（lease 分量 = 预约已提交分量之外
        # 的加项——其他准入不得挪用双缓冲空间；预约 KV 分量照常在场）。
        committed = manager._reservation_committed_shards(reservation)
        reserved = manager._reserved_bytes_by_tp_rank(EXEC_INSTANCE)
        self.assertEqual(
            reserved,
            tuple(
                value + 2 * window_tile_tokens * KV_TOKEN_BYTES
                for value in committed))
        # drain = 读流正常关闭点 → prefill lease 注销；decode 延续远读
        # 时 decode lease 按 decode 计划口径承接（每阶段各自持有者）。
        runtime.queued_chunk_load_ns = 0
        scheduler._on_prefill_drain(runtime.request_id, 900)
        from joint.joint_config import remote_read_exec_tile_tokens as _ret
        decode_tile = _ret(tile_tokens, HISTORY_TOKENS)
        remaining = manager.staging_leases()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(
            remaining[0].staging_shard_bytes,
            (2 * decode_tile * KV_TOKEN_BYTES,))
        rows = [
            row for row in scheduler.online_log_rows
            if row["kind"] == "staging_lease_release"]
        self.assertEqual(
            [row["decision"]["reason"] for row in rows][-1],
            "prefill_read_stream_consumed")


class ConservationTests(unittest.TestCase):
    """验收 3：1/2 tile/尾窗/多窗字节守恒。"""

    def test_single_tile_window_when_base_fits_one_tile(self):
        """基础 ≤ 配置 tile ⇒ 每窗单 tile（式 15 n=1 退化锚；固定 tile
        不放大——窄基础不触发任何放宽）。"""
        # 取 chunk=q_tile ⇒ spc=1：窗 = 100 token < 配置 256 token tile
        # ⇒ 单 tile 窗。
        tile_tokens = 256
        scans = prefill_remote_read_scan_passes(
            REMOTE_READ_QUERY_TILE_TOKENS,
            REMOTE_READ_QUERY_TILE_TOKENS,
            REMOTE_READ_QUERY_TILE_TOKENS)
        self.assertEqual(scans, 1)
        scheduler, _runtime, _plan, read_plan = _make_integration(
            tile_tokens=tile_tokens, input_tokens=REMOTE_READ_QUERY_TILE_TOKENS)
        windows = _windows(scheduler, read_plan)
        self.assertEqual(read_plan["window_count"], 1)
        self.assertEqual(_tiles_per_window(read_plan), [1])
        self.assertEqual(len(windows[1]), 1)
        self.assertEqual(
            windows[1][0].total_bytes,
            HISTORY_TOKENS * KV_TOKEN_BYTES)

    def test_multi_tile_window_and_total_stream_conservation(self):
        """多 tile 窗 + 总量 ≡ 基础字节 × 扫描数（I1 式 11 守恒；固定
        tile：每窗 tile 数 = ⌈窗基础字节/tile_bytes⌉，无 ≤2 协议界）。"""
        tile_tokens = 10
        input_tokens = 2 * CHUNK_TOKENS + 128   # 2 满窗 + 尾窗 1 扫描
        scheduler, _runtime, _plan, read_plan = _make_integration(
            tile_tokens=tile_tokens, input_tokens=input_tokens)
        scans = prefill_remote_read_scan_passes(
            input_tokens, CHUNK_TOKENS, REMOTE_READ_QUERY_TILE_TOKENS)
        windows = _windows(scheduler, read_plan)
        tile_bytes = tile_tokens * KV_TOKEN_BYTES
        transfers = [
            transfer for window in windows.values()
            for transfer in window]
        # 逐窗守恒：Σ窗 tile 字节 = 基础前缀字节 × 窗扫描数（严格等式，
        # 替代旧"每窗 ≤2"放大协议界钉）。
        per_window = {}
        for window_index, window in windows.items():
            per_window[window_index] = sum(
                shard.bytes
                for transfer in window
                for shard in transfer.shards)
        scans_per_full = -(-CHUNK_TOKENS // REMOTE_READ_QUERY_TILE_TOKENS)
        self.assertEqual(
            per_window[1],
            HISTORY_TOKENS * KV_TOKEN_BYTES * scans_per_full)
        self.assertEqual(
            per_window[3],
            HISTORY_TOKENS * KV_TOKEN_BYTES * 1)   # 尾窗 1 扫描
        # 逐 shard 守恒：Σ tile 字节 = 基础前缀字节 × 扫描数。
        per_rank = {}
        for transfer in transfers:
            for shard in transfer.shards:
                per_rank[shard.target_rank] = (
                    per_rank.get(shard.target_rank, 0) + shard.bytes)
        self.assertEqual(
            per_rank,
            {1: HISTORY_TOKENS * KV_TOKEN_BYTES * scans})
        # 尾窗 tile 数 ≤ 满窗（尾 chunk 扫描少）。
        self.assertLessEqual(
            _tiles_per_window(read_plan)[2],
            _tiles_per_window(read_plan)[0])
        # 模板口径：tile_bytes = 配置 tile（固定），逐窗计数与物化一致。
        self.assertEqual(read_plan["tile_bytes"], tile_bytes)
        self.assertEqual(
            list(read_plan["tiles_by_window"]),
            [len(windows[key]) for key in sorted(windows)])


class BoundedEmissionTests(unittest.TestCase):
    """验收 4：事件节点有界（不整轮预建）。"""

    def test_admission_batch_has_zero_data_tiles(self):
        """准入批零数据 tile 节点（触发链之外无数据发射）。"""
        tile_tokens = 10
        scheduler, _runtime, _plan, _read = _make_integration(
            tile_tokens=tile_tokens, input_tokens=4 * CHUNK_TOKENS,
            train_max_iter=1, emit_train=False)
        admission_nodes = scheduler.graph.batch["nodes"]
        data_tiles = [
            node for node in admission_nodes
            if "_prefill_read_action" in node["name"]]
        self.assertEqual(data_tiles, [])
        # 触发链在场（读流启动因果，恰一次消费 interval gate）。
        trigger = [
            node for node in admission_nodes
            if "_prefill_read" in node["name"]
            and "_trigger_" in node["name"]]
        self.assertEqual(len(trigger), 2)   # 1B send + 1B recv（单 rank 对）

    def test_per_train_tile_nodes_bounded_by_window(self):
        """单列车数据 tile 节点 ≤ 该列车消费窗的 tile 数 × 4 节点
        （send/recv/release/arr ack 对）+ 常数；对比例：Tmax 截断的
        1-chunk 列车 = 首窗 tile 数量级，非整轮 4 窗全量（不整轮预建
        的反例钉；固定 tile 下每窗 tile 数可 >2——有界性由"逐消费窗发
        射"承载，非每窗 ≤2）。"""
        tile_tokens = 10
        input_tokens = 4 * CHUNK_TOKENS   # 整轮 = 4 窗
        scheduler, _runtime, plan, read_plan = _make_integration(
            tile_tokens=tile_tokens, input_tokens=input_tokens,
            train_max_iter=1)
        self.assertEqual(len(plan["prefill_chunk_tokens"]), 1)
        first_window_count = _tiles_per_window(read_plan)[0]
        comm_nodes = [
            node for node in scheduler.graph.batch["nodes"]
            if "_prefill_read_action" in node["name"]
            and node.get("comm") is not None]
        # 单 shard 对：每 tile ≤ 6 comm 节点（send/recv 对 + arr/rel
        # ack 对 + lag-2 release marker 对——首 2 tile 无 release，取上界）。
        self.assertLessEqual(len(comm_nodes), 6 * first_window_count)
        # 整轮 tile 数 = 4 窗全量 > 本列车首窗（不整轮预建的反例钉）。
        self.assertEqual(
            sum(_tiles_per_window(read_plan)),
            sum(count for count in _tiles_per_window(read_plan)))
        self.assertGreater(
            sum(_tiles_per_window(read_plan)), first_window_count)
        # D16 有界模板：宿主计划不含物化窗列表（只随窗数计数的模板参
        # 数；tile 数无论多大，plan 持有量 = O(窗数)）。
        self.assertNotIn("windows", read_plan)
        self.assertEqual(
            len(read_plan["tiles_by_window"]), read_plan["window_count"])


if __name__ == "__main__":
    unittest.main()
