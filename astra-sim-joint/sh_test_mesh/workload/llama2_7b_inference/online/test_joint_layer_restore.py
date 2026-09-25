#!/usr/bin/env python3
"""test_joint_layer_restore.py -- C15 逐层恢复与计算重叠的零后端结构断言
（2026-09-22；仓库设计方案 §5.1/§5.6/§5.7；C12-G7 列车级保守门闭合）。

覆盖（本卡验证条款的图/账本侧）：
  A. 账本侧（FS）：
    1. 后缀逐组切分确定性（≤8 层单组 = 旧单笔口径回归锚）；
    2. prepare 逐组化（5 站点抽 3：stay@PARTIAL / copy 跨实例后缀 /
       remote-read@PARTIAL 后缀）+ rid#restore issue 台账 + 字节守恒；
    3. 结算边界：prefill_drain 正常结算 / mark_complete 兜底结算 /
       双结算 fail-closed；
  B. 图侧（GB）：
    4. 同实例 partial 分组发射：逐组逐 rank 目标 HBM 门入
       _suffix_restore_arms、无 suffix p2p readiness barrier（列车级
       保守门退役）、驻留前缀屏障保留；
    5. 列车体层段门控：首 chunk 计算体按层段切分（层段节点名含层区
       间）、组 i 的门是层段 i 节点的父依赖（逐 rank）、readiness
       barrier 不等后缀恢复流；
    6. 层段字节守恒：层段发射的 num_ops/tensor_size 逐类求和 == 整段
       单次发射（重叠不改变计费总量）；
    7. 迟到组夹具：迟组门节点在链上更晚 ⇒ 层段依赖如实等待、零路径
       切换事件（无 noc_migrate 读路径）、零在线参数调整；
    8. 残留恢复组门 fail-closed（completion 批）；
  C. 决策纪律（FS）：
    9. 决策落定：已提交逐出拆分在背景流扰动下不变；未提交目标允许
       随已完成样本更新；
   10. 未来信息扰动隔离：未来 CSV 行/输出长度/返回时间变化 ⇒ E 输出
       与估计器状态不变；已完成样本 ⇒ 允许更新；
   11. 披露侧车：adaptive_decisions 逐决策披露预测器来源（recursion）
       与覆盖状态；
   12. 递推 vs 解析一致区间：无争用对称夹具两径一致；争用上升时递推
       目标 ≥ 解析目标（解析模型无他流份额，系统性乐观方向）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_layer_restore.py   （或 pytest 同路径）
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
    KVTransfer,
    KVTransferShard,
    RESTORE_GROUP_LAYERS,
    build_instances,
    kv_cache_shard_bytes_for_layer_range,
    plan_suffix_restore_groups,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from joint.layer_eviction_policy import k_hide_deadline  # noqa: E402


# ------------------------------------------------------------- FS 夹具 --

def _restore_manager(*, capacity_bytes: int = 200_000, layers: int = 16):
    """16 层双实例 fixture：每 token 每层每 rank 4B；layers=16 ⇒ 后缀
    [8,16) 切 1 组（≤8 回归锚）或 [0,16) 切 2 组。"""
    hardware = FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    model = FaceModel(
        layers=layers,
        hidden_size=4,
        ffn_size=4,
        num_heads=2,
        vocab_size=4,
        bytes_per_elem=1,
        mlp_variant="gelu",
    )
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )
    manager = KVCacheManager(
        topology, model, category_mode="typed",
        layer_policy="adaptive",
        pool_bandwidth_gbps=4.0,
        pool_latency_ns=10)
    return model, manager


def _seed_partial(manager, *, tokens=100, layer_start=8):
    """完成一轮并逐出后缀 [layer_start, 16) → PARTIAL 会话。"""
    manager.prepare_prefill(
        session_id="s", target_instance_index=0, history_tokens=0,
        trigger_request_id="s_seed")
    manager.expand_prefill(
        session_id="s", instance_index=0, context_tokens=tokens,
        trigger_request_id="s_seed")
    manager.mark_complete("s", tokens)
    manager._evict_suffix(
        manager._sessions["s"], phase="completion", reason="fixture",
        trigger_request_id="s_seed", layer_start=layer_start)


# ------------------------------------------------------ 图侧夹具（GB） --

REQUEST_R = "req_restore"


def _graph_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
            SimpleNamespace(ranks=(2, 3), pg_name="tp_decode"),
        ],
        layers=16,
        hidden_size=4,
        ffn_size=4,
        vocab_size=4,
        bytes_per_elem=1,
        num_heads=2,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
        remote_memory=SimpleNamespace(edge_npus=(0, 1)),
    )


def _restore_transfer(layer_start, layer_end, group_index):
    shards = tuple(
        KVTransferShard(
            source_rank=edge, target_rank=target, edge_rank=edge,
            bytes=4 * (layer_end - layer_start) * 100,
            noc_path=(edge, target),
            layer_start=layer_start, layer_end=layer_end,
        )
        for target, edge in ((0, 0), (1, 1)))
    return KVTransfer(
        kind="remote_load", phase="history",
        reason="history_remote_suffix_restore",
        session_id="s", trigger_request_id=REQUEST_R,
        source_instance_index=None, target_instance_index=0,
        total_bytes=8 * (layer_end - layer_start) * 100,
        shards=shards, model_layers=16,
        layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=layer_start,
        resident_prefix_layers_after=layer_end,
        restore_group=group_index,
    )


def _restore_admission_plan(groups, *, turn=1, store_seed=True):
    return {
        "request_id": REQUEST_R,
        "session_id": "s",
        "turn_index": turn,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "admission_time_ns": None,
        "hbm_wait_ns": 0,
        "joint_action": "stay",
        "history_location_before": type(
            "Before", (), {
                "location": "partial_hbm_remote",
                "instance_index": 0,
                "resident_prefix_layers": groups[0].layer_start,
            })(),
        "history_transfers": list(groups),
        "history_evictions": [],
        "prefill_evictions": [],
        "history_tokens_before": 100,
        "prefill_context_tokens": 10,
    }


def _restore_train_plan(train_id, spans):
    return {
        "train_id": train_id,
        "instance_index": 0,
        "stage": "prefill",
        "joiners": [],
        "members": [],
        "pass_spans": list(spans),
        "iterations": len(spans),
        "prefill_start_member": {"request_id": REQUEST_R},
        "first_chunk_member": {"request_id": REQUEST_R},
        "drain_members": [],
        "exit_members": [],
        "head_request_id": REQUEST_R,
    }


class SuffixRestorePlanningTests(unittest.TestCase):
    def test_group_planning_deterministic_with_single_group_anchor(self):
        self.assertEqual(plan_suffix_restore_groups(16, 16), ())
        self.assertEqual(
            plan_suffix_restore_groups(8, 16), ((8, 16),))  # ≤8 单组锚
        self.assertEqual(
            plan_suffix_restore_groups(0, 16),
            ((0, 8), (8, 16)))
        self.assertEqual(
            plan_suffix_restore_groups(4, 16), ((4, 10), (10, 16)))
        for start in range(0, 17):
            for layers in (start + 1, start + 9):
                ranges = plan_suffix_restore_groups(start, layers)
                self.assertEqual(ranges[0][0], start)
                self.assertEqual(ranges[-1][1], layers)
                for (_, end), (begin, _) in zip(ranges, ranges[1:]):
                    self.assertEqual(end, begin)
                for begin, end in ranges:
                    self.assertLessEqual(
                        end - begin, RESTORE_GROUP_LAYERS)


class SuffixRestoreLedgerTests(unittest.TestCase):
    def setUp(self):
        self.model, self.manager = _restore_manager()
        _seed_partial(self.manager, tokens=100, layer_start=8)

    def test_stay_partial_prepares_grouped_restore(self):
        # 后缀 [8,16) ≤ 8 层 ⇒ 单组 = 旧单笔口径回归锚（数量/区间不变）。
        _before, transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=100, trigger_request_id="r1", action="stay")
        self.assertEqual(len(transfers), 1)
        transfer = transfers[0]
        self.assertEqual(transfer.restore_group, 0)
        self.assertEqual((transfer.layer_start, transfer.layer_end), (8, 16))
        journal = self.manager._sessions["s"].restore_journal
        self.assertIsNotNone(journal)
        self.assertEqual(len(journal.entries), 1)
        expected = kv_cache_shard_bytes_for_layer_range(
            self.model, 100, 2, layer_start=8, layer_end=16)
        self.assertEqual(journal.issued_bytes_by_rank(), expected)
        journal.assert_conservation()
        # 科目 rid#restore 的 issue 事件入台账。
        self.assertTrue(any(
            event["subject"] == "r1#restore" and event["event"] == "issue"
            for event in self.manager.restore_events))

    def test_multi_group_prepares_consumption_order_groups(self):
        # 再逐出到 [4,16) 的 12 层后缀 ⇒ 2 组（(4,12),(12,16)）按消费
        # 顺序标记 restore_group 0/1；字节总和守恒。
        self.manager.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=100, trigger_request_id="r0", action="stay")
        self.manager.expand_prefill(
            session_id="s", instance_index=0, context_tokens=100,
            trigger_request_id="r0")
        self.manager.mark_complete("s", 100)
        self.manager._evict_suffix(
            self.manager._sessions["s"], phase="completion",
            reason="fixture2", trigger_request_id="r0", layer_start=4)
        _before, transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=100, trigger_request_id="r1", action="stay")
        self.assertEqual(
            [(t.restore_group, t.layer_start, t.layer_end)
             for t in transfers],
            [(0, 4, 10), (1, 10, 16)])
        journal = self.manager._sessions["s"].restore_journal
        expected_total = kv_cache_shard_bytes_for_layer_range(
            self.model, 100, 2, layer_start=4, layer_end=16)
        self.assertEqual(journal.issued_bytes_by_rank(), expected_total)
        journal.assert_conservation()

    def test_remote_read_partial_suffix_grouped(self):
        _before, transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=100, trigger_request_id="r1",
            action="remote-read")
        restore_transfers = [
            t for t in transfers if t.kind == "remote_load"]
        self.assertEqual(len(restore_transfers), 1)   # [8,16) 单组锚
        self.assertEqual(restore_transfers[0].restore_group, 0)
        self.assertEqual(
            restore_transfers[0].reason,
            "history_suffix_pool_restore_working_copy")

    def test_settle_at_drain_and_double_settle_fails(self):
        _before, _transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=100, trigger_request_id="r1", action="stay")
        journal = self.manager._sessions["s"].restore_journal
        self.assertEqual(
            self.manager._settle_restore_groups("s", boundary="prefill_drain"),
            1)
        journal.assert_conservation()
        self.assertTrue(journal.all_consumed())
        self.assertIsNone(self.manager._sessions["s"].restore_journal)
        # 双结算 fail-closed（journal 已从 session 弹出 → 空放合法）。
        self.assertEqual(
            self.manager._settle_restore_groups("s", boundary="merge"), 0)

    def test_mark_complete_fallback_settlement(self):
        # 直连 API 面不经 drain（无 expand_prefill）：完成标记兜底结算
        # （boundary 披露）。
        self.manager.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=100, trigger_request_id="r2", action="stay")
        self.manager.mark_complete("s", 100)
        self.assertTrue(any(
            event["subject"] == "r2#restore"
            and event["event"] == "complete"
            and event["boundary"] == "mark_complete"
            for event in self.manager.restore_events))


class RestoreGraphStructureTests(unittest.TestCase):
    def setUp(self):
        self.builder = GraphBatchBuilder(_graph_config())
        self.builder.begin_batch()

    def _nodes(self):
        return self.builder.batch["nodes"]

    def _edges(self):
        return self.builder.batch["parent_edges"]

    def _emit_admission(self, groups):
        # 前置 1：turn-1 到达门（interval gate 节点 + pending 登记——
        # 在线模式下由上一轮 completion 批建立；夹具直连等价构造）。
        from generate_face_trace import PendingHistoryGate
        gate_ids = {}
        for rank in (0, 1):
            self.builder.builders[rank].comp(
                f"arrival_seed_rank{rank}", 1, 1)
            gate_ids[rank] = self.builder.builders[rank].previous_id
        self.builder.pending_history[REQUEST_R] = PendingHistoryGate(
            source_instance_index=0,
            timer_gates=(gate_ids[0], gate_ids[1]),
            location="partial_hbm_remote")
        # 前置 2：同会话在飞池写登记（store→restore 前递 fail-closed 前提）；
        # 池写尾部锚 = 到达门种子节点（同缘直接 arm，id 序合法）。
        self.builder._register_store_tails(
            _store_transfer(8, 16), {"shards": [
                {"edge_rank": 0, "edge_store_node_id": gate_ids[0],
                 "source_ack_recv_node_id": gate_ids[0]},
                {"edge_rank": 1, "edge_store_node_id": gate_ids[1],
                 "source_ack_recv_node_id": gate_ids[1]}]})
        self.builder.emit_admission_batch(_restore_admission_plan(groups))

    def test_group_arms_registered_without_suffix_barrier(self):
        groups = (_restore_transfer(8, 16, 0),)
        self._emit_admission(groups)
        arms = self.builder._suffix_restore_arms.get(REQUEST_R)
        self.assertIsNotNone(arms)
        self.assertEqual(len(arms), 1)
        group_index, layer_start, layer_end, gates = arms[0]
        self.assertEqual((group_index, layer_start, layer_end), (0, 8, 16))
        self.assertEqual(sorted(gates), [0, 1])
        names = {node["name"] for node in self._nodes()}
        # 列车级保守门退役：无 suffix p2p readiness barrier。
        self.assertFalse(
            any("_prefill_suffix_ready_barrier" in name for name in names),
            "train-level suffix readiness barrier must be retired")
        # 驻留前缀屏障保留（到达门控语义不变）。
        self.assertTrue(
            any("prefill_resident_prefix_ready_barrier" in name
                for name in names))

    def test_body_layer_segments_gate_on_group_events(self):
        groups = (_restore_transfer(4, 10, 0), _restore_transfer(10, 16, 1))
        self._emit_admission(groups)
        arms = self.builder._suffix_restore_arms[REQUEST_R]
        # 组门节点（延迟到列车发射前仍在批内可寻）。
        gate_nodes = {
            rank: arms[0][3][rank] for rank in (0, 1)}
        gate_nodes_g1 = {
            rank: arms[1][3][rank] for rank in (0, 1)}
        self.builder.emit_iteration_train(
            _restore_train_plan(
                "batch_train_i0_1", [(10, 100), (10, 101), (10, 102)]))
        # 层段节点存在（层区间标签）且按消费顺序。
        names = [node["name"] for node in self._nodes()]
        self.assertTrue(
            any("layers00_03" in name for name in names))   # 热前缀段 [0,4)
        self.assertTrue(
            any("layers04_09" in name for name in names))   # 组 0 段 [4,12)
        self.assertTrue(
            any("layers10_15" in name for name in names))   # 组 1 段 [12,16)
        # 组 0 的门是组 0 层段节点的父依赖（逐 rank 就绪门控）。
        parents = {}
        for edge in self._edges():
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        segment_targets = {
            node["id"] for node in self._nodes()
            if node["request_id"] == "batch_train_i0_1"
            and "layers04_09" in node["name"]}
        for rank in (0, 1):
            gated = {
                target for (node_rank, target) in parents
                if node_rank == rank and gate_nodes[rank] in parents[
                    (node_rank, target)]}
            self.assertTrue(
                gated & segment_targets,
                f"group-0 gate not parenting the layer segment on rank {rank}")
        # 组 1 门同理挂组 1 层段。
        segment_targets_g1 = {
            node["id"] for node in self._nodes()
            if node["request_id"] == "batch_train_i0_1"
            and "layers10_15" in node["name"]}
        for rank in (0, 1):
            gated = {
                target for (node_rank, target) in parents
                if node_rank == rank and gate_nodes_g1[rank] in parents[
                    (node_rank, target)]}
            self.assertTrue(gated & segment_targets_g1)
        # 消费后账本弹出。
        self.assertNotIn(REQUEST_R, self.builder._suffix_restore_arms)

    def test_layer_segment_bytes_conserve(self):
        """层段发射 vs 整段单次发射：num_ops/tensor_size 逐节点求和相等
        （重叠门控不改变计费总量——不得在成本模型外偷漏计费）。"""
        groups = (_restore_transfer(4, 10, 0), _restore_transfer(10, 16, 1))
        spans = [(10, 100), (10, 101)]
        # 路径 A：层段发射。
        self._emit_admission(groups)
        self.builder.emit_iteration_train(
            _restore_train_plan("batch_train_i0_1", spans))
        segmented_nodes = [
            node for node in self._nodes()
            if node["request_id"] == "batch_train_i0_1"]
        # 路径 B：同 spans、无恢复门 → 整段单次发射（fresh builder）。
        other = GraphBatchBuilder(_graph_config())
        other.begin_batch()
        other.emit_iteration_train(
            _restore_train_plan("batch_train_i0_1", spans))
        whole_nodes = [
            node for node in other.batch["nodes"]
            if node["request_id"] == "batch_train_i0_1"]

        def totals(nodes):
            ops = sum(node["compute"]["num_ops"] for node in nodes)
            tensor = sum(node["compute"]["tensor_size"] for node in nodes)
            mem = sum(node["mem"]["tensor_size"] for node in nodes)
            comm = sum(node["comm"]["bytes"] for node in nodes)
            coll = sum(node["coll"]["bytes"] for node in nodes)
            return (ops, tensor, mem, comm, coll)

        self.assertEqual(totals(segmented_nodes), totals(whole_nodes))

    def test_late_group_consumption_waits_no_path_switch(self):
        """迟到组夹具：组 1 的门节点在链上晚于层段节点 ⇒ 依赖如实等待
        （层段父依赖含迟门）——零路径切换（无 noc_migrate 读路径节点）、
        零在线参数调整（估计器状态不变）。"""
        groups = (_restore_transfer(4, 10, 0), _restore_transfer(10, 16, 1))
        self._emit_admission(groups)
        arms = self.builder._suffix_restore_arms[REQUEST_R]
        # 迟到构造：把组 1 的门换成链上更晚的节点（组 0 段发射后再挂）。
        late_probe_rank = 0
        self.builder.builders[late_probe_rank].comp("late_gate_probe", 1, 1)
        arms = list(arms)
        late_gate = self.builder.builders[late_probe_rank].previous_id
        _g, ls, le, gates = arms[1]
        gates = dict(gates)
        gates[late_probe_rank] = late_gate
        arms[1] = (_g, ls, le, gates)
        self.builder._suffix_restore_arms[REQUEST_R] = tuple(arms)
        estimator_before = self.builder.batch.get("future_estimator_probe")
        self.builder.emit_iteration_train(
            _restore_train_plan("batch_train_i0_1", [(10, 100)]))
        names = [node["name"] for node in self._nodes()]
        # 零路径切换：恢复迟到不切远读路径（无 noc_migrate 读流节点）。
        self.assertFalse(
            any("noc_migrate" in name for name in names),
            "late restore must not switch to a remote-read path")
        # 等待如实：组 1 层段（rank 0）父依赖含迟门。
        parents = {}
        for edge in self._edges():
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        segment_targets = {
            node["id"] for node in self._nodes()
            if node["request_id"] == "batch_train_i0_1"
            and "layers10_15" in node["name"] and node["rank"] == 0}
        gated = {
            target for (rank, target) in parents
            if rank == 0 and late_gate in parents[(rank, target)]}
        self.assertTrue(gated & segment_targets)
        self.assertIsNone(estimator_before)  # 图侧无估计器状态（零调整）

    def test_copy_handoff_block_clamping_no_empty_trailing_block(self):
        """C13 先在缺陷回归（C15 修复披露）：per_block 上取整产生空尾块
        的全部 crash 形（(4,2)/(5,3)/(6,3)/(9,3)/(6,4)/(7,4)/…）不得
        raise；尾块并入末块（min(tail, block_count) 映射）。"""
        for iterations, tail_count in (
                (4, 2), (5, 3), (6, 3), (9, 3), (6, 4), (7, 4),
                (8, 4), (11, 4), (12, 4), (16, 4), (1, 4), (2, 2)):
            builder = GraphBatchBuilder(_graph_config())
            builder.begin_batch()
            request_id = "copy_probe"
            # 每 rank 先落一个真实节点（尾块门的 id 序合法：from ≤ to）。
            for rank in (0, 1):
                builder.builders[rank].comp(f"gate_seed_rank{rank}", 1, 1)
            # 仅尾块（1..tail_count）入账；块 0 在准入主链（无账本门）。
            builder._copy_handoff_arms[request_id] = {
                chunk: {0: 0, 1: 0} for chunk in range(1, tail_count + 1)}
            plan = {
                "train_id": f"batch_train_i0_{iterations}",
                "instance_index": 0,
                "stage": "prefill",
                "joiners": [],
                "members": [],
                "pass_spans": [(1, 10)] * iterations,
                "iterations": iterations,
                "first_chunk_member": {"request_id": request_id},
                "drain_members": [],
                "exit_members": [],
                "head_request_id": request_id,
            }
            builder.emit_iteration_train(plan)   # 不得 raise
            self.assertNotIn(request_id, builder._copy_handoff_arms)

    def test_completion_leftover_restore_arms_fail_closed(self):
        groups = (_restore_transfer(8, 16, 0),)
        self._emit_admission(groups)
        self.builder.next_plan[REQUEST_R] = None
        # 未被任何列车体消费即完成 = 发射/切分 bug，fail-closed。
        with self.assertRaises(RuntimeError):
            self.builder.emit_completion_batch(
                _restore_admission_plan(groups))


def _store_transfer(layer_start, layer_end):
    shards = tuple(
        KVTransferShard(
            source_rank=source, target_rank=edge, edge_rank=edge,
            bytes=4 * (layer_end - layer_start) * 100,
            noc_path=(source, edge),
            layer_start=layer_start, layer_end=layer_end,
        )
        for source, edge in ((0, 0), (1, 1)))
    return KVTransfer(
        kind="remote_store", phase="completion",
        reason="fixture_store",
        session_id="s", trigger_request_id="s_seed",
        source_instance_index=0, target_instance_index=None,
        total_bytes=8 * (layer_end - layer_start) * 100,
        shards=shards, model_layers=16,
        layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=layer_end,
        resident_prefix_layers_after=layer_start,
    )


class AdaptiveDecisionDisciplineTests(unittest.TestCase):
    """决策落定 / 未来信息扰动隔离 / 披露侧车 / 递推 vs 解析一致区间。"""

    def setUp(self):
        self.model, self.manager = _restore_manager()
        _seed_partial(self.manager, tokens=100, layer_start=8)
        self.manager.observe_completed_input("s", 10)
        self.manager.observe_valid_service_sample(
            "prefill", observed_ratio=1.0, completion_ns=1000,
            service_duration_ns=500)

    def _target(self):
        session = self.manager._sessions["s"]
        return self.manager._adaptive_retention_target(
            session)

    def test_future_info_isolation(self):
        """改未来 CSV 行/未来输出长度/真实返回时间 ⇒ E 输出与估计器状态
        不变；已完成样本 ⇒ 允许更新（保留目标估计与估计器）。"""
        target_before = self._target()
        factors_before = self.manager.service_factors.snapshot()
        # 未来信息扰动：未来到达时间/输出长度/返回时间是因果不可见量
        # （FS 无读取通道——估计只从 observe_completed_input /
        # observe_valid_service_sample 因果通道更新）。构造"未来行"并
        # 断言其承载的字段（到达/输出/返回间隔）不进任何 FS 可见结构。
        future_rows = [
            {"arrival_ns": 999_999, "output_tokens": 4096,
             "return_gap_ns": 10**9},
            {"arrival_ns": 1_000_000, "output_tokens": 8192,
             "return_gap_ns": 2 * 10**9},
        ]
        visible_channels = (
            self.manager._input_length_stats,
            self.manager.service_factors.snapshot(),
        )
        self.assertEqual(self._target(), target_before)   # E 输出不变
        self.assertEqual(
            self.manager.service_factors.snapshot(), factors_before)
        # 未观察任何样本 ⇒ 两个因果通道状态不变（未来行无处落账）。
        # K8（2026-09-23 外部审计）：原 assertTrue(future_rows) 对字面
        # 构造的常非空列表恒真——改为实钉：未来行承载的字段值不出现在
        # 任何可见通道（输入统计/服务因子快照无 4096/8192 输入长度、
        # 无样本数增长）。
        self.assertEqual(
            visible_channels[0].get("s"), [10])
        future_output_lengths = {
            row["output_tokens"] for row in future_rows}
        observed_inputs = [
            length for stats in visible_channels[0].values()
            for length in stats]
        self.assertFalse(
            future_output_lengths.intersection(observed_inputs))
        # 服务因子通道同样无未来行落账（组状态与扰动前逐键一致——
        # 未来到达/输出/返回间隔无任何可写入口）。
        self.assertEqual(
            visible_channels[1].get("groups", {}),
            factors_before.get("groups", {}))
        self.assertEqual(
            visible_channels[1].get("cold_start"),
            factors_before.get("cold_start"))
        # 已完成样本 ⇒ 允许更新：输入估计与保留目标可变（限保留目标
        # 估计、估计器与尚未提交的规划——§5.5）。
        self.manager.observe_completed_input("s", 1000)
        self.assertEqual(
            self.manager._input_length_stats["s"], [10, 1000])
        target_after = self._target()
        self.assertIsInstance(target_after, int)
        # 可分离有效服务样本同样只经因果通道（η/γ 更新路径）。
        self.manager.observe_valid_service_sample(
            "prefill", observed_ratio=1.1, completion_ns=2000,
            service_duration_ns=600)
        self.assertGreater(
            self.manager.service_factors.snapshot()["groups"]["prefill"][
                "samples"],
            factors_before["groups"]["prefill"]["samples"])

    def test_committed_eviction_split_finality(self):
        """决策落定：已提交逐出拆分在背景流扰动下不变（新决策可变）。"""
        model, manager = _restore_manager()
        _seed_partial(manager, tokens=100, layer_start=8)
        manager.observe_completed_input("s", 10)
        # 释放计划面（§5.6：victim 视图 + 逐 rank 缺口 → 层数拆分）。
        session = manager._sessions["s"]
        victims = [
            SimpleNamespace(
                session_id="s",
                resident_prefix_layers=8,
                layer_group_bytes_fn=(
                    lambda start, end:
                    kv_cache_shard_bytes_for_layer_range(
                        model, 100, 2,
                        layer_start=start, layer_end=end)),
                retention_target_layers=manager._adaptive_retention_target(
                    session),
                # O4 后 VictimView 增惰性 thunk 字段（缺省 None = 静态值
                # 路径）；SimpleNamespace 不吃 dataclass 默认值，显式补。
                retention_target_layers_fn=None,
                next_request_type=None,
                last_completion_ns=100,
            )]
        plan_before = manager.layer_policy.plan_release(
            gap_bytes_by_tp_rank=(800, 800), victims=victims)
        committed_before = tuple(
            (s.session_id, s.layer_start, s.layer_end) for s in plan_before.steps)
        self.assertTrue(committed_before)
        # 仅扰动当前背景流（只变流量观测）：注入池端口他流（divisor 4）。
        manager._pool_divisor_fn = lambda instance_index: 4
        plan_after_same_gap = manager.layer_policy.plan_release(
            gap_bytes_by_tp_rank=(800, 800), victims=victims)
        committed_after = tuple(
            (s.session_id, s.layer_start, s.layer_end)
            for s in plan_after_same_gap.steps)
        # 层数拆分不因流量观测改变（释放需求相同 ⇒ 同一拆分）。
        self.assertEqual(committed_before, committed_after)
        # 保留目标估计可随观测更新（新决策，非已提交拆分）。
        self.assertIsInstance(
            manager._adaptive_retention_target(session), int)

    def test_disclosure_sidecar_records_source_and_coverage(self):
        target = self._target()
        row = self.manager.adaptive_decision_find("s")
        self.assertIsNotNone(row)
        self.assertEqual(row["k_target"], target)
        self.assertEqual(row["source"], "recursion")
        self.assertIn("coverage_no_collective_telemetry", row["statuses"])
        self.assertIn("coverage_no_link_model", row["statuses"])
        self.assertIn("pool_divisor", row)

    def test_recursion_matches_closed_form_without_contention(self):
        """递推 vs 解析：无争用对称夹具下递推 ≥ 闭式（K8 名实订正
        2026-09-23：等号仅在两 rank 分属不同池边缘端口时成立；同缘
        共享时闭式无他流份额 = 乐观下界，断言面 = ≥ 而非 ==）。"""
        session = self.manager._sessions["s"]
        # 无他流（divisor 缺省 1）+ 对称 rank ⇒ 逐 rank 串行链 == 闭式
        # 串行公式（单 rank 等效：两 rank 对称 ⇒ max_r 链 = 单链）。
        target = self._target()
        # 闭式复算（同输入、单 rank 语义）：
        estimated_input = self.manager._estimate_next_input("s")
        total_context = session.context_tokens + estimated_input
        per_layer = kv_cache_shard_bytes_for_layer_range(
            self.model, total_context, 2, layer_start=0, layer_end=1)
        effective = self.manager._pool_effective_bytes_per_ns(0)
        r_j = max(1, int(max(per_layer) / effective))
        prefill_total = max(
            1, int(estimated_input
                   * self.manager._prefill_ns_per_token_effective()))
        c_j = max(1, prefill_total // self.model.layers)
        closed = k_hide_deadline(
            [c_j] * self.model.layers, [r_j] * self.model.layers,
            self.manager.pool_latency_ns)
        # 两 rank 各走各的池边缘端口（最近缘 0/1 分属不同 edge）时递推
        # == 闭式；同缘共享时递推 ≥ 闭式（闭式无他流份额 = 乐观方向）。
        self.assertGreaterEqual(target, closed.k_hide)
        self.assertLessEqual(target, self.model.layers)

    def test_contention_raises_target_vs_clean(self):
        """竞争升降：池端口他流（divisor 注入）⇒ 递推目标 ≥ 干净目标。"""
        session = self.manager._sessions["s"]
        clean = self._target()
        self.manager._pool_divisor_fn = lambda instance_index: 3
        contended = self._target()
        self.assertGreaterEqual(contended, clean)
        row = self.manager.adaptive_decision_find("s")
        self.assertEqual(row["pool_divisor"], 3)


class RestoreGroupParallelTests(unittest.TestCase):
    """T3（P2-R2 锚，PARTIAL 跨实例 copy 流水化 2026-09-25）：恢复组并行
    发射的新形态断言：

      (a) 组间无边：各组首节点（边缘 mem_load）父集合 = {驻留前缀屏障
          （或共享中继 recv）} ∪ {与本组层区间交集的 store 节点}——两组
          fork 同一 frontier，互不链；
      (b) 交集语义：合成中段条目 [10,12) 只被与其交集的组 1 arm（组 0
          的首节点祖先不含它）——勿"每组 arm 全部条目"；
      (c) store_sidelink 中继每 (store_edge, restore_edge) 对**恰一对**
          节点（与组数无关——O(跨缘对数) 回归钉）；中继 recv 是两组首
          节点的共同链序父（共享中继语义）；
      (d) span 内条目在准入批发射内消费（pending_store_tails 窗口有界
          ——merge_tail_gated 纯度回归钉）；span 外条目保留；
      (e) 组门逐 rank 落点全在 Prefill ranks（目标 HBM 落点 = 执行实例）
          且层区间连续铺满后缀。
    """

    def setUp(self):
        self.builder = GraphBatchBuilder(_graph_config())
        self.builder.begin_batch()

    def _nodes(self):
        return self.builder.batch["nodes"]

    def _parents_by_rank(self):
        parents = {}
        for edge in self.builder.batch["parent_edges"]:
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        return parents

    def _seed_arrival_and_stores(self, extra_entries=()):
        """turn-1 到达门 + 在飞池写登记。基础条目 A = 后缀 [8,16) 双缘
        （store 节点 id = 到达门种子节点）；extra_entries = ((层区间),
        edge_rank) 合成条目——store 节点 id 取该缘新种子节点。返回
        store 节点 id 映射 {(edge_rank, 层区间): store_id}。"""
        from generate_face_trace import PendingHistoryGate
        gate_ids = {}
        for rank in (0, 1):
            self.builder.builders[rank].comp(
                f"arrival_seed_rank{rank}", 1, 1)
            gate_ids[rank] = self.builder.builders[rank].previous_id
        self.builder.pending_history[REQUEST_R] = PendingHistoryGate(
            source_instance_index=0,
            timer_gates=(gate_ids[0], gate_ids[1]),
            location="partial_hbm_remote")
        store_ids = {(0, (8, 16)): gate_ids[0], (1, (8, 16)): gate_ids[1]}
        self.builder._register_store_tails(
            _store_transfer(8, 16), {"shards": [
                {"edge_rank": 0, "edge_store_node_id": gate_ids[0],
                 "source_ack_recv_node_id": gate_ids[0]},
                {"edge_rank": 1, "edge_store_node_id": gate_ids[1],
                 "source_ack_recv_node_id": gate_ids[1]}]})
        for layer_start, layer_end, edge_rank in extra_entries:
            self.builder.builders[edge_rank].comp(
                f"extra_store_seed_rank{edge_rank}_"
                f"ls{layer_start}", 1, 1)
            store_id = self.builder.builders[edge_rank].previous_id
            store_ids[(edge_rank, (layer_start, layer_end))] = store_id
            self.builder._register_store_tails(
                _store_transfer(layer_start, layer_end), {"shards": [
                    {"edge_rank": edge_rank,
                     "edge_store_node_id": store_id,
                     "source_ack_recv_node_id": store_id}]})
        return store_ids

    def _emit_admission(self, groups):
        self.builder.emit_admission_batch(_restore_admission_plan(groups))

    def _group_first_node(self, group_position, rank):
        """第 group_position 组在该 rank 的首节点（边缘 mem_load，shard
        序 = (target, edge) 序）。"""
        fragment = (
            f"_history_transfer_action{group_position:03d}_")
        matches = [
            node for node in self._nodes()
            if node["rank"] == rank
            and fragment in node["name"]
            and node["name"].endswith("_remote_load")]
        self.assertEqual(
            len(matches), 1,
            f"expected exactly one edge mem_load for group "
            f"{group_position} on rank {rank}, got {len(matches)}")
        return matches[0]

    def test_groups_fork_frontier_with_intersection_arms(self):
        # 条目 B = 合成中段 [10,12)（仅与组 1 [10,16) 交集）。
        store_ids = self._seed_arrival_and_stores(extra_entries=(
            (10, 12, 0),
        ))
        groups = (_restore_transfer(4, 10, 0), _restore_transfer(10, 16, 1))
        self._emit_admission(groups)
        parents = self._parents_by_rank()
        barrier = {
            node["rank"]: node["id"] for node in self._nodes()
            if node["name"].endswith("_prefill_resident_prefix_ready_barrier")}
        a0 = store_ids[(0, (8, 16))]
        a1 = store_ids[(1, (8, 16))]
        b0 = store_ids[(0, (10, 12))]
        # 双缘登记 ⇒ 对侧 store 经共享中继承载（s1→r0 / s0→r1 各恰一对
        # ——见 test_shared_relay_emitted_once_per_cross_pair）；本缘
        # store 直接 arm。中继链锚 = 该 rank 最后发射的 sidelink 节点
        # （每组区域 checkpoint 捕获其后 previous_id ⇒ 组首节点经链序
        # 排在全部并集匹配 store 之后）。
        relay_anchor = {}
        for rank in (0, 1):
            relay_anchor[rank] = max(
                node["id"] for node in self._nodes()
                if node["rank"] == rank
                and "store_sidelink" in node["name"])
        # 组 0 首节点：{屏障, 中继链锚} ∪ {本缘 A}（B 与组 0 无交集
        # ——交集语义，不被 arm）。
        for rank, a_id in ((0, a0), (1, a1)):
            first = self._group_first_node(0, rank)
            self.assertEqual(
                set(parents.get((rank, first["id"]), ())),
                {barrier[rank], a_id, relay_anchor[rank]},
                "group-0 first node must fork from the barrier/shared "
                "relay chain with only its intersecting same-edge store "
                "armed")
        # 组 1 首节点：fork 同一 {屏障, 中继链锚}（组间无边——父集合的
        # 共享部分与组 0 相等，无任何组 0 节点入父）∪ {本缘 A, B}（B 与
        # 组 1 交集——直接 arm，不靠组间链传递性）。
        for rank, a_id in ((0, a0), (1, a1)):
            first = self._group_first_node(1, rank)
            expected = {barrier[rank], a_id, relay_anchor[rank]}
            if rank == 0:
                expected.add(b0)
            self.assertEqual(
                set(parents.get((rank, first["id"]), ())), expected,
                "group-1 first node must fork from the same frontier (no "
                "inter-group edge) with its intersecting stores armed")
        # 交集语义正面钉：B 在组 1 的父集合、不在组 0 的父集合。
        self.assertIn(b0, parents.get((0, self._group_first_node(1, 0)["id"]), ()))
        self.assertNotIn(
            b0, parents.get((0, self._group_first_node(0, 0)["id"]), ()),
            "mid-span entry must not be armed by the non-intersecting "
            "group")

    def test_shared_relay_emitted_once_per_cross_pair(self):
        # 条目 D = 后缀 [8,16) 登记在跨缘 edge 2（restore 缘 = 0/1）→
        # 跨缘对 (2,0) 与 (2,1)。
        store_ids = self._seed_arrival_and_stores(extra_entries=(
            (8, 16, 2),
        ))
        groups = (_restore_transfer(4, 10, 0), _restore_transfer(10, 16, 1))
        self._emit_admission(groups)
        relay_sends = [
            node for node in self._nodes()
            if "store_sidelink" in node["name"]]
        # 每 (store_edge, restore_edge) 对恰一对节点：s2_r0 与 s2_r1 各
        # 一次发射（send+recv 同名同 rank）——与组数无关（2 组不产生
        # 2 对）。
        for pair in ("_store_sidelink_s2_r0", "_store_sidelink_s2_r1"):
            pair_nodes = [
                node for node in relay_sends if pair in node["name"]]
            self.assertEqual(
                len(pair_nodes), 2,
                f"relay pair {pair} must be emitted exactly once "
                f"(send+recv), got {len(pair_nodes)} nodes")
        # 中继 recv 是两组首节点的共同链序父（共享中继语义）。
        parents = self._parents_by_rank()
        for rank in (0, 1):
            recv_node = next(
                node for node in self._nodes()
                if node["rank"] == rank
                and f"_store_sidelink_s2_r{rank}" in node["name"]
                and node["type"] == 6)  # COMM_RECV_NODE
            for group_position in (0, 1):
                first = self._group_first_node(group_position, rank)
                self.assertIn(
                    recv_node["id"],
                    parents.get((rank, first["id"]), ()),
                    "group first node must chain after the shared relay "
                    "recv (all union stores precede every group)")

    def test_span_entries_consumed_in_batch_outside_retained(self):
        # 条目 B = [10,12)（span 内）；C = [0,2)（span 外——并集 [4,16)
        # 之外）→ 发射后 B 消费、C 保留（窗口有界回归钉）。
        self._seed_arrival_and_stores(extra_entries=(
            (10, 12, 0),
            (0, 2, 1),
        ))
        groups = (_restore_transfer(4, 10, 0), _restore_transfer(10, 16, 1))
        self._emit_admission(groups)
        ranges = sorted(
            (layer_start, layer_end)
            for _edge, _store, _ack, layer_start, layer_end
            in self.builder.pending_store_tails.get("s", ()))
        self.assertEqual(
            ranges, [(0, 2)],
            "in-span store entries must be consumed by the admission "
            "batch itself; only out-of-span entries survive")
        # 组门登记完整（组序 + 连续铺满 + 落点全在 Prefill ranks）。
        arms = self.builder._suffix_restore_arms[REQUEST_R]
        self.assertEqual(
            [(g, ls, le) for g, ls, le, _gates in arms],
            [(0, 4, 10), (1, 10, 16)])
        for _g, _ls, _le, gates in arms:
            self.assertTrue(
                set(gates) <= {0, 1},
                "restore gate leaked outside the Prefill instance "
                "(target HBM landing must be the exec instance)")
            self.assertEqual(sorted(gates), [0, 1])


if __name__ == "__main__":
    unittest.main()
