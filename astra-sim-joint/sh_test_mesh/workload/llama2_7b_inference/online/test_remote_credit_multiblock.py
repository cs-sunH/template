#!/usr/bin/env python3
"""test_remote_credit_multiblock.py -- A6' 修复回归（2026-09-22，
C4b-FIX）：remote-read auto-K 多块 credit 体块的列车级集体节点
参与者计数合规（C++ commit 预检谓词的零后端镜像）。

缺陷背景（/tmp/joint_exec/C4b/BLOCKED.md，确定性复现）：
  auto-K（K = max(1, ceil(S_j/8))，S_j > 8）产生多块时，体块 phase
  缺省回落 train_id——M 个体块在同一 rank 发射 M 份同名
  ``{train_id}_all_layers_{attention,mlp}_all_reduce``；C++
  GraphBatchCommitter 预检按 (pg_name, name) 计参与者、要求每 rank
  恰 1（GraphBatchCommitter.cc:634-659），M=8 时拒批：
  "has 8 participants on rank 30 (expected exactly one)" → 次生
  ack_count != delivery_count。K ≥ S_j 单块路径完整执行通过（v1
  等价锚），缺陷特定于多块。

修复口径（GB _emit_train_body，与 C13 copy 体块 ``_cb{k}``` 同款）：
  无显式 phase 的 credit 体块在 M > 1 时逐块挂 ``{train_id}_rcb{b}``
  唯一后缀；M = 1 不加（单块 = v1 等价锚逐字节不变）；copy 体块自
  带 phase 原样透传。

覆盖：
  1. 预检谓词镜像：批内 COMM_COLL 节点按 (rank, pg_name, name) 计数
     恒 1——T1 joiner 多块（K=2，M=4）与 T2+ 续坐多块（K=3，M=2）
     两形态；
  2. 命名规则：多块集体/聚合节点名含 ``_rcb{b}``（b = 块序，1 起），
     且引用本列车 train_id；
  3. 拓扑语义不破（I2）：尾块 recv 完成门仍恰好门控对应体块（门序
     = 块序、不依赖后续块、无门消费扫描），被门控体节点名携带匹配
     的 ``_rcb{b}`` 前缀；块 1 仍走主链（无旁挂门）；
  4. 单块锚：K = S 单块无 ``_rcb`` 后缀、集体名恰为 v1 形态
     （``{train_id}_all_layers_*_all_reduce``）——I3a 等价锚的命名侧
     钉（字节级等价由 test_remote_credit_stream 的
     test_k_equals_steps_matches_plain_train 钉住）；
  5. C15 组合形态：restore_arms 首块层段发射 + 多块——层段与后续块
     的节点名互不重名（预检谓词在该组合下同样合规）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_remote_credit_multiblock.py
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_trace import COMM_COLL_NODE  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.test_graph_batch_builder import _make_config  # noqa: E402
from online.test_remote_credit_stream import (  # noqa: E402
    EXEC_INSTANCE,
    _head_runtime,
    _install_remote,
    _remote_runtime,
    _scheduler,
)


def _preflight_collective_multiplicity(nodes):
    """C++ GraphBatchCommitter 集体参与者预检谓词的零后端镜像。

    按 (rank, pg_name, name) 计批内集体节点数——每键恰 1 是预检的
    通过条件（同名多份 = 多 participants on rank，拒批）。"""
    counts = {}
    for node in nodes:
        if node["type"] != COMM_COLL_NODE:
            continue
        key = (node["rank"], node["coll"]["pg_name"], node["name"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _parents_by_rank(edges):
    parents = {}
    for edge in edges:
        parents.setdefault((edge["rank"], edge["to"]), []).append(
            edge["from"])
    return parents


class T1MultiBlockPreflightTest(unittest.TestCase):
    """T1 joiner 多块（K=2，M=4）：预检谓词 + 命名 + I2 拓扑保持。"""

    def _emit_t1(self, credit_iters="2", decode=8):
        s = _scheduler(credit_iters)
        rt = _remote_runtime(decode=decode)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        s._emit_train(state, plan, [rt], 1)
        return s, rt, plan

    def _nodes(self, s):
        return s.graph.batch["nodes"]

    def test_preflight_predicate_and_naming(self):
        s, rt, plan = self._emit_t1()
        nodes = self._nodes(s)
        train_id = plan["train_id"]
        # 预检谓词（A6' 核心回归）：同名集体每 rank 恰 1 个参与者。
        counts = _preflight_collective_multiplicity(nodes)
        self.assertTrue(counts)
        self.assertEqual(
            {key: value for key, value in counts.items() if value != 1},
            {}, "collective multiplicity preflight violated")
        # 命名规则：4 个体块逐块 _rcb1.._rcb4 后缀，引用本列车。
        for block in (1, 2, 3, 4):
            self.assertIn(
                "{}_rcb{}_all_layers_mlp_all_reduce".format(
                    train_id, block),
                {node["name"] for node in nodes},
                "block {} collective name missing".format(block))
        # 修复前形态（train_id 裸名集体）不得再出现。
        self.assertNotIn(
            "{}_all_layers_mlp_all_reduce".format(train_id),
            {node["name"] for node in nodes})
        # 每 rank 都持有每块集体（组内全员参与，跨 rank 签名一致）。
        for rank in (0, 1):
            per_rank = {
                name for node in nodes
                if node["rank"] == rank and node["type"] == COMM_COLL_NODE
                for name in (node["name"],)}
            for block in (1, 2, 3, 4):
                self.assertIn(
                    "{}_rcb{}_all_layers_attention_all_reduce".format(
                        train_id, block),
                    per_rank)

    def test_i2_gating_maps_onto_block_phases(self):
        s, rt, plan = self._emit_t1()
        nodes = self._nodes(s)
        parents = _parents_by_rank(s.graph.batch["parent_edges"])
        train_id = plan["train_id"]
        # I2：尾块 b（2..4）的 recv 门恰好门控一个体块节点，且被门控
        # 节点携带匹配的 _rcb{b} 相位；门序 = 块序（不依赖后续块）。
        gated_node_ids = {}
        for block in (2, 3, 4):
            recv = [
                node for node in nodes
                if "_credit%d_recv" % block in node["name"]
                and node["rank"] == 0][0]
            targets = [
                to for (rank, to), froms in parents.items()
                if rank == 0 and recv["id"] in froms]
            body_targets = [
                node for node in nodes
                if node["id"] in targets and node["rank"] == 0
                and node["request_id"] == train_id]
            self.assertEqual(len(body_targets), 1, block)
            gated = body_targets[0]
            self.assertTrue(
                gated["name"].startswith(
                    "{}_rcb{}_".format(train_id, block)),
                "credit block {} gate must arm its own body block phase, "
                "got {}".format(block, gated["name"]))
            gated_node_ids[block] = gated["id"]
        self.assertLess(gated_node_ids[2], gated_node_ids[3])
        self.assertLess(gated_node_ids[3], gated_node_ids[4])
        # 无门消费扫描：被门控节点均为 COMP 计算体。
        for block, node_id in gated_node_ids.items():
            node = [n for n in nodes if n["id"] == node_id][0]
            self.assertTrue(node["compute"]["num_ops"] > 0, block)
        # 块 1 主链先行（无旁挂 recv 门；既有 test_t1_multi_block_
        # structure 已钉 pd_transfer 主链位，此处只钉不存在块 1 门控）。
        self.assertFalse([
            node for node in nodes if "_credit1_recv" in node["name"]])


class T2ContinuationMultiBlockPreflightTest(unittest.TestCase):
    """T2+ 续坐多块（K=3，块 [3, 2]）：预检谓词 + 续坐发射形态保持。"""

    def test_continuation_multiblock_preflight(self):
        s = _scheduler(credit_iters="3")
        rt = _remote_runtime(decode=8)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        head = _head_runtime(chunks=3)
        s.runtime_by_request_id["rh"] = head
        state.qp.append(head)
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        # T1（iterations=3，K=3 → 单块）：先发射并结算，让 T2 成为续坐。
        plan1 = s._plan_train(state)
        s._emit_train(state, plan1, [rt], 1)
        s.kv_manager._sessions["s_rr"].context_tokens = (
            rt.joint_input_tokens + 3)
        train = state.in_flight_train
        completed = [
            request_id for request_id, _ in train["members"]
            if request_id in train["exit_set"]]
        s._finalize_completed_trains(
            drained=tuple(train["drain_members"]),
            completed_now=completed, sentinel_trains=(), tick=2)
        state.qp.remove(head)
        # T2（iterations=5，K=3 → 两块 [3, 2]，无 joiner 槽位 = D7 续坐）。
        s.graph.begin_batch()
        plan2 = s._plan_train(state)
        s._emit_train(state, plan2, [], 3)
        nodes = s.graph.batch["nodes"]
        summary = rt.remote_read_slice_summaries[1]
        self.assertEqual(summary["block_steps"], [3, 2])
        counts = _preflight_collective_multiplicity(nodes)
        self.assertTrue(counts)
        self.assertEqual(
            {key: value for key, value in counts.items() if value != 1},
            {}, "collective multiplicity preflight violated")
        train_id = plan2["train_id"]
        for block in (1, 2):
            self.assertIn(
                "{}_rcb{}_all_layers_mlp_all_reduce".format(
                    train_id, block),
                {node["name"] for node in nodes})
        self.assertNotIn(
            "{}_all_layers_mlp_all_reduce".format(train_id),
            {node["name"] for node in nodes})
        # 续坐形态保持：块 1 上主链（remote_credit 动作名在案）、
        # 无 readiness barrier（I3b）。
        self.assertTrue([
            node for node in nodes
            if "remote_credit_action" in node["name"]])
        self.assertFalse([
            node for node in nodes
            if node["name"].endswith("_decode_kv_ready_barrier")])


class SingleBlockAnchorNamingTest(unittest.TestCase):
    """K ≥ S 单块：无 _rcb 后缀（v1 等价锚命名侧钉）。"""

    def test_single_block_keeps_v1_collective_names(self):
        s = _scheduler(credit_iters="8")
        rt = _remote_runtime(decode=8)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        s._emit_train(state, plan, [rt], 1)
        nodes = s.graph.batch["nodes"]
        train_id = plan["train_id"]
        names = {node["name"] for node in nodes}
        self.assertFalse([name for name in names if "_rcb" in name])
        for collective in (
                "{}_all_layers_attention_all_reduce".format(train_id),
                "{}_all_layers_mlp_all_reduce".format(train_id)):
            self.assertIn(collective, names)
        counts = _preflight_collective_multiplicity(nodes)
        self.assertTrue(counts)
        self.assertEqual(
            {key: value for key, value in counts.items() if value != 1},
            {}, "single-block path must stay preflight-clean")


class RestoreArmsMultiBlockNamingTest(unittest.TestCase):
    """C15 组合形态：restore_arms 首块层段 + 多块 credit——预检谓词。

    直驱 _emit_train_body（夹具合成 credit_blocks + _suffix_restore_
    arms 账本），验证层段（首块）与后续块的节点名互不重名。"""

    def test_restore_first_block_segments_unique_names(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        # 合成恢复组账本：层区间 [8, layers) 连续铺到 L（组门 = 任意
        # 有效节点 id；arm_dependency 只登记 pending 依赖）。
        layers = builder.config.layers
        prefix = 1 if layers == 2 else 8
        builder.builders[0].comp("gate_seed_0", 1, 1)
        gate0 = builder.builders[0].previous_id
        builder.builders[1].comp("gate_seed_1", 1, 1)
        gate1 = builder.builders[1].previous_id
        request_id = "rr_restore"
        builder._suffix_restore_arms[request_id] = (
            (0, prefix, layers, {0: gate0, 1: gate1}),)
        # 尾块门账本（真实流程由 _emit_train_head 的旁挂支链发射登记；
        # 直驱夹具预置等价账本，块 2 的 recv 完成门 = 种子节点）。
        builder._credit_arms[request_id] = {2: {0: gate0, 1: gate1}}
        train_plan = {
            "train_id": "batch_train_i0_9",
            "instance_index": 0,
            "stage": "decode",
        }
        spans = [(4, 20)] * 4
        credit_blocks = [
            {"start_iter": 1, "end_iter": 2, "weight_passes": 2,
             "spans": spans[:2], "gates": [(request_id, 1)]},
            {"start_iter": 3, "end_iter": 4, "weight_passes": 2,
             "spans": spans[2:], "gates": [(request_id, 2)]},
        ]
        first_chunk_member = {"request_id": request_id}
        marker = builder._mark()
        builder._emit_train_body(
            train_plan, spans, 4,
            first_chunk_member=first_chunk_member,
            credit_blocks=credit_blocks)
        builder._collect(marker)
        nodes = builder.batch["nodes"]
        names_by_rank = {}
        for node in nodes:
            names_by_rank.setdefault(node["rank"], []).append(node["name"])
        # 预检谓词：集体节点 (rank, pg, name) 恒 1。
        counts = _preflight_collective_multiplicity(nodes)
        self.assertTrue(counts)
        self.assertEqual(
            {key: value for key, value in counts.items() if value != 1},
            {}, "restore+multiblock collective preflight violated")
        # 任意 rank 上节点名整体不重名（层段标签 + 块相位互斥）。
        for rank, names in names_by_rank.items():
            self.assertEqual(len(names), len(set(names)),
                             "duplicate node names on rank {}".format(rank))
        train_id = train_plan["train_id"]
        all_names = set(names_by_rank[0])
        # 首块层段（_rcb1 相位 + 层区间标签）与尾块（_rcb2 + all_layers）
        # 并存且互不重名。
        self.assertTrue([
            name for name in all_names
            if name.startswith("{}_rcb1_layers".format(train_id))])
        self.assertIn(
            "{}_rcb2_all_layers_mlp_all_reduce".format(train_id), all_names)
        # 账本消费后弹出（既有语义）。
        self.assertNotIn(request_id, builder._suffix_restore_arms)


if __name__ == "__main__":
    unittest.main(verbosity=2)
