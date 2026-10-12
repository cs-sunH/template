#!/usr/bin/env python3
"""test_r4_mixed_remote_read_train_pin.py -- 修复轮三 R4 钉测：同一列车
跨请求双远读体块源（decode credit 切片 + 队头 prefill 远读窗）的合并
发射（2026-09-28）。

背景（probe_20260927T190219Z / replay_load_first 臂红，seq 197）：
`emit_iteration_train` 曾以**列车级**断言拒收 "train carries both decode
remote-credit body blocks and prefill remote-read window blocks"——但其
注释语义（"a request streams at most one remote read per stage"）是**请
求级**互斥。跨请求并存（某 decode 成员 D 边界选 remote-read credit、同
实例队头准入选 prefill remote-read）两流分属不同请求、合法且 argmin 正
常路径可达（replay 臂未强制即命中）。

被钉语义：
  1. 混合列车（队头 prefill_read 窗 + decode 成员 credit 切片）经真实
     调度器 _plan_train/_emit_train + 真构图器发射成功；两流的 tile
     结清账本（_prefill_read_window_tiles / _credit_pending_tiles）在列
     车尾全部核销；图侧同时出现两流的服务节点；
  2. 互斥语义收窄到请求粒度：同一请求同时贡献两源的列车计划仍
     fail-closed（fail-closed 不放宽）；
  3. 合并器形状：逐迭代块界对齐、gates = 并集、span 取 credit 权威映
     射；迭代数失配 fail-closed。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_r4_mixed_remote_read_train_pin.py -q
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.test_prefill_remote_read_scheduler import (  # noqa: E402
    INPUT_TOKENS,
)
from online.test_prefill_remote_read_tile_protocol import (  # noqa: E402
    EXEC_INSTANCE,
    HOME_INSTANCE,
    _force_remote_read,
    _graph_config,
    _make_integration,
    _make_runtime,
    _model,
)
import online.sh30_online_scheduler as sh30  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    JOINT_STAGE_D_COMMITTED,
    JOINT_STAGE_D_RUNNING,
    _OnlineInstanceState,
)

TILE_TOKENS = 256
CREDIT_RID = "rc"
CREDIT_SID = "s_credit"


def _mixed_train_state(scheduler, runtime, *, input_tokens):
    """准入队头（prefill remote-read，_make_integration 已做）后，为同
    实例装配一个 decode remote-read credit 成员——构成跨请求双源列车。
    credit 成员自带会话（基础历史驻留 home，与队头会话无关——生产形态
    = 两个独立会话共乘一实例列车）。"""
    scheduler.runtime_by_request_id[runtime.request_id] = runtime
    manager = scheduler.kv_manager
    manager.prepare_prefill(
        session_id=CREDIT_SID, target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id=CREDIT_RID + "_seed")
    manager.expand_prefill(
        session_id=CREDIT_SID, instance_index=HOME_INSTANCE,
        context_tokens=64, trigger_request_id=CREDIT_RID + "_seed")
    manager.mark_complete(CREDIT_SID, 64)
    credit_runtime = _make_runtime()
    credit_runtime.request_id = CREDIT_RID
    credit_runtime.session_id = CREDIT_SID
    credit_runtime.joint_action = "remote-read"
    credit_runtime.joint_decode_action = "remote-read"
    credit_runtime.origin_home_instance = HOME_INSTANCE
    credit_runtime.prefill_instance_index = HOME_INSTANCE
    credit_runtime.decode_instance_index = EXEC_INSTANCE
    credit_runtime.joint_stage = JOINT_STAGE_D_RUNNING
    credit_runtime.completion_instance_index = EXEC_INSTANCE
    credit_runtime.joint_input_tokens = input_tokens
    credit_runtime.decode_tokens_consumed = 0
    credit_runtime.decode_length = 3
    credit_runtime.remaining_chunks = 0
    scheduler.runtime_by_request_id[CREDIT_RID] = credit_runtime
    credit_runtime.remote_read_credit_plan = (
        scheduler._joint_remote_read_credit_plan(
            credit_runtime, EXEC_INSTANCE))
    assert credit_runtime.remote_read_credit_plan is not None, (
        "fixture: credit plan must exist (home base at another instance)")
    # 实际发射身份种子面（生产在 staging lease 获取后写入 runtime
    # pipeline_identity_inputs["decode"]；plan_dict 据此生成显式种子）。
    # 目标 rank 从 credit plan 的 shard_specs 推导（与生产 _acquire 同
    # 源口径：spec[1] = 目标 rank），allocation_id 夹具固定 1。
    credit_runtime.pipeline_identity_inputs["decode"] = {
        "stage_epoch": int(credit_runtime.joint_boundary_epoch),
        "staging_allocation_id": 1,
        "target_ranks": tuple(dict.fromkeys(
            int(spec[1])
            for spec in credit_runtime.remote_read_credit_plan[
                "shard_specs"])),
    }
    state = scheduler.instances[EXEC_INSTANCE]
    credit_runtime.joint_stage = JOINT_STAGE_D_COMMITTED
    state.active_decode.append(credit_runtime)
    state.active_decode_lookup.add(credit_runtime)
    return state, credit_runtime


class MixedRemoteReadTrainEmissionTest(unittest.TestCase):
    """钉 1：跨请求双源列车发射成功且两流账本结清（修复前以
    "train carries both ..." RuntimeError 红）。"""

    def test_mixed_train_emits_and_settles_both_streams(self):
        scheduler, runtime, plan, read_plan = _make_integration(
            tile_tokens=TILE_TOKENS, input_tokens=INPUT_TOKENS,
            emit_train=False)
        self.assertIsNotNone(read_plan)
        # _make_integration 内的首次 _plan_train 已推进窗口游标（未发射
        # = 零消费）——夹具回退游标，混合列车规划从窗 1 起切窗。
        runtime.prefill_remote_read_plan["window_cursor"] = 1
        state, credit_runtime = _mixed_train_state(
            scheduler, runtime, input_tokens=INPUT_TOKENS)
        mixed_plan = scheduler._plan_train(state)
        # 列车确为双源形态（队头 chunk 窗 + credit 成员）——否则夹具
        # 失真，钉测无效。
        self.assertIsNotNone(mixed_plan.get("prefill_read"))
        self.assertTrue(
            any((rt.joint_decode_action or rt.joint_action) == "remote-read"
                and rt.remote_read_credit_plan is not None
                for rt in state.active_decode))
        scheduler._emit_train(state, mixed_plan, [], 600)
        # 两流服务面都在场（compact 发射：per-tile 事件在两流各自的
        # descriptor 草稿 endpoint_runs 内；物理节点面 = pipeline 锚
        # + 前缀读窗触发链）。
        nodes = scheduler.graph.batch["nodes"]
        self.assertTrue([
            n for n in nodes if "prefill_read" in n["name"]])
        stream_kinds = {
            draft.stream_kind
            for draft in scheduler.graph._pipeline_drafts.values()}
        self.assertIn("decode_remote_credit", stream_kinds)
        self.assertIn("prefill_read_window", stream_kinds)
        # 列车尾账本结清（emit_iteration_train 的尾检查已覆盖前缀读窗；
        # credit 待发账本亦须清空——泄漏即释放链破损）。
        self.assertEqual(scheduler.graph._credit_pending_tiles, {})
        self.assertEqual(scheduler.graph._prefill_read_window_tiles, {})

    def test_same_request_dual_stream_still_fails_closed(self):
        """钉 2：互斥语义收窄到请求粒度——同一请求双源仍 fail-closed。"""
        model = _model()
        graph = GraphBatchBuilder(_graph_config(model))
        graph.begin_batch()
        rid = "r_both"
        train_plan = {
            "train_id": "batch_train_i1_1",
            "instance_index": EXEC_INSTANCE,
            "stage": "prefill",
            "joiners": [],
            "pass_spans": [((4, 4), (4, 4)), ((4, 5), (4, 5))],
            "iterations": 2,
            "sentinel": False,
            "prefill_start_member": None,
            "first_chunk_member": None,
            "drain_members": [],
            "exit_members": [],
            "remote_credit": {"body_blocks": [{
                "start_iter": 1, "end_iter": 1, "weight_passes": 1,
                "spans": [((4, 4), (4, 4))],
                "gates": [(rid, 1)],
            }]},
            "prefill_read": {"body_blocks": [{
                "start_iter": 1, "end_iter": 1, "weight_passes": 1,
                "spans": [((4, 4), (4, 4))],
                "gates": [(rid, 1), (rid, 2)],
            }]},
        }
        with self.assertRaises(RuntimeError) as ctx:
            graph.emit_iteration_train(train_plan)
        self.assertIn("at most one remote read per stage", str(ctx.exception))
        self.assertIn(rid, str(ctx.exception))


class MergeRemoteReadBodyBlocksShapeTest(unittest.TestCase):
    """钉 3：合并器形状契约（块界对齐、门并集、迭代数失配 fail-closed）。"""

    def test_merge_unions_gates_and_keeps_credit_spans(self):
        credit = [
            {"start_iter": i, "end_iter": i, "weight_passes": 1,
             "spans": [("c", i)], "gates": [("a", i)]}
            for i in (1, 2)]
        read = [
            {"start_iter": i, "end_iter": i, "weight_passes": 1,
             "spans": [("r", i)], "gates": [("h", 2 * i - 1), ("h", 2 * i)]}
            for i in (1, 2)]
        merged = GraphBatchBuilder._merge_remote_read_body_blocks(
            credit, read)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["spans"], [("c", 1)])
        self.assertEqual(
            merged[0]["gates"], [("a", 1), ("h", 1), ("h", 2)])
        self.assertEqual(merged[1]["gates"], [("a", 2), ("h", 3), ("h", 4)])

    def test_iteration_count_mismatch_fails_closed(self):
        credit = [
            {"start_iter": i, "end_iter": i, "weight_passes": 1,
             "spans": [], "gates": []}
            for i in (1, 2)]
        read = [
            {"start_iter": i, "end_iter": i, "weight_passes": 1,
             "spans": [], "gates": []}
            for i in (1, 2, 3)]
        with self.assertRaises(RuntimeError) as ctx:
            GraphBatchBuilder._merge_remote_read_body_blocks(credit, read)
        self.assertIn("different iteration counts", str(ctx.exception))

    def test_iteration_bounds_mismatch_fails_closed(self):
        credit = [{"start_iter": 1, "end_iter": 1, "weight_passes": 1,
                   "spans": [], "gates": []}]
        read = [{"start_iter": 2, "end_iter": 2, "weight_passes": 1,
                 "spans": [], "gates": []}]
        with self.assertRaises(RuntimeError) as ctx:
            GraphBatchBuilder._merge_remote_read_body_blocks(credit, read)
        self.assertIn("iteration coverage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
