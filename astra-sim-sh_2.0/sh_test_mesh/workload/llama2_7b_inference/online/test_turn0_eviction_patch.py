#!/usr/bin/env python3
"""test_turn0_eviction_patch.py -- turn-0 在飞逐出补丁的最小合成单元测试。

背景（对比报告 §5.2 回灌，2026-08-16）：`GraphBatchBuilder.
_mark_pending_history_store` 在 remote_store 逐出命中"在飞 turn-0 会话"
（pending_request_by_session 有键、pending_history 无 gate——turn-0 的
gate 是内联 new_session 不入账本）时，旧代码直接下标
`pending_history[pending_request_id]` 抛 KeyError；补丁改挂
deferred_session_locations（与 turn>0 无登记语义一致，完成时折算）。

30s 验收输入实测该分支不可达（补验轮 2026-08-16：83 次 remote_store 全部
走 pending 无键的 A 分支；插桩运行决策日志与正式运行逐字节一致），本测试
用合成场景直接覆盖补丁分支，并回归另两条原语义分支。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_turn0_eviction_patch.py   （或 pytest 同路径）
"""
import os
import sys
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import KVTransfer, KVTransferShard  # noqa: E402
from generate_face_trace import PendingHistoryGate  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

SESSION = "session_synthetic_0"
TURN0 = f"{SESSION}_request_0"


def _make_builder() -> GraphBatchBuilder:
    cfg = SimpleNamespace(
        npus_count=2,
        remote_operand_loads=False,
        inference_groups=[SimpleNamespace(ranks=(0, 1))],
    )
    return GraphBatchBuilder(cfg)


def _remote_store(resident_after: int) -> KVTransfer:
    assert 0 <= resident_after < 32
    return KVTransfer(
        kind="remote_store",
        phase="completion",
        reason="hbm_pressure",
        session_id=SESSION,
        trigger_request_id="session_other_request_9",
        source_instance_index=0,
        target_instance_index=None,
        total_bytes=4096,
        shards=(KVTransferShard(
            source_rank=0, target_rank=None, edge_rank=2, bytes=2048,
            noc_path=(0, 2), layer_start=0, layer_end=32),) * 2,
        model_layers=32,
        layer_start=0,
        layer_end=32,
        resident_prefix_layers_before=32,
        resident_prefix_layers_after=resident_after,
    )


def test_turn0_inflight_eviction_defers_instead_of_keyerror():
    """补丁分支（§5.2）：在飞 turn-0 会话被逐出——不抛 KeyError，挂
    deferred_session_locations，位置由 resident_after 折算。"""
    b = _make_builder()
    b.pending_request_by_session[SESSION] = TURN0  # turn-0 prefill 发射登记
    assert TURN0 not in b.pending_history          # turn-0 gate 内联不入账本
    # resident_after=0 → remote_memory
    b._mark_pending_history_store(_remote_store(0))
    assert b.deferred_session_locations == {SESSION: "remote_memory"}
    # resident_after 居间 → partial_hbm_remote（覆盖补丁分支的另一半位置）
    b.pending_request_by_session[SESSION] = TURN0
    b._mark_pending_history_store(_remote_store(17))
    assert b.deferred_session_locations[SESSION] == "partial_hbm_remote"
    assert b.pending_request_by_session[SESSION] == TURN0  # 账本不被破坏


def test_no_pending_request_defers_original_semantics():
    """原 A 分支：无在飞/无登记会话被逐出 → deferred（离线同构语义）。"""
    b = _make_builder()
    b._mark_pending_history_store(_remote_store(0))
    assert b.deferred_session_locations == {SESSION: "remote_memory"}


def test_pending_request_with_gate_updates_gate():
    """原 B 分支：登记的下一 turn（gate 在账本）→ 直接更新 gate.location，
    不进 deferred。"""
    b = _make_builder()
    gate = PendingHistoryGate(
        source_instance_index=0, timer_gates=(None, None),
        location="local_hbm")
    b.pending_history[TURN0] = gate
    b.pending_request_by_session[SESSION] = TURN0
    b._mark_pending_history_store(_remote_store(0))
    assert gate.location == "remote_memory"
    assert SESSION not in b.deferred_session_locations


if __name__ == "__main__":
    test_turn0_inflight_eviction_defers_instead_of_keyerror()
    test_no_pending_request_defers_original_semantics()
    test_pending_request_with_gate_updates_gate()
    print("test_turn0_eviction_patch: 3/3 PASS")
