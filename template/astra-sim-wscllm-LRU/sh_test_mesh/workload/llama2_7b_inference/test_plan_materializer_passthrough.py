#!/usr/bin/env python3
"""test_plan_materializer_passthrough.py -- P2-13 对齐钉死测试（2026-10-01）。

契约：WP2 透传的 human_time_ns/tool_time_ns 必须写进
**metrics_manifest.json**（消费链 load_request_manifest 只读该文件；
slo_stats cmd_session 从 metrics_manifest.json 的 requests[] 取这两字段
算 T_session），不得写进 manifest.json（该文件的 requests[] 唯一消费方
在线调度器只读队列派生 9 字段，human/tool 字段在彼处零消费=写错文件；
本仓此前正是 human/tool 只写 manifest.json，SLO 多轮会话 T_session 恒
NA）。request_type/prefix_len 仍写 manifest.json（metrics_postprocess
的 _manifest_request_entries 双 manifest 合并消费）。

幂等口径与 plan_materializer 主体一致：
  - sidecar 行缺失/列空 → 字段不写（不猜测填充）；
  - 值为 int(sidecar 文本)。
"""

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SH_TEST_DIR = MODULE_DIR.parents[1]
for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plan_materializer import (  # noqa: E402
    _derive_manifest_requests,
    _derive_metrics_requests,
)
from generate_trace import RequestSpec  # noqa: E402


class _Group:
    def __init__(self, ranks):
        self.ranks = list(ranks)


class _Config:
    """plan_materializer 两个 derive 函数所需的最小 config 面。"""

    def __init__(self, specs, indices=None):
        self.request_queue = list(specs)
        self.request_queue_indices = tuple(
            range(len(specs)) if indices is None else indices)
        self.queue_position_by_index = {
            index: position
            for position, index in enumerate(self.request_queue_indices)}
        self.inference_groups = [_Group((0, 1, 2))]


def _spec(request_id, session_id, turn_index, arrival=None, interval=None):
    return RequestSpec(
        session_id=session_id,
        turn_index=turn_index,
        request_id=request_id,
        prefill_length=100,
        decode_length=10,
        session_arrival_time_ns=arrival,
        inter_request_interval_ns=interval,
    )


SIDECAR_ROWS = {
    "s0_r0": {"request_id": "s0_r0", "human_time_ns": "50000000",
              "tool_time_ns": ""},
    "s0_r1": {"request_id": "s0_r1", "human_time_ns": "",
              "tool_time_ns": "20000000"},
    # s0_r2: sidecar 无此行 → 字段不写
}


def _config():
    return _Config([
        _spec("s0_r0", "s0", 0, arrival=1000),
        _spec("s0_r1", "s0", 1, interval=500),
        _spec("s0_r2", "s0", 2, interval=500),
    ])


def test_metrics_manifest_carries_human_tool_passthrough():
    """metrics_manifest.json requests[] 必须带 human/tool_time_ns（int）。"""
    config = _config()
    records = _derive_metrics_requests(config, SIDECAR_ROWS)
    by_id = {r["request_id"]: r for r in records}
    assert by_id["s0_r0"]["human_time_ns"] == 50000000
    assert "tool_time_ns" not in by_id["s0_r0"]
    assert by_id["s0_r1"]["tool_time_ns"] == 20000000
    assert "human_time_ns" not in by_id["s0_r1"]
    # sidecar 缺行：两字段均不写（不猜测）
    assert "human_time_ns" not in by_id["s0_r2"]
    assert "tool_time_ns" not in by_id["s0_r2"]


def test_metrics_manifest_passthrough_survives_without_sidecar():
    """sidecar 整体缺席（空 dict）→ 正常合成，字段一律不写。"""
    records = _derive_metrics_requests(_config(), {})
    assert len(records) == 3
    for record in records:
        assert "human_time_ns" not in record
        assert "tool_time_ns" not in record


def test_token_manifest_requests_do_not_carry_human_tool():
    """manifest.json requests[] 不得携带 human/tool_time_ns（消费链
    在 metrics_manifest.json；manifest.json 副本属冗余写错文件）。"""
    config = _config()
    requests = _derive_manifest_requests(config, SIDECAR_ROWS)
    for entry in requests:
        assert "human_time_ns" not in entry, entry
        assert "tool_time_ns" not in entry, entry
    # 队列派生 9 字段不受影响（抽样核对）
    by_id = {e["request_id"]: e for e in requests}
    assert by_id["s0_r0"]["queue_index"] == 0
    assert by_id["s0_r0"]["history_tokens_before"] == 0
    assert by_id["s0_r1"]["history_tokens_before"] == 110


def test_manifest_request_ids_must_be_unique():
    """manifest 级 request_id 唯一性 fail-closed（4.3 对齐）：重复 id 在
    清单边界显式拒绝（下游逐 id 消费，last-wins/双计均静默）。"""
    config = _Config([
        _spec("dup_id", "s0", 0, arrival=1000),
        _spec("dup_id", "s1", 0, arrival=2000),
    ])
    try:
        _derive_manifest_requests(config, {})
    except RuntimeError as error:
        assert "duplicate request_id" in str(error)
    else:
        raise AssertionError("duplicate request_id was not rejected")


def _run_all() -> None:
    """Direct-run entry：bare ``test_*`` 函数对个别直跑场景可见性差，
    pytest 按函数名收集（run_python_unit_tests.sh 走 pytest 根清单），
    本入口仅供手动直跑，保持与 pytest 用例数一致。"""
    test_metrics_manifest_carries_human_tool_passthrough()
    test_metrics_manifest_passthrough_survives_without_sidecar()
    test_token_manifest_requests_do_not_carry_human_tool()
    test_manifest_request_ids_must_be_unique()
    print("[plan-materializer-passthrough] all 4 cases PASS")


if __name__ == "__main__":
    _run_all()
