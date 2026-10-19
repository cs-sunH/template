#!/usr/bin/env python3
"""test_config_boundary_guards.py -- 配置/输入边界的 loader 侧 fail-closed
守卫测试(2026-09-30,M-27/M-28/M-29 修复回归)。

  1. M-27:request queue 的 session_id/request_id 不得携带 C++ 名字锚点
     保留子串("kv"/"first_token")——用户数据混入锚点命名空间会令 code-7/
     code-8 名字匹配错绑(与 C-3 精确后缀匹配配套的双侧修复);
  2. M-28:拓扑白名单收紧为 Python/C++ 两栈一致的平面网格取值
     (Line/Mesh),Ring/Switch/FullyConnected 超售拒绝;
     memory-type 白名单收紧为 PER_NPU_MEMORY_EXPANSION(本仓 KV 二态
     冷热管理依赖远端池,NO_MEMORY_EXPANSION 超售在首个远端访问时
     C++ sys_panic);
  3. M-29:inference_group 跨组 rank 重叠/整网覆盖缺口 fail-closed
     (与调度器 build_instances "every NPU exactly once" 不变量同源,
     缺口在 loader 侧拦截而非物化后启动期)。
  4. Minor(2026-09-30):request queue 会话块结构不变量(session 块
     连续/turn 严格 +1/turn-0 唯一)loader 侧 fail-closed,与 C++ 读者
     WindowedTraceReader 同款。

Run: python3 test_config_boundary_guards.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
_SH_TEST_DIR = MODULE_DIR.parent.parent
for _p in (str(MODULE_DIR), str(_SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config_resolver as config_resolver_module  # noqa: E402
from config_resolver import (  # noqa: E402
    _prepare_comm_groups,
    load_hardware_config,
)
from generate_trace import (  # noqa: E402
    REQUEST_QUEUE_COLUMNS,
    load_request_queue,
)


def _hardware(rows=4, columns=4):
    return config_resolver_module.ResolvedHardware(
        source_path=Path("hardware.json"),
        capacity_profile="profile",
        slug="slug",
        label="label",
        paper_case="paper",
        mesh_rows=rows,
        mesh_cols=columns,
        topology_by_network_dimension=("Line", "Line"),
        local_hbm_capacity_bytes=1,
        local_hbm_bandwidth_gbps=1.0,
        local_hbm_latency_ns=0,
        d2d_bandwidth_gbps=1.0,
        d2d_latency_ns=0,
        remote_memory_type="PER_NPU_MEMORY_EXPANSION",
        remote_memory_bandwidth_gbps=1.0,
        remote_memory_latency_ns=1,
        remote_memory_npu_selection="mesh-boundary",
        peak_perf_tflops=1.0,
        metadata={},
    )


def _write_queue(rows):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    handle.write(",".join(REQUEST_QUEUE_COLUMNS) + "\n")
    for row in rows:
        handle.write(",".join(row) + "\n")
    handle.close()
    return Path(handle.name)


def _queue_row(session_id, request_id, turn_index="0"):
    # 列序 = REQUEST_QUEUE_COLUMNS(session_id, turn_index, request_id,
    # prefill_length, decode_length, session_arrival_time_ns,
    # inter_request_interval_ns, description)。
    return (
        session_id, turn_index, request_id, "64", "4", "1000", "", "",
    )


class LoaderReservedNamespaceGuardTests(unittest.TestCase):
    """1. M-27:session/request 标识的锚点保留子串拒绝。"""

    def test_session_id_with_kv_substring_rejected(self):
        path = _write_queue([_queue_row("my_kv_session", "request_0")])
        try:
            with self.assertRaisesRegex(ValueError, r"reserved C\+\+ anchor"):
                load_request_queue(path)
        finally:
            os.unlink(path)

    def test_request_id_with_first_token_substring_rejected(self):
        path = _write_queue([_queue_row("session_a", "first_token_probe")])
        try:
            with self.assertRaisesRegex(ValueError, r"reserved C\+\+ anchor"):
                load_request_queue(path)
        finally:
            os.unlink(path)

    def test_clean_identifiers_still_load(self):
        path = _write_queue([_queue_row("session_a", "request_0")])
        try:
            specs = load_request_queue(path)
            self.assertEqual(specs[0].session_id, "session_a")
        finally:
            os.unlink(path)


class TopologyWhitelistGuardTests(unittest.TestCase):
    """2. M-28:拓扑与 memory-type 白名单超售拒绝。"""

    @staticmethod
    def _load(topology, memory_type="PER_NPU_MEMORY_EXPANSION"):
        config = {
            "schema-version": 1, "slug": "s", "label": "l", "paper-case": "p",
            "mesh": {"rows": 2, "columns": 2,
                     "topology-by-network-dimension": list(topology)},
            "local-hbm": {
                "bandwidth-gbps": 1.0, "latency-ns": 0,
                "capacity-profiles": {
                    "profile": {"bytes": 1, "label": "l", "note": "n"}}},
            "d2d": {"bandwidth-gbps": 1.0, "latency-ns": 0},
            "remote-memory": {
                "memory-type": memory_type, "bandwidth-gbps": 1.0,
                "latency-ns": 1, "npu-selection": "mesh-boundary",
                "logical-pool": "x"},
            "compute": {"peak-perf-tflops": 1.0},
            "notes": ["n"],
        }
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(config, handle)
        handle.close()
        try:
            return load_hardware_config(Path(handle.name), "profile")
        finally:
            os.unlink(handle.name)

    def test_mesh_and_line_accepted(self):
        for topology in (("Line", "Line"), ("Mesh", "Mesh")):
            hardware = self._load(topology)
            self.assertEqual(
                hardware.topology_by_network_dimension, topology)

    def test_divergent_topologies_rejected(self):
        for topology in (("Ring", "Ring"), ("Switch", "Line"),
                         ("FullyConnected", "Line")):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                self._load(topology)

    def test_no_memory_expansion_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self._load(("Line", "Line"), "NO_MEMORY_EXPANSION")


class CommGroupCoverageGuardTests(unittest.TestCase):
    """3. M-29:跨组 rank 重叠/整网覆盖缺口拒绝。"""

    def test_cross_group_rank_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            _prepare_comm_groups(
                [("1", [0, 1, 4, 5]), ("2", [4, 5, 6, 7])], _hardware())

    def test_uncovered_mesh_rejected(self):
        with self.assertRaisesRegex(ValueError, "uncovered"):
            _prepare_comm_groups(
                [("1", [0, 1, 4, 5]), ("2", [2, 3, 6, 7])], _hardware())

    def test_exact_partition_accepted(self):
        groups = _prepare_comm_groups(
            [("1", [0, 1, 4, 5]), ("2", [2, 3, 6, 7]),
             ("3", [8, 9, 10, 11, 12, 13, 14, 15])],
            _hardware())
        self.assertEqual(sorted(groups), ["1", "2", "3"])


class LoaderQueueStructureInvariantTests(unittest.TestCase):
    """4. Minor(2026-09-30):request queue 会话块结构不变量 loader 侧
    fail-closed(C++ 读者 WindowedTraceReader 同款:session 块连续/块内
    turn 严格 +1/turn-0 每会话至多一行)——此前 Python loader 更宽,违规
    输入到 C++ 读取期才拒绝,报错远离根因。"""

    @staticmethod
    def _turn_row(session_id, request_id, turn_index):
        # turn-0 行带绝对到达、turn>0 行带 interval(turn_index 字段交叉
        # 规则的合规形态;列序 = _queue_row 注释)。
        if turn_index == 0:
            return (session_id, "0", request_id, "64", "4", "1000", "", "")
        return (session_id, str(turn_index), request_id, "64", "4", "",
                "1000", "")

    @staticmethod
    def _load(rows):
        path = _write_queue(rows)
        try:
            return load_request_queue(path)
        finally:
            os.unlink(path)

    def test_compliant_block_interleaving_loads(self):
        specs = self._load([
            self._turn_row("s_a", "a0", 0),
            self._turn_row("s_a", "a1", 1),
            self._turn_row("s_b", "b0", 0),
            self._turn_row("s_b", "b1", 1),
            self._turn_row("s_b", "b2", 2),
        ])
        self.assertEqual(
            [(spec.session_id, spec.turn_index) for spec in specs],
            [("s_a", 0), ("s_a", 1), ("s_b", 0), ("s_b", 1), ("s_b", 2)],
        )

    def test_non_contiguous_session_block_rejected(self):
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            self._load([
                self._turn_row("s_a", "a0", 0),
                self._turn_row("s_b", "b0", 0),
                self._turn_row("s_a", "a1", 1),
            ])

    def test_turn_gap_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            self._load([
                self._turn_row("s_a", "a0", 0),
                self._turn_row("s_a", "a2", 2),
            ])

    def test_duplicate_turn0_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            self._load([
                self._turn_row("s_a", "a0", 0),
                self._turn_row("s_a", "a0_again", 0),
            ])

    def test_turn0_missing_first_session_rejected(self):
        # 会话首行必须 turn-0(块内序号自 0 起)。
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            self._load([self._turn_row("s_a", "a1", 1)])


if __name__ == "__main__":
    unittest.main()
