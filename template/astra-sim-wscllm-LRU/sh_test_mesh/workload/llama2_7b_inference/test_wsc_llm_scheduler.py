#!/usr/bin/env python3
"""Focused tests for WSC-LLM PD-disaggregated trace-time request mapping."""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from session_kv_manager import SessionKVCacheManager  # noqa: E402
import wsc_llm_scheduler as wsc_llm_scheduler_module  # noqa: E402
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    MAX_EQUIVALENT_SHORTEST_PATHS,
    MAX_ROUTE_CANDIDATE_COMBINATIONS,
    PREFILL_ROLE,
    InstanceGraph,
    PrefillQueueSnapshot,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    WscLlmTopology,
    build_instances,
    build_static_pd_mapping,
    estimate_model_weight_bytes,
    kv_cache_bytes_for_tokens,
    rank_coordinates,
    select_prefill_instance,
)
from generate_wsc_llm_trace import (  # noqa: E402
    SUPPORTED_CONFIG_KEYS,
    load_wsc_llm_trace_config,
)
from generate_trace import (  # noqa: E402
    COMM_COLL_NODE,
    COMP_NODE,
    REMOTE_WEIGHT_ATTR,
    TraceBuilder,
    shard_extent,
    transformer_pass,
    transformer_pass_aggregated,
)


EXPECTED_RANKS = (
    (20, 21, 26, 27, 32, 33),
    (2, 3, 8, 9, 14, 15),
    (18, 19, 24, 25, 30, 31),
    (22, 23, 28, 29, 34, 35),
    (38, 39, 44, 45, 50, 51),
    (0, 1, 6, 7, 12, 13),
    (4, 5, 10, 11, 16, 17),
    (36, 37, 42, 43, 48, 49),
    (40, 41, 46, 47, 52, 53),
)
EXPECTED_ROLES = (
    DECODE_ROLE,
    PREFILL_ROLE,
    DECODE_ROLE,
    DECODE_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
)
EXPECTED_STATIC_ROUTES = {
    1: (0, (1, 0)),
    4: (0, (4, 0)),
    5: (2, (5, 2)),
    6: (3, (6, 3)),
    7: (2, (7, 2)),
    8: (3, (8, 3)),
}
# 真实 9 实例布局（trace_config 顺序，roles = [D,P,D,D,P,P,P,P,P]）。
# 中-1 裁决（2026-08-20）：实测该布局下每个 Prefill 实例的最近 Decode 实例
# 唯一——实例级平局不存在；布局若改动引入平局，
# test_real_layout_nearest_decode_is_unique 失败提示需重新裁决。
REAL_LAYOUT_INSTANCE_SPECS = (
    WscLlmInstanceSpec("decode_0", "1", (20, 21, 26, 27, 32, 33), DECODE_ROLE),
    WscLlmInstanceSpec("prefill_1", "2", (2, 3, 8, 9, 14, 15), PREFILL_ROLE),
    WscLlmInstanceSpec("decode_2", "3", (18, 19, 24, 25, 30, 31), DECODE_ROLE),
    WscLlmInstanceSpec("decode_3", "4", (22, 23, 28, 29, 34, 35), DECODE_ROLE),
    WscLlmInstanceSpec("prefill_4", "5", (38, 39, 44, 45, 50, 51), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_5", "6", (0, 1, 6, 7, 12, 13), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_6", "7", (4, 5, 10, 11, 16, 17), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_7", "8", (36, 37, 42, 43, 48, 49), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_8", "9", (40, 41, 46, 47, 52, 53), PREFILL_ROLE),
)


# ------------------------------------------------------------------------
# request-neutral 合成 fixture(方案 §0.3 / §3 步骤 0-1):checked-in 配置的
# request_queue_csv 为占位路径,正式入口缺失输入 fail-closed。测试需要队列
# 数据时只用本文件内手写的合成队列(绝不引用任何真实 trace 数据),通过
# 临时配置副本把 request_queue_csv 指向合成队列后加载。
# ------------------------------------------------------------------------
REQUEST_QUEUE_HEADER = (
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns,description"
)
# 8 个 turn-0 槽位 + 2 个 turn>0 槽位(槽位口径与 legacy 在线测试共用;
# turn-0 行必须带 session_arrival_time_ns,turn>0 行必须带 interval)。
# 行序合规:session 块连续 + 块内 turn 严格 +1(2026-09-30 起 loader 按
# C++ 读者同款结构不变量 fail-closed,夹具数据同步改为合规形态)。
SYNTHETIC_QUEUE_ROWS = (
    ("fixture_s0", "0", "fixture_s0_r0", "100", "5", "0", "", "synthetic fixture"),
    ("fixture_s0", "1", "fixture_s0_r1", "50", "4", "", "1000", "synthetic fixture"),
    ("fixture_s1", "0", "fixture_s1_r0", "200", "6", "0", "", "synthetic fixture"),
    ("fixture_s1", "1", "fixture_s1_r1", "60", "5", "", "2000", "synthetic fixture"),
    ("fixture_s2", "0", "fixture_s2_r0", "300", "7", "0", "", "synthetic fixture"),
    ("fixture_s3", "0", "fixture_s3_r0", "400", "8", "0", "", "synthetic fixture"),
    ("fixture_s4", "0", "fixture_s4_r0", "500", "9", "0", "", "synthetic fixture"),
    ("fixture_s5", "0", "fixture_s5_r0", "600", "10", "0", "", "synthetic fixture"),
    ("fixture_s6", "0", "fixture_s6_r0", "700", "11", "0", "", "synthetic fixture"),
    ("fixture_s7", "0", "fixture_s7_r0", "800", "12", "0", "", "synthetic fixture"),
)

_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="wscllm_test_fixture_")
# 模块级临时目录随进程退出显式清理,不留 /tmp 残留。
atexit.register(_FIXTURE_DIR.cleanup)


def _write_synthetic_queue(path: Path) -> None:
    path.write_text(
        REQUEST_QUEUE_HEADER
        + "\n"
        + "\n".join(",".join(row) for row in SYNTHETIC_QUEUE_ROWS)
        + "\n",
        encoding="utf-8",
    )


def load_checked_in_config() -> object:
    """request-neutral fixture:checked-in 配置 + 手写合成队列(一次性物化)。

    相对路径(hardware/、system/ 等)仍按 SH_TEST_DIR 解析,配置字段语义与
    checked-in 配置一致;仅输入队列槽位替换为合成数据。
    """
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    if not queue_path.exists():
        _write_synthetic_queue(queue_path)
    config_path = Path(_FIXTURE_DIR.name) / "trace_config.csv"
    if not config_path.exists():
        lines = []
        for raw_line in (MODULE_DIR / "trace_config.csv").read_text(
            encoding="utf-8"
        ).splitlines():
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


def load_config_variant(tag: str, rewrite) -> object:
    """按行改写 checked-in trace_config 后加载(loader fail-closed 回归
    钉子用例)。

    rewrite(line) 返回改写后的行,或 None 表示删除该行;queue 行固定替换
    为合成 fixture 队列(与 load_checked_in_config 同款)。
    """
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    if not queue_path.exists():
        _write_synthetic_queue(queue_path)
    lines = []
    for raw_line in (MODULE_DIR / "trace_config.csv").read_text(
        encoding="utf-8"
    ).splitlines():
        if raw_line.startswith("config,request_queue_csv,"):
            lines.append(
                "config,request_queue_csv,"
                + str(queue_path)
                + ",,,,,request-neutral synthetic fixture queue (unit test)"
            )
            continue
        rewritten = rewrite(raw_line)
        if rewritten is not None:
            lines.append(rewritten)
    config_path = Path(_FIXTURE_DIR.name) / f"trace_config_{tag}.csv"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_wsc_llm_trace_config(config_path)


def checked_in_topology() -> tuple[object, object]:
    config = load_checked_in_config()
    specs = tuple(
        WscLlmInstanceSpec(
            name=group.name,
            pg_name=group.pg_name,
            ranks=group.ranks,
            phase_role=group.phase_role,
        )
        for group in config.inference_groups
    )
    return config, build_instances(config.hardware, specs)


class WscLlmSchedulerTests(unittest.TestCase):
    def test_checked_in_config_is_request_neutral_and_fails_closed_without_input(self) -> None:
        """checked-in 配置 = request-neutral(占位队列路径,不绑定任何默认
        request 队列)+ 缺失输入 fail-closed + 配置字段语义(拓扑/角色/模型/
        硬件;不依赖任何真实 trace 计数,见方案 §0.3 与 §3 步骤 0-1)。"""
        config, topology = checked_in_topology()
        self.assertEqual(config.model_name, "llama2_7b")
        self.assertEqual(config.mlp_variant, "swiglu")
        self.assertEqual(config.npus_count, 54)
        self.assertEqual((config.hardware.mesh_rows, config.hardware.mesh_cols), (9, 6))
        self.assertEqual(config.hardware.d2d_latency_ns, 5)
        self.assertEqual(config.hardware.local_hbm_latency_ns, 100)
        # coarse_die_v2 出厂默认(阶段2,2026-10-18):checked-in 硬件带完整
        # coarse-die 描述符,容量 = 库派生 HBM 容量(144 GB =
        # 2 行 x 2 栈 x 36 GB,validation-coarse-144g 对齐;旧 64 GB /
        # 160 GiB 断言随 die 档位一同退居 legacy 臂)。
        self.assertEqual(config.hardware.local_hbm_capacity_bytes, 144000000000)
        self.assertEqual(config.hardware_config.name, "face_case5_config_c.json")
        # coarse 后端(阶段2):profile 随 coarse 派生容量对齐切换。
        self.assertEqual(config.hardware_capacity_profile, "validation-coarse-144g")
        self.assertEqual(config.system_config.name, "system.json")
        self.assertEqual(config.network_config.name, "network.yml")
        self.assertEqual(config.comm_group_config.name, "comm_group.json")
        # S3-5:remote_memory_config 镜像字段已删;物化产物断言改经
        # runtime_config 目录(system/network/comm_group 同目录物化)。
        self.assertTrue(
            config.system_config.with_name("remote_memory.json").is_file())
        with config.system_config.open(encoding="utf-8") as source:
            system_raw = json.load(source)
        self.assertEqual(system_raw["local-mem-bw"], 4710.4)
        self.assertEqual(system_raw["remote-mem-bw"], 512)
        # coarse 后端(阶段2):峰值算力为类型化速率
        # (die_peak_int4_ops_s / die_peak_vec_eqops_s),不回流通用
        # peak-perf 标量(mesh_die_v1 时代的 126.6 随该后端退役)。
        self.assertNotIn("peak-perf", system_raw)
        self.assertEqual(system_raw["compute-cost-model"], "die")
        self.assertEqual(system_raw["die-model-profile"], "coarse_die_v2")
        self.assertTrue(Path(system_raw["die-model-config"]).is_absolute())
        self.assertIn("npus_count: [ 6, 9 ]", config.network_config.read_text())
        self.assertEqual(len(config.inference_groups), 9)
        self.assertEqual(tuple(group.ranks for group in config.inference_groups), EXPECTED_RANKS)
        self.assertEqual(
            tuple(group.phase_role for group in config.inference_groups),
            EXPECTED_ROLES,
        )
        self.assertEqual(topology.indices_for_role(PREFILL_ROLE), (1, 4, 5, 6, 7, 8))
        self.assertEqual(topology.indices_for_role(DECODE_ROLE), (0, 2, 3))
        self.assertEqual({len(group.ranks) for group in config.inference_groups}, {6})
        self.assertEqual(
            (
                config.layers,
                config.hidden_size,
                config.ffn_size,
                config.num_heads,
                config.vocab_size,
            ),
            (32, 4096, 11008, 32, 32000),
        )
        self.assertEqual(config.trace_granularity, "request_aggregated")
        self.assertEqual(config.prefill_chunk_size, 512)
        # B3(2026-09-06):checked-in 交付值换 session_lru_tiered(二态
        # 冷热管理;trace_config.csv 的 kv_cache_policy 行是本批次交付物)。
        self.assertEqual(config.kv_cache_policy, "session_lru_tiered")
        # S3-5:kv_reserve_context_tokens/record_planning_iterations 等
        # 零生产读者 dataclass 字段已删(断言改落在权威 CSV 文本上)。
        # kv_reserve_context_tokens 键保留为 SLO 观测口径
        # (slo_tools/hbm_watermark.py 直读本 CSV,交付值 1_000_000);
        # record_planning_iterations 键随 S3-5 退役(行已删)。
        checked_in_config_text = (MODULE_DIR / "trace_config.csv").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "config,kv_reserve_context_tokens,1000000",
            checked_in_config_text,
        )
        self.assertNotIn(
            "record_planning_iterations", checked_in_config_text)
        # request-neutral:合成 fixture 队列的计数只反映 fixture 自身,不依赖
        # 任何真实 trace 数据(裸仓库不物化输入,见方案 §3 步骤 0-1)。
        self.assertEqual(len(config.request_queue), len(SYNTHETIC_QUEUE_ROWS))
        # checked-in 配置本体 = request-neutral:request_queue_csv 是占位路径。
        checked_in_text = (MODULE_DIR / "trace_config.csv").read_text(encoding="utf-8")
        queue_line = next(
            line
            for line in checked_in_text.splitlines()
            if line.startswith("config,request_queue_csv,")
        )
        self.assertEqual(
            queue_line.split(",")[2],
            "llama2_7b_inference/request_queue_placeholder.csv",
        )
        self.assertIn("request-neutral", queue_line)
        # 缺失输入 fail-closed:正式入口(checked-in 配置,无物化队列)必须
        # exit 非 0 并打印 missing request queue,不得回退到任何 stub 队列。
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exit_context:
                load_wsc_llm_trace_config()
        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("missing request queue", stderr.getvalue())

    def test_checked_in_config_keys_all_supported(self) -> None:
        """C-1 守卫(2026-09-30):checked-in trace_config.csv 的全部
        config 行键 ∈ SUPPORTED_CONFIG_KEYS——逐键点名的退役键测试漏新
        退役键(曾漏 output_prefix,裸仓交付态全部入口在 loader 处炸),
        全量枚举守卫把"交付键域 = loader 白名单"钉成不变量。"""
        unsupported = []
        for raw_line in (MODULE_DIR / "trace_config.csv").read_text(
            encoding="utf-8"
        ).splitlines():
            if not raw_line.strip() or raw_line.startswith("kind,"):
                continue
            columns = raw_line.split(",")
            if columns[0] != "config":
                continue  # inference_group 行不属 config 键域
            key = columns[1]
            if key not in SUPPORTED_CONFIG_KEYS:
                unsupported.append(key)
        self.assertEqual(
            unsupported,
            [],
            "checked-in trace_config.csv carries config keys outside "
            "SUPPORTED_CONFIG_KEYS (retired keys must be removed from the "
            "CSV, not just from the loader)",
        )

    def test_kv_eviction_mode_key_rejected(self) -> None:
        """主动驱逐已物理清除(2026-09-05):loader 对 kv_eviction_mode 键
        恢复"未知键 fail-closed 拒绝"——残留该行的旧配置不得静默忽略,
        更不可能借此复活水位逐出路径。"""
        with self.assertRaisesRegex(
                ValueError, "unsupported config key 'kv_eviction_mode'"):
            load_config_variant(
                "eviction_mode_key_rejected",
                lambda line: (
                    line
                    + "\nconfig,kv_eviction_mode,passive_only,,,,,stale v2 key must fail closed"
                    if line.startswith("config,kv_reserve_context_tokens,")
                    else line
                ),
            )

    def test_completion_retains_kv_without_watermark(self) -> None:
        """完成路径零逐出(主动驱逐已物理清除):mark_complete 返回空元组、
        retain_complete 事件保留(D3),会话保持 LOCAL(B2 二态)——逐出
        唯一路径 = 准入期 ensure_physical_fit 单阶段整体逐出。"""
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 300, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        # 手算:gelu 权重 144 B/rank;KV 8 B/token。a 完成(10 token=80 B)后
        # remaining = 300-144-80 = 76 B——完成边界不触发任何逐出,KV 原样
        # 驻留(准入期已按需 fit,完成期没有第二个逐出阶段)。
        manager = SessionKVCacheManager(topology, model)
        self.assertFalse(
            manager.prepare_history(
                "a", 0, 0, 10, "a0", required_context_tokens=10
            ).admission_blocked
        )
        self.assertTrue(manager.grow_prefill("a", 10, 10, "a0").admitted)
        self.assertEqual(manager.mark_complete("a", 10, "a0"), ())
        snapshot = manager.session_snapshot("a")
        self.assertEqual(snapshot.location, "local_hbm")
        self.assertFalse(snapshot.active)
        self.assertEqual(snapshot.last_completion_ns, 10)
        self.assertEqual(
            [event.event_type for event in manager.events],
            ["no_history", "retain_complete"],
        )

    def test_sufficient_physical_fit_skips_noncandidate_scan(self) -> None:
        """3.3 (2026-10-17) gate 回归钉子:noncandidate_resident 的
        O(会话数 x tp) 边际扫描已挪入 insufficient 门内——容量充足
        直达返回时零次 _local_prefix_shards 调用(挪动前对每个匹配的
        活跃/保护会话各调一次);逐出路径语义不变,由
        test_passive_only_admission_pressure_evicts_whole_lru_sessions
        等既有测试覆盖。"""
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 300, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        # 一个活跃驻留会话(正是边际扫描的计数对象):权重 144 B/rank、
        # 10 token = 80 B,remaining = 300-144-80 = 76 B;要求 8 B 远小于
        # remaining -> sufficient,直达返回。
        self.assertFalse(
            manager.prepare_history(
                "active", 0, 0, 30, "active0", required_context_tokens=10
            ).admission_blocked
        )
        self.assertTrue(
            manager.grow_prefill("active", 10, 10, "active0").admitted)
        with mock.patch.object(
                manager, "_local_prefix_shards",
                wraps=manager._local_prefix_shards) as shard_scan:
            fit = manager.ensure_physical_fit(0, (8,), 30, "active0")
        self.assertTrue(fit.admitted)
        self.assertEqual(fit.evictions, ())
        self.assertEqual(shard_scan.call_count, 0)

    def test_pg_name_guard_is_int_normalized(self) -> None:
        """pg_name 零值守卫 int 规范化(Minor 2026-09-30):零值与非规范
        十进制字形("00"/" 1"/"+1"/非整数串)一律拒绝,规范正整数放行
        ——字面量 =="0" 守卫曾放行 "00"("00" 经 int 亦为 0,且与 "0"
        异形同 id、绕开 duplicate pg_name 查重)。"""
        hardware = WscLlmHardware(1, 2, 300, 1.0, 1.0, 1.0, 0, 0)
        for bad in ("0", "00", "-1", " 1", "+1", "1.5", "abc"):
            with self.assertRaisesRegex(ValueError, "pg_name"):
                build_instances(
                    hardware,
                    (WscLlmInstanceSpec("d", bad, (0,), DECODE_ROLE),),
                )
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        self.assertEqual(len(topology.instances), 2)

    def test_passive_only_admission_pressure_evicts_whole_lru_sessions(self) -> None:
        """被动逐出唯一路径 = 准入期 ensure_physical_fit 单阶段整体 LRU
        逐出:victim = (last_completion_ns, session_id) LRU 最老完整
        session,最年轻已完成会话不动;逐笔重查、满足即停,允许整体逐出
        产生的过量释放;完成期零逐出,终态驻留全量保留(retire 后亦过)。
        逐出 = remote_store 整体外迁(层域 [0, L) 全层),victim 会话转
        REMOTE。"""
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 500, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        # 手算:权重 144;每会话 10 token = 80 B。p/q/r 完成后驻留 240 B,
        # remaining = 500-144-240 = 116;s 需 30 token = 240 B → 外迁 p(→196)
        # 仍不足 → 外迁 q(→276)≥240 即停;r(最年轻)保留 LOCAL。
        for session_id, completion_ns in (("p", 10), ("q", 20), ("r", 30)):
            self.assertFalse(
                manager.prepare_history(
                    session_id, 0, 0, completion_ns, f"{session_id}0",
                    required_context_tokens=10,
                ).admission_blocked
            )
            self.assertTrue(
                manager.grow_prefill(
                    session_id, 10, completion_ns, f"{session_id}0"
                ).admitted
            )
            self.assertEqual(
                manager.mark_complete(session_id, completion_ns,
                                      f"{session_id}0"), ())
        decision = manager.prepare_history(
            "s", 0, 0, 40, "s0", required_context_tokens=30)
        self.assertFalse(decision.admission_blocked)
        self.assertEqual(
            tuple(transfer.session_id for transfer in decision.evictions),
            ("p", "q"),
        )
        self.assertTrue(all(
            transfer.kind == "remote_store"
            for transfer in decision.evictions))
        self.assertEqual(manager.session_snapshot("r").location, "local_hbm")
        self.assertEqual(manager.session_snapshot("p").location, "remote_memory")
        self.assertEqual(
            tuple(
                snapshot.remaining_bytes
                for snapshot in manager.hbm_snapshots(0)
            ),
            (276,),
        )
        self.assertEqual(manager.remote_bytes_by_rank()[0], 160)
        self.assertEqual(manager.deep_gap_events, 0)
        self.assertTrue(manager.grow_prefill("s", 30, 40, "s0").admitted)
        self.assertEqual(
            tuple(
                snapshot.remaining_bytes
                for snapshot in manager.hbm_snapshots(0)
            ),
            (36,),
        )
        self.assertEqual(manager.mark_complete("s", 41, "s0"), ())
        # 剩余 36 B 的驻留终态合法(完成期零逐出,无第二个逐出阶段)。
        manager.assert_final_state()
        manager.retire_terminal_session("r", 42, "r0")
        manager.retire_terminal_session("s", 43, "s0")
        # 被外迁的 p/q 仍是 REMOTE 会话(远端账面 160),retire 静默核销。
        manager.retire_terminal_session("p", 44, "p0")
        manager.retire_terminal_session("q", 45, "q0")
        manager.assert_final_state()
        self.assertEqual(manager.session_ids, ())
        # retire 静默核销后远端账面归零(B2 终态守恒新口径)。
        self.assertEqual(
            set(manager.remote_bytes_by_rank().values()), {0})

    def test_passive_only_exhausted_candidates_increment_deep_gap_events(self) -> None:
        """D4(2026-09-05):活跃会话占满容量、无可逐冷会话 →
        admission_blocked 优雅推迟 + deep_gap_events 逐次递增(压力事件
        本身按 request 去重,计数不去重);活跃会话不被驱逐。"""
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 400, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        # act 长到 30 token = 240 B → remaining = 400-144-240 = 16。
        self.assertFalse(
            manager.prepare_history(
                "act", 0, 0, 40, "act0", required_context_tokens=30
            ).admission_blocked
        )
        self.assertTrue(manager.grow_prefill("act", 30, 40, "act0").admitted)
        blocked = manager.prepare_history(
            "new", 0, 0, 50, "new0", required_context_tokens=10)
        self.assertTrue(blocked.admission_blocked)
        self.assertEqual(blocked.insufficient_ranks, (0,))
        self.assertEqual(blocked.evictions, ())
        self.assertEqual(manager.deep_gap_events, 1)
        self.assertTrue(manager.session_snapshot("act").active)
        self.assertEqual(manager.session_snapshot("act").location, "local_hbm")
        blocked_event_types = [
            event.event_type
            for event in manager.events
            if event.trigger_request_id == "new0"
        ]
        self.assertEqual(blocked_event_types, ["admission_blocked"])
        # 同请求重试:压力事件去重,深缺口计数仍递增。
        retried = manager.prepare_history(
            "new", 0, 0, 60, "new0", required_context_tokens=10)
        self.assertTrue(retried.admission_blocked)
        self.assertEqual(manager.deep_gap_events, 2)

    def test_decode_instances_are_center_prioritized_and_validation_rejects_inverse(self) -> None:
        _, topology = checked_in_topology()
        decode_distances = [
            topology.instance(index).wafer_center_manhattan_distance
            for index in topology.indices_for_role(DECODE_ROLE)
        ]
        prefill_distances = [
            topology.instance(index).wafer_center_manhattan_distance
            for index in topology.indices_for_role(PREFILL_ROLE)
        ]
        self.assertLessEqual(max(decode_distances), min(prefill_distances))

        hardware = WscLlmHardware(3, 2, 100, 1.0, 1.0, 1.0, 0, 0)
        with self.assertRaisesRegex(ValueError, "center-prioritized"):
            build_instances(
                hardware,
                (
                    WscLlmInstanceSpec("outer_decode", "1", (0, 1), DECODE_ROLE),
                    WscLlmInstanceSpec("center_prefill", "2", (2, 3), PREFILL_ROLE),
                    WscLlmInstanceSpec("outer_prefill", "3", (4, 5), PREFILL_ROLE),
                ),
            )

    def test_exact_static_routes_are_one_hop_and_edge_disjoint(self) -> None:
        _, topology = checked_in_topology()
        mapping = build_static_pd_mapping(topology)
        actual = {
            route.prefill_instance_index: (
                route.decode_instance_index,
                route.path,
            )
            for route in mapping.routes
        }
        self.assertEqual(actual, EXPECTED_STATIC_ROUTES)
        self.assertEqual(mapping.total_hops, 6)
        self.assertEqual(mapping.shared_edge_occurrences, 0)
        self.assertEqual({count for _, _, count in mapping.edge_use_counts}, {1})
        self.assertTrue(all(not route.shared_edges for route in mapping.routes))

    def test_real_layout_nearest_decode_is_unique(self) -> None:
        """钉住"真实布局无实例平局"（中-1 裁决 2026-08-20）。

        本仓 decode 为静态最近实例映射；实测真实 9 实例布局下每个 Prefill
        实例的最近 Decode 实例唯一——实例级平局不存在。布局若改动引入
        平局，本测试失败提示需重新裁决（勿静默改断言迁就新布局）。
        """
        hardware = WscLlmHardware(
            mesh_rows=9,
            mesh_cols=6,
            local_hbm_capacity_bytes=50,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=2.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        topology = build_instances(hardware, REAL_LAYOUT_INSTANCE_SPECS)
        graph = InstanceGraph(topology)
        decode_indices = topology.indices_for_role(DECODE_ROLE)
        self.assertEqual(decode_indices, (0, 2, 3))
        expected_nearest_decode = {1: 0, 4: 0, 5: 2, 6: 3, 7: 2, 8: 3}
        for prefill_index in topology.indices_for_role(PREFILL_ROLE):
            decode_distances = {
                decode_index: graph.shortest_distance(prefill_index, decode_index)
                for decode_index in decode_indices
            }
            nearest_distance = min(decode_distances.values())
            nearest_decode_indices = tuple(
                decode_index
                for decode_index in decode_indices
                if decode_distances[decode_index] == nearest_distance
            )
            self.assertEqual(
                len(nearest_decode_indices),
                1,
                msg=(
                    f"real-layout nearest-Decode tie for Prefill instance "
                    f"{prefill_index} at distance {nearest_distance} "
                    f"(decode distances {decode_distances}); the 2026-08-20 "
                    "decode tie-break ruling requires re-adjudication"
                ),
            )
            self.assertEqual(
                nearest_decode_indices[0],
                expected_nearest_decode[prefill_index],
            )

    def test_prefill_selection_uses_request_count_then_config_order(self) -> None:
        queues = (
            PrefillQueueSnapshot(1, 2),
            PrefillQueueSnapshot(4, 1),
            PrefillQueueSnapshot(5, 1),
        )
        self.assertEqual(select_prefill_instance(queues), 4)
        tied_out_of_order = (
            PrefillQueueSnapshot(8, 0),
            PrefillQueueSnapshot(1, 0),
        )
        self.assertEqual(select_prefill_instance(tied_out_of_order), 1)

    def test_manager_deletes_multiple_lru_victims_in_one_pool_snapshot(self) -> None:
        """一次准入逐出多个已完成 LRU 受害者,且整池快照恰好取一次。

        2026-10-01 审查更名(原名 `..._but_not_active_kv` 名称漂移):本用例
        中 a、b 在逐出发生前均已 ``mark_complete``,场内并无 active 会话;
        "活跃会话不被驱逐"由本文件
        ``test_passive_only_exhausted_candidates_increment_deep_gap_events``
        与 test_tiered_eviction_sequence.py 的
        ``test_active_residency_counts_into_fit_feasibility`` 锁定。
        """

        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 400, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        for session_id, completion_ns in (("a", 10), ("b", 20)):
            decision = manager.prepare_history(
                session_id,
                0,
                0,
                completion_ns,
                f"{session_id}0",
                required_context_tokens=10,
            )
            self.assertFalse(decision.admission_blocked)
            self.assertTrue(
                manager.grow_prefill(
                    session_id, 10, completion_ns, f"{session_id}0"
                ).admitted
            )
            manager.mark_complete(session_id, completion_ns, f"{session_id}0")

        candidate_calls = 0
        original_candidates = manager._completed_local_candidates

        def snapshot_candidates(*args, **kwargs):
            nonlocal candidate_calls
            candidate_calls += 1
            return original_candidates(*args, **kwargs)

        manager._completed_local_candidates = snapshot_candidates
        try:
            c_decision = manager.prepare_history(
                "c", 0, 0, 30, "c0", required_context_tokens=30
            )
        finally:
            manager._completed_local_candidates = original_candidates

        # The admission needs two full offloads; the single whole-session
        # LRU pool snapshot is taken exactly once (no staging stages left).
        self.assertEqual(candidate_calls, 1)
        self.assertEqual(
            tuple(transfer.session_id for transfer in c_decision.evictions),
            ("a", "b"),
        )
        self.assertTrue(all(
            transfer.kind == "remote_store"
            for transfer in c_decision.evictions))
        # 整体逐出形态:层域 [0, L) 全层、驻留前缀 4 → 0(L=1)。
        self.assertEqual(
            [(transfer.layer_start, transfer.layer_end) for transfer in
             c_decision.evictions],
            [(0, 1), (0, 1)])
        self.assertEqual(
            [transfer.resident_prefix_layers_after
             for transfer in c_decision.evictions],
            [0, 0])
        self.assertEqual(
            [
                (event.event_type, event.session_id, event.phase,
                 event.reason, event.trigger_request_id)
                for event in manager.events
            ],
            [
                ("no_history", "a", "history", "window_first_request", "a0"),
                ("retain_complete", "a", "completion", "request_completed_keep_kv", "a0"),
                ("no_history", "b", "history", "window_first_request", "b0"),
                ("retain_complete", "b", "completion", "request_completed_keep_kv", "b0"),
                ("evict_session", "a", "history",
                 "evict_history_and_prefill_admission_session:layers0-1", "c0"),
                ("evict_session", "b", "history",
                 "evict_history_and_prefill_admission_session:layers0-1", "c0"),
                ("no_history", "c", "history", "window_first_request", "c0"),
            ],
        )
        self.assertEqual(
            tuple(snapshot.remaining_bytes for snapshot in manager.hbm_snapshots(0)),
            (256,),
        )
        self.assertEqual(manager.session_snapshot("a").logical_context_tokens, 10)
        self.assertEqual(manager.session_snapshot("a").location, "remote_memory")
        # 被逐会话全部层在远端池账面,由该 rank 的账本承载(二态)。
        self.assertEqual(manager.session_snapshot("a").remote_bytes, 80)
        self.assertTrue(manager.grow_prefill("c", 30, 30, "c0").admitted)

    def test_eviction_victim_order_is_real_lru_with_deterministic_ties(self) -> None:
        """victim 顺序 = 真实 LRU 升序(last_completion_ns);同 tick 并列
        按 session_id 确定性 tie-break。"""
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 700, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        # 完成时间故意非字母序:b(5) < c(15) < a(25) → LRU 序 = b,c,a。
        for session_id, completion_ns in (
            ("a", 25), ("b", 5), ("c", 15),
        ):
            decision = manager.prepare_history(
                session_id, 0, 0, completion_ns, f"{session_id}0",
                required_context_tokens=10)
            self.assertFalse(decision.admission_blocked)
            self.assertTrue(
                manager.grow_prefill(
                    session_id, 10, completion_ns, f"{session_id}0").admitted)
            manager.mark_complete(
                session_id, completion_ns, f"{session_id}0")
        # 同 tick tie:三个会话完成时间全部改写为同一 tick 40(mark_
        # complete 是唯一时间戳写入点;active 会话不可逐,先改 inactive
        # 会话的时间戳语义 = mark_complete 重放,这里直接构造新会话:
        # d/e/f 全部在 tick 50 完成)。
        for session_id in "def":
            decision = manager.prepare_history(
                session_id, 0, 0, 50, f"{session_id}0",
                required_context_tokens=10)
            self.assertFalse(decision.admission_blocked)
            self.assertTrue(
                manager.grow_prefill(
                    session_id, 10, 50, f"{session_id}0").admitted)
            manager.mark_complete(session_id, 50, f"{session_id}0")

        # 新请求需要 41 token(328 B;6 会话驻留后余 80)→ victim =
        # LRU 序 b(5) < c(15) < a(25),仍不足后进入同 tick(50)tie 组,
        # 组内按 session_id 字典序取 d(d < e < f)。
        decision = manager.prepare_history(
            "g", 0, 0, 60, "g0", required_context_tokens=41)
        self.assertFalse(decision.admission_blocked)
        self.assertEqual(
            tuple(transfer.session_id for transfer in decision.evictions),
            ("b", "c", "a", "d"))

    def test_terminal_retirement_releases_local_kv_and_fails_closed(self) -> None:
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 400, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        self.assertFalse(
            manager.prepare_history(
                "terminal", 0, 0, 10, "terminal_r0",
                required_context_tokens=10,
            ).admission_blocked
        )
        self.assertTrue(manager.grow_prefill("terminal", 10, 10, "terminal_r0").admitted)
        with self.assertRaisesRegex(RuntimeError, "inactive completed"):
            manager.retire_terminal_session("terminal", 10, "terminal_r0")

        manager.mark_complete("terminal", 10, "terminal_r0")
        completion_actions = manager.events
        self.assertEqual(
            manager.retire_terminal_session("terminal", 10, "terminal_r0"),
            0,
        )
        self.assertEqual(manager.events, completion_actions)
        self.assertEqual(manager.session_ids, ())
        self.assertIsNone(manager.session_snapshot("terminal"))
        self.assertTrue(
            all(snapshot.resident_kv_bytes == 0
                for snapshot in manager.hbm_snapshots())
        )
        with self.assertRaises(KeyError):
            manager.retire_terminal_session("terminal", 10, "terminal_r0")

    def test_terminal_retirement_bounds_many_single_turn_sessions(self) -> None:
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 400, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        for index in range(32):
            session_id = f"single_{index}"
            request_id = f"{session_id}_r0"
            self.assertFalse(
                manager.prepare_history(
                    session_id, 0, 0, index, request_id,
                    required_context_tokens=10,
                ).admission_blocked
            )
            self.assertTrue(manager.grow_prefill(session_id, 10, index, request_id).admitted)
            manager._pressure_event(
                now_ns=index,
                phase="admission",
                event_type="admission_blocked",
                reason="test_pressure_ownership",
                trigger_request_id=request_id,
                target_instance_index=0,
                before=(),
                after=(),
                insufficient_ranks=(),
            )
            self.assertIn(request_id, manager._pressure_event_keys_by_request)
            manager.mark_complete(session_id, index, request_id)
            self.assertNotIn(request_id, manager._pressure_event_keys_by_request)
            self.assertFalse(manager._pressure_event_keys)
            manager.retire_terminal_session(session_id, index, request_id)
            self.assertEqual(manager.session_ids, ())

    def test_pressure_event_dedup_is_request_scoped_and_retired(self) -> None:
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 1_000, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
        )
        manager = SessionKVCacheManager(topology, model)
        requests = (("first", "first_r0"), ("second", "second_r0"))
        for index, (session_id, request_id) in enumerate(requests):
            self.assertFalse(
                manager.prepare_history(
                    session_id,
                    0,
                    0,
                    index,
                    request_id,
                    required_context_tokens=10,
                ).admission_blocked
            )
            self.assertTrue(
                manager.grow_prefill(session_id, 10, index, request_id).admitted
            )

        def record_pressure(request_id: str, now_ns: int) -> None:
            manager._pressure_event(
                now_ns=now_ns,
                phase="admission",
                event_type="admission_blocked",
                reason="test_request_scoped_pressure",
                trigger_request_id=request_id,
                target_instance_index=0,
                before=(),
                after=(),
                insufficient_ranks=(),
            )

        for now_ns in range(3):
            record_pressure("first_r0", now_ns)
        for now_ns in range(3, 6):
            record_pressure("second_r0", now_ns)

        pressure_events = [
            event
            for event in manager.events
            if event.reason in {
                "test_request_scoped_pressure",
                "retry_after_previous_capacity_block",
            }
        ]
        self.assertEqual(
            [(event.event_type, event.trigger_request_id) for event in pressure_events],
            [
                ("admission_blocked", "first_r0"),
                ("admission_retry", "first_r0"),
                ("admission_blocked", "second_r0"),
                ("admission_retry", "second_r0"),
            ],
        )
        self.assertEqual(
            set(manager._pressure_event_keys_by_request),
            {"first_r0", "second_r0"},
        )
        self.assertEqual(
            manager._pressure_event_keys,
            {
                ("admission_blocked", "admission", "first_r0", 0),
                ("admission_retry", "admission", "first_r0", 0),
                ("admission_blocked", "admission", "second_r0", 0),
                ("admission_retry", "admission", "second_r0", 0),
            },
        )

        manager.mark_complete("first", 10, "first_r0")
        manager.retire_terminal_session("first", 10, "first_r0")
        self.assertEqual(set(manager._pressure_event_keys_by_request), {"second_r0"})
        self.assertEqual(
            manager._pressure_event_keys,
            {
                ("admission_blocked", "admission", "second_r0", 0),
                ("admission_retry", "admission", "second_r0", 0),
            },
        )

        manager.mark_complete("second", 11, "second_r0")
        manager.retire_terminal_session("second", 11, "second_r0")
        self.assertFalse(manager._pressure_event_keys)
        self.assertFalse(manager._pressure_event_keys_by_request)
        manager.assert_final_state()


    def test_llama2_7b_tp6_partition_is_exact_without_model_padding(self) -> None:
        config = load_checked_in_config()
        attention_heads = tuple(
            shard_extent(config.num_heads, 6, index) for index in range(6)
        )
        self.assertEqual(sum(attention_heads), 32)
        self.assertEqual(attention_heads.count(6), 2)
        self.assertEqual(attention_heads.count(5), 4)
        self.assertEqual(
            sum(shard_extent(config.ffn_size, 6, index) for index in range(6)),
            config.ffn_size,
        )
        self.assertEqual(
            sum(shard_extent(config.vocab_size, 6, index) for index in range(6)),
            config.vocab_size,
        )
        self.assertEqual(estimate_model_weight_bytes(config.model), 13_476_831_232)

    def test_aggregated_transformer_pass_preserves_expanded_totals(self) -> None:
        spans = ((3, 7), (1, 11), (5, 19))
        parameters = {
            "layers": 3,
            "hidden_size": 24,
            "ffn_size": 49,
            "tensor_parallel": 4,
            "pg_name": "1",
            "vocab_size": 101,
            "bytes_per_elem": 2,
            "num_heads": 6,
            "tensor_parallel_rank": 0,
            "mlp_variant": "swiglu",
        }
        expanded = TraceBuilder(remote_operand_loads=True)
        for index, (tokens, kv_length) in enumerate(spans):
            transformer_pass(
                expanded,
                phase=f"expanded_{index}",
                tokens=tokens,
                kv_length=kv_length,
                **parameters,
            )
        aggregated = TraceBuilder(remote_operand_loads=True)
        pass_count = transformer_pass_aggregated(
            aggregated,
            phase="aggregated",
            pass_spans=spans,
            **parameters,
        )

        def attribute(node: object, name: str) -> int:
            for attr in node.attr:
                if attr.name == name:
                    return int(attr.uint64_val)
            return 0

        def totals(builder: TraceBuilder) -> tuple[int, int, int, int]:
            compute_nodes = [node for node in builder.nodes if node.type == COMP_NODE]
            collective_nodes = [
                node for node in builder.nodes if node.type == COMM_COLL_NODE
            ]
            return (
                sum(attribute(node, "num_ops") for node in compute_nodes),
                sum(attribute(node, "tensor_size") for node in compute_nodes),
                sum(attribute(node, REMOTE_WEIGHT_ATTR) for node in compute_nodes),
                sum(attribute(node, "comm_size") for node in collective_nodes),
            )

        self.assertEqual(pass_count, len(spans))
        self.assertEqual(len(aggregated.nodes), 17)
        self.assertEqual(sum(node.type == COMP_NODE for node in aggregated.nodes), 15)
        self.assertEqual(sum(node.type == COMM_COLL_NODE for node in aggregated.nodes), 2)
        self.assertEqual(totals(aggregated), totals(expanded))

    def test_trace_builder_streams_without_retaining_nodes(self) -> None:
        streamed = []
        builder = TraceBuilder(
            remote_operand_loads=False,
            node_sink=streamed.append,
            retain_nodes=False,
        )
        builder.comp("first", num_ops=1, tensor_size=1)
        builder.comp("second", num_ops=2, tensor_size=2)
        self.assertEqual(builder.node_count, 2)
        self.assertFalse(builder.nodes)
        self.assertEqual([node.name for node in streamed], ["first", "second"])
        self.assertEqual(tuple(streamed[1].data_deps), (streamed[0].id,))


class RouteEnumerationBoundTests(unittest.TestCase):
    """SCHED-01(2026-09-26):最短路物化与候选笛卡尔积的 fail-closed 上界。

    等价最短路多的网格/对抗拓扑会组合爆炸;上界内结果必须不变(既有
    static-route 钉子测试继续覆盖合法布局)。"""

    @staticmethod
    def _grid_topology(rows: int, cols: int, decode_ranks) -> WscLlmTopology:
        """rows×cols 网格,每实例 1 个 NPU;decode_ranks 为 Decode 实例,
        其余为 Prefill(中心优先约束由 build_instances 自行校验)。"""
        decode_ranks = frozenset(decode_ranks)
        specs = tuple(
            WscLlmInstanceSpec(
                name=f"n{index}",
                pg_name=str(index + 1),
                ranks=(index,),
                phase_role=DECODE_ROLE if index in decode_ranks else PREFILL_ROLE,
            )
            for index in range(rows * cols)
        )
        return build_instances(
            WscLlmHardware(rows, cols, 100, 1.0, 1.0, 1.0, 0, 0), specs)

    def test_path_materialization_bound_fails_closed(self) -> None:
        """8×8 实例网格对角点间等价最短路 C(14,7)=3432 > 上界:递归物化
        在超界即结构化报错,而非无界消耗 CPU/内存。"""
        topology = self._grid_topology(
            8, 8, decode_ranks=set(range(64)) - {0, 7, 56, 63})
        graph = InstanceGraph(topology)
        with self.assertRaisesRegex(ValueError, "route enumeration bound"):
            graph.all_shortest_paths(0, 63)

    def test_combination_bound_fails_closed_and_default_bounds_are_inert(self) -> None:
        """候选组合上界:收紧到 1 时,多候选拓扑(2×2 网格,每 prefill 有
        2 个最近 decode 候选,组合数 4)fail-closed;缺省上界下同一拓扑
        正常产出映射(上界内结果不变)。"""
        topology = self._grid_topology(2, 2, decode_ranks=(0, 3))
        with mock.patch.object(
                wsc_llm_scheduler_module,
                "MAX_ROUTE_CANDIDATE_COMBINATIONS", 1):
            with self.assertRaisesRegex(
                    ValueError, "candidate combinations"):
                build_static_pd_mapping(topology)
        mapping = build_static_pd_mapping(topology)
        self.assertEqual(len(mapping.routes), 2)
        self.assertEqual(mapping.total_hops, 2)

        # 路径物化上界同样由常量控制:收紧到 1 时 2×2 对角(2 条等价
        # 最短路)即报错,而相邻对(1 条)不受影响。
        graph = InstanceGraph(topology)
        with mock.patch.object(
                wsc_llm_scheduler_module,
                "MAX_EQUIVALENT_SHORTEST_PATHS", 1):
            with self.assertRaisesRegex(ValueError, "route enumeration bound"):
                graph.all_shortest_paths(0, 3)
            self.assertEqual(
                graph.all_shortest_paths(0, 1), ((0, 1),))
        self.assertEqual(
            graph.shortest_distance(0, 3), 2)
        self.assertGreater(MAX_EQUIVALENT_SHORTEST_PATHS, 1)
        self.assertGreater(MAX_ROUTE_CANDIDATE_COMBINATIONS, 1)


class InputScalarValidationTests(unittest.TestCase):
    """ONLINE-05(2026-09-26):标量入口的类型/范围收紧(bool 是 int 子
    类型,显式拒绝;浮点/字符串不得经 int() 静默截断混入算术或拓扑查找)。"""

    def test_kv_cache_bytes_for_tokens_requires_non_negative_int(self) -> None:
        model = WscLlmModel(2, 8, 16, 2, 32, 2, "gelu")
        self.assertEqual(kv_cache_bytes_for_tokens(model, 0), 0)
        self.assertEqual(kv_cache_bytes_for_tokens(model, 3), 2 * 2 * 3 * 8 * 2)
        for bad in (-1, True, False, 1.5, "8", None):
            with self.assertRaises(ValueError):
                kv_cache_bytes_for_tokens(model, bad)

    def test_rank_coordinates_requires_in_range_int(self) -> None:
        hardware = WscLlmHardware(2, 3, 100, 1.0, 1.0, 1.0, 0, 0)
        self.assertEqual(rank_coordinates(hardware, 0), (0, 0))
        self.assertEqual(rank_coordinates(hardware, 5), (1, 2))
        for bad_rank in (-1, 6, True, 2.5, "3", None):
            with self.assertRaises(ValueError):
                rank_coordinates(hardware, bad_rank)


if __name__ == "__main__":
    unittest.main()
