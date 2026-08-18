#!/usr/bin/env python3
"""Focused tests for the trace-generation-time FACE scheduler."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceLut,
    FaceLutEntry,
    FaceModel,
    FaceRequest,
    KVAllocator,
    KVCacheManager,
    KVTransfer,
    KVTransferShard,
    PrefillQueueSnapshot,
    WeightedInstanceGraph,
    attention_heads_by_tp_rank,
    build_instances,
    deterministic_xy_route,
    estimate_model_weight_bytes,
    kv_cache_bytes_for_tokens,
    kv_cache_shard_bytes_for_tokens,
    manhattan_hops,
    model_weight_shard_bytes_by_tp_rank,
    nearest_edge_rank,
    physical_edge_ranks,
    plan_face_requests,
    select_decode_instance,
    select_prefill_instance,
)
from generate_face_trace import (  # noqa: E402
    TransferTagAllocator,
    TransferTriggerGate,
    _emit_kv_transfer,
    _emit_tp_readiness_barrier,
    load_face_trace_config,
    main as generate_face_trace_main,
    order_plans_for_static_emission,
    select_first_session_requests,
    write_face_trace,
)
from generate_trace import (  # noqa: E402
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_STORE_NODE,
    REMOTE_WEIGHT_ATTR,
    REQUEST_QUEUE_COLUMNS,
    RequestSpec,
    TraceBuilder,
    load_remote_memory_config,
    shard_extent,
    transformer_pass,
    transformer_pass_aggregated,
)


def line_topology() -> tuple[FaceHardware, object]:
    hardware = FaceHardware(
        mesh_rows=3,
        mesh_cols=2,
        local_hbm_capacity_bytes=100,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
            FaceInstanceSpec("ins2", "3", (4, 5)),
        ),
    )
    return hardware, topology


class FaceSchedulerTests(unittest.TestCase):
    # request-neutral（裸仓库还原，2026-08-16，阶段 7）：物化输入删除后
    # config 依赖用例跳过——按 traces/PROVENANCE.md 物化输入并在
    # trace_config.csv 指定后自动恢复（占位路径 fail-closed 由
    # test_checked_in_config_is_request_neutral_and_fails_closed_without_input
    # 常态覆盖）。
    _MATERIALIZED = (
        MODULE_DIR / "traces" /
        "astra_compute_20_first_30_seconds_request_queue_recompute.csv"
    ).is_file()

    @staticmethod
    def _request(
        session_id: str,
        turn_index: int,
        request_id: str,
    ) -> RequestSpec:
        return RequestSpec(
            session_id=session_id,
            turn_index=turn_index,
            request_id=request_id,
            prefill_length=10 + turn_index,
            decode_length=2 + turn_index,
            session_arrival_time_ns=0 if turn_index == 0 else None,
            inter_request_interval_ns=None if turn_index == 0 else 1000,
        )

    @staticmethod
    def _tiny_kv_manager(
        *,
        capacity_bytes: int = 200,
        reserve_context_tokens: int = 40,
    ) -> tuple[FaceHardware, FaceModel, object, KVCacheManager]:
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
            layers=1,
            hidden_size=2,
            ffn_size=2,
            num_heads=2,
            vocab_size=2,
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
        return (
            hardware,
            model,
            topology,
            KVCacheManager(
                topology,
                model,
                reserve_context_tokens=reserve_context_tokens,
            ),
        )

    @staticmethod
    def _seed_local_session(
        manager: KVCacheManager,
        *,
        session_id: str,
        instance_index: int,
        context_tokens: int,
        completion_ns: int | None,
    ) -> None:
        before, transfer, evictions = manager.prepare_prefill(
            session_id=session_id,
            target_instance_index=instance_index,
            history_tokens=0,
            trigger_request_id=f"{session_id}_initial",
        )
        if before is not None or transfer is not None or evictions:
            raise AssertionError("new session unexpectedly required history movement")
        growth_evictions = manager.expand_prefill(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=context_tokens,
            trigger_request_id=f"{session_id}_initial",
        )
        if growth_evictions:
            raise AssertionError("test fixture unexpectedly evicted a session")
        if completion_ns is not None:
            manager.mark_complete(session_id, completion_ns)

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_shell_config_does_not_build_full_face_plan(self) -> None:
        config = load_face_trace_config()
        output = io.StringIO()
        with (
            patch(
                "generate_face_trace.load_face_trace_config",
                return_value=config,
            ),
            patch(
                "generate_face_trace.build_face_plan",
                side_effect=AssertionError("shell config must not build a plan"),
            ),
            redirect_stdout(output),
        ):
            generate_face_trace_main(["--print-shell-config"])
        assignments = output.getvalue()
        self.assertIn("REQUEST_COUNT=1177\n", assignments)
        self.assertIn("SESSION_COUNT=112\n", assignments)
        self.assertIn("PREFILL_CHUNK_SIZE=512\n", assignments)
        self.assertIn("PREFILL_RANGE=66-169395\n", assignments)

    def test_first_n_session_selection_keeps_all_source_rows(self) -> None:
        requests = (
            self._request("s0", 0, "s0r0"),
            self._request("s1", 0, "s1r0"),
            self._request("s0", 1, "s0r1"),
            self._request("s2", 0, "s2r0"),
            self._request("s1", 1, "s1r1"),
        )
        selected, session_ids = select_first_session_requests(requests, 2)
        self.assertEqual(session_ids, ("s0", "s1"))
        self.assertEqual(
            tuple(request.request_id for request in selected),
            ("s0r0", "s1r0", "s0r1", "s1r1"),
        )
        all_requests, all_sessions = select_first_session_requests(requests, 0)
        self.assertEqual(all_requests, requests)
        self.assertEqual(all_sessions, ("s0", "s1", "s2"))
        with self.assertRaises(ValueError):
            select_first_session_requests(requests, 4)
        with self.assertRaises(ValueError):
            select_first_session_requests(requests, -1)

    def test_remote_memory_mesh_shape_rejects_malformed_legacy_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "remote_memory.json"
            source.write_text(
                json.dumps(
                    {
                        "memory-type": "PER_NPU_MEMORY_EXPANSION",
                        "npu-ids": [0],
                        "mesh-shape": 4,
                        "remote-mem-latency": 0,
                        "remote-mem-bw": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mesh-shape"):
                load_remote_memory_config(source, 4, mesh_shape=(2, 2))

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_checked_in_astra_compute_selection_uses_first_30_seconds_window(
        self,
    ) -> None:
        config = load_face_trace_config()
        self.assertEqual(config.model_name, "llama2_7b")
        self.assertEqual(config.mlp_variant, "swiglu")
        self.assertEqual(config.npus_count, 54)
        self.assertEqual((config.hardware.mesh_rows, config.hardware.mesh_cols), (9, 6))
        self.assertEqual(config.hardware.d2d_latency_ns, 5)
        self.assertEqual(config.hardware.local_hbm_latency_ns, 100)
        self.assertEqual(len(config.inference_groups), 9)
        self.assertEqual({len(group.ranks) for group in config.inference_groups}, {6})
        self.assertEqual(
            (config.layers, config.hidden_size, config.ffn_size, config.num_heads, config.vocab_size),
            (32, 4096, 11008, 32, 32000),
        )
        expected_request_queue = (
            MODULE_DIR / "traces" /
            "astra_compute_20_first_30_seconds_request_queue_recompute.csv"
        ).resolve()
        self.assertEqual(config.request_queue_csv, expected_request_queue)
        self.assertEqual(config.request_queue_session_limit, 0)
        self.assertEqual(config.prefill_chunk_size, 512)
        self.assertEqual(config.trace_granularity, "request_aggregated")
        self.assertEqual(config.source_request_count, 1177)
        self.assertEqual(config.source_session_count, 112)
        self.assertEqual(len(config.request_queue), 1177)
        self.assertEqual(len(config.selected_session_ids), 112)
        self.assertEqual(
            config.selected_session_ids[:5],
            ("session_0", "session_1", "session_2", "session_3", "session_4"),
        )
        self.assertEqual(
            config.selected_session_ids[-5:],
            ("session_107", "session_108", "session_109", "session_110", "session_111"),
        )
        self.assertEqual(
            {request.session_id for request in config.request_queue},
            {f"session_{index}" for index in range(112)},
        )
        prefill_lengths = [
            request.prefill_length for request in config.request_queue
        ]
        decode_lengths = [request.decode_length for request in config.request_queue]
        arrival_times = [
            request.session_arrival_time_ns
            for request in config.request_queue
            if request.session_arrival_time_ns is not None
        ]
        self.assertEqual((min(prefill_lengths), max(prefill_lengths)), (66, 169395))
        self.assertEqual((min(decode_lengths), max(decode_lengths)), (1, 13812))
        self.assertEqual(len(arrival_times), 112)
        self.assertEqual(
            (min(arrival_times), max(arrival_times)),
            (94835000, 25959142000),
        )
        request_arrivals: dict[str, int] = {}
        for request in config.request_queue:
            if request.turn_index == 0:
                self.assertIsNotNone(request.session_arrival_time_ns)
                request_arrival = request.session_arrival_time_ns
            else:
                self.assertIn(request.session_id, request_arrivals)
                self.assertIsNotNone(request.inter_request_interval_ns)
                request_arrival = (
                    request_arrivals[request.session_id]
                    + request.inter_request_interval_ns
                )
            self.assertIsNotNone(request_arrival)
            request_arrivals[request.session_id] = request_arrival
        self.assertEqual(max(request_arrivals.values()), 29988879000)
        self.assertLessEqual(max(request_arrivals.values()), 30000000000)

    def test_parallel_et_replay_matches_serial_et_bytes(self) -> None:
        """Workers may change CPU placement, never ET contents or FACE policy."""

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            queue_path = temporary_path / "tiny_request_queue.csv"
            with queue_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=REQUEST_QUEUE_COLUMNS)
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "session_id": "tiny_session_0",
                            "turn_index": 0,
                            "request_id": "tiny_session_0_request_0",
                            "prefill_length": 2,
                            "decode_length": 1,
                            "session_arrival_time_ns": 0,
                            "inter_request_interval_ns": "",
                            "description": "parallel replay regression fixture",
                        },
                        {
                            "session_id": "tiny_session_1",
                            "turn_index": 0,
                            "request_id": "tiny_session_1_request_0",
                            "prefill_length": 3,
                            "decode_length": 2,
                            "session_arrival_time_ns": 1000,
                            "inter_request_interval_ns": "",
                            "description": "parallel replay regression fixture",
                        },
                    )
                )

            config_path = temporary_path / "trace_config.csv"
            with (MODULE_DIR / "trace_config.csv").open(
                newline="", encoding="utf-8-sig"
            ) as source:
                reader = csv.DictReader(source)
                assert reader.fieldnames is not None
                fieldnames = reader.fieldnames
                config_rows = list(reader)
            output_dir = temporary_path / "generated"
            for row in config_rows:
                if row.get("kind") != "config":
                    continue
                if row.get("key") == "output_dir":
                    row["value"] = str(output_dir)
                elif row.get("key") == "request_queue_csv":
                    row["value"] = str(queue_path)
                elif row.get("key") == "request_queue_session_limit":
                    row["value"] = "0"
            with config_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(config_rows)

            config = load_face_trace_config(config_path)
            with redirect_stdout(io.StringIO()):
                write_face_trace(config, jobs=1)
            serial_et = {
                rank: (output_dir / f"{config.output_prefix}.{rank}.et").read_bytes()
                for rank in range(config.npus_count)
            }
            serial_lut = (output_dir / "face_lut.csv").read_bytes()
            serial_manifest = (output_dir / "manifest.json").read_bytes()

            with redirect_stdout(io.StringIO()):
                write_face_trace(config, jobs=2)
            parallel_et = {
                rank: (output_dir / f"{config.output_prefix}.{rank}.et").read_bytes()
                for rank in range(config.npus_count)
            }
            self.assertEqual(parallel_et, serial_et)
            self.assertEqual((output_dir / "face_lut.csv").read_bytes(), serial_lut)
            self.assertEqual(
                (output_dir / "manifest.json").read_bytes(), serial_manifest
            )
            self.assertEqual(list(output_dir.glob(".trace-generation-*")), [])

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_llama2_7b_tp6_partition_is_exact_without_model_padding(self) -> None:
        config = load_face_trace_config()
        attention_heads = tuple(
            shard_extent(config.num_heads, 6, index) for index in range(6)
        )
        self.assertEqual(attention_heads, (6, 6, 5, 5, 5, 5))
        self.assertEqual(
            tuple(
                shard_extent(config.ffn_size, 6, index)
                for index in range(6)
            ),
            (1835, 1835, 1835, 1835, 1834, 1834),
        )
        self.assertEqual(
            tuple(
                shard_extent(config.vocab_size, 6, index)
                for index in range(6)
            ),
            (5334, 5334, 5333, 5333, 5333, 5333),
        )
        self.assertEqual(
            estimate_model_weight_bytes(config.model), 13_476_831_232
        )
        weight_shards = model_weight_shard_bytes_by_tp_rank(config.model, 6)
        self.assertEqual(
            weight_shards,
            (
                2_335_890_091,
                2_335_890_091,
                2_201_655_979,
                2_201_655_979,
                2_200_869_546,
                2_200_869_546,
            ),
        )
        self.assertEqual(
            sum(weight_shards),
            estimate_model_weight_bytes(config.model),
        )

        topology = build_instances(
            config.hardware,
            tuple(
                FaceInstanceSpec(group.name, group.pg_name, group.ranks)
                for group in config.inference_groups
            ),
        )
        manager = KVCacheManager(
            topology,
            config.model,
            edge_ranks=config.remote_memory.edge_npus,
            reserve_context_tokens=config.kv_reserve_context_tokens,
        )
        for instance in topology.instances:
            self.assertEqual(
                tuple(
                    snapshot.model_weight_bytes
                    for snapshot in manager.hbm_snapshots(instance.index)
                ),
                weight_shards,
            )

    def test_llama2_7b_tp6_whole_head_kv_shards_are_exact(self) -> None:
        model = FaceModel(
            layers=32,
            hidden_size=4096,
            ffn_size=11008,
            num_heads=32,
            vocab_size=32000,
            bytes_per_elem=2,
            mlp_variant="swiglu",
        )
        heads = attention_heads_by_tp_rank(model.num_heads, 6)
        self.assertEqual(heads, (6, 6, 5, 5, 5, 5))

        tokens = 1_000_000
        bytes_per_head = (
            2
            * model.layers
            * tokens
            * (model.hidden_size // model.num_heads)
            * model.bytes_per_elem
        )
        shards = kv_cache_shard_bytes_for_tokens(model, tokens, 6)
        self.assertEqual(
            shards,
            tuple(head_count * bytes_per_head for head_count in heads),
        )
        self.assertEqual(sum(shards), kv_cache_bytes_for_tokens(model, tokens))

    def test_per_rank_hbm_accounting_invariants_include_weights_and_kv(self) -> None:
        _, _, _, manager = self._tiny_kv_manager()
        self._seed_local_session(
            manager,
            session_id="session",
            instance_index=0,
            context_tokens=10,
            completion_ns=None,
        )

        snapshots = manager.hbm_snapshots()
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in snapshots),
            (20, 20, 0, 0),
        )
        for snapshot in snapshots:
            self.assertEqual(
                snapshot.used_bytes,
                snapshot.model_weight_bytes + snapshot.kv_cache_bytes,
            )
            self.assertEqual(
                snapshot.remaining_bytes,
                snapshot.capacity_bytes - snapshot.used_bytes,
            )

        session = manager.session_snapshot("session")
        self.assertEqual(session.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(session.instance_index, 0)
        self.assertEqual(session.rank_bytes, ((0, 20), (1, 20)))
        self.assertEqual(sum(bytes_count for _, bytes_count in session.rank_bytes), 40)

    def test_reserve_threshold_and_fifo_can_evict_multiple_sessions(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
            reserve_context_tokens=40,
        )
        for session_id, completion_ns in (
            ("session_b", 10),
            ("session_a", 10),
            ("session_c", 20),
            ("session_d", 30),
            ("session_e", 40),
        ):
            self._seed_local_session(
                manager,
                session_id=session_id,
                instance_index=0,
                context_tokens=10,
                completion_ns=completion_ns,
            )

        self.assertEqual(
            tuple(snapshot.remaining_bytes for snapshot in manager.hbm_snapshots(0)),
            manager.reserve_bytes_by_tp_rank,
        )
        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="equal_threshold",
        )
        self.assertEqual(evictions, ())
        self.assertEqual(reserve_unmet, ())

        self._seed_local_session(
            manager,
            session_id="active_trigger",
            instance_index=0,
            context_tokens=25,
            completion_ns=None,
        )
        self.assertTrue(
            all(
                snapshot.remaining_bytes < reserve
                for snapshot, reserve in zip(
                    manager.hbm_snapshots(0),
                    manager.reserve_bytes_by_tp_rank,
                )
            )
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="below_threshold",
        )
        self.assertEqual(
            tuple(transfer.session_id for transfer in evictions),
            ("session_a", "session_b", "session_c"),
        )
        self.assertEqual(reserve_unmet, ())
        for transfer in evictions:
            self.assertEqual(transfer.kind, "remote_store")
            self.assertEqual(
                sum(shard.bytes for shard in transfer.shards),
                transfer.total_bytes,
            )
            self.assertTrue(
                all(shard.edge_rank in manager.edge_ranks for shard in transfer.shards)
            )
        for session_id in ("session_a", "session_b", "session_c"):
            self.assertEqual(
                manager.session_snapshot(session_id).location,
                KVCacheManager.REMOTE_MEMORY,
            )

    def test_unreachable_reserve_returns_reserve_unmet_without_looping(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=100,
            reserve_context_tokens=50,
        )
        self._seed_local_session(
            manager,
            session_id="old",
            instance_index=0,
            context_tokens=5,
            completion_ns=1,
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="unreachable",
        )
        self.assertEqual(tuple(transfer.session_id for transfer in evictions), ("old",))
        self.assertEqual(reserve_unmet, (0, 1))
        self.assertEqual(
            manager.session_snapshot("old").location,
            KVCacheManager.REMOTE_MEMORY,
        )

        second_evictions, second_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="unreachable_again",
        )
        self.assertEqual(second_evictions, ())
        self.assertEqual(second_unmet, reserve_unmet)

    def test_center_rank_nearest_edge_tie_and_xy_route_are_deterministic(self) -> None:
        hardware = FaceHardware(
            mesh_rows=3,
            mesh_cols=3,
            local_hbm_capacity_bytes=100,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=1.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        edges = physical_edge_ranks(hardware)
        self.assertEqual(edges, (0, 1, 2, 3, 5, 6, 7, 8))
        self.assertEqual(nearest_edge_rank(hardware, 4, edges), 1)

        nearest_route = deterministic_xy_route(hardware, 4, 1)
        self.assertEqual(nearest_route, (4, 1))
        self.assertEqual(
            len(nearest_route) - 1,
            manhattan_hops(hardware, 4, 1),
        )
        self.assertEqual(deterministic_xy_route(hardware, 4, 0), (4, 3, 0))

    def test_history_local_remote_and_other_instance_transitions(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
            reserve_context_tokens=100,
        )
        self._seed_local_session(
            manager,
            session_id="session",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
        )

        hbm_before_hit = manager.hbm_snapshots()
        before, local_hit, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="local_hit",
        )
        self.assertEqual(before.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(local_hit.kind, "local_hit")
        self.assertEqual(local_hit.shards, ())
        self.assertEqual(evictions, ())
        self.assertEqual(manager.hbm_snapshots(), hbm_before_hit)

        manager.mark_complete("session", 20)
        before, noc_transfer, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="noc_move",
        )
        self.assertEqual(before.instance_index, 0)
        self.assertEqual(noc_transfer.kind, "noc_migrate")
        self.assertEqual(evictions, ())
        self.assertEqual(
            sum(shard.bytes for shard in noc_transfer.shards),
            noc_transfer.total_bytes,
        )
        self.assertTrue(
            all(
                len(shard.noc_path) - 1
                == manhattan_hops(
                    manager.topology.hardware,
                    int(shard.source_rank),
                    int(shard.target_rank),
                )
                for shard in noc_transfer.shards
            )
        )
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (0, 0, 20, 20),
        )
        moved = manager.session_snapshot("session")
        self.assertEqual(moved.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(moved.instance_index, 1)
        self.assertEqual(tuple(rank for rank, _ in moved.rank_bytes), (2, 3))

        manager.mark_complete("session", 30)
        stores, reserve_unmet = manager.enforce_reserve(
            instance_index=1,
            trigger_request_id="remote_store",
        )
        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0].kind, "remote_store")
        self.assertEqual(reserve_unmet, (2, 3))
        self.assertTrue(
            all(shard.edge_rank in manager.edge_ranks for shard in stores[0].shards)
        )
        remote = manager.session_snapshot("session")
        self.assertEqual(remote.location, KVCacheManager.REMOTE_MEMORY)
        self.assertIsNone(remote.instance_index)
        self.assertEqual(remote.rank_bytes, ())
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (0, 0, 0, 0),
        )

        before, remote_load, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="remote_load",
        )
        self.assertEqual(before.location, KVCacheManager.REMOTE_MEMORY)
        self.assertEqual(remote_load.kind, "remote_load")
        self.assertEqual(evictions, ())
        self.assertEqual(
            sum(shard.bytes for shard in remote_load.shards),
            remote_load.total_bytes,
        )
        self.assertTrue(
            all(
                shard.edge_rank in manager.edge_ranks
                and shard.source_rank == shard.edge_rank
                for shard in remote_load.shards
            )
        )
        restored = manager.session_snapshot("session")
        self.assertEqual(restored.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(restored.instance_index, 0)
        self.assertEqual(tuple(rank for rank, _ in restored.rank_bytes), (0, 1))
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (20, 20, 0, 0),
        )

    def test_plan_preserves_face_selectors_and_records_kv_state(self) -> None:
        hardware, model, topology, _ = self._tiny_kv_manager(
            capacity_bytes=200,
            reserve_context_tokens=80,
        )
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(
                    0,
                    "session",
                    0,
                    "request",
                    10,
                    1,
                    0,
                    None,
                ),
            ),
            reserve_context_tokens=80,
        )
        request = plan.requests[0]

        expected_prefill = select_prefill_instance(
            (
                PrefillQueueSnapshot(0, 0, None),
                PrefillQueueSnapshot(1, 0, None),
            )
        )
        self.assertEqual(request.prefill_instance_index, expected_prefill)
        self.assertEqual(request.prefill_assignment_key, (0, -1, 0))
        self.assertEqual(
            request.decode_instance_index,
            min(
                request.decode_candidates,
                key=lambda cost: (cost.per_die_delta_ns, cost.instance_index),
            ).instance_index,
        )

        self.assertIsNone(request.history_location_before)
        self.assertIsNone(request.history_transfer)
        self.assertEqual(request.prefill_decode_transfer.kind, "local_hit")
        self.assertEqual(len(request.completion_evictions), 1)
        self.assertEqual(request.completion_evictions[0].kind, "remote_store")
        self.assertEqual(
            sum(
                shard.bytes
                for shard in request.completion_evictions[0].shards
            ),
            request.completion_evictions[0].total_bytes,
        )
        self.assertEqual(
            request.kv_location_after_completion,
            KVCacheManager.REMOTE_MEMORY,
        )
        self.assertIsNone(request.kv_instance_after_completion)
        self.assertEqual(request.reserve_unmet_ranks, ())
        self.assertEqual(len(request.hbm_after_completion), 4)
        for snapshot in request.hbm_after_completion:
            self.assertEqual(
                snapshot.used_bytes,
                snapshot.model_weight_bytes + snapshot.kv_cache_bytes,
            )
            self.assertEqual(
                snapshot.remaining_bytes,
                snapshot.capacity_bytes - snapshot.used_bytes,
            )
        self.assertEqual(
            plan.final_session_states[0].location,
            KVCacheManager.REMOTE_MEMORY,
        )
        self.assertEqual(plan.reserve_context_tokens, 80)

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_default_config_uses_hbm160_edge_pool_and_one_million_reserve(self) -> None:
        config = load_face_trace_config()
        self.assertEqual(
            config.hardware.local_hbm_capacity_bytes,
            160 * 1024**3,
        )
        self.assertIn("160 GiB", config.hardware.label)
        self.assertEqual(
            config.hardware_config.name,
            "face_case5_config_c.json",
        )
        self.assertEqual(
            config.hardware_capacity_profile,
            "validation-160gib",
        )
        self.assertEqual(
            config.system_template.name,
            "llama2_7b_roofline_template.json",
        )
        self.assertEqual(
            config.system_config.name,
            "system.json",
        )
        self.assertEqual(config.network_config.name, "network.yml")
        self.assertEqual(config.comm_group_config.name, "comm_group.json")
        self.assertEqual(
            config.hardware_metadata["remote-memory"]["memory-type"],
            "PER_NPU_MEMORY_EXPANSION",
        )
        self.assertEqual(len(config.configuration_digest), 8)
        self.assertEqual(config.kv_reserve_context_tokens, 1_000_000)
        self.assertEqual(
            config.remote_memory.edge_npus,
            physical_edge_ranks(config.hardware),
        )
        self.assertEqual(len(config.remote_memory.edge_npus), 26)
        self.assertEqual(config.remote_memory.remote_mem_bw_gbps, 512.0)
        self.assertEqual(config.remote_memory.remote_mem_latency_ns, 100)
        self.assertEqual(
            config.remote_memory.logical_pool,
            "unified-kv-cache-pool",
        )
        with config.remote_memory.path.open(encoding="utf-8") as source:
            remote_raw = json.load(source)
        self.assertEqual(
            remote_raw["memory-type"],
            "PER_NPU_MEMORY_EXPANSION",
        )
        self.assertEqual(
            tuple(remote_raw["npu-ids"]),
            config.remote_memory.edge_npus,
        )
        with config.system_config.open(encoding="utf-8") as source:
            system_raw = json.load(source)
        self.assertEqual(system_raw["local-mem-bw"], 1640.0)
        self.assertEqual(system_raw["local-mem-capacity-bytes"], 160 * 1024**3)
        self.assertEqual(system_raw["remote-mem-bw"], 512)
        self.assertEqual(system_raw["remote-mem-latency"], 100)
        self.assertEqual(system_raw["peak-perf"], 261.12)

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_et_kv_migration_ack_and_cross_instance_store_trigger(self) -> None:
        config = load_face_trace_config()
        group_by_index = dict(enumerate(config.inference_groups))
        builders = {
            rank: TraceBuilder(remote_operand_loads=False)
            for rank in range(config.npus_count)
        }
        control_group = group_by_index[0]
        timer_gates = tuple(
            builders[rank].timer_gate(f"control_rank{rank}", 1_000)
            for rank in control_group.ranks
        )
        store = KVTransfer(
            kind="remote_store",
            phase="completion",
            reason="test_cross_instance_trigger",
            session_id="victim",
            trigger_request_id="trigger",
            source_instance_index=1,
            target_instance_index=None,
            total_bytes=64,
            shards=(
                KVTransferShard(
                    source_rank=2,
                    target_rank=2,
                    edge_rank=2,
                    bytes=64,
                    noc_path=(2,),
                ),
            ),
        )
        store_record = _emit_kv_transfer(
            config=config,
            builders=builders,
            group_by_index=group_by_index,
            tag_allocator=TransferTagAllocator(),
            transfer=store,
            action_name="store",
            trigger_gate=TransferTriggerGate(
                control_instance_index=0,
                node_gates=timer_gates,
            ),
        )["shards"][0]

        control_rank = control_group.ranks[0]
        trigger_send = builders[control_rank].nodes[-1]
        self.assertEqual(trigger_send.type, COMM_SEND_NODE)
        self.assertIn(timer_gates[0], trigger_send.data_deps)
        self.assertIsNotNone(store_record["trigger_tag"])
        source_nodes = builders[2].nodes
        self.assertEqual(
            [node.type for node in source_nodes],
            [COMM_RECV_NODE, MEM_STORE_NODE],
        )
        self.assertIn(source_nodes[0].id, source_nodes[1].data_deps)

        noc_builders = {
            rank: TraceBuilder(remote_operand_loads=False)
            for rank in range(config.npus_count)
        }
        noc = KVTransfer(
            kind="noc_migrate",
            phase="history",
            reason="test_source_release_ack",
            session_id="moving",
            trigger_request_id="request",
            source_instance_index=0,
            target_instance_index=1,
            total_bytes=64,
            shards=(
                KVTransferShard(
                    source_rank=20,
                    target_rank=2,
                    edge_rank=None,
                    bytes=64,
                    noc_path=deterministic_xy_route(config.hardware, 20, 2),
                ),
            ),
        )
        noc_record = _emit_kv_transfer(
            config=config,
            builders=noc_builders,
            group_by_index=group_by_index,
            tag_allocator=TransferTagAllocator(),
            transfer=noc,
            action_name="noc",
        )["shards"][0]
        self.assertEqual(
            noc_record["source_release_dependency"],
            "noc_migration_ack_recv",
        )
        self.assertIsNotNone(noc_record["ack_tag"])
        self.assertEqual(
            [node.type for node in noc_builders[20].nodes],
            [COMM_SEND_NODE, COMM_RECV_NODE],
        )
        self.assertEqual(
            [node.type for node in noc_builders[2].nodes],
            [COMM_RECV_NODE, COMM_SEND_NODE],
        )
        self.assertIn(
            noc_builders[20].nodes[0].id,
            noc_builders[20].nodes[1].data_deps,
        )
        self.assertIn(
            noc_builders[2].nodes[0].id,
            noc_builders[2].nodes[1].data_deps,
        )

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_tp_readiness_barrier_waits_for_every_rank_local_predecessor(self) -> None:
        config = load_face_trace_config()
        group = config.inference_groups[0]
        builders = {
            rank: TraceBuilder(remote_operand_loads=False)
            for rank in group.ranks
        }
        predecessor_ids: dict[int, int] = {}
        for rank in group.ranks:
            builders[rank].comp(f"rank{rank}_kv_ready", 1, 1)
            predecessor = builders[rank].previous_id
            self.assertIsNotNone(predecessor)
            predecessor_ids[rank] = int(predecessor)

        record = _emit_tp_readiness_barrier(
            builders=builders,
            group=group,
            name="request_prefill_kv_ready_barrier",
        )

        self.assertEqual(record["collective"], "all_reduce")
        self.assertEqual(record["comm_size_bytes"], 1)
        self.assertEqual(record["pg_name"], group.pg_name)
        self.assertEqual(record["ranks"], list(group.ranks))
        self.assertEqual(len(record["node_ids_by_rank"]), len(group.ranks))
        node_ids_by_rank = dict(record["node_ids_by_rank"])
        for rank in group.ranks:
            barrier = builders[rank].nodes[-1]
            self.assertEqual(barrier.type, COMM_COLL_NODE)
            self.assertEqual(barrier.name, "request_prefill_kv_ready_barrier")
            self.assertEqual(barrier.id, node_ids_by_rank[rank])
            self.assertIn(predecessor_ids[rank], barrier.data_deps)
            attrs = {attr.name: attr for attr in barrier.attr}
            self.assertEqual(attrs["comm_type"].uint64_val, ALL_REDUCE)
            self.assertEqual(attrs["comm_size"].uint64_val, 1)
            self.assertEqual(attrs["pg_name"].string_val, group.pg_name)

            builders[rank].comp(f"rank{rank}_first_prefill_compute", 1, 1)
            first_compute = builders[rank].nodes[-1]
            self.assertIn(barrier.id, first_compute.data_deps)

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
        self.assertEqual(
            sum(node.type == COMP_NODE for node in aggregated.nodes),
            15,
        )
        self.assertEqual(
            sum(node.type == COMM_COLL_NODE for node in aggregated.nodes),
            2,
        )
        self.assertEqual(totals(aggregated), totals(expanded))

    def test_prefill_queue_order_and_deterministic_third_key(self) -> None:
        queues = (
            PrefillQueueSnapshot(0, 2, 10),
            PrefillQueueSnapshot(1, 1, 20),
            PrefillQueueSnapshot(2, 1, None),
        )
        self.assertEqual(select_prefill_instance(queues), 2)
        tied = (
            PrefillQueueSnapshot(2, 0, None),
            PrefillQueueSnapshot(1, 0, None),
        )
        self.assertEqual(select_prefill_instance(tied), 1)

    def test_lut_exact_filters_and_nearest_token_tie(self) -> None:
        lut = FaceLut(
            (
                FaceLutEntry(2, 0, 1, 256, 10),
                FaceLutEntry(2, 0, 1, 512, 20),
                FaceLutEntry(2, 64, 1, 256, 30),
            )
        )
        row = lut.lookup(instance_size=2, p_chunk=0, d_batch=1, d_token=384)
        self.assertEqual(row.d_token, 256)
        with self.assertRaises(KeyError):
            lut.lookup(instance_size=2, p_chunk=0, d_batch=2, d_token=384)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "face_lut.csv"
            lut.export_csv(output)
            self.assertIn("iteration_time_ns", output.read_text(encoding="utf-8"))

    def test_weighted_schedulable_range_changes_after_update(self) -> None:
        hardware, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        before = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in before), (0, 1, 2))
        graph.increase_path((0, 1))
        after = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in after), (0, 1))

    def test_decode_per_die_cost_and_config_order_tie(self) -> None:
        _, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        lut = FaceLut(
            (
                FaceLutEntry(2, 0, 0, 0, 0),
                FaceLutEntry(2, 0, 1, 256, 100),
                FaceLutEntry(2, 0, 2, 256, 300),
            )
        )
        selected, costs = select_decode_instance(
            topology=topology,
            graph=graph,
            lut=lut,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((256,), (), ()),
            new_request_token_length=256,
        )
        self.assertEqual(selected, 1)
        self.assertEqual([cost.instance_index for cost in costs], [0, 1, 2])
        self.assertGreater(costs[0].per_die_delta_ns, costs[1].per_die_delta_ns)
        self.assertEqual(costs[1].per_die_delta_ns, costs[2].per_die_delta_ns)

    def test_kv_local_first_offload_updates_and_release_restores_weight(self) -> None:
        _, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        allocator = KVAllocator(topology, graph, model_weight_bytes=0)
        allocation = allocator.allocate(
            request_id="r0",
            decode_instance_index=0,
            candidate_indices=(0, 1, 2),
            total_bytes=250,
        )
        self.assertEqual(
            [(piece.instance_index, piece.bytes) for piece in allocation.pieces],
            [(0, 200), (1, 50)],
        )
        self.assertEqual(graph.edge_weight(0, 1), 2)
        allocator.release(allocation)
        self.assertEqual(graph.edge_weight(0, 1), 1)
        self.assertEqual(tuple(allocator.remaining_capacity), (200, 200, 200))

        # At equal weighted distance, prefer the instance with more remaining
        # HBM before using config order.
        graph2 = WeightedInstanceGraph(topology)
        allocator2 = KVAllocator(topology, graph2, model_weight_bytes=0)
        allocator2.remaining_capacity[:] = [100, 0, 150]
        allocation2 = allocator2.allocate(
            request_id="r1",
            decode_instance_index=1,
            candidate_indices=(0, 1, 2),
            total_bytes=120,
        )
        self.assertEqual(allocation2.pieces[0].instance_index, 2)

    def test_hbm_blocked_request_waits_until_active_session_completes(self) -> None:
        hardware = FaceHardware(
            mesh_rows=1,
            mesh_cols=2,
            local_hbm_capacity_bytes=60,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=2.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        model = FaceModel(
            layers=1,
            hidden_size=2,
            ffn_size=2,
            num_heads=2,
            vocab_size=2,
            bytes_per_elem=1,
            mlp_variant="gelu",
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=(FaceInstanceSpec("only", "1", (0, 1)),),
            requests=(
                FaceRequest(0, "active", 0, "active_0", 10, 5, 0, None),
                FaceRequest(1, "waiting", 0, "waiting_0", 10, 5, 0, None),
            ),
            reserve_context_tokens=0,
        )
        active, waiting = plan.requests
        self.assertEqual(active.admission_time_ns, 0)
        self.assertEqual(waiting.estimated_arrival_ns, 0)
        self.assertEqual(waiting.admission_time_ns, active.completion_ns)
        self.assertEqual(waiting.hbm_wait_ns, active.completion_ns)
        self.assertGreaterEqual(waiting.prefill_start_ns, waiting.admission_time_ns)
        self.assertTrue(
            any(
                transfer.kind == "remote_store"
                and transfer.session_id == "active"
                for transfer in waiting.history_evictions
            )
        )
        self.assertTrue(
            all(snapshot.remaining_bytes >= 0 for snapshot in plan.final_hbm_states)
        )

    def test_request_larger_than_empty_instance_does_not_wait_forever(self) -> None:
        hardware = FaceHardware(1, 2, 60, 1.0, 2.0, 1.0, 0, 0)
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        with self.assertRaisesRegex(ValueError, "cannot fit on any empty instance"):
            plan_face_requests(
                hardware=hardware,
                model=model,
                instance_specs=(FaceInstanceSpec("only", "1", (0, 1)),),
                requests=(
                    FaceRequest(0, "oversized", 0, "oversized_0", 30, 1, 0, None),
                ),
                reserve_context_tokens=0,
            )

    def test_static_emission_orders_store_before_affected_following_turn(self) -> None:
        store = SimpleNamespace(
            kind="remote_store",
            phase="completion",
            session_id="victim",
        )

        def request(
            request_id: str,
            session_id: str,
            turn_index: int,
            queue_index: int,
            prefill_start_ns: int,
            admission_time_ns: int,
            completion_ns: int,
            completion_evictions: tuple[object, ...] = (),
        ) -> SimpleNamespace:
            return SimpleNamespace(
                request_id=request_id,
                session_id=session_id,
                turn_index=turn_index,
                queue_index=queue_index,
                prefill_start_ns=prefill_start_ns,
                estimated_arrival_ns=admission_time_ns,
                admission_time_ns=admission_time_ns,
                prefill_complete_ns=prefill_start_ns + 10,
                completion_ns=completion_ns,
                history_evictions=(),
                prefill_evictions=(),
                decode_evictions=(),
                completion_evictions=completion_evictions,
            )

        trigger = request("trigger", "other", 0, 0, 0, 0, 300, (store,))
        producer = request("producer", "victim", 0, 1, 100, 100, 200)
        following = request("following", "victim", 1, 2, 310, 300, 320)
        ordered = order_plans_for_static_emission(
            SimpleNamespace(requests=(trigger, producer, following))
        )
        self.assertEqual(
            [item.request_id for item in ordered],
            ["producer", "trigger", "following"],
        )

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_checked_in_shape_and_requests_plan_deterministically(self) -> None:
        config = load_face_trace_config()
        hardware = config.hardware
        model = config.model
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        requests = (
            FaceRequest(0, "session_0", 0, "session_0_request_0", 497, 42, 0, None),
            FaceRequest(
                1,
                "session_0",
                1,
                "session_0_request_1",
                494,
                58,
                None,
                1_000_000_000,
            ),
            FaceRequest(2, "session_1", 0, "session_1_request_0", 241, 57, 0, None),
            FaceRequest(
                3,
                "session_1",
                1,
                "session_1_request_1",
                152,
                46,
                None,
                1_000_000_000,
            ),
        )
        first = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=specs,
            requests=requests,
            prefill_chunk_size=config.prefill_chunk_size,
        )
        second = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=specs,
            requests=requests,
            prefill_chunk_size=config.prefill_chunk_size,
        )
        self.assertEqual(first.p_chunk, 512)
        self.assertEqual(len(first.requests), 4)
        self.assertTrue(first.iterations)
        signature = [
            (
                plan.request_id,
                plan.estimated_arrival_ns,
                plan.prefill_instance_index,
                plan.decode_instance_index,
                plan.completion_ns,
            )
            for plan in first.requests
        ]
        self.assertEqual(
            signature,
            [
                (
                    plan.request_id,
                    plan.estimated_arrival_ns,
                    plan.prefill_instance_index,
                    plan.decode_instance_index,
                    plan.completion_ns,
                )
                for plan in second.requests
            ],
        )
        by_id = {plan.request_id: plan for plan in first.requests}
        self.assertEqual(
            by_id["session_0_request_1"].estimated_arrival_ns,
            by_id["session_0_request_0"].completion_ns + 1_000_000_000,
        )
        self.assertEqual(
            by_id["session_1_request_1"].estimated_arrival_ns,
            by_id["session_1_request_0"].completion_ns + 1_000_000_000,
        )
        for plan in first.requests:
            self.assertGreaterEqual(plan.prefill_start_ns, plan.estimated_arrival_ns)
            self.assertGreater(plan.completion_ns, plan.prefill_complete_ns)
            self.assertTrue(plan.decode_candidates)
            self.assertIn(
                plan.decode_instance_index,
                [candidate.instance_index for candidate in plan.decode_candidates],
            )
            for candidate in plan.decode_candidates:
                self.assertLessEqual(
                    candidate.weighted_distance,
                    hardware.schedulable_distance_limit,
                )



    def test_checked_in_config_is_request_neutral_and_fails_closed_without_input(
            self) -> None:
        """裸仓库态常态用例（sh_2.0 回灌轮同款）：checked-in 配置必须指向
        占位路径且正式入口缺失输入 fail-closed（SystemExit 非 0），任何
        物化输入或默认队列 stub 不得回填进仓。"""
        config_path = MODULE_DIR / "trace_config.csv"
        with config_path.open(encoding="utf-8") as handle:
            lines = [
                ln for ln in handle
                if ln.startswith("config,request_queue_csv,")
            ]
        self.assertEqual(len(lines), 1)
        self.assertIn("request_queue_placeholder.csv", lines[0])
        with self.assertRaises(SystemExit) as caught:
            load_face_trace_config()
        self.assertNotEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
