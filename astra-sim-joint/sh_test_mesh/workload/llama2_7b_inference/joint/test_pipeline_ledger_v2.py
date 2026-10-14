"""Node-event 与 envelope 解析通用面钉测（远读退役收缩版）。

2026-10-14 远读退役改写（方案 §7.2 裁定"删 descriptor 用例、保留通用
work_ledger/v4 用例；所剩过少则整删并注明承接面"）：原文件的 descriptor
消费用例（SelectedShardLedgerConsumerTests / PipelineLedgerV2Test /
CompletedCursorIntervalCostTest / SparsePrepareScaling 的稀疏扫描两用例，
合计约 4600 行、descriptor 引用 516 处）随 joint.pipeline_descriptor /
joint.compact_pipeline / joint.pipeline_identity 三模块整删一并删除。
保留的通用面：

* envelope 解析纪律（数值类型严格性 / 原子性 / Mapping 兼容形态）；
* v3 archive 节点事件规范化保留（标量工厂、canonical 字节、重放幂等）。

承接面登记：

* work_ledger 通用账本（生命周期/幂等/等价/快照）→
  joint/test_work_ledger.py、joint/test_compact_ledger_direct.py、
  joint/test_compact_snapshot_index.py（node-only envelope，全绿）；
* schema2 批次发布断言 → C++ bridge loopback 系 +
  online/test_graph_batch_rank_ledger.py（B2 摘键后）；
* descriptor/slot/cursor 消费机制 → 已无被测对象（D11 最小联动档：
  0 行帧合法，pipeline 三键可缺席或空）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest joint/test_pipeline_ledger_v2.py -q
"""
import copy
import os
import sys
import unittest
from collections import UserDict
from collections.abc import Mapping as _TestMapping
from types import MappingProxyType
from typing import Mapping as _LegacyTypingMapping
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _path in (_HERE, _WL, os.path.join(_WL, "online")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from joint.pipeline_ledger import (  # noqa: E402
    ARCHIVE_V3_BRIDGE_PROTOCOL,
    ARCHIVE_V3_PROTOCOL_VERSION,
    ARCHIVE_V3_SCHEMA_VERSION,
    ARCHIVE_V3_WORK_LEDGER_SCHEMA,
    DESCRIPTOR_VERSION,
    PipelineLedger,
    PipelineLedgerError,
)
import joint.pipeline_ledger as pipeline_ledger_module  # noqa: E402
from joint.work_ledger import WorkLedger, WorkLedgerError  # noqa: E402
from joint.work_ledger import canonical_work_id, leg_work_id  # noqa: E402
import joint.work_ledger as work_ledger_module  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    OnlineSchedulerBase as LiveOnlineSchedulerBase,
)


class OnlineSchedulerBase(LiveOnlineSchedulerBase):
    """Test-only exact v3 archive adapter; live base remains strict v4."""

    def _apply_work_ledger(self, delta):
        payload = delta.get("work_ledger")
        parsed = self.joint_pipeline_ledger.validate_envelope(payload)
        work_tx = self.joint_work_ledger.begin_protocol_v3_transaction(
            delivery_sequence=payload.get("delivery_sequence"),
            tick=payload.get("tick"), snapshot_kind=payload.get("snapshot_kind"))
        prepared = self.joint_pipeline_ledger.prepare_envelope(
            payload, work_transaction=work_tx, _parsed=parsed)
        work_tx.finish(parsed.seq_end)
        work_tx.commit(notify=False)
        self.joint_pipeline_ledger.commit_prepared(prepared)
        work_tx.notify()
        self.work_ledger_absent_seen = False
        return prepared.parsed.canonical_bytes


class _TestCustomMapping(_TestMapping):
    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _TestDictSubclass(dict):
    pass


class _TestVirtualMapping:
    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def get(self, key, default=None):
        return self._values.get(key, default)


class _TestSpoofedMappingClass:
    @property
    def __class__(self):
        return _TestMapping


def _test_mapping_tree(value, factory):
    if isinstance(value, dict):
        return factory({key: _test_mapping_tree(item, factory)
                        for key, item in value.items()})
    if isinstance(value, list):
        return [_test_mapping_tree(item, factory) for item in value]
    if isinstance(value, tuple):
        return tuple(_test_mapping_tree(item, factory) for item in value)
    return value


def _empty_envelope(kind, delivery, *, base_watermark, base_version,
                    tick=0):
    return {
        "producer": "cpp_online",
        "schema_version": ARCHIVE_V3_SCHEMA_VERSION,
        "protocol_version": ARCHIVE_V3_PROTOCOL_VERSION,
        "bridge_protocol": ARCHIVE_V3_BRIDGE_PROTOCOL,
        "work_ledger_schema": ARCHIVE_V3_WORK_LEDGER_SCHEMA,
        "epoch": "test-run",
        "delivery_sequence": delivery,
        "base_watermark": base_watermark,
        "logical_seq_start": base_watermark + 1,
        "logical_seq_end": base_watermark,
        "logical_event_count": 0,
        "replay_token": f"cpp_online/test-run/{delivery}",
        "descriptor_version": DESCRIPTOR_VERSION,
        "base_state_version": base_version,
        "result_state_version": base_version,
        "snapshot_kind": kind,
        "tick": tick,
        "descriptor_commits": [],
        "node_transitions": [],
        "pipeline_transition_runs": [],
        "cursor_patches": [],
        "active_resources": [],
    }


def _node_event(seq, kind, work_id, payload=None):
    return {
        "logical_seq": seq,
        "tick": 0,
        "kind": kind,
        "work_id": work_id,
        "payload": payload or {},
    }


def _noc_resource(remaining_bytes, as_of_tick):
    request_id, stage, generation, rank, flow_id = "noc", "", 0, 0, 0
    node_id = flow_id
    node_work_id = canonical_work_id(
        request_id, stage, generation, rank, node_id)
    return {
        "resource_type": "noc_flow",
        "work_id": leg_work_id(node_work_id, f"noc:{flow_id}"),
        "request_id": request_id,
        "stage": stage,
        "generation": generation,
        "rank": rank,
        "node_id": node_id,
        "operation": "COMM_READ",
        "leg": f"noc:{flow_id}",
        "state": "Active",
        "total": {"ops": None, "bytes": 100, "latency_ns": None},
        "remaining": {"ops": None, "bytes": remaining_bytes, "latency_ns": None},
        "progress_source": "authoritative",
        "service_unit": "bytes",
        "route": [0, 1],
        "as_of_tick": as_of_tick,
        "issued_tick": 0,
        "descriptor_segment_id": None,
        "endpoint_selector": None,
        "flow_id": flow_id,
        "current_rate_Bpns": 2.5,
        "last_rate_update_time": float(as_of_tick),
    }


def _node_only_envelope(node_events):
    """node-only 帧（pipeline 三键缺席 = D11 0 行帧合法形态）。"""
    envelope = _empty_envelope(
        "full", 0, base_watermark=0, base_version=0)
    count = len(node_events)
    envelope.update({
        "logical_seq_end": count,
        "logical_event_count": count,
        "result_state_version": 1,
        "node_transitions": list(node_events),
    })
    return envelope


class EnvelopeNumericDisciplineTest(unittest.TestCase):
    def test_replayed_numeric_type_change_is_rejected(self):
        ledger = PipelineLedger()
        full = _empty_envelope(
            "full", 0, base_watermark=0, base_version=0)
        ledger.apply_envelope(full)
        changed = copy.deepcopy(full)
        changed["tick"] = 0.0
        with self.assertRaises(PipelineLedgerError):
            ledger.apply_envelope(changed)
        self.assertEqual(ledger.delivery_sequence, 0)
        self.assertEqual(ledger.state_version, 0)


class RuntimeMappingCompatibilityTest(unittest.TestCase):
    def test_mapping_forms_keep_pipeline_parse_freeze_and_typed_compare(self):
        pipeline_ledger_module._RuntimeMapping.register(_TestVirtualMapping)
        factories = (
            ("dict", dict),
            ("dict subclass", _TestDictSubclass),
            ("UserDict", UserDict),
            ("MappingProxyType", MappingProxyType),
            ("custom Mapping", _TestCustomMapping),
            ("virtual Mapping", _TestVirtualMapping),
        )
        values = {"value": 1, "nested": {"floating": 1.0}}
        expected_frozen = pipeline_ledger_module._freeze_dto(values)
        expected_resources = pipeline_ledger_module._parse_resources(
            [_noc_resource(80, 1)], "active_resources")
        for name, factory in factories:
            with self.subTest(mapping_type=name):
                wrapped = _test_mapping_tree(values, factory)
                parsed = pipeline_ledger_module._mapping(
                    wrapped, "probe", frozenset({"value", "nested"}))
                self.assertIs(parsed, wrapped)
                self.assertTrue(
                    pipeline_ledger_module._typed_equal(values, wrapped))
                self.assertEqual(
                    pipeline_ledger_module._freeze_dto(wrapped),
                    expected_frozen)

                resource_row = _test_mapping_tree(
                    _noc_resource(80, 1), factory)
                # Virtual ABC registration supplies no Mapping equality
                # mixin. Compare parsed protocol values with the same strict
                # structural rule instead of assuming wrapper __eq__.
                self.assertTrue(pipeline_ledger_module._typed_equal(
                    pipeline_ledger_module._parse_resources(
                        [resource_row], "active_resources"),
                    expected_resources))

        self.assertFalse(pipeline_ledger_module._typed_equal(
            {"value": 1}, {"value": 1.0}))
        self.assertFalse(pipeline_ledger_module._typed_equal(
            {"value": True}, {"value": 1}))
        self.assertFalse(pipeline_ledger_module._typed_equal(
            {"value": float("nan")}, {"value": float("nan")}))
        proxy = MappingProxyType({
            "value": 1, "nested": MappingProxyType({"floating": 1.0})})
        self.assertTrue(pipeline_ledger_module._typed_equal(values, proxy))
        self.assertTrue(pipeline_ledger_module._typed_equal(proxy, values))
        self.assertTrue(pipeline_ledger_module._typed_equal(
            [1, {"nested": True}], (1, MappingProxyType({"nested": True}))))
        with self.assertRaisesRegex(PipelineLedgerError, "must be an object"):
            pipeline_ledger_module._mapping(
                [], "probe", frozenset({"value"}))
        with self.assertRaisesRegex(
                PipelineLedgerError, "pipeline ledger envelope must be an object"):
            PipelineLedger().validate_envelope([])

    def test_class_spoof_matches_legacy_typing_mapping_consumer(self):
        spoofed_values = (
            mock.Mock(spec=_TestMapping),
            _TestSpoofedMappingClass(),
        )
        for value in spoofed_values:
            with self.subTest(spoof_type=type(value).__name__):
                # Bare typing.Mapping's legacy instance check uses the real
                # type(obj), while ABC isinstance may honor this spoofed class.
                self.assertTrue(isinstance(value, _TestMapping))
                self.assertFalse(isinstance(value, _LegacyTypingMapping))
                self.assertFalse(issubclass(
                    type(value), pipeline_ledger_module._RuntimeMapping))
                self.assertFalse(pipeline_ledger_module._typed_equal(
                    value, {}))
                self.assertIs(pipeline_ledger_module._freeze_dto(value), value)
                with self.assertRaisesRegex(
                        PipelineLedgerError, "must be an object"):
                    pipeline_ledger_module._mapping(
                        value, "probe", frozenset())

    def test_bad_numeric_mapping_preserves_both_ledger_states(self):
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        scheduler._apply_work_ledger({"work_ledger": _empty_envelope(
            "full", 0, base_watermark=0, base_version=0)})

        pipeline_before = scheduler.joint_pipeline_ledger.snapshot_state()
        pipeline_versions_before = (
            scheduler.joint_pipeline_ledger.delivery_sequence,
            scheduler.joint_pipeline_ledger.watermark,
            scheduler.joint_pipeline_ledger.state_version,
        )
        work_before = scheduler.joint_work_ledger.export_state()
        work_versions_before = (
            scheduler.joint_work_ledger.last_delivery_sequence,
            scheduler.joint_work_ledger.event_watermark,
            scheduler.joint_work_ledger.version,
            scheduler.joint_work_ledger.full_load_count,
            scheduler.joint_work_ledger.delta_load_count,
        )

        for invalid_schema_version in (ARCHIVE_V3_SCHEMA_VERSION + 0.0, True):
            bad = _empty_envelope(
                "delta", 1, base_watermark=0, base_version=0, tick=1)
            bad["schema_version"] = invalid_schema_version
            with self.subTest(schema_version=invalid_schema_version):
                with self.assertRaises(PipelineLedgerError):
                    scheduler._apply_work_ledger({"work_ledger": bad})
                self.assertIs(
                    scheduler.joint_pipeline_ledger.snapshot_state(),
                    pipeline_before)
                self.assertEqual(
                    (scheduler.joint_pipeline_ledger.delivery_sequence,
                     scheduler.joint_pipeline_ledger.watermark,
                     scheduler.joint_pipeline_ledger.state_version),
                    pipeline_versions_before)
                self.assertEqual(
                    scheduler.joint_work_ledger.export_state(), work_before)
                self.assertEqual(
                    (scheduler.joint_work_ledger.last_delivery_sequence,
                     scheduler.joint_work_ledger.event_watermark,
                     scheduler.joint_work_ledger.version,
                     scheduler.joint_work_ledger.full_load_count,
                     scheduler.joint_work_ledger.delta_load_count),
                    work_versions_before)


class V3NodeEventCanonicalRetentionTests(unittest.TestCase):
    def _two_node_frame(self):
        work_id = "ordinary#decode:1:r0n9"
        commit_payload = {
            "request_id": "ordinary", "stage": "decode",
            "generation": 1, "rank": 0, "node_id": 9,
            "operation": "compute",
            "total": {"ops": 10, "bytes": 0, "latency_ns": 1},
            "parents": [],
        }
        frame = _node_only_envelope((
            _node_event(1, "commit", work_id, commit_payload),
            _node_event(2, "issue", work_id),
        ))
        return work_id, frame

    @staticmethod
    def _typed_node_wire(row, delivery):
        mapped = dict(row["payload"])
        mapped.update({
            "kind": row["kind"],
            "work_id": row["work_id"],
            "tick": row["tick"],
            "event_id": {
                "tick": row["tick"],
                "delivery_sequence": delivery,
                "producer": "cpp_online",
                "local_seq": row["logical_seq"],
            },
        })
        return mapped

    def _assert_rejected_before_work_transaction(self, envelope):
        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        with mock.patch.object(
                scheduler.joint_work_ledger,
                "begin_protocol_v3_transaction") as begin_transaction:
            with self.assertRaises(PipelineLedgerError):
                scheduler._apply_work_ledger({"work_ledger": envelope})
        begin_transaction.assert_not_called()
        self.assertFalse(scheduler.joint_pipeline_ledger.initialized)
        self.assertEqual(scheduler.joint_pipeline_ledger.watermark, 0)
        self.assertEqual(scheduler.joint_pipeline_ledger.state_version, 0)
        self.assertEqual(scheduler.joint_work_ledger._works, {})
        self.assertEqual(scheduler.joint_work_ledger.version, 0)
        self.assertEqual(scheduler.joint_work_ledger.last_delivery_sequence, -1)
        self.assertIsNone(scheduler.joint_work_ledger._last_event)

    def test_v3_builds_one_event_tree_and_retains_final_node_for_legacy_replay(self):
        work_id, frame = self._two_node_frame()
        final_wire = self._typed_node_wire(frame["node_transitions"][-1], 0)
        expected_final = work_ledger_module._typed_event_from_mapping(
            final_wire).canonical_content

        expected_bytes = pipeline_ledger_module._canonical_envelope_bytes(
            frame)
        parser = PipelineLedger()
        event_canonical_builder = work_ledger_module._event_canonical_content
        with mock.patch.object(
                work_ledger_module, "_event_canonical_content",
                wraps=event_canonical_builder) as build_canonical:
            parsed = parser.validate_envelope(frame)

        self.assertEqual(build_canonical.call_count, 1)
        self.assertIsNone(parsed.node_events[0][1].canonical_content)
        self.assertEqual(parsed.node_events[-1][1].canonical_content,
                         expected_final)
        self.assertEqual(parsed.canonical_bytes, expected_bytes)

        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        accepted_bytes = scheduler._apply_work_ledger(
            {"work_ledger": frame})
        ledger = scheduler.joint_work_ledger
        self.assertEqual(accepted_bytes,
                         pipeline_ledger_module._canonical_envelope_bytes(
                             frame))
        self.assertEqual(ledger.work(work_id).state, "issued")
        self.assertEqual(scheduler.joint_pipeline_ledger.watermark, 2)
        self.assertEqual(ledger._last_event,
                         ("cpp_online", 2, expected_final))

        full_canonical_reference = WorkLedger()
        for row in frame["node_transitions"]:
            full_canonical_reference._apply_event(
                work_ledger_module._typed_event_from_mapping(
                    self._typed_node_wire(row, 0)))
        self.assertEqual(ledger.export_state(),
                         full_canonical_reference.export_state())
        self.assertEqual(ledger._last_event,
                         full_canonical_reference._last_event)

        accepted_state = ledger.export_state()
        accepted_version = ledger.version
        replayed_before = ledger.replayed_event_count
        duplicate = work_ledger_module._typed_event_from_mapping(final_wire)
        self.assertFalse(ledger._apply_event(duplicate))
        self.assertEqual(ledger.replayed_event_count, replayed_before + 1)
        self.assertEqual(ledger.export_state(), accepted_state)
        self.assertEqual(ledger.version, accepted_version)

        changed_wire = dict(final_wire)
        changed_wire["operation"] = "different-operation"
        altered = work_ledger_module._typed_event_from_mapping(changed_wire)
        with self.assertRaisesRegex(WorkLedgerError, "different content"):
            ledger._apply_event(altered)
        self.assertEqual(ledger.export_state(), accepted_state)
        self.assertEqual(ledger.version, accepted_version)

    def test_v3_parser_uses_scalar_factory_once_per_node_and_generic_never(self):
        for node_count in (8, 32, 128):
            with self.subTest(node_count=node_count):
                rows = tuple(
                    _node_event(
                        index + 1, "issue",
                        f"probe#decode:0:r0n{index}",
                        {"operation": "compute", "parents": []})
                    for index in range(node_count))
                envelope = _node_only_envelope(rows)
                expected_events = tuple(
                    work_ledger_module._typed_event_from_mapping(
                        self._typed_node_wire(row, 0))
                    for row in envelope["node_transitions"])
                expected_bytes = (
                    pipeline_ledger_module._canonical_envelope_bytes(envelope))
                scalar_factory = (
                    pipeline_ledger_module
                    ._typed_v3_node_event_from_validated_scalars)
                with mock.patch.object(
                        pipeline_ledger_module,
                        "_typed_v3_node_event_from_validated_scalars",
                        wraps=scalar_factory) as scalar_calls, \
                        mock.patch.object(
                            work_ledger_module, "_typed_event_from_mapping",
                            wraps=work_ledger_module._typed_event_from_mapping
                        ) as generic_calls, \
                        mock.patch.object(
                            work_ledger_module, "_event_canonical_content",
                            wraps=work_ledger_module._event_canonical_content
                        ) as canonical_calls:
                    parsed = PipelineLedger().validate_envelope(envelope)

                self.assertEqual(scalar_calls.call_count, node_count)
                self.assertEqual(generic_calls.call_count, 0)
                self.assertEqual(canonical_calls.call_count, 1)
                self.assertEqual(parsed.canonical_bytes, expected_bytes)
                self.assertEqual(len(parsed.node_events), node_count)
                for index, ((seq, actual), expected) in enumerate(zip(
                        parsed.node_events, expected_events)):
                    self.assertEqual(seq, expected.local_seq)
                    self.assertEqual(actual.kind, expected.kind)
                    self.assertEqual(actual.work_id, expected.work_id)
                    self.assertEqual(actual.tick, expected.tick)
                    self.assertEqual(actual.producer, expected.producer)
                    self.assertEqual(actual.delivery_sequence,
                                     expected.delivery_sequence)
                    self.assertEqual(actual.local_seq, expected.local_seq)
                    self.assertEqual(actual.delta_served,
                                     expected.delta_served)
                    self.assertEqual(
                        actual.payload,
                        {key: value for key, value in expected.payload.items()
                         if key not in ("kind", "work_id", "tick", "event_id")})
                    if index == node_count - 1:
                        self.assertEqual(actual.canonical_content,
                                         expected.canonical_content)
                    else:
                        self.assertIsNone(actual.canonical_content)

    def test_bad_node_rows_still_reject_before_either_transaction_begins(self):
        def bad_sequence_float(row):
            row["logical_seq"] = float(row["logical_seq"])

        def bad_sequence_bool(row):
            row["logical_seq"] = True

        def bad_tick_float(row):
            row["tick"] = float(row["tick"])

        def bad_tick_bool(row):
            row["tick"] = True

        def bad_kind_type(row):
            row["kind"] = 1

        def bad_kind_value(row):
            row["kind"] = "unknown-kind"

        def empty_work_id(row):
            row["work_id"] = ""

        def bad_work_id_type(row):
            row["work_id"] = 1

        def unknown_payload_key(row):
            row["payload"]["unknown_field"] = 1

        def remaining_on_wrong_kind(row):
            row["kind"] = "issue"
            row["payload"]["remaining_ops"] = 1

        def nonfinite_remaining(row):
            row["kind"] = "progress"
            row["payload"]["remaining_ops"] = float("nan")

        def bad_delta_keys(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"unknown_amount": 1}

        def bad_delta_type(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = 1

        def bad_delta_bool(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"ops": True}

        def negative_delta(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"ops": -1}

        def nonfinite_delta(row):
            row["kind"] = "progress"
            row["payload"]["delta_served"] = {"ops": float("nan")}

        def unsupported_nested_json(row):
            row["payload"]["total"] = {"ops": object()}

        def nonfinite_nested_json(row):
            row["payload"]["total"] = {"ops": float("nan")}

        invalidations = (
            ("float logical sequence", bad_sequence_float),
            ("bool logical sequence", bad_sequence_bool),
            ("float event tick", bad_tick_float),
            ("bool event tick", bad_tick_bool),
            ("non-string event kind", bad_kind_type),
            ("unknown event kind", bad_kind_value),
            ("malformed work id", empty_work_id),
            ("non-string work id", bad_work_id_type),
            ("unknown event field", unknown_payload_key),
            ("remaining field on wrong kind", remaining_on_wrong_kind),
            ("nonfinite remaining amount", nonfinite_remaining),
            ("unknown delta amount key", bad_delta_keys),
            ("non-mapping delta amount", bad_delta_type),
            ("bool delta amount", bad_delta_bool),
            ("negative delta amount", negative_delta),
            ("nonfinite delta amount", nonfinite_delta),
            ("unsupported nested JSON", unsupported_nested_json),
            ("nonfinite nested JSON", nonfinite_nested_json),
        )
        for node_index in (0, 1):
            for name, invalidate in invalidations:
                with self.subTest(node_index=node_index, invalidation=name):
                    _work_id, envelope = self._two_node_frame()
                    invalidate(envelope["node_transitions"][node_index])
                    self._assert_rejected_before_work_transaction(envelope)

    def test_negative_remaining_parses_but_prepare_rejects_atomically(self):
        _work_id, envelope = self._two_node_frame()
        row = envelope["node_transitions"][1]
        row["kind"] = "progress"
        row["payload"]["delta_served"] = {"ops": 0}
        row["payload"]["remaining_ops"] = -1

        # The generic legacy parser and v3 scalar factory both accept finite
        # negative remaining values at parse time; progress conservation is
        # checked later against the committed work during transaction prepare.
        generic = work_ledger_module._typed_event_from_mapping(
            self._typed_node_wire(row, envelope["delivery_sequence"]))
        parsed = PipelineLedger().validate_envelope(envelope)
        accelerated = parsed.node_events[-1][1]
        self.assertEqual(accelerated.kind, generic.kind)
        self.assertEqual(accelerated.work_id, generic.work_id)
        self.assertEqual(accelerated.tick, generic.tick)
        self.assertEqual(accelerated.producer, generic.producer)
        self.assertEqual(accelerated.delivery_sequence,
                         generic.delivery_sequence)
        self.assertEqual(accelerated.local_seq, generic.local_seq)
        self.assertEqual(accelerated.delta_served, generic.delta_served)
        generic_body = {
            key: value for key, value in generic.payload.items()
            if key not in ("kind", "work_id", "tick", "event_id")}
        self.assertEqual(accelerated.payload, generic_body)
        self.assertEqual(accelerated.canonical_content,
                         generic.canonical_content)

        scheduler = object.__new__(OnlineSchedulerBase)
        scheduler.joint_pipeline_ledger = PipelineLedger()
        scheduler.joint_work_ledger = WorkLedger()
        scheduler.work_ledger_absent_seen = False
        pipeline = scheduler.joint_pipeline_ledger
        work = scheduler.joint_work_ledger
        pipeline_before = pipeline.snapshot_state()
        work_before = work.snapshot_state()
        pipeline_marks = (
            pipeline.initialized, pipeline.delivery_sequence,
            pipeline.watermark, pipeline.state_version)
        work_marks = (
            work.last_delivery_sequence, work.event_watermark,
            work.version, work.full_load_count, work.delta_load_count)

        with self.assertRaisesRegex(
                WorkLedgerError, "remaining_ops disagrees with delta"):
            scheduler._apply_work_ledger({"work_ledger": envelope})

        self.assertIs(pipeline.snapshot_state(), pipeline_before)
        self.assertIs(work.snapshot_state(), work_before)
        self.assertEqual(
            (pipeline.initialized, pipeline.delivery_sequence,
             pipeline.watermark, pipeline.state_version),
            pipeline_marks)
        self.assertEqual(
            (work.last_delivery_sequence, work.event_watermark,
             work.version, work.full_load_count, work.delta_load_count),
            work_marks)
        self.assertEqual(work._works, {})
        self.assertIsNone(work._last_event)

    def test_bad_outer_integer_fields_reject_before_work_transaction(self):
        invalid_fields = (
            ("delivery_sequence", 0.0),
            ("delivery_sequence", True),
            ("tick", 0.0),
            ("tick", True),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                _work_id, envelope = self._two_node_frame()
                envelope[field] = value
                self._assert_rejected_before_work_transaction(envelope)

    def test_float_and_int_payloads_remain_distinct_in_frame_replay_bytes(self):
        _work_id, integer_frame = self._two_node_frame()
        float_frame = copy.deepcopy(integer_frame)
        float_frame["node_transitions"][0]["payload"]["total"]["ops"] = 10.0
        integer_parsed = PipelineLedger().validate_envelope(integer_frame)
        float_parsed = PipelineLedger().validate_envelope(float_frame)
        self.assertIsNone(integer_parsed.node_events[0][1].canonical_content)
        self.assertIsNone(float_parsed.node_events[0][1].canonical_content)
        self.assertNotEqual(integer_parsed.canonical_bytes,
                            float_parsed.canonical_bytes)


if __name__ == "__main__":
    unittest.main()
