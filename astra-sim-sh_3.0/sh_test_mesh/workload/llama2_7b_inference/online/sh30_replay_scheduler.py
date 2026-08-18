#!/usr/bin/env python3
"""sh30_replay_scheduler.py -- sh_3.0 决策日志回放调度器（replay 模式）。

方案 §4 步骤 1-8 操作 3（sh_3.0 版）。不跑任何策略：每个决策边界从
decision_log.jsonl（阶段 0 --replay-record 产出，replay 权威）取对应决策，
交给 graph_batch_builder 翻译成与离线 ET 相同的节点集。

  边界                    登记(条件集)         动作
  ARRIVAL                prefill 流 arrived    prefill 记录 + 发射 prefill 整段 +
                                               PREFILL_DRAIN watch（q-吸收同蓝本裁决 12）
  PREFILL_DRAIN          decode 流 drained     decode 记录 + 发射 decode 整段 +
                                               DECODE_COMPLETION watch
  DECODE_COMPLETION      completion 流 completed 无（收尾由 REQUEST_COMPLETE 处理）
  REQUEST_COMPLETE       --                    completion 记录 + 发射 completion 批
                                               （completion_evictions + 下一 turn
                                               interval gate）+ 下一次 session arrival 排程

决策事实（实例/KV 动作）来自 manifest.json（离线 plan 的逐请求记录，
transfers_by_stage 携带全部 KVTransfer 事实）；时序权威 = decision_log tick。
"""

from face_scheduler import KVTransfer, KVTransferShard
from online.online_scheduler_base import (
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)

# 准入排队吸收阈值（蓝本裁决 12 同款）：q > 1us 时 prefill 相位时长吸收 q。
# SH30_DEFER_THRESHOLD_NS 仅用于决定性验证实验（B1 round 2 扫掠），生产
# 缺省 1000 不变。
_DEFER_THRESHOLD_NS = int(__import__("os").environ.get(
    "SH30_DEFER_THRESHOLD_NS", "1000"))


class _LocationShim:
    """history_location_before 的 dict -> 属性面 shim（builder/
    reconcile_pending_history_location 消费）。"""

    __slots__ = ("location", "instance_index", "resident_prefix_layers",
                 "total_bytes", "context_tokens")

    def __init__(self, record):
        self.location = record["location"]
        self.instance_index = record["instance_index"]
        self.resident_prefix_layers = record["resident_prefix_layers"]
        self.total_bytes = record["total_bytes"]
        self.context_tokens = record["context_tokens"]


def _shard_from_dict(shard: dict) -> KVTransferShard:
    return KVTransferShard(
        source_rank=shard["source_rank"],
        target_rank=shard["target_rank"],
        edge_rank=shard["edge_rank"],
        bytes=shard["bytes"],
        noc_path=tuple(shard["noc_path"]),
        layer_start=shard["layer_start"],
        layer_end=shard["layer_end"],
    )


def _transfer_from_dict(record: dict) -> KVTransfer:
    return KVTransfer(
        kind=record["kind"],
        phase=record["phase"],
        reason=record["reason"],
        session_id=record["session_id"],
        trigger_request_id=record["trigger_request_id"],
        source_instance_index=record["source_instance_index"],
        target_instance_index=record["target_instance_index"],
        total_bytes=record["total_bytes"],
        shards=tuple(
            _shard_from_dict(shard) for shard in record["shards"]),
        model_layers=record["model_layers"],
        layer_start=record["layer_start"],
        layer_end=record["layer_end"],
        resident_prefix_layers_before=record["resident_prefix_layers_before"],
        resident_prefix_layers_after=record["resident_prefix_layers_after"],
    )


def _transfers(records, key):
    return tuple(
        _transfer_from_dict(record)
        for record in records["transfers_by_stage"][key])


def build_plan_dict(record: dict) -> dict:
    """manifest 逐请求记录 -> graph_batch_builder 消费的 plan dict。"""
    timing = record["planned_timing_ns"]
    history_location = record.get("history_location_before")
    pd_records = record["transfers_by_stage"]["prefill_decode_transfer"]
    return {
        "request_id": record["request_id"],
        "session_id": record["session_id"],
        "turn_index": record["turn_index"],
        "queue_index": record["queue_index"],
        "prefill_length": record["prefill_length"],
        "decode_length": record["decode_length"],
        "prefill_instance_index":
            record["prefill_assignment"]["instance_index"],
        "decode_instance_index":
            record["decode_assignment"]["instance_index"],
        "prefill_affinity_reason":
            record["prefill_assignment"].get("affinity_reason"),
        "history_tokens_before": record["history_tokens_before"],
        "prefill_context_tokens": record["prefill_context_tokens"],
        "final_context_tokens": record["final_context_tokens"],
        "admission_time_ns": timing["hbm_admission"],
        "hbm_wait_ns": timing["hbm_wait"],
        "history_source_instance_index":
            record["history_source_instance_index"],
        "history_transfer_bytes": record["history_transfer_bytes"],
        "history_tokens_discarded": record["history_tokens_discarded"],
        "history_location_before": (
            None if history_location is None
            else _LocationShim(history_location)),
        "history_evictions":
            _transfers(record, "history_evictions"),
        "history_transfer": (
            _transfer_from_dict(record["transfers_by_stage"]
                                ["history_transfer"][0])
            if record["transfers_by_stage"]["history_transfer"] else None),
        "prefill_evictions": _transfers(record, "prefill_evictions"),
        "decode_evictions": _transfers(record, "decode_evictions"),
        "prefill_decode_transfer": (
            _transfer_from_dict(pd_records[0]) if pd_records else None),
        "completion_evictions": _transfers(record, "completion_evictions"),
        "kv_location_after_completion":
            record["completion_kv_location"]["location"],
        "kv_instance_after_completion":
            record["completion_kv_location"]["instance_index"],
    }


class Sh30ReplayScheduler(OnlineSchedulerBase):
    """replay 变体：决策全部来自离线日志与 manifest，登记式消费，无策略判据。"""

    def __init__(self, *, manifest, config, replay, graph, digest_sink=None,
                 mode: str = "replay"):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=replay,
            digest_sink=digest_sink,
            mode=mode,
        )
        self.graph = graph
        self.plans = {
            record["request_id"]: build_plan_dict(record)
            for record in manifest["requests"]
        }
        by_turn = {
            (record["session_id"], record["turn_index"]): record
            for record in manifest["requests"]
        }
        # request_id -> 下一 turn plan dict（interval gate 发射用）。
        self.graph.set_next_plan({
            record["request_id"]:
                self.plans[by_turn[
                    (record["session_id"], record["turn_index"] + 1)
                ]["request_id"]]
            if (record["session_id"], record["turn_index"] + 1) in by_turn
            else None
            for record in manifest["requests"]
        })

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        tick = delta["tick"]
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if stage == STAGE_PREFILL:
                self._on_prefill_drain(request_id, tick)
            elif stage == STAGE_DECODE:
                self.replay.register("completion", request_id)
            elif stage == STAGE_REQUEST:
                self._on_request_complete(request_id, tick)
            else:
                raise ValueError(
                    "unknown completion stage {!r} for request {!r}".format(
                        stage, request_id))
        for arrival in delta["arrivals"]:
            self._on_arrival(arrival, tick)

    # ------------------------------------------------------------- 边界 --

    def _on_arrival(self, arrival: dict, tick: int) -> None:
        request_id = arrival["request_id"]
        plan = self.plans[request_id]
        is_new = self.replay.register("prefill", request_id)
        if not is_new:
            raise RuntimeError(
                "request {!r} arrived twice (replay)".format(request_id))
        record = self.replay.record_for("prefill", request_id)
        self.log_decision(record, tick)
        if record["tick"] > tick + _DEFER_THRESHOLD_NS:
            self._emit_prefill(request_id, plan,
                               absorb_queue_ns=record["tick"] - tick)
            return
        self._emit_prefill(request_id, plan)

    def _emit_prefill(self, request_id: str, plan: dict,
                      absorb_queue_ns: int = 0) -> None:
        prefill_dur, _ = self.replay.phase_durations(request_id)
        prefill_dur += absorb_queue_ns
        members = self.graph.emit_prefill_batch(
            plan, phase_duration_ns=prefill_dur)
        self._batch["watches"].append({
            "request_id": request_id,
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": request_id,
            "prefill_instance_index": plan["prefill_instance_index"],
            "decode_instance_index": plan["decode_instance_index"],
            "prefill_affinity_reason": plan["prefill_affinity_reason"],
        })

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        self.replay.register("decode", request_id)
        record = self.replay.record_for("decode", request_id)
        self.log_decision(record, tick)
        plan = self.plans[request_id]
        _, decode_dur = self.replay.phase_durations(request_id)
        members = self.graph.emit_decode_batch(
            plan, phase_duration_ns=decode_dur)
        self._batch["watches"].append({
            "request_id": request_id,
            "stage": STAGE_DECODE,
            "generation": 1,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": request_id,
            "prefill_instance_index": plan["prefill_instance_index"],
            "decode_instance_index": plan["decode_instance_index"],
        })

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        record = self.replay.record_for("completion", request_id)
        self.log_decision(record, tick)
        plan = self.plans[request_id]
        # completion 批：completion_evictions + 下一 turn interval gate
        # （合同① 两段式发射的 completion 边界）。
        self.graph.emit_completion_batch(plan)
        next_plan = self.graph.next_plan.get(request_id)
        if next_plan is None:
            return
        next_record = self.replay.peek_prefill(next_plan["request_id"])
        self._batch["future_alarms"].append({
            "arrival_world_ns": next_record["tick"],
            "envelope": {
                "request_id": next_plan["request_id"],
                "session_id": next_plan["session_id"],
                "turn_index": next_plan["turn_index"],
                "prefill_length": next_plan["prefill_length"],
                "decode_length": next_plan["decode_length"],
                "inter_request_interval_ns": self._interval_ns(next_plan),
            },
        })

    def _interval_ns(self, plan: dict):
        spec = self.config.request_queue[plan["queue_index"]]
        return spec.inter_request_interval_ns
