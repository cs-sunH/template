#!/usr/bin/env python3
"""sh20_replay_scheduler.py -- sh_2.0 决策日志回放调度器（replay 模式）。

方案 §4 步骤 1-8 操作 3（蓝本 wsc_llm_replay_scheduler 同构）。不跑任何
策略：每个决策边界从 decision_log.jsonl（阶段 0 产出，replay 权威）取对应
决策，交给 graph_batch_builder 翻译成与离线 ET 相同的节点集。

  边界                    登记(条件集)            记日志 + 动作
  ARRIVAL                prefill 流 arrived      prefill 记录 + 发射 prefill 整段 +
                                                 PREFILL_DRAIN watch
  PREFILL_DRAIN          decode 流 drained       decode 记录 + 发射 decode 整段 +
                                                 DECODE_COMPLETION watch
  DECODE_COMPLETION      completion 流 completed 无（收尾边界由 REQUEST_COMPLETE 处理）
  REQUEST_COMPLETE       --                      completion 记录 + 发射 completion 段
                                                 （逐出 + 下一 turn interval gates）+
                                                 下一次 session arrival 排程

"到达条件集 + 头部推进"消费语义（蓝本修复后版本）：边界事件只登记，游标按
记录序头部推进；失步由 consumed_all() 兜底 fail-closed。

下一次 session arrival 排程（replay 口径，蓝本同款）：alarm 时刻 = 下一
turn 的 prefill 决策记录 tick（= 离线计划 prefill_start_ns，含准入排队）。
"""

from online.online_scheduler_base import (
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)

# 准入排队吸收阈值（蓝本同款；turn-0 准入排队 q = prefill 记录 tick -
# 到达边界 tick）：q 超过阈值时 prefill 相位时长吸收 q（链终点 = decode
# 记录 tick，LUT 时钟逐请求对齐；不产生"本请求已在飞"的 future alarm）。
_DEFER_THRESHOLD_NS = 1000


class Sh20ReplayScheduler(OnlineSchedulerBase):
    """replay 变体：决策全部来自离线日志，登记式消费，无策略判据。"""

    def __init__(self, *, manifest, config, replay, graph, digest_sink=None,
                 mode: str = "replay"):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=replay,
            digest_sink=digest_sink,
            mode=mode,
        )
        self.graph = graph  # GraphBatchBuilder（构图状态跨批次）
        # (session_id, turn_index) -> manifest 记录 索引（O(1) 定位下一 turn）。
        self._by_turn = {
            (record["session_id"], record["turn_index"]): record
            for record in manifest["requests"]
        }

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """completion 批先于 arrival 批（与离线事件循环一致）。"""
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
        plan = self.request_by_id[request_id]
        is_new = self.replay.register("prefill", request_id)
        if not is_new:
            raise RuntimeError(
                "request {!r} arrived twice (replay)".format(request_id))
        record = self.replay.record_for("prefill", request_id)
        # sh_2.0 特有（实录 bug#10 终态裁决）：turn-0 准入排队达秒级
        # （task-load 均衡 + HBM 等待，离线 prefill 记录 tick = admission
        # 时刻，远晚于 CSV 到达）。发射在到达边界进行（链起点 = 到达），
        # 相位时长吸收准入排队（wscllm 蓝本 _DEFER_THRESHOLD 同款）；
        # 在线决策日志行序 = 触发序，与记录序的解耦由 online_service 的
        # replay 排序出口统一（按权威 record tick 排序，B1 比较口径）。
        absorb = 0
        if record["tick"] > tick + _DEFER_THRESHOLD_NS:
            absorb = record["tick"] - tick
        self._emit_prefill(request_id, plan, record, tick, absorb)

    def _emit_prefill(self, request_id: str, plan: dict, record: dict,
                      tick: int, absorb_queue_ns: int = 0) -> None:
        self.log_decision(record, tick)
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
        })

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        self.replay.register("decode", request_id)
        record = self.replay.record_for("decode", request_id)
        self.log_decision(record, tick)
        plan = self.request_by_id[request_id]
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

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        record = self.replay.record_for("completion", request_id)
        self.log_decision(record, tick)
        plan = self.request_by_id[request_id]
        next_plan = self._next_turn_plan(request_id)
        self.graph.emit_completion_batch(plan, next_plan)
        if next_plan is None:
            return  # session 最后一 turn：无下一次 arrival
        next_record = self.replay.peek_prefill(next_plan["request_id"])
        # replay 权威：alarm 时刻 = 下一 prefill 决策记录 tick。interval==0
        # 的 turn 在在线时钟相对 LUT 时钟漂移数 ns 后，记录 tick 可能落在
        # 当前边界之前——钳制到 tick+1（ingress 的迟到钳制同款策略，
        # RequestIngress::late_arrival_count 计数可观测；顺序仍严格递增）。
        alarm_tick = max(next_record["tick"], tick + 1)
        self._batch["future_alarms"].append({
            "arrival_world_ns": alarm_tick,
            "envelope": {
                "request_id": next_plan["request_id"],
                "session_id": next_plan["session_id"],
                "turn_index": next_plan["turn_index"],
                "prefill_length": next_plan["prefill_length"],
                "decode_length": next_plan["decode_length"],
                "inter_request_interval_ns": self._interval_ns(next_plan),
            },
        })

    # ------------------------------------------------------------- 助手 --

    def _next_turn_plan(self, request_id: str):
        plan = self.request_by_id[request_id]
        return self._by_turn.get(
            (plan["session_id"], plan["turn_index"] + 1))

    def _interval_ns(self, plan: dict):
        spec = self.config.request_queue[plan["queue_index"]]
        return spec.inter_request_interval_ns
