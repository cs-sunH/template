#!/usr/bin/env python3
"""wsc_llm_replay_scheduler.py -- 决策日志回放调度器(replay 模式)。

方案 §4 步骤 1-8 操作 3。不跑任何策略:每个决策边界从 decision_log.jsonl
(阶段 0 产出,replay 权威)取对应决策,交给 graph_batch_builder 翻译成与
离线 ET 相同的节点集。变体策略(run_variant_policy)的 replay 实现:

  边界                    登记(条件集)            记日志 + 动作
  ARRIVAL                prefill 流 arrived      prefill 记录(本边界)+ 发射 prefill 整段 +
                                                 PREFILL_DRAIN watch(turn-0 准入排队 q>1us 时
                                                 prefill 相位时长吸收 q:链仍从到达边界起算、
                                                 终点=decode 记录 tick,LUT 时钟逐请求对齐)
  PREFILL_DRAIN          decode 流 drained       decode 记录(本边界)+ 发射 decode 整段 +
                                                 DECODE_COMPLETION watch
  DECODE_COMPLETION      completion 流 completed 无(收尾边界由 REQUEST_COMPLETE 处理)
  REQUEST_COMPLETE       --                      completion 记录(本边界)+ 下一次 session arrival 排程

阶段 7 §10.8(3min 在线验证,方案 B):边界事件只经 replay.register 登记进
条件集合,replay_source 游标按记录序头部推进——触发序与记录序的反转(离
线准入排队使 turn-0 CSV 到达早于其 prefill 记录 tick)只造成等待,绝不
失步;在线决策日志行序 = 触发序(与修复前逐字节一致,30s 回归门)。

下一次 session arrival 排程(inter_request_interval_ns 语义不变,replay 口径):
alarm 的时刻 = 下一 turn 的 prefill 决策记录 tick(= 离线计划 prefill_start_ns,
含准入排队)。图上的 interval gate 仍依赖上一 turn 的完成 barrier,实际
执行不会早于真实完成。
"""

from online.online_scheduler_base import (
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)

# 准入排队吸收阈值(turn-0 准入排队 q = prefill 记录 tick - 到达边界 tick):
# q 超过该阈值时 prefill 相位时长校准吸收 q(时长 = decode 记录 tick - 到达
# 时刻,链仍从到达边界起算、终点 = decode 记录 tick——与 LUT 时钟逐请求
# 对齐,且不产生"本请求已在飞"的 future alarm,C++ GraphBatch 校验拒绝)。
# 30s 输入实测 q<=17ns,阈值 1us 下不触发——行为与修复前逐字节一致(链
# 完成时刻 = decode 记录 tick - q,偏差 <=17ns,30s 规律内);3min 输入
# q 达 23.6s 全部吸收。turn>0 到达边界 tick == 记录 tick(q==0),恒不触发。
_DEFER_THRESHOLD_NS = 1000


class WscLlmReplayScheduler(OnlineSchedulerBase):
    """replay 变体:决策全部来自离线日志,登记式消费,无策略判据。"""

    def __init__(self, *, manifest, config, replay, graph, digest_sink=None,
                 mode: str = "replay"):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=replay,
            digest_sink=digest_sink,
            mode=mode,
        )
        self.graph = graph  # GraphBatchBuilder(构图状态跨批次)
        # 阶段 4 §7.3:(session_id, turn_index) -> manifest 记录 索引,
        # 替换 _next_turn_plan 的 O(N) 全量扫描(turn+1 精确语义一致)。
        self._by_turn = {
            (record["session_id"], record["turn_index"]): record
            for record in manifest["requests"]
        }

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """completion 批先于 arrival 批(与离线事件循环一致)。"""
        tick = delta["tick"]
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if stage == STAGE_PREFILL:
                self._on_prefill_drain(request_id, tick)
            elif stage == STAGE_DECODE:
                # 阶段 7 §10.8:DECODE_COMPLETION 只登记 completion 流条件
                # 集合(记日志与下一次 arrival 排程由同 tick 一并交付的
                # REQUEST_COMPLETE 处理)。
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
            # 第二次 ARRIVAL:同一请求的重复到达事件。修复后发射对齐不再
            # 排本请求的 alarm,正常路径不会出现;出现即失步(fail-closed)。
            raise RuntimeError(
                "request {!r} arrived twice (replay)".format(request_id))
        # 首次到达:本边界记日志(行序 = 触发序,修复前逐字节一致)。
        record = self.replay.record_for("prefill", request_id)
        self.log_decision(record, tick)
        if record["tick"] > tick + _DEFER_THRESHOLD_NS:
            # turn-0 准入排队 q>1us:prefill 相位时长吸收 q(链终点 =
            # decode 记录 tick,LUT 时钟对齐;起点仍在到达边界,无本请求
            # future alarm)。
            self._emit_prefill(request_id, plan, absorb_queue_ns=record["tick"] - tick)
            return
        self._emit_prefill(request_id, plan)

    def _emit_prefill(self, request_id: str, plan: dict,
                      absorb_queue_ns: int = 0) -> None:
        # 步骤 1-8 计时校准:prefill 相位时长 = 日志 LUT 时钟(在线引擎
        # 的 COMP 链据此校准,replay 消费才不失步);准入排队吸收时 += q。
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
        # 本边界记日志(链完成时刻 = decode 记录 tick ± ns 抖动,行序 =
        # 触发序,修复前逐字节一致)。
        record = self.replay.record_for("decode", request_id)
        self.log_decision(record, tick)
        plan = self.request_by_id[request_id]
        # 步骤 1-8 计时校准:decode 相位时长 = 日志 LUT 时钟。
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
        # 本边界记日志(completion 记录;条件登记在 DECODE_COMPLETION)。
        record = self.replay.record_for("completion", request_id)
        self.log_decision(record, tick)
        next_plan = self._next_turn_plan(request_id)
        if next_plan is None:
            return  # session 最后一 turn:无下一次 arrival
        next_record = self.replay.peek_prefill(next_plan["request_id"])
        # replay 权威:alarm 时刻 = 下一 prefill 决策记录 tick(计划
        # prefill_start,含准入排队)。
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

    # ------------------------------------------------------------- 助手 --

    def _next_turn_plan(self, request_id: str):
        """同 session 的下一 turn 的 manifest 记录(没有则返回 None)。
        阶段 4 §7.3:经 (session_id, turn_index) 索引 O(1) 定位
        (替换 O(N) 全量扫描)。"""
        plan = self.request_by_id[request_id]
        return self._by_turn.get(
            (plan["session_id"], plan["turn_index"] + 1))

    def _interval_ns(self, plan: dict):
        spec = self.config.request_queue[plan["queue_index"]]
        return spec.inter_request_interval_ns
