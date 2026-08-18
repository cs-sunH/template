#!/usr/bin/env python3
"""sh10_replay_scheduler.py -- sh_1.0 决策日志回放调度器(replay 模式)。

方案 §4 步骤 1-8 操作 3(sh_1.0 适配新写;蓝本 wsc_llm_replay_scheduler.py
为模板)。不跑任何策略:每个决策边界从 decision_log.jsonl(阶段 0 产出,
replay 权威)取对应决策,交给 graph_batch_builder 翻译成与离线 ET 相同的
节点集。三段式发射(本仓与蓝本两段式的差异):

  边界                    登记(条件集)      记日志 + 动作
  ARRIVAL                prefill 流 arrived  prefill 记录(本边界)+ turn-0 到达门 +
                                             段 1(history_evictions/history_transfer/
                                             prefill_evictions/prefill 屏障/prefill)+
                                             PREFILL_DRAIN watch(准入排队 q>1us 时
                                             prefill 相位时长吸收 q,蓝本 q-吸收同款)
  PREFILL_DRAIN          decode 流 drained   decode 记录(本边界)+ 段 2(decode_evictions/
                                             prefill→decode 迁移/decode 屏障/decode)+
                                             DECODE_COMPLETION watch
  DECODE_COMPLETION      completion 流 completed 无(收尾边界由 REQUEST_COMPLETE 处理)
  REQUEST_COMPLETE       --                  completion 记录(本边界)+ 段 3
                                             (completion_evictions + 下一 turn interval 门)+
                                             REQUEST_COMPLETE watch(无 completion_evictions
                                             时成员 = 段 2 末节点,口径显式记录)+
                                             下一次 session arrival 排程
"""

from online.online_scheduler_base import (
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)

# 准入排队吸收阈值(蓝本同款):turn-0 准入排队 q = prefill 记录 tick -
# 到达边界 tick;q 超过阈值时 prefill 相位时长吸收 q。本仓 admission 同样
# 可因 HBM 暂不可行排队(pending_admissions,方案 §0.4 #17)。
_DEFER_THRESHOLD_NS = 1000


class Sh10ReplayScheduler(OnlineSchedulerBase):
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
        self._by_turn = {
            (record["session_id"], record["turn_index"]): record
            for record in manifest["requests"]
        }

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """completion 批先于 arrival 批(与离线事件循环 :2970 一致)。"""
        tick = delta["tick"]
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if stage == STAGE_PREFILL:
                self._on_prefill_drain(request_id, tick)
            elif stage == STAGE_DECODE:
                # DECODE_COMPLETION 只登记 completion 流条件集合(记日志与
                # 下一次 arrival 排程由同 tick 的 REQUEST_COMPLETE 处理)。
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
        self.log_decision(record, tick)
        # 段 1 需要 prefill 决策的全部构图输入:实例/历史迁移/逐出序列。
        seg1_plan = self._segment1_plan(request_id, plan, record)
        absorb_queue_ns = 0
        if record["tick"] > tick + _DEFER_THRESHOLD_NS:
            absorb_queue_ns = record["tick"] - tick
        self._emit_segment1(request_id, seg1_plan, absorb_queue_ns)

    def _segment1_plan(self, request_id: str, plan: dict, record: dict) -> dict:
        decision = record["decision"]
        seg = dict(plan)
        seg["prefill_instance_index"] = (
            plan["prefill_assignment"]["instance_index"])
        seg["decode_instance_index"] = (
            plan["decode_assignment"]["instance_index"])
        seg["admission_time_ns"] = decision["admission_time_ns"]
        seg["history_location_before"] = decision["history_location_before"]
        seg["history_transfer"] = decision["history_transfer"]
        seg["history_evictions"] = decision["history_evictions"]
        seg["prefill_evictions"] = decision["prefill_evictions"]
        return seg

    def _emit_segment1(self, request_id: str, seg1_plan: dict,
                       absorb_queue_ns: int = 0) -> None:
        prefill_dur, _ = self.replay.phase_durations(request_id)
        prefill_dur += absorb_queue_ns
        members = self.graph.emit_prefill_batch(
            seg1_plan, phase_duration_ns=prefill_dur)
        self._batch["watches"].append({
            "request_id": request_id,
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": request_id,
            "prefill_instance_index": seg1_plan["prefill_instance_index"],
            "decode_instance_index": self.request_by_id[request_id][
                "decode_assignment"]["instance_index"],
        })
        if self.sensing_enabled:
            self._note_emitted(request_id, STAGE_PREFILL)
            self._ledger_admit(request_id, self._batch["tick"], {
                "type": "prefill_qp",
                "instance_index": seg1_plan["prefill_instance_index"]})
            self._ledger_issue(request_id, self._batch["tick"],
                               STAGE_PREFILL,
                               seg1_plan["prefill_instance_index"])

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        self.replay.register("decode", request_id)
        record = self.replay.record_for("decode", request_id)
        self.log_decision(record, tick)
        plan = self.request_by_id[request_id]
        decision = record["decision"]
        seg2_plan = dict(plan)
        seg2_plan["prefill_instance_index"] = (
            plan["prefill_assignment"]["instance_index"])
        seg2_plan["decode_instance_index"] = (
            plan["decode_assignment"]["instance_index"])
        seg2_plan["decode_evictions"] = decision["decode_evictions"]
        seg2_plan["prefill_decode_transfer"] = decision["prefill_decode_transfer"]
        _, decode_dur = self.replay.phase_durations(request_id)
        members = self.graph.emit_decode_batch(
            seg2_plan, phase_duration_ns=decode_dur)
        self._batch["watches"].append({
            "request_id": request_id,
            "stage": STAGE_DECODE,
            "generation": 1,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        if self.sensing_enabled:
            self._ledger_unissue(request_id, STAGE_PREFILL)
            self._note_emitted(request_id, STAGE_DECODE)
            self._ledger_admit(request_id, tick, {
                "type": "active_decode",
                "instance_index": seg2_plan["decode_instance_index"]})
            self._ledger_issue(request_id, tick, STAGE_DECODE,
                               seg2_plan["decode_instance_index"])

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        record = self.replay.record_for("completion", request_id)
        self.log_decision(record, tick)
        plan = self.request_by_id[request_id]
        decision = record["decision"]
        seg3_plan = dict(plan)
        seg3_plan["decode_instance_index"] = (
            plan["decode_assignment"]["instance_index"])
        seg3_plan["completion_evictions"] = decision["completion_evictions"]
        seg3_plan["kv_location_after_completion"] = (
            plan["completion_kv_location"]["location"])
        next_plan = self._next_turn_plan(request_id)
        if next_plan is None:
            seg3_plan["following"] = None
        else:
            seg3_plan["following"] = {
                "queue_index": next_plan["queue_index"],
                "request_id": next_plan["request_id"],
                "hbm_wait_ns": self._hbm_wait_ns(next_plan),
            }
        members = self.graph.emit_completion_batch(seg3_plan)
        if not members:
            # 无 completion_evictions:REQUEST_COMPLETE 成员 = 段 2 末节点
            # (该口径在合同④/方案步骤 1-8 显式记录)。
            members = dict(self.graph._block_ends.get(request_id, {}).get(
                "seg2", {}))
        # REQUEST_COMPLETE 不注册独立 watch:decode watch fire 已同时推送
        # DECODE_COMPLETION + REQUEST_COMPLETE(main_online.cc 机制)。
        if next_plan is not None:
            next_record = self.replay.peek_prefill(next_plan["request_id"])
            # replay 权威:alarm 时刻 = 下一 prefill 决策记录 tick(计划
            # admission_time,含准入排队)。
            # 边界:下 turn 记录 tick == 本完成边界 tick(interval=0 的
            # session)时,在线完成事件比 LUT 时钟晚 1ns(1ns 事件粒度),
            # alarm 会被校验拒绝(past arrival)。按 RequestIngress 迟到
            # 钳制同款口径 clamp 到 current+1(登记为 sh_1.0 边界口径)。
            self._batch["future_alarms"].append({
                "arrival_world_ns": max(next_record["tick"], tick + 1),
                "envelope": {
                    "request_id": next_plan["request_id"],
                    "session_id": next_plan["session_id"],
                    "turn_index": next_plan["turn_index"],
                    "prefill_length": next_plan["prefill_length"],
                    "decode_length": next_plan["decode_length"],
                    "inter_request_interval_ns": self._interval_ns(next_plan),
                },
            })
        if self.sensing_enabled:
            self._ledger_unissue(request_id, STAGE_DECODE)

    # ------------------------------------------------------------- 助手 --

    def _next_turn_plan(self, request_id: str):
        plan = self.request_by_id[request_id]
        return self._by_turn.get(
            (plan["session_id"], plan["turn_index"] + 1))

    def _interval_ns(self, plan: dict):
        spec = self.config.request_queue[plan["queue_index"]]
        return spec.inter_request_interval_ns

    def _hbm_wait_ns(self, plan: dict) -> int:
        return plan["planned_timing_ns"]["hbm_wait"]
