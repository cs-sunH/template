#!/usr/bin/env python3
"""replay_source.py -- 决策日志回放源(方案 §4 步骤 1-8 操作 2)。

顺序读阶段 0 产出的 decision_log.jsonl(每行 {seq, tick, priority, kind,
request_id, decision},按 (tick, priority, seq) 排序,kind 为 prefill /
iteration / decode / completion 四种;iteration 是离线的 LUT 计时计划,在线
删除计时部分,不消费)。

消费语义 = 阶段 7 §10.8(3min 在线验证)"到达条件集 + 头部推进":

    register(kind, request_id)    边界事件只登记:请求进入该流条件集合
                                  (arrived / drained / completed);游标按
                                  记录序推进——头部记录所属请求已在集合
                                  内则越过并继续,否则游标停在头部等待。
                                  返回 True = 新登记(首次到达);False =
                                  已登记(同一请求的第二次 ARRIVAL——发射
                                  对齐 alarm 的合法重复,调度器据此只发射
                                  不再登记/记日志)。
    record_for(kind, request_id)  该 request 的决策记录——本边界记日志用
                                  (在线决策日志的行序 = 边界触发序,与
                                  LUT 时钟的纳秒抖动一致,同 30s 输入)。

边界事件与记录序解耦:记录序反转(离线准入排队使 turn-0 的 CSV 到达早于
其 prefill 记录 tick,四档 3min 全部非零——到达参考序 vs 记录序的成对
反转 prefill 63/29677/48649/233705,方法见 validation_3min_online/REPORT.md
§3.1 与 /tmp/wscllm3min/analyze_inversions.py)不再失步——游标只等待,
不失步;失步只剩"永不满足"的运行结束形态,由 consumed_all() 兜底
(fail-closed: 未消费完即 ReplayDesyncError 式报错)。

peek_prefill(request_id):非消费查找某 request 的 prefill 决策——REQUEST_COMPLETE
边界用它取"下一次 session arrival"的 alarm 时刻(= 该 request 的 prefill
记录 tick;其 arrival 事件仍由未来的 ARRIVAL 边界经 register 登记)。
"""

import json
from pathlib import Path

# 消费的决策 kind 闭集(离线日志的 iteration 行是 LUT 计时计划,在线不消费)。
_CONSUMED_KINDS = ("prefill", "decode", "completion")


class ReplayDesyncError(RuntimeError):
    """回放源与当前边界不一致:replay 失步(fail-closed)。"""


class ReplaySource:
    def __init__(self, decision_log_path):
        self.path = Path(decision_log_path)
        with open(self.path, "r", encoding="utf-8") as source:
            lines = [json.loads(line) for line in source if line.strip()]
        if not lines:
            raise ReplayDesyncError("decision log is empty: {}".format(self.path))
        self._streams = {
            kind: [record for record in lines if record["kind"] == kind]
            for kind in _CONSUMED_KINDS
        }
        for kind in _CONSUMED_KINDS:
            if not self._streams[kind]:
                raise ReplayDesyncError(
                    "decision log has no {} records: {}".format(kind, self.path))
        self._cursors = {kind: 0 for kind in _CONSUMED_KINDS}
        # 阶段 7 §10.8(3min 在线验证,方案 B):"到达条件集 + 头部推进"——
        # 边界事件只登记请求进入条件集合(arrived/drained/completed);游标
        # 按记录序推进,头部记录所属请求在集合内才越过(不在则等待,绝不
        # 因触发序与记录序反转而失步)。30s 输入 0 结构反转,行为与严格
        # next-match 逐字节一致(同 tick 组的纳秒抖动由"本边界记日志"保留
        # 触发序,见 register/record_for 文档)。
        self._condition_sets = {
            "prefill": set(),  # ARRIVAL 边界登记
            "decode": set(),   # PREFILL_DRAIN 边界登记
            "completion": set(),  # DECODE_COMPLETION 边界登记
        }
        # 每 kind 的 request_id -> 记录 索引(本边界记日志用 O(1) 定位)。
        self._records_by_request = {}
        for kind in _CONSUMED_KINDS:
            self._records_by_request[kind] = {
                record["request_id"]: record
                for record in self._streams[kind]
            }
        self._request_to_prefill = self._records_by_request["prefill"]
        # 阶段 4 §7.3:prefill 流内下标索引(peek_prefill 的"已消费判定"
        # O(N) list.index -> O(1) 字典定位)。
        self._prefill_positions = {
            record["request_id"]: index
            for index, record in enumerate(self._streams["prefill"])
        }
        # 日志覆盖的 request 集合(结束校验:全部消费)。
        self.covered_request_ids = set(self._request_to_prefill)
        # LUT 时钟的相位时长索引(步骤 1-8 在线计时校准):prefill 时长 =
        # decode 记录 tick - prefill 记录 tick,decode 时长 = completion
        # 记录 tick - decode 记录 tick。日志的 (tick, priority, seq) 排序
        # 即离线规划器的 LUT 时钟(decision_log 权威);三个 tick 之差与
        # iteration 记录的 iteration_time_ns 求和逐请求一致(实测 1177/1177
        # 三条目齐全,相位边界含准入排队口径)。在线引擎的 COMP 链据此
        # 校准到同一时钟,replay 顺序消费才不会失步。
        self._ticks_by_request = {}
        for kind in _CONSUMED_KINDS:
            for record in self._streams[kind]:
                self._ticks_by_request.setdefault(record["request_id"], {})[
                    kind] = record["tick"]

    def phase_durations(self, request_id: str) -> tuple:
        """(prefill_dur_ns, decode_dur_ns):request 的 LUT 时钟相位时长。

        缺条目或非正(相位无真实内容)返回 (0, 0)——调用方据此跳过校准,
        保持 roofline 回退。时长来自日志 tick 差,绝不从 estimated_arrival
        或 interval 推算(会与流顺序失配,见模块 docstring)。
        """
        ticks = self._ticks_by_request.get(request_id)
        if ticks is None:
            return (0, 0)
        prefill_dur = ticks.get("decode", 0) - ticks.get("prefill", 0)
        decode_dur = ticks.get("completion", 0) - ticks.get("decode", 0)
        return (prefill_dur if prefill_dur > 0 else 0,
                decode_dur if decode_dur > 0 else 0)

    def register(self, kind: str, request_id: str) -> bool:
        """边界事件登记(到达条件集 + 头部推进;阶段 7 §10.8 方案 B)。

        请求进入该 kind 流的条件集合(ARRIVAL->prefill 流 arrived 集,
        PREFILL_DRAIN->decode 流 drained 集,DECODE_COMPLETION->completion
        流 completed 集);随后游标按记录序推进:头部记录所属请求已在集合
        内则越过并继续,否则停在头部等待(触发序与记录序的反转只造成
        等待,绝不失步——30s 输入 0 反转,推进路径与严格 next-match 逐
        字节一致)。

        返回 True = 新登记(该请求的首次边界事件);False = 已登记(同一
        请求的第二次 ARRIVAL——发射对齐 alarm 的合法重复,调度器据此只
        发射、不再登记与记日志)。失步判定不在此处:条件永不满足的形态
        由 consumed_all() 在运行结束兜底(fail-closed)。
        """
        condition_set = self._condition_sets[kind]
        if request_id in condition_set:
            return False
        condition_set.add(request_id)
        stream = self._streams[kind]
        cursor = self._cursors[kind]
        while (cursor < len(stream) and
               stream[cursor]["request_id"] in condition_set):
            cursor += 1
        self._cursors[kind] = cursor
        return True

    def record_for(self, kind: str, request_id: str) -> dict:
        """该 request 的决策记录(本边界记日志用,不消费)。

        在线决策日志的行序 = 边界触发序(含同 tick 组的纳秒抖动,与修复
        前逐字节一致;记录序的反转不改变行序——日志按触发序写,游标按
        记录序推进,两者解耦)。日志覆盖集保证每条记录恰好一个边界。
        """
        record = self._records_by_request[kind].get(request_id)
        if record is None:
            raise ReplayDesyncError(
                "no {} decision in the log for request {!r}".format(
                    kind, request_id))
        return record

    def peek_prefill(self, request_id: str) -> dict:
        """非消费查找某 request 的 prefill 决策(下一次 arrival 排程用)。

        该 request 的 prefill 决策必须尚未被游标越过(它将在未来的
        ARRIVAL 边界经 register 登记);已越过 = 失步(排程的 request
        不应已到达)。§7.3:位置索引 O(1) 定位。
        """
        record = self._request_to_prefill.get(request_id)
        if record is None:
            raise ReplayDesyncError(
                "no prefill decision in the log for request {!r}".format(request_id))
        if self._prefill_positions[request_id] < self._cursors["prefill"]:
            raise ReplayDesyncError(
                "request {!r} prefill decision already consumed -- its "
                "arrival was scheduled twice?".format(request_id))
        return record

    def consumed_all(self) -> bool:
        return all(
            cursor >= len(self._streams[kind])
            for kind, cursor in self._cursors.items())

    def consumed_counts(self) -> dict:
        return {
            kind: self._cursors[kind]
            for kind in _CONSUMED_KINDS
        }

    def total_counts(self) -> dict:
        return {
            kind: len(self._streams[kind])
            for kind in _CONSUMED_KINDS
        }
