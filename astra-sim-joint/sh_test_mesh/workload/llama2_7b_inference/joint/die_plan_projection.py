#!/usr/bin/env python3
"""die_plan_projection.py -- §9.6 预测工作图的 die 计划描述与推进协议
（joint 仓二期；设计文档 §9.6 两层接口第 2/3 条）。

License: MIT (see LICENSE in the repository root).

本模块是 joint 预测层对 extern/die_model 共享成本库的**协议层**桥接：

* ``DiePlanDescriptor``：一条投影 compute leg 的 die 工作描述——若干
  DieOpSpec（契约 §2 schema）按执行序排列 + HBM 端口资源路由 + 执行
  工厂。描述符不携带任何时间/字节公式（§8.1：Python 不实现第二套
  公式）——孤立估值与计划推进全部来自共享库；
* ``DieExecutionProtocol``：A3 执行协议的最小 Python 面
  （next_action/advance_time/complete_memory/close），供
  native 投影内核驱动（NativeProjectionWorld facade 同源）；真实实现 =
  online.die_model_oracle 的 ctypes 客户端，测试替身 = ``StubExecution``；
* 名义 service 语义（§9.6 第 1 条）：``nominal_service_ns`` 只把预测
  内存服务率变为 ``B/s``（service_units ÷ (B/s)），不整段 ×s、不降速
  NoC/PE/vec——非内存资源时间（PE/VEC/NOC/launch）原样来自库的
  CostResult（total − mem）；
* 退役纪律（§9.6 第 3/4 条）：die 腿不携带 ``(pure_compute_ns,
  hbm_bytes)`` 聚合量（TaskSpec 层强制 bytes_amount=0、base_ns=0），
  不用 ``service×bandwidth`` 伪造字节；service factor 只缩放
  service_units（``scale_service_units``），物理 actual_bytes 不动。

分层纪律：本模块位于 joint/（预测内核层），不 import online/；ctypes
客户端由调度器（sh30._joint_cost_model）注入 ``execution_factory``。
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

#: A3 执行协议动作类别（include/DieCApi.h 的 JSON 形态）。
ACTION_TIMER = "timer"
ACTION_MEMORY = "memory"
ACTION_DONE = "done"

#: 披露标记：die 腿的聚合口径（不改变计价，只进 notes/审计面）。
NOTE_DIE_PLAN_LEG = "die_plan_compute_leg"
#: 披露标记：名义 service 查询走同源 DiePlan（§9.6 第 1 条）。
NOTE_DIE_NOMINAL_SERVICE = "die_nominal_service"
#: 披露标记：退役的聚合等效字节回退在 die 后端不出现（§9.6 第 3 条）。
NOTE_DIE_AGGREGATE_RETIRED = "die_aggregate_equivalent_retired"


class DiePlanError(ValueError):
    """die 计划描述/推进协议的 fail-closed 错误。"""


class DieExecutionProtocol:
    """A3 执行协议的最小驱动面（ctypes 客户端与测试替身共用）。

    语义（include/DieCApi.h / src/Execution.cc）：

    * ``next_action()`` -> ``{"kind": "timer", "ns": int}`` 或
      ``{"kind": "memory", "stage_id": int, "actual_bytes": int,
      "service_units": float}`` 或 ``{"kind": "done", "elapsed_ns": int}``；
     状态不变时幂等（同动作）；
    * ``advance_time(ns)``：内部时钟推进 ns（完成全部 end≤now 的内部
      阶段）；
    * ``complete_memory(stage_id)``：完成当前未决内存请求（调用前时钟
      须已推进到完成时刻）；
    * 一个执行同时最多一个未决内存请求。
    """

    def next_action(self) -> Dict[str, Any]:
        raise NotImplementedError

    def advance_time(self, ns: int) -> None:
        raise NotImplementedError

    def complete_memory(self, stage_id: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ScaledDieExecution(DieExecutionProtocol):
    """service factor 包装（§9.6 第 4 条）：只缩放 memory 动作的
    service_units（额外折损 = 服务时间变长），timer 时长与物理
    actual_bytes 原样透传。
    """

    def __init__(self, inner: DieExecutionProtocol,
                 service_factor: float) -> None:
        if not math.isfinite(float(service_factor)) or service_factor <= 0:
            raise DiePlanError("service factor must be finite and positive")
        self._inner = inner
        self._factor = float(service_factor)

    def next_action(self) -> Dict[str, Any]:
        action = self._inner.next_action()
        if action.get("kind") == ACTION_MEMORY:
            action = dict(action)
            action["service_units"] = (
                float(action.get("service_units", 0.0)) * self._factor)
        return action

    def advance_time(self, ns: int) -> None:
        self._inner.advance_time(ns)

    def complete_memory(self, stage_id: int) -> None:
        self._inner.complete_memory(stage_id)

    def close(self) -> None:
        self._inner.close()


class StubExecution(DieExecutionProtocol):
    """测试替身：按给定阶段序列回放协议（无共享库依赖）。

    ``phases`` = ``("timer", ns)`` / ``("memory", stage_id, actual_bytes,
    service_units)`` 序列；全排空后 Done(elapsed=Σtimer+Σservice)。内存
    完成时刻由驱动方决定（替身只登记 advance 的时钟），与真实协议一致。
    """

    def __init__(self, phases: Sequence[Tuple[Any, ...]]) -> None:
        self._phases = tuple(phases)
        self._index = 0
        self._clock_ns = 0
        self._elapsed_ns = 0
        self._pending: Optional[Dict[str, Any]] = None
        for phase in self._phases:
            if phase[0] == ACTION_TIMER:
                if not isinstance(phase[1], int) or phase[1] < 0:
                    raise DiePlanError(
                        f"stub timer phase must be (timer, ns>=0): {phase!r}")
            elif phase[0] == ACTION_MEMORY:
                if (len(phase) != 4 or not isinstance(phase[1], int)
                        or phase[1] < 0):
                    raise DiePlanError(
                        "stub memory phase must be (memory, stage_id, "
                        f"actual_bytes, service_units): {phase!r}")
            else:
                raise DiePlanError(f"unknown stub phase kind: {phase[0]!r}")

    def next_action(self) -> Dict[str, Any]:
        if self._index >= len(self._phases):
            return {"kind": ACTION_DONE, "elapsed_ns": self._elapsed_ns}
        phase = self._phases[self._index]
        if phase[0] == ACTION_TIMER:
            return {"kind": ACTION_TIMER, "ns": int(phase[1])}
        return {
            "kind": ACTION_MEMORY,
            "stage_id": int(phase[1]),
            "actual_bytes": int(phase[2]),
            "service_units": float(phase[3]),
        }

    def advance_time(self, ns: int) -> None:
        ns = int(ns)
        if ns < 0:
            raise DiePlanError("advance_time requires ns >= 0")
        self._clock_ns += ns
        phase = self._phases[self._index]
        if phase[0] == ACTION_TIMER and ns >= int(phase[1]):
            self._index += 1
            self._elapsed_ns += int(phase[1])

    def complete_memory(self, stage_id: int) -> None:
        phase = self._phases[self._index]
        if phase[0] != ACTION_MEMORY or int(phase[1]) != int(stage_id):
            raise DiePlanError(
                f"complete_memory stage {stage_id} does not match the "
                f"pending request {phase!r}")
        self._index += 1
        self._elapsed_ns = self._clock_ns


class SequencedDieExecution(DieExecutionProtocol):
    """顺序执行若干 spec 的计划（一个层段/block 的宏组序列）。

    当前 spec 的计划排空（Done）即切下一个 spec 的执行；对外协议面
    不变（timer/memory 逐个交付；全部排空后 Done(累计 elapsed)）。
    各 spec 的执行互相独立（层段内宏组串行——与执行侧聚合链的
    previous_id 线性链一致）。
    """

    def __init__(self, executions: Sequence[DieExecutionProtocol]) -> None:
        self._executions = tuple(executions)
        if not self._executions:
            raise DiePlanError(
                "sequenced die execution requires at least one execution")
        self._index = 0
        self._elapsed_total_ns = 0

    def _current(self) -> DieExecutionProtocol:
        return self._executions[self._index]

    def next_action(self) -> Dict[str, Any]:
        action = self._current().next_action()
        if action.get("kind") == ACTION_DONE:
            self._elapsed_total_ns += int(action.get("elapsed_ns", 0))
            if self._index + 1 >= len(self._executions):
                return {
                    "kind": ACTION_DONE,
                    "elapsed_ns": self._elapsed_total_ns,
                }
            self._index += 1
            return self.next_action()
        return action

    def advance_time(self, ns: int) -> None:
        self._current().advance_time(ns)

    def complete_memory(self, stage_id: int) -> None:
        self._current().complete_memory(stage_id)

    def close(self) -> None:
        for execution in self._executions:
            try:
                execution.close()
            except Exception:  # noqa: BLE001 -- 关闭期聚合,不阻断
                pass


@dataclass(frozen=True)
class DiePlanDescriptor:
    """一条投影 compute leg 的 die 工作描述（§9.6 第 2 条）。

    * ``specs``：按执行序排列的 DieOpSpec dict 序列（契约 §2 schema；
      一个层段/block 的多个宏组 = 多个 spec 顺序执行）；
    * ``route``：内存阶段的 HBM 端口资源键（native 投影内核的分享
      集合/除数裁决面）；
    * ``execution_factory``：``(descriptor) -> DieExecutionProtocol``；
      None = 未注入（测试可用 ``stub_phases`` 直构；生产由调度器注入
      ctypes 客户端工厂）；
    * ``stub_phases``：测试替身的阶段序列（仅测试；与 factory 二选一）；
    * ``label``：披露用标识（notes/审计）。

    驱动纪律：每条腿每候选各创建独立执行（"执行句柄不入缓存"，§8.5）；
    计划编译缓存由共享库内 PlanCache 承担（同 spec 复用）。
    """

    specs: Tuple[Mapping[str, Any], ...]
    route: Tuple[str, ...]
    label: str = ""
    execution_factory: Optional[
        Callable[["DiePlanDescriptor"], DieExecutionProtocol]] = None
    stub_phases: Optional[Tuple[Tuple[Any, ...], ...]] = None
    hardware_config: Optional[Mapping[str, Any]] = None
    service_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.specs and self.stub_phases is None:
            raise DiePlanError(
                "die plan descriptor requires at least one DieOpSpec "
                "(or stub phases for kernel tests)")
        for index, spec in enumerate(self.specs):
            if not isinstance(spec, Mapping):
                raise DiePlanError(
                    f"die plan spec[{index}] must be a DieOpSpec mapping")
            if spec.get("schema_version") != 1:
                raise DiePlanError(
                    f"die plan spec[{index}] schema_version must be 1")
            if not spec.get("op_kind"):
                raise DiePlanError(
                    f"die plan spec[{index}] requires op_kind")
            if not spec.get("spans"):
                raise DiePlanError(
                    f"die plan spec[{index}] requires at least one span")
        if not self.route:
            raise DiePlanError(
                "die plan descriptor requires the HBM port route")
        for resource in self.route:
            if not isinstance(resource, str) or not resource.strip():
                raise DiePlanError(
                    "die plan route entries must be non-empty resource ids")
        if len(set(self.route)) != len(self.route):
            raise DiePlanError(
                "die plan route contains duplicate resources")
        if self.execution_factory is not None and self.stub_phases is not None:
            raise DiePlanError(
                "die plan descriptor cannot carry both an execution factory "
                "and stub phases")
        if (isinstance(self.service_factor, bool) or
                not isinstance(self.service_factor, (int, float)) or
                not math.isfinite(float(self.service_factor)) or
                float(self.service_factor) <= 0):
            raise DiePlanError("die plan service_factor must be finite and positive")
        if self.hardware_config is not None and not isinstance(
                self.hardware_config, Mapping):
            raise DiePlanError("die plan hardware_config must be an object or null")

    def to_wire(self) -> Dict[str, Any]:
        """Return the complete deterministic native DiePlan input.

        Execution factories are host handles and are intentionally omitted.
        Production providers supply the resolved hardware object; null hardware
        and stub phases exist only for controlled protocol tests.
        """
        try:
            specs = json.loads(json.dumps(
                [dict(spec) for spec in self.specs], allow_nan=False))
            hardware = (None if self.hardware_config is None else
                        json.loads(json.dumps(
                            dict(self.hardware_config), allow_nan=False)))
            stub_phases = (None if self.stub_phases is None else
                           json.loads(json.dumps(
                               [list(phase) for phase in self.stub_phases],
                               allow_nan=False)))
        except (TypeError, ValueError) as exc:
            raise DiePlanError(
                "die plan contains a non-JSON or non-finite value") from exc
        return {
            "specs": specs,
            "route": list(self.route),
            "label": str(self.label),
            "hardware_config": hardware,
            "service_factor": float(self.service_factor),
            "stub_phases": stub_phases,
        }

    def create_execution(self) -> DieExecutionProtocol:
        """按描述符创建协议执行（生产 = ctypes；测试 = 替身回放）。

        多 spec 描述符返回 ``SequencedDieExecution``（层段宏组串行）。
        """
        if self.execution_factory is not None:
            if len(self.specs) == 1:
                return self.execution_factory(self)
            return SequencedDieExecution(
                self.execution_factory(
                    DiePlanDescriptor(
                        specs=(spec,), route=self.route, label=self.label,
                        stub_phases=self.stub_phases,
                        hardware_config=self.hardware_config,
                        service_factor=self.service_factor))
                for spec in self.specs)
        if self.stub_phases is not None:
            return StubExecution(self.stub_phases)
        raise DiePlanError(
            "die plan descriptor has no execution factory; the prediction "
            "host must inject the shared-library client (contract 7.1: "
            "fail-closed, no silent roofline fallback)")


# ------------------------------------------------ 名义 service 语义（§9.6-1） --

def result_non_memory_ns(result: Mapping[str, Any]) -> int:
    """CostResult 的非内存资源时间（PE+VEC+NOC+launch；ns）。

    ``hbm_share_divisor`` 只作用于内存服务率，不降速这些资源（§9.6
    第 1 条）——预测侧把它们整体作为 pure compute 侧常量。
    """
    seconds = float(result["total_seconds"]) - float(result["mem_seconds"])
    if seconds < 0:
        raise DiePlanError(
            "die CostResult mem_seconds exceeds total_seconds")
    return int(math.ceil(seconds * 1e9))


def result_service_units(result: Mapping[str, Any]) -> float:
    """CostResult 的内存服务量（matrix+vector service_units 之和）。"""
    units = (float(result["matrix_service_units"])
             + float(result["vector_service_units"]))
    if units < 0:
        raise DiePlanError("die CostResult service units must be >= 0")
    return units


def result_actual_bytes(result: Mapping[str, Any]) -> int:
    """CostResult 的实际读写字节（matrix+vector actual；物理量，不缩放）。"""
    return int(result["matrix_actual_bytes"]) + int(
        result["vector_actual_bytes"])


def nominal_service_ns(
        result: Mapping[str, Any],
        hbm_bytes_per_ns: float,
        hbm_share_divisor: int = 1) -> int:
    """名义 service：内存服务率按 ``B/s`` 推进该计划（§9.6 第 1 条）。

    = 非内存资源时间 + service_units ÷ (B/s)。**不**把总时长 ×s（NoC/
    PE/vec 不降速）、**不**回退 service×bandwidth 等效字节。
    """
    hbm_bytes_per_ns = float(hbm_bytes_per_ns)
    divisor = int(hbm_share_divisor)
    if hbm_bytes_per_ns <= 0:
        raise DiePlanError("hbm_bytes_per_ns must be positive")
    if divisor < 1:
        raise DiePlanError("hbm_share_divisor must be >= 1")
    units = result_service_units(result)
    service_ns = units / (hbm_bytes_per_ns / divisor)
    return int(math.ceil(result_non_memory_ns(result) + service_ns))


def scale_service_units(units: float, factor: float) -> float:
    """service factor 只缩放预测 service_units（§9.6 第 4 条）。

    物理字节（actual_bytes）与硬件校准系数不参与——额外折损体现为
    服务时间变长，不伪造带宽流量。
    """
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise DiePlanError("service factor must be finite and positive")
    return float(units) * factor
