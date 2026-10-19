"""Frame builder and binding helpers for atlang AST capture."""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable

from .actions import capture_process_call
from .source import RuntimeScope
from ..ir.expr import collect_ceildiv_operands, make_binary_expr, symbol
from ..ir.kernel import AtlangKernel
from ..ir.nodes import CoreArrayContext, KernelRegion, LoopRegion, OpAction, OpRegion, SimulatorMetadataSnapshot, TensorDecl
from ..language import CoreArray, Kernel, MPMD, ProcessStyleCall, SPMD, SerialRange, Tensor, TensorRegion


# Frame builder

class FrameBuilder:
    def __init__(self, source_function: Callable[..., Any], runtime_scope: RuntimeScope) -> None:
        self.source_function = source_function
        self.runtime_scope = runtime_scope
        self.snapshot = SimulatorMetadataSnapshot(attrs={"source_name": source_function.__name__})
        self.parameter_names = tuple(inspect.signature(source_function).parameters)
        self.active_core_context: CoreArrayContext | None = None
        self.active_op_region: OpRegion | None = None
        self.active_kernel_region: KernelRegion | None = None
        self.active_loop_region: LoopRegion | None = None
        self.condition_stack: list[Any] = []

    # ------------------------------------------------------------------
    # AST capture: create and maintain neutral IR objects
    # ------------------------------------------------------------------

    def register_parameters(self) -> None:
        for parameter_name in self.parameter_names:
            tensor_decl = TensorDecl(
                name=parameter_name,
                from_parameter=True,
                declaration_site="main_parameter",
                attrs={"declared_in_parameter": True},
            )
            self.snapshot.tensor_declarations[parameter_name] = tensor_decl
            self.runtime_scope.set(parameter_name, TensorParameter(parameter_name))

    def enter_core_array(self, core_array: CoreArray, body: Callable[[], None]) -> None:
        previous_context = self.active_core_context
        core_context = CoreArrayContext(
            core_array_shape=tuple(int(axis_extent) for axis_extent in core_array.shape),
            global_kwargs=dict(core_array.attrs),
            source_name=self.source_function.__name__,
        )
        for tensor_decl in self.snapshot.tensor_declarations.values():
            core_context.tensors.setdefault(tensor_decl.name, tensor_decl)
        self.snapshot.core_array_contexts.append(core_context)
        self.active_core_context = core_context
        try:
            body()
        finally:
            self.active_core_context = previous_context

    def enter_spmd_region(
        self,
        spmd: SPMD,
        target: ast.AST | None,
        bind_target: Callable[[ast.AST, Any], None],
        body: Callable[[], None],
    ) -> None:
        if self.active_core_context is None:
            raise RuntimeError("A.SPMD must be nested inside A.CoreArray.")
        bound_values = _derive_spmd_bindings(spmd.type, spmd.kwargs, self.active_core_context.core_array_shape)
        op_region = OpRegion(
            name=spmd.name,
            region_kind="spmd",
            op_category=spmd.type,
            region_kwargs=dict(spmd.kwargs),
            attrs={"bound_values": bound_values},
        )
        self.active_core_context.register_op_region(op_region)
        if target is not None:
            bind_target(target, bound_values)
        self._with_op_region(op_region, body)

    def enter_mpmd_region(
        self,
        mpmd: MPMD,
        target: ast.AST | None,
        bind_target: Callable[[ast.AST, Any], None],
        body: Callable[[], None],
    ) -> None:
        if self.active_core_context is None:
            raise RuntimeError("A.MPMD must be nested inside A.CoreArray.")
        core_symbol = symbol("CORE_ID")
        op_region = OpRegion(
            name=mpmd.name,
            region_kind="mpmd",
            op_category=mpmd.type,
            region_kwargs=dict(mpmd.kwargs),
            attrs={"core_identifier": "CORE_ID"},
        )
        self.active_core_context.register_op_region(op_region)
        if target is not None:
            bind_target(target, core_symbol)
        self._with_op_region(op_region, body)

    def enter_kernel_region(
        self,
        kernel: Kernel,
        target: ast.AST | None,
        bind_target: Callable[[ast.AST, Any], None],
        body: Callable[[], None],
    ) -> None:
        if self.active_op_region is None:
            raise RuntimeError("A.Kernel must be nested inside A.SPMD or A.MPMD.")
        bound_symbol_names = _target_names(target)
        core_list = self._normalize_kernel_core_list(kernel.core_list)
        kernel_region = KernelRegion(
            block_expressions=kernel.blocks,
            autotune_enabled=kernel.autotune,
            core_list=core_list,
            bound_symbols=tuple(bound_symbol_names),
            attrs={
                "is_blockless": len(kernel.blocks) == 0,
                "block_ceildivs": collect_ceildiv_operands(kernel.blocks),
                "conditions": tuple(self.condition_stack),
            },
        )
        self.active_op_region.kernel_regions.append(kernel_region)
        if target is not None:
            if len(bound_symbol_names) == 1:
                bind_target(target, symbol(bound_symbol_names[0]))
            else:
                bind_target(target, tuple(symbol(name) for name in bound_symbol_names))
        previous_kernel = self.active_kernel_region
        self.active_kernel_region = kernel_region
        try:
            body()
        finally:
            self.active_kernel_region = previous_kernel

    def _normalize_kernel_core_list(self, raw_core_list: tuple[Any, ...] | None) -> list[int] | None:
        if raw_core_list is None:
            return None
        if self.active_core_context is None:
            raise RuntimeError("A.Kernel core_list validation requires an active CoreArray.")
        core_ids: list[int] = []
        seen_core_ids: set[int] = set()
        core_num = self.active_core_context.core_num
        for value in raw_core_list:
            if isinstance(value, bool):
                raise ValueError("A.Kernel core_list entries must be integer core ids, got bool.")
            try:
                core_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"A.Kernel core_list entries must be static integer core ids, got {value!r}.") from exc
            if core_id < 0 or core_id >= core_num:
                raise ValueError(f"A.Kernel core_list core id {core_id} is outside CoreArray core range [0, {core_num}).")
            if core_id in seen_core_ids:
                raise ValueError(f"A.Kernel core_list contains duplicate core id {core_id}.")
            seen_core_ids.add(core_id)
            core_ids.append(core_id)
        return core_ids

    def enter_serial_loop(self, target_name: str, serial_range: SerialRange, body: Callable[[], None]) -> None:
        if self.active_kernel_region is None:
            raise RuntimeError("A.Serial loops must be nested inside A.Kernel.")
        loop_symbol = symbol(target_name)
        loop_region = LoopRegion(
            loop_variable=target_name,
            extent_expression=_serial_extent(serial_range),
            loop_kind="serial",
            attrs={
                "start": serial_range.start,
                "step": serial_range.step,
                "iter_var": target_name,
                "bound_expr": loop_symbol,
                "conditions": tuple(self.condition_stack),
            },
        )
        self._append_loop(loop_region)
        previous_loop = self.active_loop_region
        previous_value = self.runtime_scope.get(target_name) if self._name_exists(target_name) else _MISSING
        self.active_loop_region = loop_region
        self.runtime_scope.set(target_name, loop_symbol)
        try:
            body()
        finally:
            self.active_loop_region = previous_loop
            if previous_value is _MISSING:
                self.runtime_scope.scopes[-1].pop(target_name, None)
            else:
                self.runtime_scope.set(target_name, previous_value)

    def enter_python_range_loop(self, target_name: str, host_range: "PythonRange", body: Callable[[], None]) -> None:
        if self.active_kernel_region is None:
            raise RuntimeError("Symbolic range loops must be nested inside A.Kernel.")
        loop_symbol = symbol(target_name)
        loop_region = LoopRegion(
            loop_variable=target_name,
            extent_expression=_range_extent(host_range),
            loop_kind="python-range",
            attrs={
                "start": host_range.start,
                "step": host_range.step,
                "iter_var": target_name,
                "bound_expr": loop_symbol,
                "conditions": tuple(self.condition_stack),
            },
        )
        self._append_loop(loop_region)
        previous_loop = self.active_loop_region
        previous_value = self.runtime_scope.get(target_name) if self._name_exists(target_name) else _MISSING
        self.active_loop_region = loop_region
        self.runtime_scope.set(target_name, loop_symbol)
        try:
            body()
        finally:
            self.active_loop_region = previous_loop
            if previous_value is _MISSING:
                self.runtime_scope.scopes[-1].pop(target_name, None)
            else:
                self.runtime_scope.set(target_name, previous_value)

    def record_process_call(self, process_call: ProcessStyleCall, source_line: int | None) -> None:
        if self.active_kernel_region is None:
            return
        action = capture_process_call(
            process_call,
            source_line=source_line,
            conditions=tuple(self.condition_stack),
        )
        self._append_action(action)

    def register_tensor_assignment(self, name: str, tensor: Tensor, source_line: int | None) -> None:
        tensor_decl = self.snapshot.tensor_declarations.get(name)
        if tensor_decl is None:
            tensor_decl = TensorDecl(name=name, from_parameter=False, declaration_site="body_tensor")
            self.snapshot.tensor_declarations[name] = tensor_decl
        if tensor_decl.attrs.get("defined_in_body"):
            raise ValueError(f"Tensor '{name}' already has a body layout declaration.")
        tensor_decl.shape = tuple(tensor.shape)
        tensor_decl.strides = None if tensor.strides is None else tuple(tensor.strides)
        tensor_decl.dtype = tensor.dtype
        tensor_decl.declaration_site = "body_tensor"
        tensor_decl.attrs.update({"defined_in_body": True, "source_line": source_line})
        if self.active_core_context is not None:
            self.active_core_context.tensors[name] = tensor_decl

    def push_condition(self, condition: Any) -> None:
        self.condition_stack.append(condition)

    def pop_condition(self) -> None:
        self.condition_stack.pop()

    def _with_op_region(self, op_region: OpRegion, body: Callable[[], None]) -> None:
        previous_region = self.active_op_region
        self.active_op_region = op_region
        try:
            body()
        finally:
            self.active_op_region = previous_region

    def _append_loop(self, loop_region: LoopRegion) -> None:
        if self.active_loop_region is not None:
            index = len(self.active_loop_region.child_loops)
            self.active_loop_region.child_loops.append(loop_region)
            self.active_loop_region.body_sequence.append(("loop", index))
            return
        if self.active_kernel_region is None:
            raise RuntimeError("Loop capture requires an active kernel region.")
        index = len(self.active_kernel_region.child_loops)
        self.active_kernel_region.child_loops.append(loop_region)
        self.active_kernel_region.body_sequence.append(("loop", index))

    def _append_action(self, action: OpAction) -> None:
        if self.active_loop_region is not None:
            index = len(self.active_loop_region.actions)
            self.active_loop_region.actions.append(action)
            self.active_loop_region.body_sequence.append(("action", index))
            return
        if self.active_kernel_region is None:
            raise RuntimeError("Action capture requires an active kernel region.")
        index = len(self.active_kernel_region.actions)
        self.active_kernel_region.actions.append(action)
        self.active_kernel_region.body_sequence.append(("action", index))

    def _name_exists(self, name: str) -> bool:
        try:
            self.runtime_scope.get(name)
        except NameError:
            return False
        return True

    # ------------------------------------------------------------------
    # Kernel lowering: build the shell kernel and trigger simulator output
    # ------------------------------------------------------------------

    def to_kernel(self) -> AtlangKernel:
        self.validate_tensor_declarations()
        metadata = self._kernel_metadata()
        kernel = AtlangKernel.simulator_shell(
            name=self.source_function.__name__,
            source_function=self.source_function,
            simulator_metadata_snapshot=self.snapshot,
            **metadata,
        )
        if metadata.get("system_config") is not None:
            kernel.extract_simulator_outputs()
        return kernel

    def validate_tensor_declarations(self) -> None:
        for tensor_decl in self.snapshot.tensor_declarations.values():
            if tensor_decl.from_parameter and not tensor_decl.has_layout:
                raise ValueError(f"Tensor parameter '{tensor_decl.name}' is missing a body layout declaration.")

    def _kernel_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.snapshot.core_array_contexts:
            global_kwargs = self.snapshot.core_array_contexts[0].global_kwargs
            for field_name in ("system_config", "intermediate_result_dir", "gemm_tiling_cache_dir"):
                if field_name in global_kwargs:
                    metadata[field_name] = global_kwargs[field_name]
        for field_name in ("operator_dict", "dtype"):
            try:
                metadata[field_name] = self.runtime_scope.get(field_name)
            except NameError:
                pass
        return metadata


class TensorParameter:
    def __init__(self, name: str) -> None:
        self.name = name

    def __getitem__(self, key: Any) -> TensorRegion:
        return TensorRegion(base=self, indices=(key,) if not isinstance(key, tuple) else key)


@dataclass(frozen=True, slots=True)
class PythonRange:
    start: Any
    stop: Any
    step: Any

# Expression and binding helpers

def _compare_values(op: ast.cmpop, lhs: Any, rhs: Any) -> Any:
    if isinstance(op, ast.Lt):
        return lhs < rhs
    if isinstance(op, ast.LtE):
        return lhs <= rhs
    if isinstance(op, ast.Gt):
        return lhs > rhs
    if isinstance(op, ast.GtE):
        return lhs >= rhs
    if isinstance(op, ast.Eq):
        return lhs == rhs
    if isinstance(op, ast.NotEq):
        return lhs != rhs
    raise NotImplementedError(f"Unsupported comparison operator {type(op).__name__}.")


def _single_target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    raise NotImplementedError("Loop targets must be simple names in atlang kernels.")


def _target_names(target: ast.AST | None) -> list[str]:
    if target is None:
        return []
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [_single_target_name(element) for element in target.elts]
    raise NotImplementedError(f"Unsupported context binding target {type(target).__name__}.")


def _serial_extent(serial_range: SerialRange) -> Any:
    if isinstance(serial_range.start, int) and isinstance(serial_range.step, int) and serial_range.start == 0 and serial_range.step == 1:
        return serial_range.stop
    return math.ceil((serial_range.stop - serial_range.start) / serial_range.step)


def _make_python_range(args: list[Any]) -> PythonRange:
    if len(args) == 1:
        return PythonRange(0, args[0], 1)
    if len(args) == 2:
        return PythonRange(args[0], args[1], 1)
    if len(args) == 3:
        return PythonRange(args[0], args[1], args[2])
    raise TypeError(f"range expected 1 to 3 arguments, got {len(args)}.")


def _range_extent(host_range: PythonRange) -> Any:
    if isinstance(host_range.start, int) and isinstance(host_range.step, int) and host_range.start == 0 and host_range.step == 1:
        return host_range.stop
    return (host_range.stop - host_range.start + host_range.step - 1) // host_range.step


def _derive_spmd_bindings(
    operator_category: str,
    region_kwargs: dict[str, Any],
    core_array_shape: tuple[int, ...],
) -> tuple[Any, ...] | None:
    if operator_category == "gemm":
        return _derive_gemm_bindings(region_kwargs, core_array_shape)
    if operator_category == "decode-attention":
        return _normalize_binding_tuple(region_kwargs.get("attention_shape"), "attention_shape", expected_rank=5)
    if operator_category == "general":
        return None
    raise ValueError(f"Unsupported SPMD operator category {operator_category!r}.")


def _derive_gemm_bindings(region_kwargs: dict[str, Any], core_array_shape: tuple[int, ...]) -> tuple[Any, ...]:
    gemm_shape = _normalize_binding_tuple(region_kwargs.get("gemm_shape"), "gemm_shape", expected_rank=3)
    core_dim_mapping = _normalize_binding_tuple(
        region_kwargs.get("core_dim_mapping", (None, None, None)),
        "core_dim_mapping",
        expected_rank=3,
    )
    per_core_dimensions = []
    for base_extent, mapped_core_axes in zip(gemm_shape, core_dim_mapping):
        normalized_core_axes = _normalize_core_axis_mapping(mapped_core_axes, len(core_array_shape))
        if normalized_core_axes is None:
            per_core_dimensions.append(base_extent)
            continue
        split_factor = 1
        for core_axis in normalized_core_axes:
            split_factor *= core_array_shape[core_axis]
        if isinstance(base_extent, int):
            per_core_dimensions.append(math.ceil(base_extent / split_factor))
        else:
            per_core_dimensions.append(make_binary_expr("floordiv", base_extent + split_factor - 1, split_factor))
    return tuple(per_core_dimensions)


def _normalize_binding_tuple(value: Any, field_name: str, expected_rank: int) -> tuple[Any, ...]:
    if value is None:
        raise ValueError(f"{field_name} is required for atlang SPMD binding capture.")
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple/list, got {type(value).__name__}.")
    normalized = tuple(value)
    if len(normalized) != expected_rank:
        raise ValueError(f"{field_name} must have rank {expected_rank}, got {len(normalized)}.")
    return normalized


def _normalize_core_axis_mapping(mapped_core_axes: Any, core_array_rank: int) -> tuple[int, ...] | None:
    if mapped_core_axes is None:
        return None
    if isinstance(mapped_core_axes, int):
        normalized_core_axes = (mapped_core_axes,)
    elif isinstance(mapped_core_axes, (tuple, list)):
        normalized_core_axes = tuple(mapped_core_axes)
    else:
        raise TypeError("core_dim_mapping entries must be int, tuple/list-like, or None.")
    if not normalized_core_axes:
        raise ValueError("core_dim_mapping entries cannot be empty tuples.")
    for core_axis in normalized_core_axes:
        if not isinstance(core_axis, int):
            raise TypeError("core_dim_mapping axis values must be integers.")
        if core_axis < 0 or core_axis >= core_array_rank:
            raise ValueError(f"core_dim_mapping axis {core_axis} is out of range for CoreArray rank {core_array_rank}.")
    return normalized_core_axes


_MISSING = object()

__all__ = [
    "FrameBuilder",
    "PythonRange",
    "TensorParameter",
    "_MISSING",
    "_compare_values",
    "_make_python_range",
    "_range_extent",
    "_single_target_name",
    "_target_names",
]
