"""Neutral simulator-facing IR containers for the atlang frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# Type aliases

RegionKind = Literal["spmd", "mpmd"]
OpCategory = Literal["gemm", "communication", "decode-attention", "general"]
ActionKind = Literal[
    "matrix_compute",
    "vector_compute",
    "buffer_load",
    "buffer_store",
    "dram_access",
    "noc_send",
    "noc_recv",
    "tensor_copy",
    "general_call",
]
LoopKind = Literal["serial", "python-range", "general"]
TensorAccessMode = Literal["read", "write", "readwrite"]


# Tensor IR

@dataclass(slots=True)
class TensorDecl:
    """Unique declaration record for one tensor symbol in the simulator view."""

    name: str
    from_parameter: bool
    shape: tuple[Any, ...] | None = None
    strides: tuple[Any, ...] | None = None
    dtype: Any | None = None
    allow_cross_region_reuse: bool = False
    declaration_site: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shape is not None:
            self.shape = tuple(self.shape)
        if self.strides is not None:
            self.strides = tuple(self.strides)
        if self.shape is not None and self.strides is not None and len(self.shape) != len(self.strides):
            raise ValueError("TensorDecl shape and strides must have the same rank.")

    @property
    def has_layout(self) -> bool:
        return self.shape is not None and self.strides is not None


@dataclass(slots=True)
class TensorAccess:
    """Normalized tensor or buffer access record shared by later extraction."""

    tensor_name: str
    access_mode: TensorAccessMode
    index_expressions: tuple[Any, ...] = ()
    slice_shape: tuple[Any, ...] | None = None
    slice_strides: tuple[Any, ...] | None = None
    byte_count: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index_expressions = tuple(self.index_expressions)
        if self.slice_shape is not None:
            self.slice_shape = tuple(self.slice_shape)
        if self.slice_strides is not None:
            self.slice_strides = tuple(self.slice_strides)
        if self.slice_shape is not None and self.slice_strides is not None and len(self.slice_shape) != len(self.slice_strides):
            raise ValueError("TensorAccess slice_shape and slice_strides must have the same rank.")

    @property
    def has_slice_layout(self) -> bool:
        return self.slice_shape is not None and self.slice_strides is not None


# Region IR

@dataclass(slots=True)
class OpAction:
    """One simulator-relevant action inside a kernel or loop region."""

    action_kind: ActionKind
    input_accesses: list[TensorAccess] = field(default_factory=list)
    output_accesses: list[TensorAccess] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoopRegion:
    """Loop container used to recover simulator-visible iteration structure."""

    loop_variable: str | None = None
    extent_expression: Any | None = None
    loop_kind: LoopKind = "serial"
    child_loops: list["LoopRegion"] = field(default_factory=list)
    actions: list[OpAction] = field(default_factory=list)
    body_sequence: list[tuple[str, int]] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KernelRegion:
    """One `with A.Kernel(...)` execution region in the atlang IR."""

    block_expressions: tuple[Any, ...] = ()
    autotune_enabled: bool | None = None
    core_list: list[int] | None = None
    bound_symbols: tuple[str, ...] = ()
    child_loops: list[LoopRegion] = field(default_factory=list)
    actions: list[OpAction] = field(default_factory=list)
    body_sequence: list[tuple[str, int]] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.block_expressions = tuple(self.block_expressions)
        self.bound_symbols = tuple(self.bound_symbols)
        if self.core_list is not None:
            self.core_list = list(self.core_list)


@dataclass(slots=True)
class OpRegion:
    """Uniform operator-region container for SPMD, MPMD, and general regions."""

    name: str
    region_kind: RegionKind
    op_category: OpCategory
    region_kwargs: dict[str, Any] = field(default_factory=dict)
    kernel_regions: list[KernelRegion] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CoreArrayContext:
    """Top-level simulator context shared by all later extraction steps."""

    core_array_shape: tuple[int, ...]
    global_kwargs: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorDecl] = field(default_factory=dict)
    op_regions: list[OpRegion] = field(default_factory=list)
    source_name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.core_array_shape = tuple(self.core_array_shape)
        if not self.core_array_shape:
            raise ValueError("CoreArrayContext requires a non-empty core_array_shape.")
        if any(axis_extent <= 0 for axis_extent in self.core_array_shape):
            raise ValueError("CoreArrayContext shape extents must be positive integers.")

    @property
    def core_num(self) -> int:
        core_count = 1
        for axis_extent in self.core_array_shape:
            core_count *= int(axis_extent)
        return core_count

    def register_tensor(self, tensor_decl: TensorDecl) -> None:
        if tensor_decl.name in self.tensors:
            raise ValueError(f"Tensor '{tensor_decl.name}' is already registered.")
        self.tensors[tensor_decl.name] = tensor_decl

    def register_op_region(self, op_region: OpRegion) -> None:
        self.op_regions.append(op_region)


# Snapshot and extraction result

@dataclass(slots=True)
class SimulatorExtractionResult:
    """Container for simulator-facing outputs produced from an atlang snapshot."""

    data_placement_list: list[Any] = field(default_factory=list)
    task_description_list: list[Any] = field(default_factory=list)
    inter_chip_communication_list: list[Any] = field(default_factory=list)
    attn_input: Any | None = None
    operator_description_config_path: str | None = None
    data_placement_config_path: str | None = None
    intra_chip_computation_performance: Any | None = None
    intra_chip_computation_latency: Any | None = None
    intra_chip_computation_energy: Any | None = None
    inter_channel_communication_total_latency: Any | None = None
    intra_channel_accumulation_total_latency: Any | None = None
    inter_channel_communication_total_energy: Any | None = None
    intra_channel_accumulation_total_energy: Any | None = None
    intra_channel_accumulation_latency_dict: dict[str, Any] = field(default_factory=dict)
    intra_channel_accumulation_energy_dict: dict[str, Any] = field(default_factory=dict)
    inter_channel_communication_latency_dict: dict[str, Any] = field(default_factory=dict)
    inter_channel_communication_energy_dict: dict[str, Any] = field(default_factory=dict)
    softmax_latency: Any | None = None
    softmax_energy: Any | None = None
    inter_chip_communication_performance: Any | None = None
    inter_chip_communication_latency: Any | None = None
    inter_chip_communication_energy: Any | None = None
    latency: Any | None = None
    energy: Any | None = None


@dataclass(slots=True)
class SimulatorMetadataSnapshot:
    """Stable frontend snapshot attached to an `AtlangKernel` before extraction."""

    core_array_contexts: list[CoreArrayContext] = field(default_factory=list)
    tensor_declarations: dict[str, TensorDecl] = field(default_factory=dict)
    extraction_result: SimulatorExtractionResult = field(default_factory=SimulatorExtractionResult)
    attrs: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ActionKind",
    "CoreArrayContext",
    "KernelRegion",
    "LoopKind",
    "LoopRegion",
    "OpAction",
    "OpCategory",
    "OpRegion",
    "RegionKind",
    "SimulatorExtractionResult",
    "SimulatorMetadataSnapshot",
    "TensorAccess",
    "TensorAccessMode",
    "TensorDecl",
]
