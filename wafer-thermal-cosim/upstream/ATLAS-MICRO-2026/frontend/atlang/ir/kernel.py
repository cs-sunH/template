"""Atlang simulator shell kernel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .nodes import SimulatorExtractionResult, SimulatorMetadataSnapshot


# Public shell fields

_EXTRACTION_RESULT_FIELDS: tuple[str, ...] = (
    "data_placement_list",
    "task_description_list",
    "inter_chip_communication_list",
    "attn_input",
    "operator_description_config_path",
    "data_placement_config_path",
    "intra_chip_computation_performance",
    "intra_chip_computation_latency",
    "intra_chip_computation_energy",
    "inter_channel_communication_total_latency",
    "intra_channel_accumulation_total_latency",
    "inter_channel_communication_total_energy",
    "intra_channel_accumulation_total_energy",
    "intra_channel_accumulation_latency_dict",
    "intra_channel_accumulation_energy_dict",
    "inter_channel_communication_latency_dict",
    "inter_channel_communication_energy_dict",
    "softmax_latency",
    "softmax_energy",
    "inter_chip_communication_performance",
    "inter_chip_communication_latency",
    "inter_chip_communication_energy",
    "latency",
    "energy",
)

_KERNEL_METADATA_FIELDS: tuple[str, ...] = (
    "system_config",
    "operator_dict",
    "dtype",
    "intermediate_result_dir",
    "gemm_tiling_cache_dir",
)


# Kernel shell

class AtlangKernel:
    """Simulator-only kernel shell returned by the future `@A.main` path."""

    def __init__(
        self,
        *,
        name: str | None = None,
        source_function: Callable[..., Any] | None = None,
        simulator_metadata_snapshot: SimulatorMetadataSnapshot | None = None,
        **metadata: Any,
    ) -> None:
        self.name = name or getattr(source_function, "__name__", None)
        self.source_function = source_function
        self.is_simulator_shell = True
        self.simulator_metadata_snapshot: SimulatorMetadataSnapshot | None = None
        self.simulator_extraction_result: SimulatorExtractionResult | None = None
        for field_name in _KERNEL_METADATA_FIELDS:
            setattr(self, field_name, metadata.get(field_name))
        for field_name in _EXTRACTION_RESULT_FIELDS:
            setattr(self, field_name, None)
        self.config = None
        self.ref_latency = None
        if simulator_metadata_snapshot is not None:
            self.attach_simulator_metadata_snapshot(simulator_metadata_snapshot)

    @classmethod
    def simulator_shell(
        cls,
        *,
        name: str | None = None,
        source_function: Callable[..., Any] | None = None,
        simulator_metadata_snapshot: SimulatorMetadataSnapshot | None = None,
        **metadata: Any,
    ) -> "AtlangKernel":
        return cls(
            name=name,
            source_function=source_function,
            simulator_metadata_snapshot=simulator_metadata_snapshot,
            **metadata,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("AtlangKernel is a simulator shell and does not have a compiled runtime backend.")

    # Metadata and extraction handling

    def attach_simulator_metadata_snapshot(
        self,
        snapshot: SimulatorMetadataSnapshot,
    ) -> SimulatorMetadataSnapshot:
        self.simulator_metadata_snapshot = snapshot
        self.simulator_extraction_result = snapshot.extraction_result
        return snapshot

    def apply_extraction_result(
        self,
        extraction_result: SimulatorExtractionResult,
    ) -> SimulatorExtractionResult:
        self.simulator_extraction_result = extraction_result
        for field_name in _EXTRACTION_RESULT_FIELDS:
            setattr(self, field_name, getattr(extraction_result, field_name))
        return extraction_result

    def extract_simulator_outputs(
        self,
        extractor: Callable[..., SimulatorExtractionResult] | None = None,
    ) -> SimulatorExtractionResult:
        if self.simulator_metadata_snapshot is None:
            raise RuntimeError("Simulator metadata snapshot is not attached to this AtlangKernel.")
        if extractor is None:
            from ..simulator.extract import extract_simulator_outputs

            extractor = extract_simulator_outputs
        extraction_result = extractor(
            self.simulator_metadata_snapshot,
            system_config=self.system_config,
            operator_dict=self.operator_dict,
            dtype=self.dtype,
            intermediate_result_dir=self.intermediate_result_dir,
            gemm_tiling_cache_dir=self.gemm_tiling_cache_dir,
        )
        return self.apply_extraction_result(extraction_result)


__all__ = ["AtlangKernel"]
