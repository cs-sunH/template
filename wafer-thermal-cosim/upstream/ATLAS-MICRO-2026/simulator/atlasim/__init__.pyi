from typing import List


# --- Enums ---
class SimulatorOperatorType:
    GEMM: 'SimulatorOperatorType'
    DecodeAttention: 'SimulatorOperatorType'
    Communication: 'SimulatorOperatorType'
    SPMDGeneral: 'SimulatorOperatorType'
    MPMDGeneral: 'SimulatorOperatorType'

    def __init__(self, value: int) -> None: ...
    @property
    def value(self) -> int: ...
    @property
    def name(self) -> str: ...
    def __str__(self) -> str: ...
    @staticmethod
    def from_str(str: str) -> 'SimulatorOperatorType': ...


# --- Classes ---
class ControllerConfig:
    power: float
    area: float

    def __init__(self, **kwargs) -> None: ...


class MatrixConfig:
    mac_num: int
    power: float
    area: float

    def __init__(self, **kwargs) -> None: ...


class VectorConfig:
    vec_num: int
    power: float
    area: float

    def __init__(self, **kwargs) -> None: ...


class BufferConfig:
    buffer_size: int
    read_bw: float
    write_bw: float
    power: float
    area: float

    def __init__(self, **kwargs) -> None: ...


class DRAMConfig:
    config_path: str
    power: float
    area: float

    def __init__(self, **kwargs) -> None: ...


class NoCConfig:
    topology: str
    config_path: str
    flit_size: int
    power: float
    area: float

    def __init__(self, **kwargs) -> None: ...


class ChipConfig:
    frequency: float
    core_num: int
    controller_config: ControllerConfig
    matrix_config: MatrixConfig
    vector_config: VectorConfig
    buffer_config: BufferConfig
    dram_config: DRAMConfig
    has_noc: bool
    noc_config: NoCConfig

    def __init__(self, **kwargs) -> None: ...
    def display_stats(self, indent: int = 0) -> None: ...


class AttnInput:
    q_head_num: int
    kv_head_num: int
    q_head_num_per_kv: int
    qk_head_dim: int
    v_head_dim: int
    input_tensor_name: str
    output_tensor_name: str
    kv_cache_tensor_name: str
    total_slot_num: int
    kv_head_addr_offset: int

    def __init__(self, **kwargs) -> None: ...
    def display_stats(self, indent: int = 0) -> None: ...


class DAttnCoreInput:
    core_id: int
    request_indices: List[int]
    request_tile_count: List[int]
    # request_tile_info_list type is complex (vector of vector of vector), simplifying as list
    request_tile_info_list: List[List[List[int]]]

    def __init__(self, **kwargs) -> None: ...
    def display_stats(self, indent: int = 0) -> None: ...


class DAttnInput(AttnInput):
    token_tile_size: int
    block_size: int
    total_block_num: int
    core_input_list: List[DAttnCoreInput]
    def __init__(self, **kwargs) -> None: ...
    def display_stats(self, indent: int = 0) -> None: ...


class Stats:
    e2e_cycles: int
    matrix_cycles: int
    vector_cycles: int
    buffer_cycles: int
    dram_cycles: int
    noc_cycles: int
    e2e_energy: float
    controller_energy: float
    matrix_energy: float
    vector_energy: float
    buffer_energy: float
    dram_energy: float
    noc_energy: float
    flop_count: float
    memory_access_bytes: float
    compute_non_overlap_cycles: int
    matrix_bubble_on_chip_cycles: int
    matrix_bubble_dram_cycles: int

    def __init__(self, **kwargs) -> None: ...
    def matrix_util(self) -> float: ...
    def vector_util(self) -> float: ...
    def buffer_util(self) -> float: ...
    def dram_util(self) -> float: ...
    def noc_util(self) -> float: ...
    def dram_bw(self, frequency: float = 1000) -> float: ...
    def dram_active_bw(self, frequency: float = 1000) -> float: ...
    def arithmetic_intensity(self) -> float: ...
    def compute_non_overlap_ratio(self) -> float: ...
    def update_e2e_energy(self) -> None: ...
    def display_stats(self, max_bw: float = 0, frequency: float = 1000, indent: int = 0) -> None: ...


class Performance:
    core_bw: float
    chip_bw: float
    chip_frequency: float
    e2e_stats: Stats
    core_stats: List[Stats]
    operator_stats: List[tuple] # vector<pair<string, Stats>>
    operator_stats_per_core: List[tuple] # vector<pair<string, vector<Stats>>>

    def __init__(self) -> None: ...
    def display_stats(self, indent: int = 0) -> None: ...


class Chip:
    def __init__(self, arch_config_path: str, operator_list_path: str, placement_map_path: str, log_file_path: str = "") -> None: ...
    def get_arch_config(self) -> ChipConfig: ...
    def reset_execution_status(self, operator_list_path: str = "") -> None: ...
    def simulate(self) -> Performance: ...
    def get_max_dram_bw_per_core(self) -> float: ...
    def get_max_compute_capacity_per_core(self) -> float: ...
    def get_comp_bw_ratio(self) -> float: ...
    def get_max_dram_bw(self) -> float: ...
