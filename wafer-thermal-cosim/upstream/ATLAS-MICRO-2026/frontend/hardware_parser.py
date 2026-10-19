import yaml
import sys
from dataclasses import dataclass, field
from typing import Dict, Any

from frontend.model_parser import ParallelConfig

from atlasim import (
    ControllerConfig, MatrixConfig, VectorConfig,
    BufferConfig, DRAMConfig, NoCConfig, ChipConfig
)


def load_controller_config_from_yaml(config: Dict[str, Any]) -> ControllerConfig:
    try:
        return ControllerConfig(
            power=float(config.get("power", 0)),
            area=float(config.get("area", 0)),
        )
    except Exception as e:
        print(f"[Error] Controller config is invalid: {e}", file=sys.stderr)
        sys.exit(-1)


def load_matrix_config_from_yaml(config: Dict[str, Any]) -> MatrixConfig:
    try:
        return MatrixConfig(
            mac_num=int(config["mac_num"]),
            power=float(config.get("power", 0)),
            area=float(config.get("area", 0)),
        )
    except Exception as e:
        print(f"[Error] Matrix config is invalid: {e}", file=sys.stderr)
        sys.exit(-1)


def load_vector_config_from_yaml(config: Dict[str, Any]) -> VectorConfig:
    try:
        return VectorConfig(
            vec_num=int(config["vec_num"]),
            power=float(config.get("power", 0)),
            area=float(config.get("area", 0)),
        )
    except Exception as e:
        print(f"[Error] Vector config is invalid: {e}", file=sys.stderr)
        sys.exit(-1)


def load_buffer_config_from_yaml(config: Dict[str, Any]) -> BufferConfig:
    try:
        return BufferConfig(
            buffer_size=int(config["buffer_size"]),
            read_bw=float(config["read_bw"]),
            write_bw=float(config["write_bw"]),
            power=float(config.get("power", 0)),
            area=float(config.get("area", 0)),
        )
    except Exception as e:
        print(f"[Error] Buffer config is invalid: {e}", file=sys.stderr)
        sys.exit(-1)


def load_dram_config_from_yaml(config: Dict[str, Any]) -> DRAMConfig:
    try:
        return DRAMConfig(
            config_path=str(config["config_path"]),
            power=float(config.get("power", 0)),
            area=float(config.get("area", 0)),
        )
    except Exception as e:
        print(f"[Error] DRAM config is invalid: {e}", file=sys.stderr)
        sys.exit(-1)


def load_noc_config_from_yaml(config: Dict[str, Any]) -> NoCConfig:
    try:
        return NoCConfig(
            topology=str(config["topology"]),
            config_path=str(config["config_path"]),
            flit_size=int(config["flit_size"]),
            power=float(config.get("power", 0)),
            area=float(config.get("area", 0)),
        )
    except Exception as e:
        print(f"[Error] NoC config is invalid: {e}", file=sys.stderr)
        sys.exit(-1)


def load_chip_config_from_yaml(config: Dict[str, Any]) -> ChipConfig:
    try:
        arch_config = config["architecture"]
        frequency = float(arch_config["frequency"])
        core_num = int(arch_config["core_num"])

        controller_config = load_controller_config_from_yaml(arch_config["core"]["controller"])
        matrix_config = load_matrix_config_from_yaml(arch_config["core"]["matrix"])
        vector_config = load_vector_config_from_yaml(arch_config["core"]["vector"])
        buffer_config = load_buffer_config_from_yaml(arch_config["core"]["buffer"])
        dram_config = load_dram_config_from_yaml(arch_config["dram"])
        has_noc = "noc" in arch_config
        noc_config = load_noc_config_from_yaml(arch_config["noc"]) if has_noc else NoCConfig()

        return ChipConfig(
            frequency=frequency,
            core_num=core_num,
            controller_config=controller_config,
            matrix_config=matrix_config,
            vector_config=vector_config,
            buffer_config=buffer_config,
            dram_config=dram_config,
            has_noc=has_noc,
            noc_config=noc_config
        )
    except Exception as e:
        print(f"[Error] Arch config is invalid: {e}", file=sys.stderr)
        sys.exit(-1)


def extract_noc_topology(noc_config_path: str) -> str:
    topology = ""
    k = -1
    n = -1
    with open(noc_config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("topology") and "=" in line:
                topology = line.split("=")[1].split(";")[0].strip()
            elif line.startswith("k") and "=" in line:
                k = int(line.split("=")[1].split(";")[0].strip())
            elif line.startswith("n") and "=" in line:
                n = int(line.split("=")[1].split(";")[0].strip())
    return (topology, k, n)


@dataclass
class InterconnectConfig:
    scale_up_latency: float = 0.0  # s
    scale_up_bandwidth: float = 0.0  # GB/s
    scale_up_energy: float = 0.0  # pJ/bit

    scale_out_latency: float = 0.0  # s
    scale_out_bandwidth: float = 0.0  # GB/s
    scale_out_energy: float = 0.0  # pJ/bit

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "InterconnectConfig":
        obj = cls()

        try:
            obj.scale_up_latency = float(config["scale_up_latency"])
            obj.scale_up_bandwidth = float(config["scale_up_bandwidth"])
            obj.scale_up_energy = float(config["scale_up_energy"])
            
            obj.scale_out_latency = float(config["scale_out_latency"])
            obj.scale_out_bandwidth = float(config["scale_out_bandwidth"])
            obj.scale_out_energy = float(config["scale_out_energy"])
        except Exception as e:
            print(f"[Error] Interconnect config is invalid: {e}", file=sys.stderr)
            sys.exit(-1)

        return obj


@dataclass
class CloudSystemConfig:
    chip_config_path: str = ""
    chip_config: ChipConfig = field(default_factory=ChipConfig)

    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    tp_scope: str = ""
    ep_scope: str = ""

    interconnect_config: InterconnectConfig = field(default_factory=InterconnectConfig)

    max_context_length: int = 0 # token count
    block_size: int = 0 # token count

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "CloudSystemConfig":
        obj = cls()

        try:
            system_config = config["system"]
            obj.chip_config_path = str(system_config["chip"]["config_path"])
            with open(obj.chip_config_path, "r") as f:
                chip_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
            obj.chip_config = load_chip_config_from_yaml(chip_config_yaml)
            obj.parallel_config = ParallelConfig.from_yaml(system_config["parallelism"])
            obj.tp_scope = str(system_config["parallelism"]["tp_scope"])
            obj.ep_scope = str(system_config["parallelism"]["ep_scope"])
            obj.interconnect_config = InterconnectConfig.from_yaml(system_config["interconnect"])
            obj.max_context_length = int(system_config["kv_cache"]["max_context_length"])
            obj.block_size = int(system_config["kv_cache"]["block_size"])
        except Exception as e:
            print(f"[Error] CloudSystem config is invalid: {e}", file=sys.stderr)
            sys.exit(-1)

        return obj


@dataclass
class EdgeSystemConfig:
    chip_config_path: str = ""
    chip_config: ChipConfig = field(default_factory=ChipConfig)
    intra_channel_vector_length: int = 0 # vector count
    channel_num: int = 0
    channel_bandwidth: float = 0 # GB/s
    channel_energy: float = 0 # pJ/bit

    npu_frequency: float = 0 # MHz
    npu_vec_length: int = 0
    npu_vec_energy: float = 0 # pJ/op

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> "EdgeSystemConfig":
        obj = cls()
        try:
            system_config = config["system"]
            obj.chip_config_path = str(system_config["chip"]["config_path"])
            with open(obj.chip_config_path, "r") as f:
                chip_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
            obj.chip_config = load_chip_config_from_yaml(chip_config_yaml)
            obj.intra_channel_vector_length = int(system_config["chip"]["intra_channel_vector_length"])
            obj.channel_num = int(system_config["chip"]["channel_num"])
            obj.channel_bandwidth = float(system_config["chip"]["channel_bandwidth"])
            obj.channel_energy = float(system_config["chip"].get("channel_energy", 0))

            obj.npu_frequency = float(system_config["npu"]["frequency"])
            obj.npu_vec_length = int(system_config["npu"]["vec_length"])
            obj.npu_vec_energy = float(system_config["npu"].get("vec_energy", 0))
        except Exception as e:
            print(f"[Error] EdgeSystem config is invalid: {e}", file=sys.stderr)
            sys.exit(-1)
        return obj


if __name__ == "__main__":
    with open("configs/architecture/chip/test_chip_16ch.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    chip_config = load_chip_config_from_yaml(config)
    print(chip_config)

    with open("configs/architecture/system/test_cloud_system.yaml", "r") as f:
        system_config = yaml.load(f, Loader=yaml.FullLoader)
    cloud_system_config = CloudSystemConfig.from_yaml(system_config)
    print(cloud_system_config)
