import os
from typing import List, Dict, Any

from .floorplan import Block, visualize_floorplan, ParentInfo
from .layer import LayerConfig


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PytaConfig():
    """
    Top-level configuration for PyTA
    """

    def __init__(
        self,
        name: str,
        layer_list: List[LayerConfig],
        thermal_params: Dict[str, Any],
    ) -> None:
        self.name = name
        self.layer_list = layer_list
        self.thermal_params = thermal_params
        self.ptrace = None
    
    def dump_hotspot_config(self, rundir: str) -> None:
        """
        Dump lcf, flp and other hotspot config files
        """
        os.makedirs(rundir, exist_ok=True)
        self.dump_lcf(rundir)
        self.dump_flp(rundir)
        self.dump_thermal_params(rundir)
        self.dump_ptrace(rundir)

    def dump_flp(self, rundir: str) -> None:
        """
        Dump flp files for each layer
        """
        for layer in self.layer_list:
            parent_info_list = [
                ParentInfo(
                    parent_name=f"layer_{layer.layer_number}",
                    index=layer.layer_number,
                    offset=(0.0, 0.0),
                    alias_suffix=f"_L{layer.layer_number}",
                )
            ]
            block_list = layer.floorplan.flatten(parent_info_list=parent_info_list)
            flp_codes = "\n".join(block.get_hotspot_desc() for block in block_list) + "\n"
            with open(layer.get_flp_path(rundir), "w") as f:
                f.write(flp_codes)
            visualize_floorplan(block_list, layer.get_vis_path(rundir))

    def dump_lcf(self, rundir: str) -> None:
        """
        Dump lcf file
        """
        lcf_codes_list = [layer.to_hotspot_desc(rundir) for layer in self.layer_list]
        lcf_codes = f"""
# File Format:
#<Layer Number>
#<Lateral heat flow Y/N?>
#<Power Dissipation Y/N?>
#<Specific heat capacity in J/(m^3K)>
#<Resistivity in (m-K)/W>
#<Thickness in m>
#<floorplan file>

"""
        lcf_codes += "\n".join(lcf_codes_list)
        with open(os.path.join(rundir, f"hotspot.lcf"), "w") as f:
            f.write(lcf_codes)

    def dump_thermal_params(self, rundir: str) -> None:
        """
        Dump thermal params file
        """
        template_config_path = os.path.join(PROJECT_DIR, "thirdparty", "hotspot", "template.config")
        assert os.path.exists(template_config_path), "Template config file not found"

        new_config_path = os.path.join(rundir, f"hotspot.config")
        
        with open(template_config_path, "r") as fin:
            with open(new_config_path, "w") as fout:
                for line in fin:
                    if line.strip().startswith("-"):
                        key = line.split("-")[1].split()[0]
                        if key in self.thermal_params:
                            value = self.thermal_params[key]
                            fout.write(f"        -{key} {value}\n")
                        else:
                            fout.write(line)
                    else:
                        fout.write(line)

    def get_block_list(self, skip_powerless_block: bool = True) -> List[Block]:
        """
        Get the list of flattened blocks. Can also choose those with power dissipation.
        """
        block_list: List[Block] = []
        for layer in self.layer_list:
            if not layer.power_dissipation and skip_powerless_block:
                continue
            parent_info_list = [
                ParentInfo(
                    parent_name=f"layer_{layer.layer_number}",
                    index=layer.layer_number,
                    offset=(0.0, 0.0),
                    alias_suffix=f"_L{layer.layer_number}",
                )
            ]
            block_list.extend(
                layer.floorplan.flatten(parent_info_list=parent_info_list)
            )
        return block_list

    def set_ptrace(self, ptrace: Dict[str, List[float]]) -> None:
        """
        Set the power trace
        """
        # check if all the powerful blocks are included in the ptrace
        power_block_list = self.get_block_list(skip_powerless_block=True)
        block_names = [block.get_full_name() for block in power_block_list]
        assert set(block_names) == set(
            ptrace.keys()
        ), "All powerful blocks must be included in the ptrace"

        # check if all the length of the ptrace are the same
        assert len(
            set(len(ptrace[name]) for name in block_names)
        ) == 1, "All powerful blocks must have the same length of ptrace"

        self.ptrace = ptrace

    def dump_ptrace(self, rundir: str) -> None:
        """
        Dump ptrace file
        """
        if self.ptrace is None:
            return

        key_list = list(self.ptrace.keys())
        total_time_steps = len(self.ptrace[key_list[0]])

        with open(os.path.join(rundir, f"hotspot.ptrace"), "w") as f:
            f.write("\t".join(key_list) + "\n")
            for time_step in range(total_time_steps):
                f.write("\t".join(str(self.ptrace[block][time_step]) for block in key_list) + "\n")
