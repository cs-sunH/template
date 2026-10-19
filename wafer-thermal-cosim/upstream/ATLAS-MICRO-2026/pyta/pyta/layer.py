import os

from .material import Material
from .floorplan import Floorplan


class LayerConfig():
    """
    Layer configuration for 3D thermal simulation
    """

    def __init__(
        self, 
        layer_number: int,
        thickness: float,
        default_material: Material,
        floorplan: Floorplan,
        lateral_heat_flow: bool, 
        power_dissipation: bool, 
    ) -> None:
        self.layer_number = layer_number
        self.thickness = thickness
        self.default_material = default_material
        self.floorplan = floorplan
        self.lateral_heat_flow = lateral_heat_flow
        self.power_dissipation = power_dissipation

    def to_hotspot_desc(self, rundir: str) -> str:
        """
        Dump to hotspot lcf format
        """
        desc = ""
        desc += f"{self.layer_number}\n"
        desc += f"{'Y' if self.lateral_heat_flow else 'N'}\n"
        desc += f"{'Y' if self.power_dissipation else 'N'}\n"
        desc += f"{self.default_material.heat_capacity}\n"
        desc += f"{1 / self.default_material.heat_conductivity}\n"
        desc += f"{self.thickness}\n"
        desc += f"{self.get_flp_path(rundir)}\n"
        return desc

    def get_flp_path(self, rundir) -> str:
        return os.path.join(rundir, f"L{self.layer_number}_{self.floorplan.name}.flp")

    def get_vis_path(self, rundir) -> str:
        return os.path.join(rundir, f"L{self.layer_number}_{self.floorplan.name}.png")
