from dataclasses import dataclass
from typing import Tuple, List, Optional

from ..material import Material


@dataclass
class ParentInfo:
    """
    Record the relationship between a block and one parent floorplan.

    Attributes:
        parent_name: Name of the parent floorplan.
        index: Order of this child under the parent (e.g., array/stack index).
        offset: Offset of this child inside the parent floorplan.
        alias_suffix: Suffix contributed by this parent, used to build
                      a globally unique block name.
    """

    parent_name: str
    index: int
    offset: Tuple[float, float]
    alias_suffix: str


ParentInfoList = List[ParentInfo]


class Block:
    """
    A block is a basic unit of a floorplan.
    """

    def __init__(
        self,
        name: str,
        material: Material,
        shape: Tuple[float, float],
        parent_info_list: Optional[ParentInfoList] = None,
    ) -> None:
        self.name = name
        self.material = material
        self.shape = shape
        self.parent_info_list: ParentInfoList = parent_info_list or []

    def get_shape(self) -> Tuple[float, float]:
        return self.shape

    def get_global_offset(self) -> Tuple[float, float]:
        """
        Compute the global (x, y) position of this block in the full floorplan
        by accumulating all ParentInfo offsets along the hierarchy.
        """
        x, y = 0.0, 0.0
        for info in self.parent_info_list:
            x += info.offset[0]
            y += info.offset[1]
        return (x, y)

    @property
    def offset(self) -> Tuple[float, float]:
        """
        Global offset w.r.t. the root floorplan. (Keeping this property for compatibility)
        """
        return self.get_global_offset()

    def get_full_name(self) -> str:
        """
        Build a globally unique name using all parent alias suffixes.
        """
        if not self.parent_info_list:
            return self.name
        suffix = "".join(
            info.alias_suffix for info in self.parent_info_list if info.alias_suffix
        )
        return f"{self.name}{suffix}"

    def get_hotspot_desc(self) -> str:
        """
        Dump to hotspot description format.
        """
        width, height = self.shape
        left, bottom = self.get_global_offset()
        heat_capacity = self.material.heat_capacity
        resistivity = 1 / self.material.heat_conductivity
        full_name = self.get_full_name()
        desc = (
            f"{full_name}\t{width:.6f}\t{height:.6f}\t"
            f"{left:.6f}\t{bottom:.6f}\t{heat_capacity}\t{resistivity}"
        )
        return desc
