from typing import Tuple, List, Optional

from .base_floorplan import Floorplan
from .block import Block, ParentInfo, ParentInfoList


class ArrayFloorplan(Floorplan):
    """
    A one-dimensional array of child floorplans along a given axis (X or Y).
    """

    def __init__(
        self,
        name: str,
        child_floorplan: Floorplan,
        count: int,
        axis: str = "X",
    ):
        super().__init__(name)
        self.child_floorplan = child_floorplan
        self.count = count
        if axis not in ("X", "Y"):
            raise ValueError("axis must be 'X' or 'Y', got %r" % (axis,))
        self.axis = axis

    def get_shape(self) -> Tuple[float, float]:
        child_width, child_height = self.child_floorplan.get_shape()
        if self.axis == "X":
            return (self.count * child_width, child_height)
        else:
            return (child_width, self.count * child_height)

    def flatten(
        self,
        parent_info_list: Optional[ParentInfoList] = None,
    ) -> List[Block]:
        block_list: List[Block] = []
        base_parent_infos: ParentInfoList = list(parent_info_list or [])
        child_width, child_height = self.child_floorplan.get_shape()

        for i in range(self.count):
            if self.axis == "X":
                local_child_offset = (i * child_width, 0.0)
                alias_suffix = f"_AX{i}"
            else:
                local_child_offset = (0.0, i * child_height)
                alias_suffix = f"_AY{i}"
            child_parent_info = ParentInfo(
                parent_name=self.name,
                index=i,
                offset=local_child_offset,
                alias_suffix=alias_suffix,
            )
            child_infos = base_parent_infos + [child_parent_info]
            child_block_list = self.child_floorplan.flatten(
                parent_info_list=child_infos
            )
            block_list.extend(child_block_list)

        return block_list
