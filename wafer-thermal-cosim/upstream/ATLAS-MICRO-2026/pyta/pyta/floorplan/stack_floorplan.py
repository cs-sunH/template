from typing import Tuple, List, Optional

from .base_floorplan import Floorplan
from .block import Block, ParentInfo, ParentInfoList


class StackFloorplan(Floorplan):
    """
    A stack floorplan is a stack of child floorplans.
    """

    def __init__(
        self,
        name: str,
        child_floorplan_list: List[Floorplan],
        direction: str,
    ):
        super().__init__(name)
        self.child_floorplan_list = child_floorplan_list
        self.direction = direction
        assert direction in ["X", "Y"], "Direction must be either 'X' or 'Y'"
        if not self._validate():
            raise ValueError(
                "Invalid stack floorplan, check if the width/height of all children are the same"
            )

    def _validate(self) -> bool:
        """
        Check if the width/height of all children are the same.
        """
        if self.direction == "X":
            # height must be the same
            child_shapes = [
                child.get_shape()[1] for child in self.child_floorplan_list
            ]
        else:
            # width must be the same
            child_shapes = [
                child.get_shape()[0] for child in self.child_floorplan_list
            ]
        return len(set(child_shapes)) == 1

    def get_shape(self) -> Tuple[float, float]:
        if self.direction == "X":
            total_width = sum(
                child.get_shape()[0] for child in self.child_floorplan_list
            )
            total_height = max(
                child.get_shape()[1] for child in self.child_floorplan_list
            )
        else:
            total_width = max(
                child.get_shape()[0] for child in self.child_floorplan_list
            )
            total_height = sum(
                child.get_shape()[1] for child in self.child_floorplan_list
            )
        return (total_width, total_height)

    def flatten(
        self,
        parent_info_list: Optional[ParentInfoList] = None,
    ) -> List[Block]:
        block_list: List[Block] = []
        base_parent_infos: ParentInfoList = list(parent_info_list or [])
        accum_offset = 0.0

        for i, child in enumerate(self.child_floorplan_list):
            child_shape = child.get_shape()
            # calculate offsets within this StackFloorplan
            if self.direction == "X":
                offset_x = accum_offset
                offset_y = 0.0
                accum_offset += child_shape[0]
            else:
                offset_x = 0.0
                offset_y = accum_offset
                accum_offset += child_shape[1]

            local_child_offset = (offset_x, offset_y)

            child_parent_info = ParentInfo(
                parent_name=self.name,
                index=i,
                offset=local_child_offset,
                alias_suffix=f"_S{self.direction}{i}",
            )
            child_infos = base_parent_infos + [child_parent_info]
            child_block_list = child.flatten(parent_info_list=child_infos)
            block_list.extend(child_block_list)

        return block_list
