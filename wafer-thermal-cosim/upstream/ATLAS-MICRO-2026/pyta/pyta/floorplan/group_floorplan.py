from copy import deepcopy
from typing import Tuple, List, Optional

from .base_floorplan import Floorplan
from .block import Block, ParentInfo, ParentInfoList
from .utils import is_block_list_overlap_with_offsets


class GroupFloorplan(Floorplan):
    """
    Grouping multiple blocks manually by specifying explicit offsets for each block.
    (Grouping sub-floorplans is not supported yet)
    """

    def __init__(
        self,
        name: str,
        block_list: List[Block],
        offset_list: List[Tuple[float, float]],
    ):
        """
        Args:
            name: Name of this floorplan.
            block_list: List of child blocks (without any ParentInfo attached).
            offset_list: List of (x, y) offsets for each block inside this floorplan.
        """
        super().__init__(name)
        if len(block_list) != len(offset_list):
            raise ValueError(
                "block_list and offset_list must have the same length, "
                f"got {len(block_list)} blocks and {len(offset_list)} offsets"
            )
        self.block_list = block_list
        self.offset_list = offset_list
        assert self._validate(), "Invalid group floorplan"

    def _validate(self, eps: float = 1e-6) -> bool:
        """
        Check if the blocks overlap and whether they exactly cover the parent floorplan
        (up to a small numerical tolerance on area).
        """
        # check if the blocks have parent info
        for block in self.block_list:
            assert len(block.parent_info_list) == 0, "Block must not have parent info"

        # check for pairwise overlap using the provided offsets
        if is_block_list_overlap_with_offsets(self.block_list, self.offset_list):
            raise ValueError("Blocks overlap")

        # check if they exactly cover the parent floorplan (area-wise)
        parent_shape = self.get_shape()
        total_area = sum(block.shape[0] * block.shape[1] for block in self.block_list)
        if abs(total_area - parent_shape[0] * parent_shape[1]) > eps:
            raise ValueError("Blocks do not cover the parent floorplan")

        return True

    def get_shape(self) -> Tuple[float, float]:
        """
        Compute the bounding box of all blocks using their local offsets.
        """
        if not self.block_list:
            return (0.0, 0.0)

        min_x = min(offset[0] for offset in self.offset_list)
        min_y = min(offset[1] for offset in self.offset_list)
        max_x = max(
            offset[0] + block.shape[0]
            for block, offset in zip(self.block_list, self.offset_list)
        )
        max_y = max(
            offset[1] + block.shape[1]
            for block, offset in zip(self.block_list, self.offset_list)
        )
        return (max_x - min_x, max_y - min_y)

    def flatten(
        self,
        parent_info_list: Optional[ParentInfoList] = None,
    ) -> List[Block]:
        """
        Flatten to blocks and attach full ParentInfoList for each block.

        Each child block receives a single ParentInfo entry from this GroupFloorplan,
        with offset taken from the corresponding entry in offset_list.
        """
        base_parent_infos: ParentInfoList = list(parent_info_list or [])

        new_block_list: List[Block] = []
        for idx, (block, offset) in enumerate(zip(self.block_list, self.offset_list)):
            new_block = deepcopy(block)
            block_parent_info = ParentInfo(
                parent_name=self.name,
                index=idx,
                offset=offset,
                alias_suffix="",
            )
            new_block.parent_info_list = base_parent_infos + [block_parent_info]
            new_block_list.append(new_block)
        return new_block_list
