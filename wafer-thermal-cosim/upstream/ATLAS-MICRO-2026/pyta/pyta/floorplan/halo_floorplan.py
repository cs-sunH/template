from typing import Tuple, List, Optional

from ..material import Material
from .base_floorplan import Floorplan
from .block import Block, ParentInfo, ParentInfoList


class HaloFloorplan(Floorplan):
    """
    A halo floorplan creates a halo around the child floorplan.
    """

    def __init__(
        self,
        name: str,
        child_floorplan: Floorplan,
        shape: Tuple[float, float],
        halo_material: Material,
    ):
        super().__init__(name)
        self.child_floorplan = child_floorplan
        self.shape = shape
        self.halo_material = halo_material
        assert (
            shape[0] >= child_floorplan.get_shape()[0]
        ), f"Halo width must be greater than or equal to the child floorplan width, {shape[0]} < {child_floorplan.get_shape()[0]}"
        assert (
            shape[1] >= child_floorplan.get_shape()[1]
        ), f"Halo height must be greater than or equal to the child floorplan height, {shape[1]} < {child_floorplan.get_shape()[1]}"

    def get_shape(self) -> Tuple[float, float]:
        return self.shape

    def flatten(
        self,
        parent_info_list: Optional[ParentInfoList] = None,
    ) -> List[Block]:
        block_list: List[Block] = []
        base_parent_infos: ParentInfoList = list(parent_info_list or [])

        # add the halo blocks
        child_shape = self.child_floorplan.get_shape()
        wl = (self.shape[0] + child_shape[0]) / 2  # width of the longer halo
        ws = (self.shape[0] - child_shape[0]) / 2  # width of the shorter halo
        hl = (self.shape[1] + child_shape[1]) / 2  # height of the longer halo
        hs = (self.shape[1] - child_shape[1]) / 2  # height of the shorter halo

        # central child blocks
        central_parent_info = ParentInfo(
            parent_name=self.name,
            index=0,
            offset=(ws, hs),
            alias_suffix="_H",
        )
        child_block_list = self.child_floorplan.flatten(
            parent_info_list=base_parent_infos + [central_parent_info],
        )
        block_list.extend(child_block_list)

        halo_bl_parent_info = ParentInfo(
            parent_name=self.name,
            index=1,
            offset=(0.0, 0.0),
            alias_suffix="_H",
        )
        halo_bl = Block(
            "halo_bl",
            self.halo_material,
            (wl, hs),
            parent_info_list=base_parent_infos + [halo_bl_parent_info],
        )  # bottom left

        halo_br_parent_info = ParentInfo(
            parent_name=self.name,
            index=2,
            offset=(wl, 0.0),
            alias_suffix="_H",
        )
        halo_br = Block(
            "halo_br",
            self.halo_material,
            (ws, hl),
            parent_info_list=base_parent_infos + [halo_br_parent_info],
        )  # bottom right

        halo_tr_parent_info = ParentInfo(
            parent_name=self.name,
            index=3,
            offset=(ws, hl),
            alias_suffix="_H",
        )
        halo_tr = Block(
            "halo_tr",
            self.halo_material,
            (wl, hs),
            parent_info_list=base_parent_infos + [halo_tr_parent_info],
        )  # top right


        halo_tl_parent_info = ParentInfo(
            parent_name=self.name,
            index=4,
            offset=(0.0, hs),
            alias_suffix="_H",
        )
        halo_tl = Block(
            "halo_tl",
            self.halo_material,
            (ws, hl),
            parent_info_list=base_parent_infos + [halo_tl_parent_info],
        )  # top left

        block_list.extend([halo_bl, halo_br, halo_tr, halo_tl])

        return block_list
