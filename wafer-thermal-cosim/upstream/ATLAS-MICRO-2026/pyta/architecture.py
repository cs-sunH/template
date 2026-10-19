import math
from typing import Tuple, List, Optional

from pyta.material import SILICON, UBUMP_T16UM_D25UM_P50UM, HB_T1300NM_P1750NM, COPPER_TSV, BEOL_CU_P28NM
from pyta.floorplan import (
    Floorplan, Block, GroupFloorplan, ArrayFloorplan, StackFloorplan,
    HaloFloorplan, TreeFloorplan, SlicingTree,
)
from pyta.layer import LayerConfig
from pyta.pyta_config import PytaConfig


def _count_tree_leaves(tree: SlicingTree) -> int:
    """Count leaf nodes in a SlicingTree."""
    if isinstance(tree, int):
        return 1
    return _count_tree_leaves(tree[1]) + _count_tree_leaves(tree[2])


def _reindex_slicing_tree(tree: SlicingTree, offset: int) -> SlicingTree:
    """Add offset to all leaf indices in the tree."""
    if isinstance(tree, int):
        return tree + offset
    cut, left, right = tree
    return (cut, _reindex_slicing_tree(left, offset), _reindex_slicing_tree(right, offset))


def get_default_slicing_tree(n: int, cut: str = "V") -> SlicingTree:
    """
    Generate a default SlicingTree for n blocks using binary recursive splitting.
    Block indices 0..n-1. Each split divides blocks into left and right halves.

    Args:
        n: Number of blocks (must be >= 1).
        cut: Cut direction for root, "V" (left|right) or "H" (top|bottom).

    Returns:
        SlicingTree with leaves 0..n-1.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 0
    mid = n // 2
    left = get_default_slicing_tree(mid, cut)
    right = get_default_slicing_tree(n - mid, cut)
    right = _reindex_slicing_tree(right, mid)
    return (cut, left, right)


class AtlasArch():
    """
    Atlas architecture featuring 3D stacked memory.

    An Arch contains a tile_array of tiles. Each tile consists of:
    - Logic tile (below): layout defined by TreeFloorplan
    - Memory tiles (above): each memory_tile = left bank_group | IO stripe | right bank_group

    Tile size is fixed at init. Memory tile uses io_stripe_ratio and io_stripe_orientation
    to build [bank_group, io_stripe, bank_group] via StackFloorplan; each bank_group
    is built with ArrayFloorplan of dram_bank blocks.
    """

    def __init__(
        self,
        name: str,
        tile_array_size: Tuple[int, int],
        tile_shape: Tuple[float, float],
        num_memory_layer: int,
        cooling_style: str,
        interposer_size: float,
        num_banks_per_group: int = 1,
        io_stripe_ratio: float = 0.3,
        io_stripe_orientation: str = "Y",
        d2d_bonding_style: str = "hb",
        tile_block_list: Optional[List[str]] = None,
        split_tree: Optional[SlicingTree] = None,
    ):
        self.name = name
        self.tile_array_size = tile_array_size
        self.tile_shape = tile_shape
        self.num_banks_per_group = num_banks_per_group
        self.io_stripe_ratio = io_stripe_ratio
        self.io_stripe_orientation = io_stripe_orientation
        self.num_memory_layer = num_memory_layer
        self.d2d_bonding_style = d2d_bonding_style
        self.cooling_style = cooling_style
        self.interposer_size = interposer_size
        self.tile_block_list = tile_block_list if tile_block_list is not None else ["logic_blk_0", "logic_blk_1", "logic_blk_2"]
        if split_tree is not None:
            self.split_tree = split_tree
            assert _count_tree_leaves(split_tree) == len(self.tile_block_list), \
                "split_tree leaf count must match tile_block_list length"
        else:
            self.split_tree = get_default_slicing_tree(len(self.tile_block_list))

        assert 0 < io_stripe_ratio < 1, "io_stripe_ratio must be in (0, 1)"
        assert io_stripe_orientation in ("X", "Y"), "io_stripe_orientation must be 'X' or 'Y'"
        assert num_banks_per_group == 1, "num_banks_per_group must be 1"

        self._tile_area_limit = tile_shape[0] * tile_shape[1]
        self._logic_tile_flp = self._build_logic_tile_flp()

    def _build_logic_tile_flp(self) -> TreeFloorplan:
        """Build logic tile as TreeFloorplan from split_tree and tile_block_list."""
        block_names = self.tile_block_list
        num_blocks = len(block_names)
        default_area = self._tile_area_limit / num_blocks
        side = math.sqrt(default_area)
        block_list = [
            Block(name, SILICON, (side, side), parent_info_list=[])
            for name in block_names
        ]
        aspect_ratio = self.tile_shape[0] / self.tile_shape[1]
        return TreeFloorplan("logic_tile", block_list, self.split_tree, aspect_ratio)


    def _build_memory_tile_flp(self, flp_bank: Floorplan) -> Floorplan:
        """
        Build memory tile: [left bank_group | io_stripe | right bank_group].
        io_stripe_orientation: "Y" -> stripe extends along Y, split along X (left|stripe|right);
                               "X" -> stripe extends along X, split along Y (top|stripe|bottom).
        """
        w, h = self.tile_shape
        r = self.io_stripe_ratio
        orient = self.io_stripe_orientation

        if orient == "Y":
            # Stripe extends along Y (vertical bar), split along X
            group_width = (1 - r) / 2 * w
            stripe_width = r * w
            group_shape = (group_width, h)
            stripe_shape = (stripe_width, h)
            stack_axis = "X"
        else:
            # orient == "X": stripe extends along X (horizontal bar), split along Y
            group_height = (1 - r) / 2 * h
            stripe_height = r * h
            group_shape = (w, group_height)
            stripe_shape = (w, stripe_height)
            stack_axis = "Y"

        flp_bank_group = self._build_bank_group_flp(flp_bank, group_shape)
        block_io_stripe = Block("io_stripe", COPPER_TSV, stripe_shape)
        flp_io_stripe = GroupFloorplan("io_stripe", [block_io_stripe], [(0.0, 0.0)])

        flp_core = StackFloorplan(
            "core",
            [flp_bank_group, flp_io_stripe, flp_bank_group],
            stack_axis,
        )
        return flp_core

    def _build_bank_group_flp(self, flp_bank: Floorplan, group_shape: Tuple[float, float]) -> Floorplan:
        """Build bank group as ArrayFloorplan. Resize bank to fit group_shape."""
        gw, gh = group_shape
        orient = self.io_stripe_orientation
        n = self.num_banks_per_group

        if orient == "Y":
            bank_w = gw / n
            bank_h = gh
            axis = "X"
        else:
            bank_w = gw
            bank_h = gh / n
            axis = "Y"

        if hasattr(flp_bank, "block_list") and flp_bank.block_list:
            src = flp_bank.block_list[0]
            bank_block = Block(src.name, src.material, (bank_w, bank_h))
        else:
            bank_block = Block("dram_bank", SILICON, (bank_w, bank_h))
        flp_single_bank = GroupFloorplan("dram_bank", [bank_block], [(0.0, 0.0)])
        return ArrayFloorplan("bank_group", flp_single_bank, n, axis)

    def get_logic_tile_flp(self) -> TreeFloorplan:
        """Return the logic tile TreeFloorplan."""
        return self._logic_tile_flp

    def get_memory_bank_flp(self) -> Floorplan:
        w, h = self.tile_shape
        r = self.io_stripe_ratio
        orient = self.io_stripe_orientation
        n = self.num_banks_per_group
        if orient == "Y":
            gw = (1 - r) / 2 * w
            bank_w = gw / n
            bank_h = h
        else:
            gh = (1 - r) / 2 * h
            bank_w = w
            bank_h = gh / n
        return GroupFloorplan(
            "dram_bank",
            [Block("dram_bank", SILICON, (bank_w, bank_h))],
            [(0.0, 0.0)],
        )

    def get_beol_bank_flp(self) -> Floorplan:
        w, h = self.tile_shape
        r = self.io_stripe_ratio
        orient = self.io_stripe_orientation
        n = self.num_banks_per_group
        if orient == "Y":
            gw = (1 - r) / 2 * w
            bank_w = gw / n
            bank_h = h
        else:
            gh = (1 - r) / 2 * h
            bank_w = w
            bank_h = gh / n
        return GroupFloorplan(
            "beol_bank",
            [Block("beol_bank", BEOL_CU_P28NM, (bank_w, bank_h))],
            [(0.0, 0.0)],
        )

    def get_memory_floorplan(self, flp_bank: Floorplan, flp_name: str) -> Floorplan:
        """Build memory layer: tile_array of memory tiles."""
        flp_memory_tile = self._build_memory_tile_flp(flp_bank)
        nx, ny = self.tile_array_size
        flp_core_row = ArrayFloorplan("core_row", flp_memory_tile, nx, "X")
        flp_core_array = ArrayFloorplan("core_array", flp_core_row, ny, "Y")
        flp_interposer = HaloFloorplan(flp_name, flp_core_array, (self.interposer_size, self.interposer_size), SILICON)
        return flp_interposer

    def get_logic_floorplan(self, flp_name: str) -> Floorplan:
        """Build logic layer: tile_array of logic tiles."""
        flp_core = self._logic_tile_flp
        nx, ny = self.tile_array_size
        flp_core_row = ArrayFloorplan("core_row", flp_core, nx, "X")
        flp_core_array = ArrayFloorplan("core_array", flp_core_row, ny, "Y")
        flp_interposer = HaloFloorplan(flp_name, flp_core_array, (self.interposer_size, self.interposer_size), SILICON)
        return flp_interposer

    def set_tile_block_area(self, areas: List[float]) -> bool:
        """
        Update each block's area in the logic tile. Returns False if total area
        exceeds tile area limit.
        """
        total = sum(areas)
        if total > self._tile_area_limit:
            return False
        num_blocks = len(self._logic_tile_flp.block_list)
        if len(areas) != num_blocks:
            return False
        for i, a in enumerate(areas):
            if a <= 0:
                continue
            side = math.sqrt(a)
            self._logic_tile_flp.block_list[i].shape = (side, side)
        self._logic_tile_flp._total_area = total
        return True

    def set_logic_tile_power(
        self,
        config: PytaConfig,
        power_list: List[float],
        row_id: int,
        col_id: int,
    ) -> None:
        """
        Update power for each block in the logic tile at (row_id, col_id).
        power_list[i] is the power (W) for block i (steady-state).
        """
        block_list = self._logic_tile_flp.block_list
        num_blocks = len(block_list)
        if config.ptrace is None or len(power_list) != num_blocks:
            return
        for i in range(num_blocks):
            block_name = block_list[i].name
            full_name = f"{block_name}_L0_H_AY{row_id}_AX{col_id}_T{i}"
            if full_name in config.ptrace:
                p = power_list[i]
                config.ptrace[full_name] = [p] if isinstance(p, (int, float)) else p

    def set_dram_tile_power(
        self,
        config: PytaConfig,
        power_list: List[float],
        row_id: int,
        col_id: int,
        layer_id: int = 0,
    ) -> None:
        """
        Update memory bank power for tile at (row_id, col_id). Power is split
        50/50 to left and right bank groups; center TSV stripe gets 0.
        """
        if config.ptrace is None or len(power_list) == 0:
            return
        half_power = [p / 2.0 for p in power_list]
        banks_per_group = self.num_banks_per_group
        power_per_bank = [p / banks_per_group for p in half_power]

        layer_num = 3
        stack_axis = "X" if self.io_stripe_orientation == "Y" else "Y"
        stack_suffix = "SX" if stack_axis == "X" else "SY"
        bank_suffix = "AX" if self.io_stripe_orientation == "Y" else "AY"
        for i in range(banks_per_group):
            for side in (0, 2):
                full_name = f"dram_bank_L{layer_num}_H_AY{row_id}_AX{col_id}_{stack_suffix}{side}_{bank_suffix}{i}"
                if full_name in config.ptrace:
                    config.ptrace[full_name] = power_per_bank

    def get_pyta_config(self) -> PytaConfig:
        """Build PytaConfig (simple stacking mode only)."""
        logic_layer_flp = self.get_logic_floorplan("logic_die")
        logic_layer = LayerConfig(
            layer_number=0,
            thickness=10e-6,
            default_material=SILICON,
            floorplan=logic_layer_flp,
            lateral_heat_flow=True,
            power_dissipation=True,
        )

        logic_shape = logic_layer_flp.get_shape()
        if self.d2d_bonding_style == "ubump":
            d2d_material = UBUMP_T16UM_D25UM_P50UM
            d2d_thickness = 16e-6
        elif self.d2d_bonding_style == "hb":
            d2d_material = HB_T1300NM_P1750NM
            d2d_thickness = 1.3e-6
        else:
            raise NotImplementedError("Only ubump and hb are supported for now")

        flp_d2d = GroupFloorplan(
            "d2d",
            [Block("d2d", d2d_material, logic_shape)],
            [(0.0, 0.0)],
        )
        d2d_layer = LayerConfig(
            layer_number=1,
            thickness=d2d_thickness * self.num_memory_layer,
            default_material=d2d_material,
            floorplan=flp_d2d,
            lateral_heat_flow=True,
            power_dissipation=False,
        )

        beol_bank_flp = self.get_beol_bank_flp()
        beol_layer = LayerConfig(
            layer_number=2,
            thickness=5e-6 * self.num_memory_layer,
            default_material=BEOL_CU_P28NM,
            floorplan=self.get_memory_floorplan(beol_bank_flp, "beol_die"),
            lateral_heat_flow=True,
            power_dissipation=False,
        )

        memory_bank_flp = self.get_memory_bank_flp()
        memory_layer = LayerConfig(
            layer_number=3,
            thickness=40e-6 * self.num_memory_layer,
            default_material=SILICON,
            floorplan=self.get_memory_floorplan(memory_bank_flp, "memory_die"),
            lateral_heat_flow=True,
            power_dissipation=True,
        )

        total_height = sum(layer.thickness for layer in [logic_layer, memory_layer, beol_layer, d2d_layer])
        remaining_height = 750e-6 - total_height
        assert remaining_height > 0, "Remaining height must be positive"

        dummy_flp = GroupFloorplan(
            "dummy",
            [Block("dummy", SILICON, logic_shape)],
            [(0.0, 0.0)],
        )
        dummy_layer = LayerConfig(
            layer_number=4,
            thickness=remaining_height,
            default_material=SILICON,
            floorplan=dummy_flp,
            lateral_heat_flow=True,
            power_dissipation=False,
        )

        layer_list = [logic_layer, memory_layer, beol_layer, d2d_layer, dummy_layer]

        if self.cooling_style == "air":
            htc = 500
        elif self.cooling_style == "liquid":
            htc = 10000
        else:
            raise NotImplementedError("Only air and liquid cooling are supported for now")

        s_interposer = self.interposer_size
        s_spreader = 2 * s_interposer
        s_sink = 2 * s_spreader
        r_convec = 1 / (htc * s_sink * s_sink)

        thermal_params = {
            "r_convec": r_convec,
            "s_spreader": s_spreader,
            "s_sink": s_sink,
        }

        pyta_config = PytaConfig(
            name=self.name,
            layer_list=layer_list,
            thermal_params=thermal_params,
        )

        block_list = pyta_config.get_block_list(skip_powerless_block=True)
        ptrace = {block.get_full_name(): [0.0] for block in block_list}
        pyta_config.set_ptrace(ptrace)

        return pyta_config
