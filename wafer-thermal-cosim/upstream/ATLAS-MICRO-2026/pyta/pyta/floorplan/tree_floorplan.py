from copy import deepcopy
from typing import Tuple, List, Optional, Union

from .base_floorplan import Floorplan
from .block import Block, ParentInfo, ParentInfoList


# Tree: leaf = block index (int); internal = ("H"|"V", left_tree, right_tree)
# "H" = horizontal cut: left subtree is ABOVE right (same width)
# "V" = vertical cut: left subtree is LEFT of right (same height)
SlicingTree = Union[int, Tuple[str, "SlicingTree", "SlicingTree"]]


def _is_leaf(node: SlicingTree) -> bool:
    return isinstance(node, int)


def _area_of_node(node: SlicingTree, block_list: List[Block]) -> float:
    """Total area of the subtree rooted at node."""
    if _is_leaf(node):
        w, h = block_list[node].shape
        return w * h
    _, left, right = node
    return _area_of_node(left, block_list) + _area_of_node(right, block_list)


def _collect_leaf_indices(node: SlicingTree) -> List[int]:
    """Return leaf indices in left-to-right order (for consistent ParentInfo index)."""
    if _is_leaf(node):
        return [node]
    _, left, right = node
    return _collect_leaf_indices(left) + _collect_leaf_indices(right)


class TreeFloorplan(Floorplan):
    """
    A floorplan defined by a binary slicing tree over a list of blocks.
    Internal nodes are "H" (left above right) or "V" (left left of right).
    Block shapes are resized to preserve area and fill the bounding box
    with the given aspect_ratio (width / height).
    """

    def __init__(
        self,
        name: str,
        block_list: List[Block],
        tree: SlicingTree,
        aspect_ratio: float,
    ):
        super().__init__(name)
        self.block_list = block_list
        self.tree = tree
        if aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        self.aspect_ratio = aspect_ratio
        self._leaf_order = _collect_leaf_indices(tree)
        if set(self._leaf_order) != set(range(len(block_list))):
            raise ValueError(
                "Tree leaves must be exactly indices 0..len(block_list)-1, "
                "each appearing once"
            )
        self._total_area = _area_of_node(tree, block_list)

    def get_shape(self) -> Tuple[float, float]:
        """Return (width, height) so that width/height = aspect_ratio and area is preserved."""
        area = self._total_area
        # width / height = aspect_ratio, width * height = area
        # => height^2 * aspect_ratio = area => height = sqrt(area / aspect_ratio)
        height = (area / self.aspect_ratio) ** 0.5
        width = area / height
        return (width, height)

    def _flatten_rec(
        self,
        node: SlicingTree,
        x: float,
        y: float,
        w: float,
        h: float,
        base_parent_infos: ParentInfoList,
        leaf_counter: List[int],
    ) -> None:
        """Recursively assign (x,y,w,h) and append blocks to self._result_blocks."""
        if _is_leaf(node):
            idx = node
            block = self.block_list[idx]
            # New block with assigned shape (w, h); area w*h equals block's area
            new_block = deepcopy(block)
            new_block.shape = (w, h)
            pos = leaf_counter[0]
            leaf_counter[0] += 1
            child_info = ParentInfo(
                parent_name=self.name,
                index=pos,
                offset=(x, y),
                alias_suffix=f"_T{pos}",
            )
            new_block.parent_info_list = base_parent_infos + [child_info]
            self._result_blocks.append(new_block)
            return
        cut, left, right = node
        a_left = _area_of_node(left, self.block_list)
        a_right = _area_of_node(right, self.block_list)
        a_total = a_left + a_right
        if a_total <= 0:
            return
        if cut == "H":
            # Left above right: same width, split height
            h_left = h * (a_left / a_total)
            h_right = h * (a_right / a_total)
            self._flatten_rec(left, x, y, w, h_left, base_parent_infos, leaf_counter)
            self._flatten_rec(
                right, x, y + h_left, w, h_right, base_parent_infos, leaf_counter
            )
        elif cut == "V":
            # Left left of right: same height, split width
            w_left = w * (a_left / a_total)
            w_right = w * (a_right / a_total)
            self._flatten_rec(left, x, y, w_left, h, base_parent_infos, leaf_counter)
            self._flatten_rec(
                right, x + w_left, y, w_right, h, base_parent_infos, leaf_counter
            )
        else:
            raise ValueError("Cut must be 'H' or 'V', got %r" % (cut,))

    def flatten(
        self,
        parent_info_list: Optional[ParentInfoList] = None,
    ) -> List[Block]:
        base_parent_infos: ParentInfoList = list(parent_info_list or [])
        self._result_blocks = []
        width, height = self.get_shape()
        self._flatten_rec(
            self.tree,
            0.0,
            0.0,
            width,
            height,
            base_parent_infos,
            [0],
        )
        return self._result_blocks
