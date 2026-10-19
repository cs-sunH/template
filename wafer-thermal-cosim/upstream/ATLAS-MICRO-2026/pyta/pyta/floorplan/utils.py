from typing import List, Tuple

from .block import Block


def is_block_pair_overlap(block1: Block, block2: Block) -> bool:
    """
    Check if the two blocks overlap, using their global offsets.
    """
    x1, y1 = block1.offset
    x2, y2 = x1 + block1.shape[0], y1 + block1.shape[1]
    x3, y3 = block2.offset
    x4, y4 = x3 + block2.shape[0], y3 + block2.shape[1]
    return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)


def is_block_list_overlap(block_list: List[Block]) -> bool:
    """
    Check if the blocks in the list overlap, using their global offsets.
    """
    for i in range(len(block_list)):
        for j in range(i + 1, len(block_list)):
            if is_block_pair_overlap(block_list[i], block_list[j]):
                return True
    return False


def is_block_list_overlap_with_offsets(
    block_list: List[Block],
    offset_list: List[Tuple[float, float]],
) -> bool:
    """
    Check if blocks would overlap when placed at the given local offsets.

    This is similar to is_block_list_overlap, but uses an explicit offset_list
    instead of the blocks' own ParentInfo/global offsets.
    """
    if len(block_list) != len(offset_list):
        raise ValueError(
            "block_list and offset_list must have the same length, "
            f"got {len(block_list)} blocks and {len(offset_list)} offsets"
        )

    n = len(block_list)
    for i in range(n):
        x1, y1 = offset_list[i]
        w1, h1 = block_list[i].shape
        x2_r = x1 + w1
        y2_r = y1 + h1
        for j in range(i + 1, n):
            x3, y3 = offset_list[j]
            w2, h2 = block_list[j].shape
            x4_r = x3 + w2
            y4_r = y3 + h2
            if not (x2_r <= x3 or x4_r <= x1 or y2_r <= y3 or y4_r <= y1):
                return True

    return False
