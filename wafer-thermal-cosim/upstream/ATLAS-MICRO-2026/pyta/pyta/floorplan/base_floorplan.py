import abc
from typing import Tuple, List, Optional

from .block import Block, ParentInfoList


class Floorplan(abc.ABC):
    """
    A floorplan is a collection of blocks.
    """

    def __init__(
        self,
        name: str,
    ):
        self.name = name

    @abc.abstractmethod
    def get_shape(self) -> Tuple[float, float]:
        pass

    @abc.abstractmethod
    def flatten(
        self,
        parent_info_list: Optional[ParentInfoList] = None,
    ) -> List[Block]:
        """
        Flatten the floorplan to a list of instantiated blocks.
        """
        pass
