import abc


class Material(abc.ABC):
    def __init__(
        self,
        name: str,
        heat_capacity: float,
        heat_conductivity: float,
    ):
        """
        Args:
            name: str, name of the material
            heat_capacity: float, in J/(m^3-K)
            heat_conductivity: float, in W/(m-K)
        """
        self.name = name
        self.heat_capacity = heat_capacity
        self.heat_conductivity = heat_conductivity


SILICON = Material(
    name="silicon",
    heat_capacity=1750000,
    heat_conductivity=100.0,
)


COPPER = Material(
    name="copper",
    heat_capacity=3494400,
    heat_conductivity=400.0,
)


UNDERFILL = Material(
    name="underfill",
    heat_capacity=2320000,
    heat_conductivity=0.4,
)


UBUMP_T16UM_D25UM_P50UM = Material(
    name="ubump_t16um_d25um_p50um",
    heat_capacity=2320000,
    heat_conductivity=2.0,
)


HB_T1300NM_P1750NM = Material(
    name="hb_t1300nm_p1750nm",
    heat_capacity=2320000,
    heat_conductivity=1.625,
)


COPPER_TSV = Material(
    name="copper_tsv",
    heat_capacity=3494400,
    heat_conductivity=190.0,
)


BEOL_CU_P28NM = Material(
    name="beol_cu_p28nm",
    heat_capacity=2320000,
    heat_conductivity=3.0,
)
