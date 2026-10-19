from atlasim import Chip


def test_gemm():
    chip = Chip(
        "configs/architecture/chip/test_chip_16ch.yaml",
        "configs/operator_yaml/gemm_comp/gemm.yaml",
        "configs/operator_yaml/gemm_comp/gemm_data.yaml"
    )
    performance = chip.simulate()
    performance.display_stats()


if __name__ == "__main__":
    test_gemm()
