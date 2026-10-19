#include <filesystem>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
// atlasim include
#include "common/arch.h"
#include "core/component.h"
#include "core/matrix_unit.h"


int main(int argc, char** argv) {
    if(argc < 3) {
        std::cout << "Usage: ./test_component_input [CHIP_CONFIG] [CASE]" << std::endl;
        exit(-1);
    }

    std::string chip_config_path(argv[1]);
    if(!std::filesystem::exists(chip_config_path)) {
        std::cerr << "Chip config path " << chip_config_path << " does not exist." << std::endl;
        exit(-1);
    }
    YAML::Node chip_config = YAML::LoadFile(chip_config_path);
    atlasim::ArchConfig arch_config = std::make_shared<atlasim::ChipConfig>(chip_config);

    std::string test_case(argv[2]);
    if(test_case == "matrix_missing_mac_count") {
        atlasim::MatrixUnit matrix("test", arch_config);
        atlasim::ComponentInput input = atlasim::make_component_input({
            {{"not_mac_count", 1}}
        });
        matrix.simulate(input);
    }
    else {
        std::cerr << "Invalid test case: " << test_case << std::endl;
        exit(-1);
    }

    return 0;
}
