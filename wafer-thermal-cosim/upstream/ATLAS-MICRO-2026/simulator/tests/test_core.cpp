#include <iostream>
// atlasim include
#include "chip/chip.h"


int main(int argc, char** argv) {
    if(argc<4) {
        std::cout << "Usage: ./test_core "
                  << "[CHIP_CONFIG] "
                  << "[OPERATOR_LIST] "
                  << "[PLACEMENT_MAP] "
                  << "[(optional) LOG_FILE]" 
                  << std::endl;
        exit(-1);
    }

    std::string chip_config_path(argv[1]);
    std::string operator_list_path(argv[2]);
    std::string placement_map_path(argv[3]);
    std::string log_file_path = "";
    if(argc>=5) {
        log_file_path = std::string(argv[4]);
    }

    atlasim::Chip chip(
        chip_config_path, 
        operator_list_path, 
        placement_map_path, 
        log_file_path
    );
    atlasim::Performance performance = chip.simulate();

    chip.display_stats();
    performance.display_stats();
    return 0;
}
