#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cassert>
#include <yaml-cpp/yaml.h>
// atlasim include
#include "common/arch.h"
#include "common/task.h"
#include "common/data.h"


int parse_arch_config(std::string& config_path) {
    if(!std::filesystem::exists(config_path)) {
        std::cerr << "Arch config path " << config_path << " does not exist." << std::endl;
        return 0;
    }

    YAML::Node config = YAML::LoadFile(config_path);
    atlasim::ChipConfig arch_config(config);
    arch_config.display_stats();
    return arch_config.core_num;
}


void parse_operator_config(std::string& config_path, int core_num) {
    if(!std::filesystem::exists(config_path)) {
        std::cerr << "Operator config path " << config_path << " does not exist." << std::endl;
        return;
    }

    std::vector<std::shared_ptr<atlasim::OperatorTask>> task_list;

    // sanity check
    YAML::Node config = YAML::LoadFile(config_path);
    if(!config["operator"].IsSequence()) {
        std::cerr << "Operator config should be a sequence of all operators' description." << std::endl;
        return;
    }
    std::cout << "This operator config has " << config["operator"].size() << " operators." << std::endl;

    // parse each task
    for(int i=0; i<config["operator"].size(); ++i) {
        YAML::Node tmp_yaml = config["operator"][i];
        std::shared_ptr<atlasim::OperatorTask> task = atlasim::OperatorTaskFactory::create(tmp_yaml, core_num);
        task_list.push_back(task);
    }

    // print task information
    std::cout << "All Operator Configs:" << std::endl;
    for(int i=0; i<task_list.size(); ++i) {
        std::cout << "Operator " << i << ":" << std::endl;
        int indent = 4;
        task_list[i]->display_stats(std::cout, indent);
    }
}


void parse_data_config(std::string config_path, int core_num) {
    if(!std::filesystem::exists(config_path)) {
        std::cerr << "Data config path " << config_path << " does not exist." << std::endl;
        return;
    }

    YAML::Node config = YAML::LoadFile(config_path);
    atlasim::PlacementMapCollection placement_maps = atlasim::parse_placement_map_collection(config, core_num);

    // print tensot placement information
    std::cout << "All Tensor Placement Configs:" << std::endl;
    if(config["core_tensor"]) {
        std::cout << "This tensor placement config has per-core placements for "
                  << placement_maps->size() << " cores." << std::endl;
        for(int core_id=0; core_id<placement_maps->size(); ++core_id) {
            std::cout << "Core " << core_id << " Tensor Placement Configs:" << std::endl;
            int index = 0;
            for(auto& placement: *placement_maps->at(core_id)) {
                std::cout << "Tensor Placement " << index << ":" << std::endl;
                int indent = 4;
                placement.second.display_stats(std::cout, indent);
                index++;
            }
        }
    }
    else {
        std::cout << "This tensor placement config has " << config["tensor"].size() << " tensors." << std::endl;
        for(int i=0; i<config["tensor"].size(); ++i) {
            atlasim::TensorPlacement placement(config["tensor"][i]);
            std::cout << "Tensor Placement " << i << ":" << std::endl;
            int indent = 4;
            placement.display_stats(std::cout, indent);
        }
    }
}


int main(int argc, char** argv) {
    if(argc<4) {
        std::cout << "Usage: ./test_config_parser ARCH_CONFIG OP_CONFIG DATA_CONFIG" << std::endl;
        exit(-1);
    }

    std::string arch_config_path(argv[1]);
    std::string operator_config_path(argv[2]);
    std::string data_config_path(argv[3]);

    int core_num = parse_arch_config(arch_config_path);
    parse_operator_config(operator_config_path, core_num);
    parse_data_config(data_config_path, core_num);

    return 0;
}
