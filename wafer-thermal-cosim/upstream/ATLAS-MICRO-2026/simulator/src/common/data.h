#ifndef _ATLASIM_DATA_H
#define _ATLASIM_DATA_H

#include <memory>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>


namespace atlasim {


struct MemoryAccess {
    std::string tensor_name;
    bool is_write;
    std::vector<int64_t> access_base;
    std::vector<int64_t> access_extent;
};


struct TensorPlacement {
    std::string name;

    int64_t base_addr = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int element_size = 0;

    TensorPlacement() { }
    TensorPlacement(YAML::Node config) {
        name = config["name"].as<std::string>();
        base_addr = config["base_addr"].as<int64_t>();
        element_size = config["element_size"].as<int>();

        bool has_general_layout = config["shape"] && config["strides"];
        if(!has_general_layout) {
            std::cerr << "TensorPlacement must provide shape/strides." << std::endl;
            exit(-1);
        }
        shape = config["shape"].as<std::vector<int64_t>>();
        strides = config["strides"].as<std::vector<int64_t>>();
        if(shape.empty() || shape.size() != strides.size()) {
            std::cerr << "TensorPlacement shape/strides must be non-empty and have the same rank." << std::endl;
            exit(-1);
        }
        validate_layout();
    }

    int rank() const {
        return static_cast<int>(shape.size());
    }

    void validate_layout() const {
        if(element_size <= 0) {
            std::cerr << "TensorPlacement element_size must be positive." << std::endl;
            exit(-1);
        }

        struct DimInfo {
            size_t dim;
            int64_t stride;
            int64_t extent;
        };

        std::vector<DimInfo> active_dims;
        active_dims.reserve(shape.size());
        for(size_t dim = 0; dim < shape.size(); ++dim) {
            if(shape[dim] <= 0) {
                std::cerr << "TensorPlacement shape[" << dim << "] must be positive, got " << shape[dim] << "." << std::endl;
                exit(-1);
            }
            if(strides[dim] < 0) {
                std::cerr << "TensorPlacement strides[" << dim << "] must be non-negative, got " << strides[dim] << "." << std::endl;
                exit(-1);
            }
            if(shape[dim] > 1) {
                active_dims.push_back({dim, strides[dim], shape[dim]});
            }
        }

        std::sort(active_dims.begin(), active_dims.end(), [](const DimInfo& lhs, const DimInfo& rhs) {
            if(lhs.stride != rhs.stride) {
                return lhs.stride < rhs.stride;
            }
            return lhs.dim < rhs.dim;
        });

        int64_t covered_offset_span = 0;
        for(const auto& dim_info : active_dims) {
            if(dim_info.stride <= covered_offset_span) {
                std::cerr << "TensorPlacement shape/strides cause overlapping element mapping at dimension "
                          << dim_info.dim << ": stride " << dim_info.stride
                          << " does not exceed previously covered offset span " << covered_offset_span << "."
                          << std::endl;
                exit(-1);
            }
            covered_offset_span += (dim_info.extent - 1) * dim_info.stride;
        }
    }

    int64_t offset_elements(const std::vector<int64_t>& indices) const {
        if(indices.size() != shape.size()) {
            std::cerr << "TensorPlacement offset_elements received rank " << indices.size()
                      << " for tensor rank " << shape.size() << "." << std::endl;
            exit(-1);
        }
        int64_t offset = 0;
        for(size_t dim = 0; dim < indices.size(); ++dim) {
            offset += indices[dim] * strides[dim];
        }
        return offset;
    }

    int64_t offset_bytes(const std::vector<int64_t>& indices) const {
        return offset_elements(indices) * element_size;
    }

    void display_stats(std::ostream& os=std::cout, int indent=0) {
        os << std::setw(indent) << "" << "Tensor Name: " << name << std::endl;
        os << std::setw(indent) << "" << "Base Address: " << base_addr << std::endl;
        os << std::setw(indent) << "" << "Element Size: " << element_size << std::endl;
        os << std::setw(indent) << "" << "Shape: [";
        for(size_t i=0; i<shape.size(); ++i) {
            if(i > 0) os << ", ";
            os << shape[i];
        }
        os << "]" << std::endl;
        os << std::setw(indent) << "" << "Strides: [";
        for(size_t i=0; i<strides.size(); ++i) {
            if(i > 0) os << ", ";
            os << strides[i];
        }
        os << "]" << std::endl;
    }
};


typedef std::shared_ptr<std::unordered_map<std::string, TensorPlacement>> PlacementMap;
typedef std::shared_ptr<std::vector<PlacementMap>> PlacementMapCollection;


inline PlacementMap parse_tensor_placement_list(const YAML::Node& tensor_config, const std::string& context) {
    if(!tensor_config || !tensor_config.IsSequence()) {
        std::cerr << context << " should be a sequence of all tensors' description." << std::endl;
        exit(-1);
    }

    PlacementMap placement_map = std::make_shared<std::unordered_map<std::string, TensorPlacement>>();
    for(int i=0; i<tensor_config.size(); ++i) {
        YAML::Node tensor_node = tensor_config[i];
        std::string tensor_name = tensor_node["name"].as<std::string>();
        auto inserted = placement_map->emplace(tensor_name, TensorPlacement(tensor_node));
        if(!inserted.second) {
            std::cerr << context << " has duplicate tensor placement for `" << tensor_name << "`." << std::endl;
            exit(-1);
        }
    }
    return placement_map;
}


inline PlacementMapCollection parse_placement_map_collection(const YAML::Node& config, int core_num) {
    if(core_num <= 0) {
        std::cerr << "Placement map parser requires a positive core_num." << std::endl;
        exit(-1);
    }

    PlacementMapCollection placement_maps = std::make_shared<std::vector<PlacementMap>>();
    if(config["core_tensor"]) {
        if(!config["core_tensor"].IsSequence()) {
            std::cerr << "Placement map core_tensor should be a sequence of per-core tensor descriptions." << std::endl;
            exit(-1);
        }
        placement_maps->assign(core_num, nullptr);
        std::set<int> seen_core_ids;
        for(int i=0; i<config["core_tensor"].size(); ++i) {
            YAML::Node core_tensor_node = config["core_tensor"][i];
            if(!core_tensor_node["core_id"]) {
                std::cerr << "Placement map core_tensor entry must provide core_id." << std::endl;
                exit(-1);
            }
            int core_id = core_tensor_node["core_id"].as<int>();
            if(core_id < 0 || core_id >= core_num) {
                std::cerr << "Placement map core_tensor has invalid core_id " << core_id
                          << " for core_num " << core_num << "." << std::endl;
                exit(-1);
            }
            if(seen_core_ids.count(core_id) > 0) {
                std::cerr << "Placement map core_tensor has duplicate core_id " << core_id << "." << std::endl;
                exit(-1);
            }
            seen_core_ids.insert(core_id);
            (*placement_maps)[core_id] = parse_tensor_placement_list(
                core_tensor_node["tensor"],
                "Placement map core_tensor[" + std::to_string(core_id) + "].tensor"
            );
        }
        if(static_cast<int>(seen_core_ids.size()) != core_num) {
            std::cerr << "Placement map core_tensor must provide one complete tensor list for each of "
                      << core_num << " cores." << std::endl;
            exit(-1);
        }
        return placement_maps;
    }

    if(!config["tensor"].IsSequence()) {
        std::cerr << "Placement map should be a sequence of all tensors' description." << std::endl;
        exit(-1);
    }
    PlacementMap shared_placement_map = parse_tensor_placement_list(config["tensor"], "Placement map tensor");
    for(int i=0; i<core_num; ++i) {
        placement_maps->push_back(shared_placement_map);
    }
    return placement_maps;
}


}


#endif // _ATLASIM_DATA_H
