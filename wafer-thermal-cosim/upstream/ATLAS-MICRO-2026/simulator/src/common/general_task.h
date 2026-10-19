#ifndef _ATLASIM_GENERAL_TASK_H
#define _ATLASIM_GENERAL_TASK_H

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/component_task.h"
#include "common/operator_task.h"


namespace atlasim {


inline YAML::Node require_general_sequence(
    const YAML::Node& config,
    const std::string& key,
    const std::string& context
) {
    if(!config[key] || !config[key].IsSequence()) {
        std::cerr << context << " must provide `" << key << "` as a YAML sequence." << std::endl;
        exit(-1);
    }
    return config[key];
}


struct GeneralIterationTask {
    std::vector<MatrixTask> matrix_tasks;
    std::vector<VectorTask> vector_tasks;
    std::vector<BufferTask> buffer_load_tasks;
    std::vector<BufferTask> buffer_store_tasks;
    std::vector<DRAMTask> DRAM_tasks;
    std::vector<NoCTask> noc_tx_tasks;
    std::vector<NoCTask> noc_rx_tasks;

    GeneralIterationTask() { }
    GeneralIterationTask(const YAML::Node config, const std::string& context) {
        YAML::Node matrix_config = require_general_sequence(config, "matrix", context);
        for(int i=0; i<matrix_config.size(); ++i) {
            matrix_tasks.emplace_back(matrix_config[i]);
        }

        YAML::Node vector_config = require_general_sequence(config, "vector", context);
        for(int i=0; i<vector_config.size(); ++i) {
            vector_tasks.emplace_back(vector_config[i]);
        }

        YAML::Node buffer_load_config = require_general_sequence(config, "buffer_load", context);
        for(int i=0; i<buffer_load_config.size(); ++i) {
            buffer_load_tasks.emplace_back(buffer_load_config[i]);
            if(buffer_load_tasks.back().is_write) {
                std::cerr << "Buffer load task should set is_write to false." << std::endl;
                exit(-1);
            }
        }

        YAML::Node buffer_store_config = require_general_sequence(config, "buffer_store", context);
        for(int i=0; i<buffer_store_config.size(); ++i) {
            buffer_store_tasks.emplace_back(buffer_store_config[i]);
            if(!buffer_store_tasks.back().is_write) {
                std::cerr << "Buffer store task should set is_write to true." << std::endl;
                exit(-1);
            }
        }

        YAML::Node dram_config = require_general_sequence(config, "dram", context);
        for(int i=0; i<dram_config.size(); ++i) {
            DRAM_tasks.emplace_back(dram_config[i]);
            if(DRAM_tasks.back().init_iter != 0 ||
               DRAM_tasks.back().stride_iter != 1 ||
               DRAM_tasks.back().total_iter != 1) {
                std::cerr << "General DRAM task must set init_iter=0, stride_iter=1, total_iter=1." << std::endl;
                exit(-1);
            }
        }

        YAML::Node noc_tx_config = require_general_sequence(config, "noc_tx", context);
        for(int i=0; i<noc_tx_config.size(); ++i) {
            noc_tx_tasks.emplace_back(noc_tx_config[i]);
            if(!noc_tx_tasks.back().is_send) {
                std::cerr << "NoC send (Tx) task should set is_send to true." << std::endl;
                exit(-1);
            }
        }

        YAML::Node noc_rx_config = require_general_sequence(config, "noc_rx", context);
        for(int i=0; i<noc_rx_config.size(); ++i) {
            noc_rx_tasks.emplace_back(noc_rx_config[i]);
            if(noc_rx_tasks.back().is_send) {
                std::cerr << "NoC receive (Rx) task should set is_send to false." << std::endl;
                exit(-1);
            }
        }
    }
};


struct GeneralCoreTask {
    std::vector<GeneralIterationTask> per_iter_tasks;

    GeneralCoreTask() { }
    GeneralCoreTask(int iteration) {
        per_iter_tasks.assign(iteration, GeneralIterationTask());
    }
    GeneralCoreTask(const YAML::Node config, int iteration, const std::string& source_path) {
        if(config["iteration"] && config["iteration"].as<int>() != iteration) {
            std::cerr << "General execution file " << source_path
                      << " has iteration " << config["iteration"].as<int>()
                      << ", but operator iteration is " << iteration << "." << std::endl;
            exit(-1);
        }
        YAML::Node execution_config = require_general_sequence(config, "execution", source_path);
        if(execution_config.size() > static_cast<size_t>(iteration)) {
            std::cerr << "General execution file " << source_path << " has "
                      << execution_config.size() << " iterations, but operator iteration is "
                      << iteration << "." << std::endl;
            exit(-1);
        }
        for(int i=0; i<execution_config.size(); ++i) {
            per_iter_tasks.emplace_back(
                execution_config[i],
                source_path + " execution[" + std::to_string(i) + "]"
            );
        }
        while(per_iter_tasks.size() < static_cast<size_t>(iteration)) {
            per_iter_tasks.emplace_back();
        }
    }
};


struct GeneralTask: OperatorTask {
    std::string file_prefix;
    std::shared_ptr<GeneralCoreTask> shared_core_task;
    std::vector<GeneralCoreTask> core_tasks;

    GeneralTask() { }
    GeneralTask(const YAML::Node config, int core_num): OperatorTask(config) {
        if(type != OperatorType::SPMDGeneral && type != OperatorType::MPMDGeneral) {
            std::cerr << "Find unsupported general operator type: " << optype2str.at(type) << std::endl;
            exit(-1);
        }
        if(!config["file_prefix"]) {
            std::cerr << "General operator must set file_prefix." << std::endl;
            exit(-1);
        }
        if(core_num <= 0) {
            std::cerr << "General operator parser requires a positive core_num." << std::endl;
            exit(-1);
        }
        file_prefix = config["file_prefix"].as<std::string>();

        if(type == OperatorType::SPMDGeneral) {
            if(!std::filesystem::exists(file_prefix) || !std::filesystem::is_regular_file(file_prefix)) {
                std::cerr << "SPMD general execution file does not exist: " << file_prefix << std::endl;
                exit(-1);
            }
            YAML::Node execution_node = YAML::LoadFile(file_prefix);
            shared_core_task = std::make_shared<GeneralCoreTask>(execution_node, iteration, file_prefix);
        }
        else {
            if(!std::filesystem::exists(file_prefix) || !std::filesystem::is_directory(file_prefix)) {
                std::cerr << "MPMD general execution directory does not exist: " << file_prefix << std::endl;
                exit(-1);
            }
            for(int i=0; i<core_num; ++i) {
                std::string config_file = file_prefix + "/core_" + std::to_string(i) + ".yaml";
                if(std::filesystem::exists(config_file)) {
                    YAML::Node execution_node = YAML::LoadFile(config_file);
                    core_tasks.emplace_back(execution_node, iteration, config_file);
                }
                else {
                    core_tasks.emplace_back(iteration);
                }
            }
        }
    }

    const GeneralCoreTask& get_core_task(int core_id) const {
        if(type == OperatorType::SPMDGeneral) {
            return *shared_core_task;
        }
        if(core_id < 0 || core_id >= static_cast<int>(core_tasks.size())) {
            std::cerr << "MPMD general task does not contain core " << core_id << "." << std::endl;
            exit(-1);
        }
        return core_tasks[core_id];
    }

    void display_stats(std::ostream& os=std::cout, int indent=0) override {
        os << std::setw(indent) << "" << "Name: " << name << std::endl;
        os << std::setw(indent) << "" << "Type: " << optype2str.at(type) << std::endl;
        os << std::setw(indent) << "" << "Loop Iteration Count: " << iteration << std::endl;
        os << std::setw(indent) << "" << "Execution File Prefix: " << file_prefix << std::endl;
        if(type == OperatorType::SPMDGeneral) {
            os << std::setw(indent) << "" << "Shared Core Iteration Tasks: "
               << shared_core_task->per_iter_tasks.size() << std::endl;
        }
        else {
            os << std::setw(indent) << "" << "Per Core Execution Tasks:" << std::endl;
            for(int i=0; i<core_tasks.size(); ++i) {
                os << std::setw(indent) << "" << "    Core (" << i+1 << "): "
                   << core_tasks[i].per_iter_tasks.size() << " iterations" << std::endl;
            }
        }
    }
};


}


#endif // _ATLASIM_GENERAL_TASK_H
