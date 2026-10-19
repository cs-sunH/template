#ifndef _ATLASIM_COMPUTATION_TASK_H
#define _ATLASIM_COMPUTATION_TASK_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/component_task.h"
#include "common/input.h"
#include "common/operator_task.h"


namespace atlasim {


struct ComputationTask: OperatorTask {
    // Only for non-NoC operators
    std::vector<MatrixTask> matrix_tasks;
    std::vector<VectorTask> vector_tasks;
    std::vector<BufferTask> buffer_load_tasks;
    std::vector<BufferTask> buffer_store_tasks;
    std::vector<DRAMTask> DRAM_tasks;
    std::string attention_input_path;
    std::shared_ptr<DAttnInput> decode_attn_input;

    ComputationTask() { }
    ComputationTask(const YAML::Node config): OperatorTask(config) {
        if(config["attention_input"] && type != OperatorType::DecodeAttention) {
            std::cerr << "Only decode-attention operators can set attention_input." << std::endl;
            exit(-1);
        }
        if(type == OperatorType::DecodeAttention) {
            if(!config["attention_input"]) {
                std::cerr << "Decode-attention operator must set attention_input." << std::endl;
                exit(-1);
            }
            attention_input_path = config["attention_input"].as<std::string>();
            YAML::Node attention_input_node = YAML::LoadFile(attention_input_path);
            if(!attention_input_node["attention_input"]) {
                std::cerr << "Decode-attention input YAML must contain root key `attention_input`." << std::endl;
                exit(-1);
            }
            decode_attn_input = std::make_shared<DAttnInput>(attention_input_node["attention_input"]);
        }

        YAML::Node execution_config = config["execution"];
        if(type==OperatorType::GEMM || type==OperatorType::DecodeAttention) {
            // Parse matrix tasks
            try {
                YAML::Node matrix_task_config = execution_config["matrix"];
                for(int i=0; i<matrix_task_config.size(); ++i) {
                    matrix_tasks.emplace_back(matrix_task_config[i]);
                }
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }

            // Parse vector tasks
            try {
                YAML::Node vector_task_config = execution_config["vector"];
                for(int i=0; i<vector_task_config.size(); ++i) {
                    vector_tasks.emplace_back(vector_task_config[i]);
                }
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }

            // Parse buffer load tasks
            try {
                YAML::Node buffer_load_config = execution_config["buffer_load"];
                for(int i=0; i<buffer_load_config.size(); ++i) {
                    buffer_load_tasks.emplace_back(buffer_load_config[i]);
                    if(buffer_load_tasks[i].is_write) {
                        std::cerr << "Buffer load task should set is_write to false." << std::endl;
                        exit(-1);
                    }
                }
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }

            // Parse buffer store tasks
            try {
                YAML::Node buffer_store_config = execution_config["buffer_store"];
                for(int i=0; i<buffer_store_config.size(); ++i) {
                    buffer_store_tasks.emplace_back(buffer_store_config[i]);
                    if(!buffer_store_tasks[i].is_write) {
                        std::cerr << "Buffer store task should set is_write to true." << std::endl;
                        exit(-1);
                    }
                }
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }

            // Parse DRAM load/store tasks
            try {
                YAML::Node dram_access_config = execution_config["dram"];
                for(int i=0; i<dram_access_config.size(); ++i) {
                    DRAM_tasks.emplace_back(dram_access_config[i]);
                }
            }
            catch(const std::exception& e) {
                std::cerr << e.what() << std::endl;
            }
        }
        else {
            std::cerr << "Find unsupported computation operator type: "  << type << std::endl;
            exit(-1);
        }
    }

    void display_stats(std::ostream& os=std::cout, int indent=0) override {
        os << std::setw(indent) << "" << "Name: " << name << std::endl;
        os << std::setw(indent) << "" << "Type: " << optype2str.at(type) << std::endl;
        os << std::setw(indent) << "" << "Loop Iteration Count: " << iteration << std::endl;
        if(type == OperatorType::DecodeAttention) {
            os << std::setw(indent) << "" << "Attention Input: " << attention_input_path << std::endl;
        }
        os << std::setw(indent) << "" << "Per Loop Execution Tasks:" << std::endl;

        os << std::setw(indent) << "" << "    Matrix Tasks:" << std::endl;
        for(int i=0; i<matrix_tasks.size(); ++i) {
            os << std::setw(indent) << "" << "    (" << i+1 << ") Name: " << matrix_tasks[i].name << std::endl;
            os << std::setw(indent) << "" << "        MAC Count: " << matrix_tasks[i].mac_count << std::endl;
        }

        os << std::setw(indent) << "" << "    Vector Tasks:" << std::endl;
        for(int i=0; i<vector_tasks.size(); ++i) {
            os << std::setw(indent) << "" << "    (" << i+1 << ") Name: " << vector_tasks[i].name << std::endl;
            os << std::setw(indent) << "" << "        Vec Count: " << vector_tasks[i].vec_count << std::endl;
        }

        os << std::setw(indent) << "" << "    Buffer Load Tasks:" << std::endl;
        for(int i=0; i<buffer_load_tasks.size(); ++i) {
            os << std::setw(indent) << "" << "    (" << i+1 << ") Name: " << buffer_load_tasks[i].name << std::endl;
            os << std::setw(indent) << "" << "        Is Write: " << (buffer_load_tasks[i].is_write ? "true" : "false") << std::endl;
            os << std::setw(indent) << "" << "        Byte Count: " << buffer_load_tasks[i].byte_count << std::endl;
        }

        os << std::setw(indent) << "" << "    Buffer Store Tasks:" << std::endl;
        for(int i=0; i<buffer_store_tasks.size(); ++i) {
            os << std::setw(indent) << "" << "    (" << i+1 << ") Name: " << buffer_store_tasks[i].name << std::endl;
            os << std::setw(indent) << "" << "        Is Write: " << (buffer_store_tasks[i].is_write ? "true" : "false") << std::endl;
            os << std::setw(indent) << "" << "        Byte Count: " << buffer_store_tasks[i].byte_count << std::endl;
        }

        os << std::setw(indent) << "" << "    DRAM Tasks:" << std::endl;
        for(int i=0; i<DRAM_tasks.size(); ++i) {
            os << std::setw(indent) << "" << "    (" << i+1 << ") Name: " << DRAM_tasks[i].name << std::endl;
            os << std::setw(indent) << "" << "        Is Write: " << (DRAM_tasks[i].is_write ? "true" : "false") << std::endl;
            os << std::setw(indent) << "" << "        Access Rank: " << DRAM_tasks[i].access_rank() << std::endl;
            os << std::setw(indent) << "" << "        Access Base: [";
            for(size_t dim = 0; dim < DRAM_tasks[i].access_base.size(); ++dim) {
                if(dim > 0) os << ", ";
                os << DRAM_tasks[i].access_base[dim];
            }
            os << "]" << std::endl;
            os << std::setw(indent) << "" << "        Access Extent: [";
            for(size_t dim = 0; dim < DRAM_tasks[i].access_extent.size(); ++dim) {
                if(dim > 0) os << ", ";
                os << DRAM_tasks[i].access_extent[dim];
            }
            os << "]" << std::endl;
            os << std::setw(indent) << "" << "        Access Iter Stride to Add: [";
            for(size_t dim = 0; dim < DRAM_tasks[i].access_stride_add.size(); ++dim) {
                if(dim > 0) os << ", ";
                os << DRAM_tasks[i].access_stride_add[dim];
            }
            os << "]" << std::endl;
            os << std::setw(indent) << "" << "        Offset to Add: [";
            for(size_t dim = 0; dim < DRAM_tasks[i].access_offset_add.size(); ++dim) {
                if(dim > 0) os << ", ";
                os << DRAM_tasks[i].access_offset_add[dim];
            }
            os << "]" << std::endl;
            os << std::setw(indent) << "" << "        Init Iteration to Access: " << DRAM_tasks[i].init_iter << std::endl;
            os << std::setw(indent) << "" << "        Iter Stride to Access: " << DRAM_tasks[i].stride_iter << std::endl;
            os << std::setw(indent) << "" << "        Total Access Iteration Count: " << DRAM_tasks[i].total_iter << std::endl;
        }
    }
};


}


#endif // _ATLASIM_COMPUTATION_TASK_H
