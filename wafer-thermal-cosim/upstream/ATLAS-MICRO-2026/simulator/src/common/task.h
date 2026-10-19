#ifndef _ATLASIM_TASK_H
#define _ATLASIM_TASK_H

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>
// atlasim include
#include "common/operator_task.h"
#include "common/computation_task.h"
#include "common/communication_task.h"
#include "common/general_task.h"


namespace atlasim {


struct OperatorTaskFactory {
    static bool is_computation(std::string type) {
        return type == optype2str.at(OperatorType::GEMM) ||
               type == optype2str.at(OperatorType::DecodeAttention);
    }

    static bool is_communication(std::string type) {
        return type == optype2str.at(OperatorType::Communication);
    }

    static bool is_general(std::string type) {
        return type == optype2str.at(OperatorType::SPMDGeneral) ||
               type == optype2str.at(OperatorType::MPMDGeneral);
    }

    static std::shared_ptr<OperatorTask> create(const YAML::Node config, int core_num = 0) {
        std::string type = config["type"].as<std::string>();
        if(is_computation(type)) {
            return std::make_shared<ComputationTask>(config);
        }
        if(is_communication(type)) {
            return std::make_shared<CommunicationTask>(config);
        }
        if(is_general(type)) {
            return std::make_shared<GeneralTask>(config, core_num);
        }

        std::cerr << "Find unsupported operator type: " << type << std::endl;
        exit(-1);
    }
};


}


#endif // _ATLASIM_TASK_H
