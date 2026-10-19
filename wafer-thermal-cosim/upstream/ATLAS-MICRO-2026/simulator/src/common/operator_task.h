#ifndef _ATLASIM_OPERATOR_TASK_H
#define _ATLASIM_OPERATOR_TASK_H

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>


namespace atlasim {


enum OperatorType {
    GEMM,
    DecodeAttention,
    Communication,
    SPMDGeneral,
    MPMDGeneral,
};

static const std::unordered_map<std::string, OperatorType> str2optype = {
    {"gemm", OperatorType::GEMM},
    {"decode-attention", OperatorType::DecodeAttention},
    {"communication", OperatorType::Communication},
    {"spmd-general", OperatorType::SPMDGeneral},
    {"mpmd-general", OperatorType::MPMDGeneral},
};

static const std::unordered_map<OperatorType, std::string> optype2str = {
    {OperatorType::GEMM, "gemm"},
    {OperatorType::DecodeAttention, "decode-attention"},
    {OperatorType::Communication, "communication"},
    {OperatorType::SPMDGeneral, "spmd-general"},
    {OperatorType::MPMDGeneral, "mpmd-general"},
};


struct OperatorTask {
    std::string name;
    OperatorType type;
    int iteration;

    OperatorTask() { }
    OperatorTask(const YAML::Node config) {
        name = config["name"].as<std::string>();
        type = str2optype.at(config["type"].as<std::string>());
        iteration = config["iteration"].as<int>();
    }

    virtual void display_stats(std::ostream& os=std::cout, int indent=0) = 0;
};


typedef std::shared_ptr<std::vector<std::shared_ptr<OperatorTask>>> OperatorList;


}


#endif // _ATLASIM_OPERATOR_TASK_H
