#ifndef _ATLASIM_COMMUNICATION_TASK_H
#define _ATLASIM_COMMUNICATION_TASK_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/component_task.h"
#include "common/operator_task.h"


namespace atlasim {


// Describe one NoC iteration's send (Tx) tasks
struct NoCTxTask {
    std::vector<BufferTask> buffer_load_tasks;
    std::vector<NoCTask> noc_tasks;

    NoCTxTask() { }
    NoCTxTask(const YAML::Node config) {
        try {
            YAML::Node buffer_load_config = config["buffer_load"];
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

        try {
            YAML::Node noc_task_config = config["noc"];
            for(int i=0; i<noc_task_config.size(); ++i) {
                noc_tasks.emplace_back(noc_task_config[i]);
                if(!noc_tasks[i].is_send) {
                    std::cerr << "NoC send (Tx) task should set is_send to true." << std::endl;
                    exit(-1);
                }
            }
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
};


// Describe one NoC iteration's receive (Rx) tasks
struct NoCRxTask {
    std::vector<VectorTask> vector_tasks;
    std::vector<BufferTask> buffer_load_tasks;
    std::vector<BufferTask> buffer_store_tasks;
    std::vector<NoCTask> noc_tasks;

    NoCRxTask() { }
    NoCRxTask(const YAML::Node config) {
        try {
            YAML::Node vector_task_config = config["vector"];
            for(int i=0; i<vector_task_config.size(); ++i) {
                vector_tasks.emplace_back(vector_task_config[i]);
            }
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }

        try {
            YAML::Node buffer_load_config = config["buffer_load"];
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

        try {
            YAML::Node buffer_store_config = config["buffer_store"];
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

        try {
            YAML::Node noc_task_config = config["noc"];
            for(int i=0; i<noc_task_config.size(); ++i) {
                noc_tasks.emplace_back(noc_task_config[i]);
                if(noc_tasks[i].is_send) {
                    std::cerr << "NoC receive (Rx) task should set is_send to true." << std::endl;
                    exit(-1);
                }
            }
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
};


struct CoreNoCTask {
    std::vector<NoCTxTask> per_iter_tx_tasks;
    std::vector<NoCRxTask> per_iter_rx_tasks;
    std::vector<DRAMTask> DRAM_tasks;

    CoreNoCTask() { }
    CoreNoCTask(const YAML::Node config) {
        YAML::Node on_chip_config = config["communication"]["on_chip"];
        try {
            for(int i=0; i<on_chip_config.size(); ++i) {
                YAML::Node tmp_tx_config = on_chip_config[i]["tx"];
                per_iter_tx_tasks.emplace_back(tmp_tx_config);
                YAML::Node tmp_rx_config = on_chip_config[i]["rx"];
                per_iter_rx_tasks.emplace_back(tmp_rx_config);
            }
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }

        YAML::Node dram_config = config["communication"]["dram"];
        try {
            for(int i=0; i<dram_config.size(); ++i) {
                DRAM_tasks.emplace_back(dram_config[i]);
            }
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
};


struct CommunicationTask: OperatorTask {
    // Only for NoC operators
    // Since NoC adopts MPMD programming model, we need to maintain different objects for each core
    std::vector<CoreNoCTask> core_noc_tasks;

    CommunicationTask() { }
    CommunicationTask(const YAML::Node config): OperatorTask(config) {
        std::string communication_file_prefix = config["file_prefix"].as<std::string>();
        int core_num = config["core_num"].as<int>();
        for(int i=0; i<core_num; ++i) {
            std::string config_file = communication_file_prefix + "/core_" + std::to_string(i) + ".yaml";
            YAML::Node config = YAML::LoadFile(config_file);
            core_noc_tasks.emplace_back(config);
        }
    }

    void display_stats(std::ostream& os=std::cout, int indent=0) override {
        os << std::setw(indent) << "" << "Name: " << name << std::endl;
        os << std::setw(indent) << "" << "Type: " << optype2str.at(type) << std::endl;
        os << std::setw(indent) << "" << "Loop Iteration Count: " << iteration << std::endl;
        os << std::setw(indent) << "" << "Per PE Execution Tasks:" << std::endl;

        for(int i=0; i<core_noc_tasks.size(); ++i) {
            os << std::setw(indent) << "" << "    Core (" << i+1 << "):" << std::endl;

            os << std::setw(indent) << "" << "    Per Iteration NoC Tasks:" << std::endl;
            for(int j=0; j<iteration; ++j) {
                os << std::setw(indent) << "" << "        (" << j+1 << ") Tx:" << std::endl;
                os << std::setw(indent) << "" << "            Buffer Load Tasks:" << std::endl;
                for(int k=0; k<core_noc_tasks[i].per_iter_tx_tasks[j].buffer_load_tasks.size(); ++k) {
                    os << std::setw(indent) << "" << "                (" << k+1 << ") Name: " << core_noc_tasks[i].per_iter_tx_tasks[j].buffer_load_tasks[k].name << std::endl;
                    os << std::setw(indent) << "" << "                    Byte Count: " << core_noc_tasks[i].per_iter_tx_tasks[j].buffer_load_tasks[k].byte_count << std::endl;
                }
                os << std::setw(indent) << "" << "            NoC Send Tasks:" << std::endl;
                for(int k=0; k<core_noc_tasks[i].per_iter_tx_tasks[j].noc_tasks.size(); ++k) {
                    os << std::setw(indent) << "" << "                (" << k+1 << ") Name: " << core_noc_tasks[i].per_iter_tx_tasks[j].noc_tasks[k].name << std::endl;
                    os << std::setw(indent) << "" << "                    Is Send: " << (core_noc_tasks[i].per_iter_tx_tasks[j].noc_tasks[k].is_send ? "true" : "false") << std::endl;
                    os << std::setw(indent) << "" << "                    Src: " << core_noc_tasks[i].per_iter_tx_tasks[j].noc_tasks[k].src << std::endl;
                    os << std::setw(indent) << "" << "                    Dst: " << core_noc_tasks[i].per_iter_tx_tasks[j].noc_tasks[k].dst << std::endl;
                    os << std::setw(indent) << "" << "                    Flit Num: " << core_noc_tasks[i].per_iter_tx_tasks[j].noc_tasks[k].flit_num << std::endl;
                    os << std::setw(indent) << "" << "                    Init Packet Id: " << core_noc_tasks[i].per_iter_tx_tasks[j].noc_tasks[k].init_flit_id << std::endl;
                }

                os << std::setw(indent) << "" << "            Rx:" << std::endl;
                os << std::setw(indent) << "" << "            Vector Tasks:" << std::endl;
                for(int k=0; k<core_noc_tasks[i].per_iter_rx_tasks[j].vector_tasks.size(); ++k) {
                    os << std::setw(indent) << "" << "                (" << k+1 << ") Name: " << core_noc_tasks[i].per_iter_rx_tasks[j].vector_tasks[k].name << std::endl;
                    os << std::setw(indent) << "" << "                    Vec Count: " << core_noc_tasks[i].per_iter_rx_tasks[j].vector_tasks[k].vec_count << std::endl;
                }
                os << std::setw(indent) << "" << "            Buffer Load Tasks:" << std::endl;
                for(int k=0; k<core_noc_tasks[i].per_iter_rx_tasks[j].buffer_load_tasks.size(); ++k) {
                    os << std::setw(indent) << "" << "                (" << k+1 << ") Name: " << core_noc_tasks[i].per_iter_rx_tasks[j].buffer_load_tasks[k].name << std::endl;
                    os << std::setw(indent) << "" << "                    Byte Count: " << core_noc_tasks[i].per_iter_rx_tasks[j].buffer_load_tasks[k].byte_count << std::endl;
                }
                os << std::setw(indent) << "" << "            Buffer Store Tasks:" << std::endl;
                for(int k=0; k<core_noc_tasks[i].per_iter_rx_tasks[j].buffer_store_tasks.size(); ++k) {
                    os << std::setw(indent) << "" << "                (" << k+1 << ") Name: " << core_noc_tasks[i].per_iter_rx_tasks[j].buffer_store_tasks[k].name << std::endl;
                    os << std::setw(indent) << "" << "                    Byte Count: " << core_noc_tasks[i].per_iter_rx_tasks[j].buffer_store_tasks[k].byte_count << std::endl;
                }
                os << std::setw(indent) << "" << "            NoC Receive Tasks:" << std::endl;
                for(int k=0; k<core_noc_tasks[i].per_iter_rx_tasks[j].noc_tasks.size(); ++k) {
                    os << std::setw(indent) << "" << "                (" << k+1 << ") Name: " << core_noc_tasks[i].per_iter_rx_tasks[j].noc_tasks[k].name << std::endl;
                    os << std::setw(indent) << "" << "                    Is Send: " << (core_noc_tasks[i].per_iter_rx_tasks[j].noc_tasks[k].is_send ? "true" : "false") << std::endl;
                    os << std::setw(indent) << "" << "                    Src: " << core_noc_tasks[i].per_iter_rx_tasks[j].noc_tasks[k].src << std::endl;
                    os << std::setw(indent) << "" << "                    Dst: " << core_noc_tasks[i].per_iter_rx_tasks[j].noc_tasks[k].dst << std::endl;
                    os << std::setw(indent) << "" << "                    Flit Num: " << core_noc_tasks[i].per_iter_rx_tasks[j].noc_tasks[k].flit_num << std::endl;
                    os << std::setw(indent) << "" << "                    Init Packet Id: " << core_noc_tasks[i].per_iter_rx_tasks[j].noc_tasks[k].init_flit_id << std::endl;
                }
            }

            os << std::setw(indent) << "" << "    DRAM Tasks:" << std::endl;
            for(int j=0; j<core_noc_tasks[i].DRAM_tasks.size(); ++j) {
                os << std::setw(indent) << "" << "        (" << j+1 << ") Name: " << core_noc_tasks[i].DRAM_tasks[j].name << std::endl;
                os << std::setw(indent) << "" << "            Is Write: " << (core_noc_tasks[i].DRAM_tasks[j].is_write ? "true" : "false") << std::endl;
                os << std::setw(indent) << "" << "            ######## General Access" << std::endl;
                os << std::setw(indent) << "" << "            Access Rank: " << core_noc_tasks[i].DRAM_tasks[j].access_rank() << std::endl;
                os << std::setw(indent) << "" << "            Access Base: [";
                for(size_t dim = 0; dim < core_noc_tasks[i].DRAM_tasks[j].access_base.size(); ++dim) {
                    if(dim > 0) os << ", ";
                    os << core_noc_tasks[i].DRAM_tasks[j].access_base[dim];
                }
                os << "]" << std::endl;
                os << std::setw(indent) << "" << "            Access Extent: [";
                for(size_t dim = 0; dim < core_noc_tasks[i].DRAM_tasks[j].access_extent.size(); ++dim) {
                    if(dim > 0) os << ", ";
                    os << core_noc_tasks[i].DRAM_tasks[j].access_extent[dim];
                }
                os << "]" << std::endl;
                os << std::setw(indent) << "" << "            Access Iter Stride to Add: [";
                for(size_t dim = 0; dim < core_noc_tasks[i].DRAM_tasks[j].access_stride_add.size(); ++dim) {
                    if(dim > 0) os << ", ";
                    os << core_noc_tasks[i].DRAM_tasks[j].access_stride_add[dim];
                }
                os << "]" << std::endl;
                os << std::setw(indent) << "" << "            Offset to Add: [";
                for(size_t dim = 0; dim < core_noc_tasks[i].DRAM_tasks[j].access_offset_add.size(); ++dim) {
                    if(dim > 0) os << ", ";
                    os << core_noc_tasks[i].DRAM_tasks[j].access_offset_add[dim];
                }
                os << "]" << std::endl;
                os << std::setw(indent) << "" << "            Init Iteration to Access: " << core_noc_tasks[i].DRAM_tasks[j].init_iter << std::endl;
                os << std::setw(indent) << "" << "            Iter Stride to Access: " << core_noc_tasks[i].DRAM_tasks[j].stride_iter << std::endl;
                os << std::setw(indent) << "" << "            Total Access Iteration Count: " << core_noc_tasks[i].DRAM_tasks[j].total_iter << std::endl;
            }
        }
    }
};


}


#endif // _ATLASIM_COMMUNICATION_TASK_H
