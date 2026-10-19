#include <memory>
#include <vector>
#include <fstream>
#include <filesystem>
// yaml-cpp include
#include <yaml-cpp/yaml.h>
// booksim include
#include "include/booksim_config.hpp"
// atlasim include
#include "chip/chip.h"
#include "common/arch.h"


namespace atlasim {


Chip::Chip(
    std::string _arch_config_path,
    std::string _operator_list_path,
    std::string _placement_map_path,
    std::string _log_file_path
) : arch_config_path(_arch_config_path),
    operator_list_path(_operator_list_path),
    placement_map_path(_placement_map_path) {

    YAML::Node arch_config_node = YAML::LoadFile(arch_config_path);
    arch_config = std::make_shared<ChipConfig>(arch_config_node);

    // Initialize operator list
    YAML::Node operator_list_node = YAML::LoadFile(operator_list_path);
    if(!operator_list_node["operator"].IsSequence()) {
        std::cerr << "Operator list should be a sequence of all operators' description." << std::endl;
        exit(-1);
    }
    std::cout << "Operator list " << _operator_list_path << " has " << operator_list_node["operator"].size() << " operators." << std::endl;
    operator_list = std::make_shared<std::vector<std::shared_ptr<OperatorTask>>>();
    for(int i=0; i<operator_list_node["operator"].size(); ++i) {
        YAML::Node operator_node = operator_list_node["operator"][i];
        operator_list->push_back(OperatorTaskFactory::create(operator_node, arch_config->core_num));
    }

    // Initialize placement map
    YAML::Node placement_map_node = YAML::LoadFile(placement_map_path);
    placement_maps = parse_placement_map_collection(placement_map_node, arch_config->core_num);
    if(placement_map_node["core_tensor"]) {
        std::cout << "Placement map " << _placement_map_path << " has per-core tensor placements for "
                  << arch_config->core_num << " cores." << std::endl;
    }
    else {
        std::cout << "Placement map " << _placement_map_path << " has "
                  << placement_map_node["tensor"].size() << " tensors." << std::endl;
    }

    // Extract booksim config
    int threshold = 0;
    if(arch_config->has_noc) {
        std::string booksim_config_path = arch_config->noc_config.config_path;
        booksim_config.ParseFile(booksim_config_path);
        threshold = booksim_config.GetInt("threshold");

        // Check configuration consistency
        int k = booksim_config.GetInt("k");
        int n = booksim_config.GetInt("n");
        bool config_match = (int)std::pow(k, n) == arch_config->core_num;
        if(!config_match) {
            std::cerr << "Booksim configuration does not match the architecture configuration:" << std::endl;
            std::cerr << "    Booksim configuration: k=" << k << ", n=" << n << std::endl;
            std::cerr << "    Architecture configuration: core_num=" << arch_config->core_num << std::endl;
            exit(-1);
        }
    }

    // Initialize interface classes between PEs and NoC wrapper
    typedef std::deque<Packet> T;
    send_queues = std::make_shared<std::vector<CNInterface> >();
    received_queues = std::make_shared<std::vector<CNInterface> >();
    credit_board = std::make_shared<std::vector<bool> >();
    for(int i=0; i<arch_config->core_num; ++i) {
        send_queues->push_back(std::make_shared<T>());
        received_queues->push_back(std::make_shared<T>());
        credit_board->push_back(true);
    }

    // Redirect the standard output stream into log file
    if(_log_file_path != "") {
        cout_buf = std::cout.rdbuf();
        cerr_buf = std::cerr.rdbuf();
        log_file = std::make_shared<std::ofstream>(_log_file_path, std::ios::out);
        std::cout.rdbuf(log_file->rdbuf());
        std::cerr.rdbuf(log_file->rdbuf());
    }
    else {
        log_file = nullptr;
    }

    // Initialize Components
    if(arch_config->has_noc) {
        noc = std::make_shared<NoCWrapper>(
            booksim_config,
            send_queues,
            received_queues
        );
    }
    else {
        noc = nullptr;
    }
    core_array = std::make_shared<CoreArray>(
        arch_config, operator_list, placement_maps,
        threshold,
        send_queues, received_queues, credit_board
    );

    // Initialize clocks
    prev_clk.clear();
    clk = 0;
}


Chip::~Chip() {
    if(cout_buf) {
        std::cout.rdbuf(cout_buf);
    }
    if(cerr_buf) {
        std::cerr.rdbuf(cerr_buf);
    }

    // Explicit destruction order to avoid crashes
    
    // 1. Destroy CoreArray first. 
    // This stops cores from running and releasing any internal resources.
    // DEBUG: Reset CoreArray
    // std::cerr << "Chip Destructor: Reset CoreArray..." << std::endl;
    core_array.reset();

    // 2. Clear shared queues.
    // These queues might contain Packets which might contain Flit pointers.
    // We must clear them BEFORE destroying NoC (which destroys the Flit pool).
    // DEBUG: Clear queues
    // std::cerr << "Chip Destructor: Clear Queues..." << std::endl;
    if (send_queues) {
        for (auto& q : *send_queues) {
            if (q) q->clear();
        }
        send_queues->clear(); // Clear the vector itself
    }
    if (received_queues) {
        for (auto& q : *received_queues) {
            if (q) q->clear();
        }
        received_queues->clear(); // Clear the vector itself
    }

    // 3. Destroy NoCWrapper.
    // This triggers TrafficManager destruction and Flit::FreeAll().
    // DEBUG: Reset NoC
    // std::cerr << "Chip Destructor: Reset NoC..." << std::endl;
    noc.reset();
}


void Chip::reset_execution_status(std::string _operator_list_path) {
    if(std::filesystem::exists(_operator_list_path)) {
        std::cout << "Reset operator list from " << operator_list_path << " to " << _operator_list_path << std::endl;
        // Reset operator list
        operator_list_path = _operator_list_path;
        YAML::Node operator_list_node = YAML::LoadFile(operator_list_path);
        if(!operator_list_node["operator"].IsSequence()) {
            std::cerr << "Operator list should be a sequence of all operators' description." << std::endl;
            exit(-1);
        }
        std::cout << "Operator list " << operator_list_path << " has " << operator_list_node["operator"].size() << " operators." << std::endl;
        operator_list->clear();
        for(int i=0; i<operator_list_node["operator"].size(); ++i) {
            YAML::Node operator_node = operator_list_node["operator"][i];
            operator_list->push_back(OperatorTaskFactory::create(operator_node, arch_config->core_num));
        }
    }

    prev_clk.emplace_back(clk);
    core_array->reset_execution_status();
    if(noc) {
        noc->reset_execution_status();
    }
    // Clear history to ensure the first check in the new run doesn't match stale state
    last_queues_state = {};
}


Performance Chip::simulate() {
    reset_execution_status();

    int check_frequency = arch_config->has_noc ? booksim_config.GetInt("deadlock_check_freq") : 10000;
    while(!is_finished()) {
        tick();

        if(clk % check_frequency == 0) {
            std::cout << "Simulate " << clk << " cycles" << std::endl;
            if (check_deadlock()) {
                std::cerr << "Deadlock detected: the chip state keeps unchanged over " << check_frequency << " cycles" << std::endl;
            }
        }
    }

    return core_array->get_performance();
}


void Chip::tick() {
    clk++;
    core_array->tick();
    if(noc) {
        noc->tick();
    }
}


bool Chip::is_finished() {
    return core_array->all_core_finished() && (!noc || noc->traffic_drained());
}


bool Chip::check_deadlock() {
    auto create_snapshot = [](const PCNInterfaceSet& queues) -> QueueSnapshot {
        QueueSnapshot snapshot;
        if (queues) {
            for (const auto& q_ptr : *queues) {
                if (q_ptr) {
                    snapshot.push_back(*q_ptr);
                } else {
                    snapshot.push_back(std::deque<Packet>());
                }
            }
        }
        return snapshot;
    };

    QueueSnapshot current_sq = create_snapshot(send_queues);
    QueueSnapshot current_rq = create_snapshot(received_queues);

    bool queues_empty = true;
    for (const auto& q : current_sq) {
        if (!q.empty()) {
            queues_empty = false;
            break;
        }
    }
    if (queues_empty) {
        for (const auto& q : current_rq) {
            if (!q.empty()) {
                queues_empty = false;
                break;
            }
        }
    }
    
    bool deadlock = false;
    deadlock |= !core_array->state_changed();
    if (!queues_empty) {
        deadlock |= (last_queues_state.first == current_sq) && (last_queues_state.second == current_rq);
    }

    last_queues_state.first = current_sq;
    last_queues_state.second = current_rq;

    return deadlock;
}


void Chip::display_stats(std::ostream& os) {
    os << std::endl << " ================== Dumped Stats ================== " << std::endl;
    core_array->display_stats(os);
    if(noc) {
        noc->display_stats(os);
    }
    os << std::endl << " ================================================== " << std::endl;
}


}
